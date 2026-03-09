"""
/api/v1/tnav — tNavigator bridge endpoints (v2.0).

Allows Jupyter notebooks and CI pipelines to:
  1. Convert a .pfproj project to a tNavigator .sim ASCII deck.
  2. Import a tNavigator .sim deck and register it as a PhysicsFlow project.
  3. Run tNavigator as a subprocess and stream back its exit code + log tail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import require_api_key
from ..schemas import StatusResponse, tNavigatorImportResponse

router = APIRouter(prefix="/tnav", tags=["tnavigator"])
_AUTH = [Depends(require_api_key)]


@router.post("/import/{project_id}", response_model=tNavigatorImportResponse,
             dependencies=_AUTH)
async def import_sim(request: Request, project_id: str, sim_path: str):
    """
    Parse a tNavigator .sim deck from *sim_path* and register it as a
    PhysicsFlow project, linking it to *project_id*.

    Returns summary metadata extracted from the deck.
    """
    from physicsflow.io.tnavigator_bridge import TNavigatorBridge
    try:
        bridge  = TNavigatorBridge(sim_path)
        summary = bridge.to_summary()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {sim_path!r}")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Parse error: {exc}")

    db_svc = request.app.state.db_svc
    db_svc.audit(
        "tnav.import",
        f"Imported tNavigator deck: {sim_path}",
        project_id=project_id,
        metadata=summary,
    )

    return tNavigatorImportResponse(
        project_id=project_id,
        sim_path=sim_path,
        **summary,
    )


@router.get("/export/{project_id}", dependencies=_AUTH)
async def export_sim(request: Request, project_id: str):
    """
    Convert the .pfproj project to a tNavigator ASCII .sim deck.
    Returns the generated .sim content as plain text.
    """
    db_svc = request.app.state.db_svc
    project = db_svc.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")

    from physicsflow.io.tnavigator_bridge import TNavigatorBridge
    try:
        sim_text = TNavigatorBridge.from_pfproj(project.pfproj_path).to_sim()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export error: {exc}")

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=sim_text,
        media_type="text/plain",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{project_id}.sim"'
            )
        },
    )


@router.post("/run/{project_id}", response_model=StatusResponse, dependencies=_AUTH)
async def run_tnavigator(request: Request, project_id: str, sim_path: str):
    """
    Execute tNavigator on *sim_path* as a subprocess (blocking, max 10 min).

    Requires ``tnavigator_exe`` to be set in EngineConfig / environment.
    Returns exit code and last 50 lines of stdout/stderr.
    """
    cfg = request.app.state.cfg

    if not cfg.tnavigator_exe:
        raise HTTPException(
            status_code=503,
            detail=(
                "tNavigator executable not configured. "
                "Set PHYSICSFLOW_TNAVIGATOR_EXE in the environment or config."
            ),
        )

    if not Path(sim_path).exists():
        raise HTTPException(status_code=404, detail=f"Sim file not found: {sim_path!r}")

    env = {}
    if cfg.tnavigator_license_server:
        env["LM_LICENSE_FILE"] = cfg.tnavigator_license_server

    try:
        result = subprocess.run(
            [cfg.tnavigator_exe, sim_path],
            capture_output=True,
            text=True,
            timeout=600,
            env=env or None,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="tNavigator run timed out (>10 min).")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Subprocess error: {exc}")

    output_lines = (result.stdout + result.stderr).splitlines()
    tail = "\n".join(output_lines[-50:])

    db_svc = request.app.state.db_svc
    db_svc.audit(
        "tnav.run",
        f"tNavigator run finished (exit={result.returncode})",
        project_id=project_id,
        metadata={"exit_code": result.returncode, "sim_path": sim_path},
    )

    if result.returncode != 0:
        return StatusResponse(
            status="failed",
            message=f"tNavigator exited with code {result.returncode}",
            data={"exit_code": result.returncode, "log_tail": tail},
        )

    return StatusResponse(
        status="success",
        message="tNavigator run completed successfully.",
        data={"exit_code": 0, "log_tail": tail},
    )
