"""
/api/v1/io — File upload and project export endpoints.

Supported uploads: .DATA / .EGRID / .UNRST (Eclipse), .LAS, .pfproj, .sim (tNavigator).
All uploads are written to cfg.projects_dir / "uploads" / {project_id} /
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ..auth import require_api_key
from ..schemas import StatusResponse

router = APIRouter(prefix="/io", tags=["io"])
_AUTH = [Depends(require_api_key)]

_ALLOWED_SUFFIXES = {
    ".data", ".egrid", ".unrst", ".unsmry", ".smspec",
    ".las", ".pfproj", ".pfproj.enc", ".sim", ".csv", ".txt",
}


def _upload_dir(cfg, project_id: str) -> Path:
    p = Path(cfg.projects_dir) / "uploads" / project_id
    p.mkdir(parents=True, exist_ok=True)
    return p


@router.post("/upload/{project_id}", response_model=StatusResponse,
             dependencies=_AUTH)
async def upload_file(
    request: Request,
    project_id: str,
    file: UploadFile = File(...),
):
    """
    Upload any supported reservoir file and save it under the project's upload directory.

    Returns the saved file path so the client can reference it in subsequent API calls.
    """
    cfg = request.app.state.cfg
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type {suffix!r} not supported. "
                f"Allowed: {sorted(_ALLOWED_SUFFIXES)}"
            ),
        )

    dest = _upload_dir(cfg, project_id) / (file.filename or "upload")
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return StatusResponse(
        status="uploaded",
        message=f"Saved to {dest}",
        data={"path": str(dest), "size_bytes": dest.stat().st_size},
    )


@router.post("/parse/eclipse/{project_id}", dependencies=_AUTH)
async def parse_eclipse(request: Request, project_id: str, deck_path: str):
    """
    Parse an Eclipse .DATA deck and return grid dimensions, well list, and PVT
    as a JSON object.  deck_path must be the server-side path returned by /upload.
    """
    from physicsflow.io.eclipse_reader import EclipseReader
    try:
        reader = EclipseReader(deck_path)
        return {
            "dims":  {"nx": reader.nx, "ny": reader.ny, "nz": reader.nz},
            "wells": reader.well_names() if hasattr(reader, "well_names") else [],
            "pvt":   {"initial_pressure_bar": reader.initial_pressure()
                       if hasattr(reader, "initial_pressure") else None},
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Eclipse parse error: {exc}")


@router.get("/export/pfproj/{project_id}", dependencies=_AUTH)
async def export_pfproj(request: Request, project_id: str):
    """
    Download the .pfproj project file for the given project.
    """
    db_svc = request.app.state.db_svc
    project = db_svc.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")

    import os
    file_path = project.pfproj_path
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Project file not on disk: {file_path!r}",
        )

    return FileResponse(
        path=file_path,
        media_type="application/json",
        filename=Path(file_path).name,
    )
