"""
/api/v1/simulation — Forward simulation submission and live state queries.
"""

from __future__ import annotations

import base64
import threading

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import require_api_key
from ..schemas import JobSubmittedResponse, SimulationRunRequest, StatusResponse

router = APIRouter(prefix="/simulation", tags=["simulation"])
_AUTH = [Depends(require_api_key)]


@router.post("/run", response_model=JobSubmittedResponse, dependencies=_AUTH)
async def submit_simulation(request: Request, body: SimulationRunRequest):
    """
    Submit a forward simulation run.  Returns a run_id immediately;
    the simulation executes asynchronously in a background thread.
    """
    db_svc  = request.app.state.db_svc
    context = request.app.state.context
    cfg     = request.app.state.cfg

    run_id = db_svc.start_run(
        body.project_id, "forward",
        config={"n_timesteps": body.n_timesteps, "use_surrogate": body.use_surrogate},
    )

    def _run():
        try:
            from physicsflow.services.simulation_service import _run_forward_surrogate
            _run_forward_surrogate(cfg, context, run_id, body.n_timesteps)
            db_svc.complete_run(run_id)
        except Exception as exc:
            db_svc.fail_run(run_id, str(exc))

    threading.Thread(target=_run, daemon=True, name=f"sim-{run_id[:8]}").start()

    return JobSubmittedResponse(
        run_id=run_id,
        status="queued",
        message=f"Forward run queued with {body.n_timesteps} timesteps.",
    )


@router.get("/status", dependencies=_AUTH)
async def simulation_status(request: Request):
    """
    Return live simulation state from the shared ReservoirContextProvider.
    Equivalent to the get_simulation_status agent tool.
    """
    context = request.app.state.context
    try:
        return {
            "status":          context.simulation_status,
            "progress":        context.simulation_progress,
            "current_epoch":   context.current_epoch,
            "total_epochs":    context.total_epochs,
            "current_loss":    context.current_loss,
            "model_type":      context.model_type,
            "training_active": context.training_active,
        }
    except AttributeError as e:
        raise HTTPException(status_code=503, detail=f"Context unavailable: {e}")


@router.get("/wells", dependencies=_AUTH)
async def all_well_performance(request: Request):
    """
    Return time-series performance data for every well in the current context.
    """
    context = request.app.state.context
    try:
        return {"wells": context.well_performance or {}}
    except AttributeError:
        return {"wells": {}}


@router.get("/wells/{well_name}", dependencies=_AUTH)
async def well_performance(request: Request, well_name: str):
    """Return time-series performance data for a single well."""
    context = request.app.state.context
    try:
        perf = (context.well_performance or {}).get(well_name)
    except AttributeError:
        perf = None

    if perf is None:
        raise HTTPException(
            status_code=404,
            detail=f"Well {well_name!r} not found in current simulation context.",
        )
    return {"well_name": well_name, "performance": perf}


@router.get("/field/{timestep}", dependencies=_AUTH)
async def field_snapshot(request: Request, timestep: int):
    """
    Return pressure and Sw field arrays at the given timestep as
    base64-encoded numpy byte strings (float32, C-order).

    Clients deserialise with:
        import numpy as np, base64
        arr = np.frombuffer(base64.b64decode(data), dtype=np.float32)
    """
    context = request.app.state.context
    try:
        fields = context.field_snapshots
        if fields is None or timestep >= len(fields):
            raise HTTPException(
                status_code=404,
                detail=f"Timestep {timestep} not available (n={len(fields) if fields else 0}).",
            )
        snap = fields[timestep]
        return {
            "timestep": timestep,
            "pressure_b64":  base64.b64encode(snap["pressure"].astype("float32").tobytes()).decode(),
            "sw_b64":        base64.b64encode(snap["sw"].astype("float32").tobytes()).decode(),
            "shape":         list(snap["pressure"].shape),
        }
    except AttributeError:
        raise HTTPException(status_code=503, detail="Field snapshots not available.")
