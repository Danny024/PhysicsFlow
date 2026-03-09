"""
/api/v1/hm — History matching submission and iteration monitoring.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import require_api_key
from ..schemas import HMIterationSchema, HMStartRequest, JobSubmittedResponse, StatusResponse

router = APIRouter(prefix="/hm", tags=["history_matching"])
_AUTH = [Depends(require_api_key)]


@router.post("/start", response_model=JobSubmittedResponse, dependencies=_AUTH)
async def start_hm(request: Request, body: HMStartRequest):
    """
    Submit an αREKI history matching run.
    Returns a run_id immediately; HM executes in a background thread.
    Each iteration's mismatch, α, and P10/P50/P90 are streamed to the DB
    and visible via GET /api/v1/hm/{hm_run_id}/iterations.
    """
    db_svc  = request.app.state.db_svc
    context = request.app.state.context
    cfg     = request.app.state.cfg

    run_id = db_svc.new_hm_run_id(body.project_id)
    sim_run_id = db_svc.start_run(
        body.project_id, "hm",
        config=body.model_dump(),
        n_ensemble=body.n_ensemble,
    )

    def _hm():
        try:
            from physicsflow.services.hm_service import _run_areki
            _run_areki(cfg, context, db_svc, body.project_id, run_id,
                       sim_run_id, body)
            db_svc.complete_run(sim_run_id)
        except Exception as exc:
            db_svc.fail_run(sim_run_id, str(exc))

    threading.Thread(target=_hm, daemon=True, name=f"hm-{run_id[:8]}").start()

    return JobSubmittedResponse(
        run_id=run_id,
        status="queued",
        message=(
            f"αREKI started: {body.n_ensemble} members, "
            f"max {body.max_iterations} iterations."
        ),
    )


@router.get("/status", dependencies=_AUTH)
async def hm_status(request: Request):
    """Live αREKI state from the shared ReservoirContextProvider."""
    context = request.app.state.context
    return {
        "hm_active":      getattr(context, "hm_active", False),
        "current_iter":   getattr(context, "hm_current_iter", 0),
        "max_iter":       getattr(context, "hm_max_iter", 0),
        "current_mismatch": getattr(context, "hm_current_mismatch", None),
        "s_cumulative":   getattr(context, "hm_s_cumulative", None),
        "converged":      getattr(context, "hm_converged", False),
    }


@router.get("/{hm_run_id}/iterations",
            response_model=list[HMIterationSchema], dependencies=_AUTH)
async def hm_iterations(request: Request, hm_run_id: str, project_id: str):
    """Return all recorded iterations for a history matching run."""
    db_svc = request.app.state.db_svc
    iters = db_svc.get_hm_history(project_id=project_id, hm_run_id=hm_run_id)
    if not iters:
        raise HTTPException(
            status_code=404,
            detail=f"No iterations found for HM run {hm_run_id!r}.",
        )
    return [HMIterationSchema.model_validate(i) for i in iters]


@router.get("/{hm_run_id}/ensemble", dependencies=_AUTH)
async def hm_ensemble_summary(request: Request, hm_run_id: str, project_id: str):
    """
    Return P10/P50/P90 EUR evolution across all recorded iterations.
    Useful for plotting convergence fan charts in Jupyter.
    """
    db_svc = request.app.state.db_svc
    iters = db_svc.get_hm_history(project_id=project_id, hm_run_id=hm_run_id)
    return {
        "hm_run_id": hm_run_id,
        "iterations": [
            {
                "iteration": i.iteration,
                "mismatch":  i.mismatch,
                "eur_p10":   i.eur_p10,
                "eur_p50":   i.eur_p50,
                "eur_p90":   i.eur_p90,
            }
            for i in iters
        ],
    }


@router.post("/stop", response_model=StatusResponse, dependencies=_AUTH)
async def stop_hm(request: Request):
    """Signal the active αREKI run to stop after the current iteration."""
    context = request.app.state.context
    if hasattr(context, "request_hm_stop"):
        context.request_hm_stop()
        return StatusResponse(status="stop_requested",
                              message="HM will stop after current iteration.")
    return StatusResponse(status="no_op", message="No active HM run.")
