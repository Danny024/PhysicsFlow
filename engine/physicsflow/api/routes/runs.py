"""
/api/v1/runs — Simulation run history and epoch loss curves.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..auth import require_api_key
from ..schemas import RunListResponse, RunSchema, TrainingEpochSchema

router = APIRouter(prefix="/runs", tags=["runs"])
_AUTH = [Depends(require_api_key)]


@router.get("", response_model=RunListResponse, dependencies=_AUTH)
async def list_runs(
    request: Request,
    project_id: str | None = Query(default=None),
    run_type: str | None = Query(default=None, description="'training' | 'hm' | 'forward'"),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, le=500),
):
    """List simulation runs with optional filters."""
    db_svc = request.app.state.db_svc
    runs = db_svc.list_runs(
        project_id=project_id, run_type=run_type, status=status, limit=limit
    )
    return RunListResponse(
        runs=[RunSchema.model_validate(r) for r in runs],
        total=len(runs),
    )


@router.get("/{run_id}", response_model=RunSchema, dependencies=_AUTH)
async def get_run(request: Request, run_id: str):
    """Get details for a single run."""
    db_svc = request.app.state.db_svc
    run = db_svc.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return RunSchema.model_validate(run)


@router.get("/{run_id}/epochs", response_model=list[TrainingEpochSchema],
            dependencies=_AUTH)
async def run_epoch_history(request: Request, run_id: str):
    """
    Return per-epoch loss history for a training run.
    Useful for plotting live loss curves in Jupyter.
    """
    db_svc = request.app.state.db_svc
    epochs = db_svc.get_epoch_history(run_id)
    return [TrainingEpochSchema.model_validate(e) for e in epochs]
