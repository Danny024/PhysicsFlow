"""
/api/v1/training — PINO surrogate training submission and monitoring.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, Request

from ..auth import require_api_key
from ..schemas import JobSubmittedResponse, StatusResponse, TrainingStartRequest

router = APIRouter(prefix="/training", tags=["training"])
_AUTH = [Depends(require_api_key)]


@router.post("/start", response_model=JobSubmittedResponse, dependencies=_AUTH)
async def start_training(request: Request, body: TrainingStartRequest):
    """
    Submit a PINO training job.  Returns a run_id immediately.
    Training executes in a background thread and records per-epoch metrics
    in the database (queryable via GET /api/v1/runs/{run_id}/epochs).
    """
    db_svc  = request.app.state.db_svc
    cfg     = request.app.state.cfg

    run_id = db_svc.start_run(
        body.project_id, "training",
        config=body.model_dump(),
    )

    def _train():
        try:
            from physicsflow.training.pretrain_norne import PretrainConfig, pretrain_norne
            pcfg = PretrainConfig(
                epochs=body.epochs,
                batch_size=body.batch_size,
                learning_rate=body.learning_rate,
                w_pde=body.w_pde, w_data=body.w_data,
                w_well=body.w_well, w_ic=body.w_ic, w_bc=body.w_bc,
                device=body.device,
                output_dir=str(cfg.models_dir),
            )
            pretrain_norne(pcfg, project_id=body.project_id, run_id=run_id)
            db_svc.complete_run(run_id)
        except Exception as exc:
            db_svc.fail_run(run_id, str(exc))

    threading.Thread(target=_train, daemon=True, name=f"train-{run_id[:8]}").start()

    return JobSubmittedResponse(
        run_id=run_id,
        status="queued",
        message=f"Training queued for {body.epochs} epochs on device={body.device}.",
    )


@router.get("/status", dependencies=_AUTH)
async def training_status(request: Request):
    """Live training state from the shared ReservoirContextProvider."""
    context = request.app.state.context
    return {
        "training_active": getattr(context, "training_active", False),
        "current_epoch":   getattr(context, "current_epoch", 0),
        "total_epochs":    getattr(context, "total_epochs", 0),
        "current_loss":    getattr(context, "current_loss", None),
        "loss_pde":        getattr(context, "current_loss_pde", None),
        "loss_data":       getattr(context, "current_loss_data", None),
    }


@router.post("/stop", response_model=StatusResponse, dependencies=_AUTH)
async def stop_training(request: Request):
    """Signal an active training run to stop after the current epoch."""
    context = request.app.state.context
    if hasattr(context, "request_training_stop"):
        context.request_training_stop()
        return StatusResponse(status="stop_requested",
                              message="Training will stop after the current epoch.")
    return StatusResponse(status="no_op", message="No active training run.")
