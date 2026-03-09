"""
/api/v1/models — Model version registry (list, activate, download).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ..auth import require_api_key
from ..schemas import ModelVersionSchema, StatusResponse

router = APIRouter(prefix="/models", tags=["models"])
_AUTH = [Depends(require_api_key)]


@router.get("/projects/{project_id}", response_model=list[ModelVersionSchema],
            dependencies=_AUTH)
async def list_models(request: Request, project_id: str):
    """List all model versions registered for a project, newest first."""
    db_svc = request.app.state.db_svc
    models = db_svc.list_models(project_id)
    return [ModelVersionSchema.model_validate(m) for m in models]


@router.get("/{model_id}", response_model=ModelVersionSchema, dependencies=_AUTH)
async def get_model(request: Request, model_id: str):
    """Get metadata for a single model version."""
    db_svc = request.app.state.db_svc
    model = db_svc.get_model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")
    return ModelVersionSchema.model_validate(model)


@router.post("/{model_id}/activate", response_model=StatusResponse,
             dependencies=_AUTH)
async def activate_model(request: Request, model_id: str):
    """
    Mark a model version as active (deactivates all others of the same type
    within the same project).
    """
    db_svc = request.app.state.db_svc
    ok = db_svc.activate_model(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")
    return StatusResponse(status="activated",
                          message=f"Model {model_id} is now active.")


@router.get("/{model_id}/download", dependencies=_AUTH)
async def download_model(request: Request, model_id: str):
    """
    Stream the model checkpoint file (.pt) to the client.
    Useful for retrieving a trained model from a remote engine server.
    """
    import os
    db_svc = request.app.state.db_svc
    model  = db_svc.get_model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")

    file_path = model.file_path
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found on disk: {file_path!r}",
        )

    filename = os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=filename,
    )
