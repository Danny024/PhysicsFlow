"""GET /api/v1/health — server health and configuration summary."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    """
    Return server health status, version, and backend configuration.

    This endpoint requires no authentication so the .NET desktop and
    external monitoring tools can poll it without a key.
    """
    from physicsflow.db.database import db_backend

    cfg = request.app.state.cfg
    return HealthResponse(
        status="ok",
        version="2.0.0",
        grpc_port=cfg.grpc_port,
        rest_port=cfg.rest_port,
        db_backend=db_backend(),
        team_mode=cfg.effective_team_mode(),
        ollama_model=cfg.default_llm_model,
    )
