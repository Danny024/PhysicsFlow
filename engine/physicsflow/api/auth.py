"""
PhysicsFlow REST API — Authentication.

Single API-key scheme via X-API-Key request header.

When PHYSICSFLOW_REST_API_KEY is empty (default for single-user local installs)
every request passes through without a key check.  In team/server mode set the
env var and require all clients to supply the header.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request,
    x_api_key: str | None = Security(_api_key_scheme),
) -> None:
    """
    FastAPI dependency — inject into any route that requires authentication.

    Usage:
        @router.get("/sensitive", dependencies=[Depends(require_api_key)])
        async def sensitive_endpoint(): ...
    """
    cfg = request.app.state.cfg
    expected = cfg.rest_api_key

    if not expected:
        # Auth disabled — single-user local mode
        return

    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-Key header.",
        )
