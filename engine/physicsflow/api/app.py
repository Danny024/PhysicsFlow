"""
PhysicsFlow REST API — application factory (v2.0).

create_rest_app(cfg, context, db_svc) builds the FastAPI app, wires up all
routers under /api/v1, and configures CORS + startup/shutdown hooks.

The function is called once from server.py which launches Uvicorn in a daemon
thread alongside the existing gRPC server.  Both the gRPC servicers and the
REST handlers share the same ReservoirContextProvider Python object — zero
serialisation overhead.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from ..agent.context_provider import ReservoirContextProvider
    from ..config import EngineConfig
    from ..db.db_service import DatabaseService

log = logging.getLogger(__name__)

# ── Router imports ────────────────────────────────────────────────────────────

from .routes.health      import router as health_router
from .routes.projects    import router as projects_router
from .routes.runs        import router as runs_router
from .routes.simulation  import router as simulation_router
from .routes.training    import router as training_router
from .routes.history_matching import router as hm_router
from .routes.models      import router as models_router
from .routes.io          import router as io_router
from .routes.agent       import router as agent_router
from .routes.tnavigator  import router as tnav_router


# ── Factory ───────────────────────────────────────────────────────────────────

def create_rest_app(
    cfg: "EngineConfig",
    context: "ReservoirContextProvider",
    db_svc: "DatabaseService",
) -> FastAPI:
    """
    Build the FastAPI application.

    Args:
        cfg:     Shared EngineConfig (from config.py).
        context: Shared ReservoirContextProvider (same object used by gRPC).
        db_svc:  Shared DatabaseService singleton.

    Returns:
        Configured FastAPI instance ready to be passed to Uvicorn.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # Startup
        db_svc.audit(
            "rest_api.started",
            f"REST API started on {cfg.rest_host}:{cfg.rest_port}",
        )
        log.info("PhysicsFlow REST API ready — http://%s:%s/docs",
                 cfg.rest_host, cfg.rest_port)
        yield
        # Shutdown
        db_svc.audit("rest_api.stopped", "REST API shutting down")
        log.info("PhysicsFlow REST API stopped.")

    app = FastAPI(
        title="PhysicsFlow Engine API",
        version="2.0.0",
        description=(
            "On-premise REST API for PhysicsFlow reservoir simulation engine. "
            "Provides programmatic access to PINO simulation, PINO training, "
            "αREKI history matching, and the AI assistant."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    # ── Inject shared state ───────────────────────────────────────────────────
    app.state.cfg     = cfg
    app.state.context = context
    app.state.db_svc  = db_svc
    app.state._agent  = None   # lazy-initialised by agent router on first use

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Allow Jupyter (8888), React dev server (3000), and any custom origins
    # specified in cfg.rest_cors_origins.
    origins = list(cfg.rest_cors_origins) or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    PREFIX = "/api/v1"

    # Health — no auth, no prefix beyond /api/v1
    app.include_router(health_router,     prefix=PREFIX)

    # Core domain routers
    app.include_router(projects_router,   prefix=PREFIX)
    app.include_router(runs_router,       prefix=PREFIX)
    app.include_router(simulation_router, prefix=PREFIX)
    app.include_router(training_router,   prefix=PREFIX)
    app.include_router(hm_router,         prefix=PREFIX)
    app.include_router(models_router,     prefix=PREFIX)
    app.include_router(io_router,         prefix=PREFIX)
    app.include_router(agent_router,      prefix=PREFIX)
    app.include_router(tnav_router,       prefix=PREFIX)

    return app


# ── Stand-alone entry point ───────────────────────────────────────────────────

def run_standalone(port: int = 8000, host: str = "0.0.0.0") -> None:
    """
    Launch the REST API without the gRPC server.

    Useful for running just the REST API in tests or Docker health-check
    environments.  All services are initialised with defaults.
    """
    import uvicorn

    from ..agent.context_provider import ReservoirContextProvider
    from ..config import EngineConfig
    from ..db.db_service import DatabaseService

    cfg     = EngineConfig()
    context = ReservoirContextProvider()
    db_svc  = DatabaseService.instance()

    app = create_rest_app(cfg, context, db_svc)
    uvicorn.run(app, host=host, port=port, log_level="info")
