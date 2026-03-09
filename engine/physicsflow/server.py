"""
PhysicsFlow gRPC Server — main entry point.

Starts all gRPC services:
  - SimulationService   (PINO forward model)
  - TrainingService     (PINO training)
  - HistoryMatchingService (αREKI)
  - AgentService        (Ollama LLM assistant)

The .NET desktop application starts this process automatically via EngineManager.
"""

from __future__ import annotations
import signal
import sys
import threading
from concurrent import futures
from typing import NoReturn

import grpc
import click
from loguru import logger

from .agent.context_provider import ReservoirContextProvider
from .config import EngineConfig
from .services import (
    SimulationServicer, TrainingServicer,
    HistoryMatchingServicer, AgentServicer,
)


def create_server(
    port: int,
    max_workers: int,
    context: ReservoirContextProvider,
    cfg: EngineConfig | None = None,
) -> grpc.Server:
    """Build and configure the gRPC server with all services."""
    if cfg is None:
        cfg = EngineConfig()

    # Import generated stubs (produced by grpcio-tools at build time)
    try:
        from .proto import simulation_pb2_grpc, history_matching_pb2_grpc, agent_pb2_grpc
        stubs_available = True
    except ImportError:
        logger.warning(
            "gRPC stubs not generated yet. Run:\n"
            "  python -m grpc_tools.protoc -I physicsflow/proto "
            "--python_out=physicsflow/proto "
            "--grpc_python_out=physicsflow/proto "
            "physicsflow/proto/*.proto"
        )
        stubs_available = False

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length",    256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 5_000),
        ],
    )

    if stubs_available:
        simulation_pb2_grpc.add_SimulationServiceServicer_to_server(
            SimulationServicer(cfg, context), server
        )
        simulation_pb2_grpc.add_TrainingServiceServicer_to_server(
            TrainingServicer(cfg, context), server
        )
        history_matching_pb2_grpc.add_HistoryMatchingServiceServicer_to_server(
            HistoryMatchingServicer(cfg, context), server
        )
        agent_pb2_grpc.add_AgentServiceServicer_to_server(
            AgentServicer(cfg, context), server
        )
        logger.info("All 4 gRPC services registered.")
    else:
        logger.warning("Running without gRPC services (stubs not generated).")

    server.add_insecure_port(f"[::]:{port}")
    return server


def _start_rest_api(cfg: EngineConfig,
                    context: ReservoirContextProvider,
                    db_svc) -> None:
    """
    Launch the FastAPI REST server in a daemon background thread.

    The thread is daemonised so it automatically dies when the main
    gRPC process exits — no cleanup required.
    """
    if not cfg.rest_enabled:
        logger.info("REST API disabled (rest_enabled=false).")
        return

    try:
        import uvicorn
        from .api.app import create_rest_app
    except ImportError as exc:
        logger.warning(
            "REST API dependencies not installed (%s). "
            "Install with: pip install fastapi uvicorn[standard] python-multipart",
            exc,
        )
        return

    app = create_rest_app(cfg, context, db_svc)

    uv_config = uvicorn.Config(
        app,
        host=cfg.rest_host,
        port=cfg.rest_port,
        log_level="warning",       # Uvicorn's own logs; application uses loguru
        access_log=False,
    )
    server = uvicorn.Server(uv_config)

    def _run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    t = threading.Thread(target=_run, daemon=True, name="rest-api")
    t.start()
    logger.info(
        "REST API starting on http://%s:%s  (docs: /docs)",
        cfg.rest_host, cfg.rest_port,
    )


@click.command()
@click.option("--port", default=50051, show_default=True, help="gRPC listen port")
@click.option("--workers", default=8, show_default=True, help="Thread pool workers")
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(port: int, workers: int, log_level: str) -> NoReturn:
    """Start the PhysicsFlow gRPC engine server."""

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}",
    )
    logger.add(
        "physicsflow_engine.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )

    logger.info(f"PhysicsFlow Engine starting on port {port} ({workers} workers)")

    # Initialise database (creates tables if not present)
    from .db.db_service import DatabaseService
    db_svc = DatabaseService.instance()
    db_svc.audit("engine.started", f"gRPC engine started on port {port}")
    logger.info("Database ready.")

    # Shared context store
    context = ReservoirContextProvider()

    # Build and start servers
    cfg = EngineConfig()
    server = create_server(port, workers, context, cfg)
    server.start()
    logger.info(f"gRPC server listening on 0.0.0.0:{port}")

    # v2.0 — REST API (optional, daemon thread)
    _start_rest_api(cfg, context, db_svc)

    # Write ready signal for .NET EngineManager to detect
    with open("engine.ready", "w") as f:
        f.write(f"ready:{port}")

    # Graceful shutdown on SIGTERM / SIGINT
    def _shutdown(signum, frame):
        logger.info("Shutdown signal received, stopping server...")
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    main()
