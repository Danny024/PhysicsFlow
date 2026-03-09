"""
PhysicsFlow — DatabaseService

Singleton service that integrates the database with the gRPC engine.
Called by all service handlers to record runs, iterations, and audit events.

Usage:
    from physicsflow.db.db_service import DatabaseService
    svc = DatabaseService.instance()
    run_id = svc.start_training_run(project_id="...", config={...})
    svc.record_epoch(run_id, epoch=1, loss_total=0.5, ...)
    svc.complete_run(run_id, loss_total=0.12)
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Optional

from .database import init_db, get_session
from .repositories import AuditRepo, HMRepo, ModelRepo, ProjectRepo, RunRepo

log = logging.getLogger(__name__)


class DatabaseService:
    """Thread-safe singleton wrapping all repository operations."""

    _instance: Optional["DatabaseService"] = None
    _lock = threading.Lock()

    def __init__(self):
        db_path = init_db()
        log.info("DatabaseService ready — %s", db_path)

    @classmethod
    def instance(cls) -> "DatabaseService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Project ────────────────────────────────────────────────────────────

    def register_project(self, name: str, pfproj_path: str, **kwargs) -> str:
        with get_session() as db:
            proj = ProjectRepo.get_or_create(db, name=name,
                                              pfproj_path=pfproj_path, **kwargs)
            AuditRepo.log(db, "project.opened", f"Project opened: {name}",
                           project_id=proj.id, project_name=name)
            return proj.id

    def update_project_grid(self, project_id: str, nx: int, ny: int, nz: int,
                             n_wells: int = 0) -> None:
        with get_session() as db:
            proj = db.get(ProjectRepo.__class__, project_id)
            proj = db.query(__import__("physicsflow.db.models", fromlist=["Project"])
                             .Project).get(project_id)
            if proj:
                proj.nx, proj.ny, proj.nz, proj.n_wells = nx, ny, nz, n_wells

    def get_recent_projects(self, limit: int = 20) -> list:
        with get_session() as db:
            return ProjectRepo.all_recent(db, limit)

    # ── Simulation / Training runs ─────────────────────────────────────────

    def start_run(self, project_id: str, run_type: str,
                   config: Optional[dict] = None, seed: Optional[int] = None) -> str:
        with get_session() as db:
            run = RunRepo.start(db, project_id=project_id,
                                 run_type=run_type, config=config, seed=seed)
            AuditRepo.log(db, f"run.started",
                           f"{run_type.upper()} run started",
                           project_id=project_id,
                           entity_type="run", entity_id=run.id)
            return run.id

    def record_epoch(self, run_id: str, epoch: int, **metrics) -> None:
        with get_session() as db:
            RunRepo.add_epoch(db, run_id=run_id, epoch=epoch, **metrics)

    def complete_run(self, run_id: str, **metrics) -> None:
        with get_session() as db:
            RunRepo.complete(db, run_id=run_id, **metrics)
            run = db.get(
                __import__("physicsflow.db.models", fromlist=["SimulationRun"])
                .SimulationRun, run_id
            )
            if run:
                AuditRepo.log(db, "run.completed",
                               f"{run.run_type.upper()} run completed in "
                               f"{run.duration_seconds:.1f}s" if run.duration_seconds else "",
                               project_id=run.project_id,
                               entity_type="run", entity_id=run_id)
                if run.run_type == "training":
                    ProjectRepo.mark_pino_trained(db, run.project_id)

    def fail_run(self, run_id: str, error: str, tb: str = "") -> None:
        with get_session() as db:
            RunRepo.fail(db, run_id=run_id, error=error, tb=tb)
            run = db.get(
                __import__("physicsflow.db.models", fromlist=["SimulationRun"])
                .SimulationRun, run_id
            )
            if run:
                AuditRepo.log(db, "run.failed", f"Run failed: {error[:200]}",
                               project_id=run.project_id,
                               entity_type="run", entity_id=run_id,
                               success=False, error_message=error[:500])

    # ── History matching ────────────────────────────────────────────────────

    def new_hm_run_id(self) -> str:
        return HMRepo.new_run_id()

    def record_hm_iteration(self, project_id: str, hm_run_id: str,
                              iteration: int, mismatch: float, alpha: float,
                              s_cumulative: float, converged: bool = False,
                              improvement_pct: float = 0.0,
                              p10=None, p50=None, p90=None,
                              per_well_rmse: Optional[dict] = None) -> None:
        with get_session() as db:
            HMRepo.record_iteration(
                db, project_id=project_id, hm_run_id=hm_run_id,
                iteration=iteration, mismatch=mismatch, alpha=alpha,
                s_cumulative=s_cumulative, converged=converged,
                improvement_pct=improvement_pct,
                p10=p10, p50=p50, p90=p90,
                per_well_rmse=per_well_rmse,
            )
            if converged:
                ProjectRepo.update_hm_status(db, project_id,
                                              best_mismatch=mismatch,
                                              converged=True)
                AuditRepo.log(db, "hm.converged",
                               f"History match converged at iteration {iteration}, "
                               f"mismatch={mismatch:.4f}",
                               project_id=project_id)

    def get_hm_history(self, project_id: str, hm_run_id: str) -> list:
        with get_session() as db:
            return HMRepo.get_run_history(db, project_id, hm_run_id)

    # ── Model versions ──────────────────────────────────────────────────────

    def register_model(self, project_id: str, model_type: str,
                        file_path: str, version_tag: str = "latest",
                        training_run_id: Optional[str] = None,
                        **metrics) -> str:
        with get_session() as db:
            mv = ModelRepo.register(
                db, project_id=project_id, model_type=model_type,
                file_path=file_path, version_tag=version_tag,
                training_run_id=training_run_id, **metrics
            )
            AuditRepo.log(db, "model.registered",
                           f"{model_type.upper()} model registered: {file_path}",
                           project_id=project_id,
                           entity_type="model", entity_id=mv.id,
                           metadata={"version_tag": version_tag, **metrics})
            return mv.id

    # ── Audit ───────────────────────────────────────────────────────────────

    def audit(self, event_type: str, description: str,
               project_id: Optional[str] = None, **kwargs) -> None:
        with get_session() as db:
            AuditRepo.log(db, event_type=event_type,
                           description=description,
                           project_id=project_id, **kwargs)

    def get_audit_log(self, project_id: Optional[str] = None,
                       limit: int = 200) -> list:
        with get_session() as db:
            return AuditRepo.recent(db, limit=limit, project_id=project_id)
