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
from .models import ModelVersion, Project, SimulationRun
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
            proj = db.get(Project, project_id)
            if proj:
                proj.nx, proj.ny, proj.nz, proj.n_wells = nx, ny, nz, n_wells

    def get_recent_projects(self, limit: int = 20) -> list:
        with get_session() as db:
            return ProjectRepo.all_recent(db, limit)

    def list_projects(self, limit: int = 100) -> list:
        with get_session() as db:
            return ProjectRepo.all_recent(db, limit)

    def get_project(self, project_id: str):
        with get_session() as db:
            return ProjectRepo.get(db, project_id)

    def update_project(self, project_id: str, **kwargs) -> bool:
        with get_session() as db:
            proj = db.get(Project, project_id)
            if proj is None:
                return False
            allowed = {"name", "description", "pfproj_path", "nx", "ny", "nz",
                       "n_wells", "field_name", "country", "operator"}
            for k, v in kwargs.items():
                if k in allowed and hasattr(proj, k):
                    setattr(proj, k, v)
            return True

    def delete_project(self, project_id: str) -> bool:
        with get_session() as db:
            ok = ProjectRepo.delete(db, project_id)
            if ok:
                AuditRepo.log(db, "project.deleted",
                               f"Project deleted: {project_id}",
                               project_id=project_id)
            return ok

    # ── Simulation / Training runs ─────────────────────────────────────────

    def list_runs(self, project_id: str, limit: int = 50) -> list:
        with get_session() as db:
            return RunRepo.recent(db, project_id, limit)

    def get_run(self, run_id: str):
        with get_session() as db:
            return db.get(SimulationRun, run_id)

    def get_epoch_history(self, run_id: str) -> list:
        with get_session() as db:
            return RunRepo.get_epoch_history(db, run_id)

    def start_run(self, project_id: str, run_type: str,
                   config: Optional[dict] = None, seed: Optional[int] = None,
                   **_extra) -> str:
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
            run = db.get(SimulationRun, run_id)
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
            run = db.get(SimulationRun, run_id)
            if run:
                AuditRepo.log(db, "run.failed", f"Run failed: {error[:200]}",
                               project_id=run.project_id,
                               entity_type="run", entity_id=run_id,
                               success=False, error_message=error[:500])

    # ── History matching ────────────────────────────────────────────────────

    def new_hm_run_id(self, project_id: str = "") -> str:
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

    def list_models(self, project_id: str) -> list:
        with get_session() as db:
            from sqlalchemy import desc
            return (db.query(ModelVersion)
                      .filter(ModelVersion.project_id == project_id)
                      .order_by(desc(ModelVersion.created_at))
                      .all())

    def get_model_by_id(self, model_id: str):
        with get_session() as db:
            return db.get(ModelVersion, model_id)

    def activate_model(self, model_id: str) -> bool:
        with get_session() as db:
            mv = db.get(ModelVersion, model_id)
            if mv is None:
                return False
            # Deactivate siblings of same type in same project
            (db.query(ModelVersion)
               .filter(ModelVersion.project_id == mv.project_id,
                       ModelVersion.model_type  == mv.model_type,
                       ModelVersion.is_active   == True)
               .update({"is_active": False}))
            mv.is_active = True
            AuditRepo.log(db, "model.activated",
                           f"Model {model_id} activated",
                           project_id=mv.project_id,
                           entity_type="model", entity_id=model_id)
            return True

    def get_active_model(self, project_id: str, model_type: str):
        with get_session() as db:
            return ModelRepo.get_active(db, project_id, model_type)

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
