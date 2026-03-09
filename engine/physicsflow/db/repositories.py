"""
PhysicsFlow — Repository layer (data access objects).

Each repository wraps one ORM model and provides domain-specific
query methods. Business logic stays in services; repositories
only talk to the database.

Usage:
    from physicsflow.db.repositories import ProjectRepo, RunRepo, AuditRepo
    from physicsflow.db.database import get_session

    with get_session() as db:
        project = ProjectRepo.create(db, name="Norne Q4", pfproj_path="/data/norne.pfproj")
        AuditRepo.log(db, "project.created", project_id=project.id, description="...")
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from .models import (
    AuditLog, HMIteration, ModelVersion,
    Project, SimulationRun, TrainingEpoch, WellObservation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Project repository
# ─────────────────────────────────────────────────────────────────────────────

class ProjectRepo:

    @staticmethod
    def create(db: Session, name: str, pfproj_path: str, **kwargs) -> Project:
        proj = Project(name=name, pfproj_path=pfproj_path, **kwargs)
        db.add(proj)
        db.flush()
        return proj

    @staticmethod
    def get(db: Session, project_id: str) -> Optional[Project]:
        return db.get(Project, project_id)

    @staticmethod
    def get_by_path(db: Session, pfproj_path: str) -> Optional[Project]:
        return db.query(Project).filter(Project.pfproj_path == pfproj_path).first()

    @staticmethod
    def get_or_create(db: Session, name: str, pfproj_path: str, **kwargs) -> Project:
        proj = ProjectRepo.get_by_path(db, pfproj_path)
        if proj is None:
            proj = ProjectRepo.create(db, name=name, pfproj_path=pfproj_path, **kwargs)
        return proj

    @staticmethod
    def all_recent(db: Session, limit: int = 20) -> List[Project]:
        return (db.query(Project)
                  .order_by(desc(Project.last_opened_at.nullslast()),
                            desc(Project.modified_at))
                  .limit(limit)
                  .all())

    @staticmethod
    def mark_opened(db: Session, project_id: str) -> None:
        proj = db.get(Project, project_id)
        if proj:
            proj.last_opened_at = datetime.now(timezone.utc)

    @staticmethod
    def update_hm_status(db: Session, project_id: str,
                          best_mismatch: float, converged: bool) -> None:
        proj = db.get(Project, project_id)
        if proj:
            proj.hm_completed = True
            proj.hm_converged = converged
            proj.best_mismatch = best_mismatch

    @staticmethod
    def mark_pino_trained(db: Session, project_id: str) -> None:
        proj = db.get(Project, project_id)
        if proj:
            proj.pino_trained = True

    @staticmethod
    def search(db: Session, query: str) -> List[Project]:
        pattern = f"%{query}%"
        return (db.query(Project)
                  .filter(Project.name.ilike(pattern))
                  .order_by(desc(Project.modified_at))
                  .all())

    @staticmethod
    def delete(db: Session, project_id: str) -> bool:
        proj = db.get(Project, project_id)
        if proj:
            db.delete(proj)
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Simulation run repository
# ─────────────────────────────────────────────────────────────────────────────

class RunRepo:

    @staticmethod
    def start(db: Session, project_id: str, run_type: str,
              config: Optional[dict] = None, seed: Optional[int] = None) -> SimulationRun:
        run = SimulationRun(
            project_id=project_id,
            run_type=run_type,
            status="running",
            random_seed=seed,
            input_hash=_hash_config(config) if config else None,
            config_snapshot=config,
        )
        db.add(run)
        db.flush()
        return run

    @staticmethod
    def complete(db: Session, run_id: str, **metrics) -> None:
        run = db.get(SimulationRun, run_id)
        if run:
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            if run.started_at:
                delta = run.completed_at - run.started_at
                run.duration_seconds = delta.total_seconds()
            for k, v in metrics.items():
                if hasattr(run, k):
                    setattr(run, k, v)

    @staticmethod
    def fail(db: Session, run_id: str, error: str, tb: str = "") -> None:
        run = db.get(SimulationRun, run_id)
        if run:
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = error[:2000]
            run.traceback = tb[:10000]

    @staticmethod
    def add_epoch(db: Session, run_id: str, epoch: int, **metrics) -> None:
        ep = TrainingEpoch(run_id=run_id, epoch=epoch, **{
            k: v for k, v in metrics.items()
            if k in TrainingEpoch.__table__.columns.keys()
        })
        db.add(ep)

    @staticmethod
    def get_epoch_history(db: Session, run_id: str) -> List[TrainingEpoch]:
        return (db.query(TrainingEpoch)
                  .filter(TrainingEpoch.run_id == run_id)
                  .order_by(TrainingEpoch.epoch)
                  .all())

    @staticmethod
    def recent(db: Session, project_id: str, limit: int = 50) -> List[SimulationRun]:
        return (db.query(SimulationRun)
                  .filter(SimulationRun.project_id == project_id)
                  .order_by(desc(SimulationRun.started_at))
                  .limit(limit)
                  .all())

    @staticmethod
    def last_training_run(db: Session, project_id: str) -> Optional[SimulationRun]:
        return (db.query(SimulationRun)
                  .filter(SimulationRun.project_id == project_id,
                          SimulationRun.run_type == "training",
                          SimulationRun.status == "completed")
                  .order_by(desc(SimulationRun.completed_at))
                  .first())


# ─────────────────────────────────────────────────────────────────────────────
# History matching iteration repository
# ─────────────────────────────────────────────────────────────────────────────

class HMRepo:

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def record_iteration(db: Session, project_id: str, hm_run_id: str,
                          iteration: int, mismatch: float, alpha: float,
                          s_cumulative: float, converged: bool = False,
                          improvement_pct: float = 0.0,
                          p10: Optional[list] = None,
                          p50: Optional[list] = None,
                          p90: Optional[list] = None,
                          per_well_rmse: Optional[dict] = None) -> HMIteration:
        row = HMIteration(
            project_id=project_id,
            hm_run_id=hm_run_id,
            iteration=iteration,
            mismatch=mismatch,
            alpha=alpha,
            s_cumulative=s_cumulative,
            converged=converged,
            improvement_pct=improvement_pct,
            p10_snapshot=p10,
            p50_snapshot=p50,
            p90_snapshot=p90,
            per_well_rmse=per_well_rmse,
        )
        db.add(row)
        return row

    @staticmethod
    def get_run_history(db: Session, project_id: str,
                         hm_run_id: str) -> List[HMIteration]:
        return (db.query(HMIteration)
                  .filter(HMIteration.project_id == project_id,
                          HMIteration.hm_run_id == hm_run_id)
                  .order_by(HMIteration.iteration)
                  .all())

    @staticmethod
    def get_all_runs(db: Session, project_id: str) -> List[str]:
        """Return distinct hm_run_id values ordered by first iteration date."""
        rows = (db.query(HMIteration.hm_run_id,
                         func.min(HMIteration.recorded_at).label("first"))
                  .filter(HMIteration.project_id == project_id)
                  .group_by(HMIteration.hm_run_id)
                  .order_by(desc("first"))
                  .all())
        return [r.hm_run_id for r in rows]

    @staticmethod
    def best_mismatch(db: Session, project_id: str) -> Optional[float]:
        result = (db.query(func.min(HMIteration.mismatch))
                    .filter(HMIteration.project_id == project_id)
                    .scalar())
        return float(result) if result is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Well observation repository
# ─────────────────────────────────────────────────────────────────────────────

class WellObsRepo:

    @staticmethod
    def upsert_timeseries(db: Session, project_id: str, well_name: str,
                           records: List[dict]) -> int:
        """
        Insert or update well observations for a well.
        Each record dict must have: date, timestep, and any obs_/sim_ columns.
        Returns count of rows inserted.
        """
        count = 0
        for rec in records:
            date = rec.get("date")
            if isinstance(date, str):
                date = datetime.fromisoformat(date.replace("Z", "+00:00"))

            existing = (db.query(WellObservation)
                          .filter(WellObservation.project_id == project_id,
                                  WellObservation.well_name == well_name,
                                  WellObservation.date == date)
                          .first())
            if existing is None:
                row = WellObservation(
                    project_id=project_id,
                    well_name=well_name,
                    date=date,
                    **{k: v for k, v in rec.items()
                       if k != "date" and k in WellObservation.__table__.columns.keys()}
                )
                db.add(row)
                count += 1
            else:
                for k, v in rec.items():
                    if k not in ("date",) and hasattr(existing, k):
                        setattr(existing, k, v)
        db.flush()
        return count

    @staticmethod
    def get_well_timeseries(db: Session, project_id: str,
                             well_name: str) -> List[WellObservation]:
        return (db.query(WellObservation)
                  .filter(WellObservation.project_id == project_id,
                          WellObservation.well_name == well_name)
                  .order_by(WellObservation.date)
                  .all())

    @staticmethod
    def get_well_names(db: Session, project_id: str) -> List[str]:
        rows = (db.query(WellObservation.well_name)
                  .filter(WellObservation.project_id == project_id)
                  .distinct()
                  .all())
        return [r.well_name for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Model version repository
# ─────────────────────────────────────────────────────────────────────────────

class ModelRepo:

    @staticmethod
    def register(db: Session, project_id: str, model_type: str,
                  file_path: str, version_tag: str = "latest",
                  training_run_id: Optional[str] = None,
                  **metrics) -> ModelVersion:
        # Deactivate previous versions of same type
        (db.query(ModelVersion)
           .filter(ModelVersion.project_id == project_id,
                   ModelVersion.model_type == model_type,
                   ModelVersion.is_active == True)
           .update({"is_active": False}))

        sha = None
        size = None
        try:
            size = os.path.getsize(file_path)
            with open(file_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
        except (OSError, IOError):
            pass

        mv = ModelVersion(
            project_id=project_id,
            model_type=model_type,
            version_tag=version_tag,
            file_path=file_path,
            file_size_bytes=size,
            file_sha256=sha,
            training_run_id=training_run_id,
            is_active=True,
            **{k: v for k, v in metrics.items()
               if k in ModelVersion.__table__.columns.keys()}
        )
        db.add(mv)
        db.flush()
        return mv

    @staticmethod
    def get_active(db: Session, project_id: str,
                    model_type: str) -> Optional[ModelVersion]:
        return (db.query(ModelVersion)
                  .filter(ModelVersion.project_id == project_id,
                          ModelVersion.model_type == model_type,
                          ModelVersion.is_active == True)
                  .first())

    @staticmethod
    def history(db: Session, project_id: str, model_type: str) -> List[ModelVersion]:
        return (db.query(ModelVersion)
                  .filter(ModelVersion.project_id == project_id,
                          ModelVersion.model_type == model_type)
                  .order_by(desc(ModelVersion.created_at))
                  .all())


# ─────────────────────────────────────────────────────────────────────────────
# Audit log repository
# ─────────────────────────────────────────────────────────────────────────────

class AuditRepo:
    """
    Append-only audit log. Every call inserts one row — no updates or deletes.
    """

    @staticmethod
    def log(db: Session, event_type: str, description: str,
             project_id: Optional[str] = None,
             project_name: Optional[str] = None,
             entity_type: Optional[str] = None,
             entity_id: Optional[str] = None,
             success: bool = True,
             error_message: Optional[str] = None,
             metadata: Optional[dict] = None) -> AuditLog:
        entry = AuditLog(
            event_type=event_type,
            description=description,
            project_id=project_id,
            project_name=project_name,
            entity_type=entity_type,
            entity_id=entity_id,
            username=_current_user(),
            hostname=_hostname(),
            process_id=os.getpid(),
            success=success,
            error_message=error_message,
            metadata=metadata,
        )
        db.add(entry)
        db.flush()
        return entry

    @staticmethod
    def recent(db: Session, limit: int = 200,
                project_id: Optional[str] = None) -> List[AuditLog]:
        q = db.query(AuditLog)
        if project_id:
            q = q.filter(AuditLog.project_id == project_id)
        return q.order_by(desc(AuditLog.timestamp)).limit(limit).all()

    @staticmethod
    def search(db: Session, event_type: Optional[str] = None,
                project_id: Optional[str] = None,
                since: Optional[datetime] = None,
                limit: int = 500) -> List[AuditLog]:
        q = db.query(AuditLog)
        if event_type:
            q = q.filter(AuditLog.event_type == event_type)
        if project_id:
            q = q.filter(AuditLog.project_id == project_id)
        if since:
            q = q.filter(AuditLog.timestamp >= since)
        return q.order_by(desc(AuditLog.timestamp)).limit(limit).all()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash_config(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _current_user() -> str:
    try:
        return os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    except Exception:
        return "unknown"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"
