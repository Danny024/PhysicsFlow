"""
PhysicsFlow — SQLAlchemy ORM Models (SQLite backend).

Database schema:
    projects         — registry of all .pfproj studies
    simulation_runs  — every forward/surrogate run with inputs + outputs
    hm_iterations    — per-iteration αREKI metrics (persistent across restarts)
    well_observations — production/injection time series per well per project
    model_versions   — trained model file registry with performance metadata
    audit_log        — immutable append-only compliance record

All tables use UTC timestamps. The database file is stored alongside the
project or in the configured data directory (PHYSICSFLOW_DB_PATH env var).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, JSON, event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Project registry
# ─────────────────────────────────────────────────────────────────────────────

class Project(Base):
    """
    Central registry entry for one PhysicsFlow study.
    One row per .pfproj file; updated whenever the project is saved.
    """
    __tablename__ = "projects"

    id: Mapped[str]          = mapped_column(String(36), primary_key=True, default=_new_uuid)
    name: Mapped[str]        = mapped_column(String(255), nullable=False, index=True)
    pfproj_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime]  = mapped_column(DateTime(timezone=True), default=_utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow,
                                                   onupdate=_utcnow)
    last_opened_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Grid summary (denormalised for fast UI display without reading .pfproj)
    nx: Mapped[Optional[int]]    = mapped_column(Integer)
    ny: Mapped[Optional[int]]    = mapped_column(Integer)
    nz: Mapped[Optional[int]]    = mapped_column(Integer)
    n_wells: Mapped[Optional[int]] = mapped_column(Integer)

    # Status flags
    pino_trained: Mapped[bool]   = mapped_column(Boolean, default=False)
    hm_completed: Mapped[bool]   = mapped_column(Boolean, default=False)
    hm_converged: Mapped[bool]   = mapped_column(Boolean, default=False)
    best_mismatch: Mapped[Optional[float]] = mapped_column(Float)

    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    runs: Mapped[List["SimulationRun"]]         = relationship(back_populates="project",
                                                                cascade="all, delete-orphan")
    hm_iterations: Mapped[List["HMIteration"]]  = relationship(back_populates="project",
                                                                cascade="all, delete-orphan")
    well_observations: Mapped[List["WellObservation"]] = relationship(back_populates="project",
                                                                       cascade="all, delete-orphan")
    model_versions: Mapped[List["ModelVersion"]] = relationship(back_populates="project",
                                                                 cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Project name={self.name!r} id={self.id[:8]}>"


# ─────────────────────────────────────────────────────────────────────────────
# Simulation run log
# ─────────────────────────────────────────────────────────────────────────────

class SimulationRun(Base):
    """
    Record of every forward simulation or surrogate evaluation.
    Provides full audit trail: who, when, what inputs, what outputs, how long.
    """
    __tablename__ = "simulation_runs"

    id: Mapped[str]             = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str]     = mapped_column(ForeignKey("projects.id"), index=True)
    run_type: Mapped[str]       = mapped_column(String(32))   # 'pino', 'opm_flow', 'training'
    status: Mapped[str]         = mapped_column(String(32), default="pending")
                                                              # pending|running|completed|failed

    # Timing
    started_at: Mapped[datetime]           = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)

    # Input fingerprint (for reproducibility)
    input_hash: Mapped[Optional[str]]  = mapped_column(String(64))  # SHA-256 of input config
    random_seed: Mapped[Optional[int]] = mapped_column(Integer)
    n_timesteps: Mapped[Optional[int]] = mapped_column(Integer)
    n_ensemble: Mapped[Optional[int]]  = mapped_column(Integer)

    # Output summary metrics
    rmse_pressure: Mapped[Optional[float]] = mapped_column(Float)
    rmse_sw: Mapped[Optional[float]]       = mapped_column(Float)
    loss_total: Mapped[Optional[float]]    = mapped_column(Float)
    loss_pde: Mapped[Optional[float]]      = mapped_column(Float)
    loss_data: Mapped[Optional[float]]     = mapped_column(Float)

    # Training-specific
    epochs_completed: Mapped[Optional[int]] = mapped_column(Integer)
    best_epoch: Mapped[Optional[int]]       = mapped_column(Integer)
    model_path: Mapped[Optional[str]]       = mapped_column(Text)

    # Error capture
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    traceback: Mapped[Optional[str]]     = mapped_column(Text)

    # Full config snapshot (JSON)
    config_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)

    # Relationships
    project: Mapped["Project"]               = relationship(back_populates="runs")
    epoch_history: Mapped[List["TrainingEpoch"]] = relationship(back_populates="run",
                                                                  cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sim_runs_project_type", "project_id", "run_type"),
        Index("ix_sim_runs_started",      "started_at"),
    )

    def __repr__(self) -> str:
        return f"<SimulationRun type={self.run_type} status={self.status}>"


class TrainingEpoch(Base):
    """Per-epoch training metrics for loss curve visualisation."""
    __tablename__ = "training_epochs"

    id: Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str]      = mapped_column(ForeignKey("simulation_runs.id"), index=True)
    epoch: Mapped[int]       = mapped_column(Integer, nullable=False)
    loss_total: Mapped[float]  = mapped_column(Float)
    loss_pde: Mapped[float]    = mapped_column(Float)
    loss_data: Mapped[float]   = mapped_column(Float)
    loss_well: Mapped[float]   = mapped_column(Float)
    loss_ic: Mapped[float]     = mapped_column(Float)
    loss_bc: Mapped[float]     = mapped_column(Float)
    learning_rate: Mapped[Optional[float]] = mapped_column(Float)
    gpu_util: Mapped[Optional[float]]      = mapped_column(Float)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped["SimulationRun"] = relationship(back_populates="epoch_history")


# ─────────────────────────────────────────────────────────────────────────────
# History matching iterations
# ─────────────────────────────────────────────────────────────────────────────

class HMIteration(Base):
    """
    Per-iteration αREKI metrics — persisted to DB so they survive engine restarts.
    The UI fan chart and convergence plot are rebuilt from this table on re-open.
    """
    __tablename__ = "hm_iterations"

    id: Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str]  = mapped_column(ForeignKey("projects.id"), index=True)
    hm_run_id: Mapped[str]   = mapped_column(String(36), index=True)   # groups one HM run

    iteration: Mapped[int]          = mapped_column(Integer, nullable=False)
    mismatch: Mapped[float]         = mapped_column(Float)
    alpha: Mapped[float]            = mapped_column(Float)
    s_cumulative: Mapped[float]     = mapped_column(Float)
    improvement_pct: Mapped[Optional[float]] = mapped_column(Float)
    converged: Mapped[bool]         = mapped_column(Boolean, default=False)

    # Ensemble statistics snapshot
    p10_snapshot: Mapped[Optional[list]] = mapped_column(JSON)   # list of floats
    p50_snapshot: Mapped[Optional[list]] = mapped_column(JSON)
    p90_snapshot: Mapped[Optional[list]] = mapped_column(JSON)

    # Per-well mismatch at this iteration
    per_well_rmse: Mapped[Optional[dict]] = mapped_column(JSON)  # {well_name: rmse}

    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    project: Mapped["Project"] = relationship(back_populates="hm_iterations")

    __table_args__ = (
        Index("ix_hm_iter_project_run", "project_id", "hm_run_id"),
    )

    def __repr__(self) -> str:
        return f"<HMIteration iter={self.iteration} mismatch={self.mismatch:.4f}>"


# ─────────────────────────────────────────────────────────────────────────────
# Well observations (production / injection time series)
# ─────────────────────────────────────────────────────────────────────────────

class WellObservation(Base):
    """
    Observed and simulated well rates at each timestep.
    Loaded from Eclipse SUMMARY / field data and stored for comparison.
    """
    __tablename__ = "well_observations"

    id: Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str]  = mapped_column(ForeignKey("projects.id"), index=True)
    well_name: Mapped[str]   = mapped_column(String(64), nullable=False, index=True)
    date: Mapped[datetime]   = mapped_column(DateTime(timezone=True), nullable=False)
    timestep: Mapped[int]    = mapped_column(Integer)

    # Observed (from field or Eclipse reference)
    obs_wopr: Mapped[Optional[float]] = mapped_column(Float)   # oil rate stb/day
    obs_wwpr: Mapped[Optional[float]] = mapped_column(Float)   # water rate stb/day
    obs_wgpr: Mapped[Optional[float]] = mapped_column(Float)   # gas rate Mscf/day
    obs_wbhp: Mapped[Optional[float]] = mapped_column(Float)   # BHP bar
    obs_wwct: Mapped[Optional[float]] = mapped_column(Float)   # water cut fraction

    # Simulated P50 (updated after HM)
    sim_wopr: Mapped[Optional[float]] = mapped_column(Float)
    sim_wwpr: Mapped[Optional[float]] = mapped_column(Float)
    sim_wgpr: Mapped[Optional[float]] = mapped_column(Float)
    sim_wbhp: Mapped[Optional[float]] = mapped_column(Float)

    # Ensemble bounds
    p10_wopr: Mapped[Optional[float]] = mapped_column(Float)
    p90_wopr: Mapped[Optional[float]] = mapped_column(Float)

    data_source: Mapped[str] = mapped_column(String(64), default="eclipse")

    project: Mapped["Project"] = relationship(back_populates="well_observations")

    __table_args__ = (
        Index("ix_well_obs_project_well_date", "project_id", "well_name", "date"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model version registry
# ─────────────────────────────────────────────────────────────────────────────

class ModelVersion(Base):
    """
    Registry of trained model checkpoints with performance metadata.
    Supports model comparison and rollback.
    """
    __tablename__ = "model_versions"

    id: Mapped[str]          = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str]  = mapped_column(ForeignKey("projects.id"), index=True)
    model_type: Mapped[str]  = mapped_column(String(32))   # 'pino', 'ccr', 'vcae', 'ddim'
    version_tag: Mapped[str] = mapped_column(String(64))   # e.g. 'v1', 'best', 'epoch_300'

    file_path: Mapped[str]   = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    file_sha256: Mapped[Optional[str]]     = mapped_column(String(64))

    # Training provenance
    training_run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("simulation_runs.id"))
    epochs_trained: Mapped[Optional[int]]  = mapped_column(Integer)

    # Performance metrics at save time
    loss_total: Mapped[Optional[float]]    = mapped_column(Float)
    loss_pde: Mapped[Optional[float]]      = mapped_column(Float)
    rmse_pressure: Mapped[Optional[float]] = mapped_column(Float)
    rmse_sw: Mapped[Optional[float]]       = mapped_column(Float)

    # Architecture config snapshot
    architecture_config: Mapped[Optional[dict]] = mapped_column(JSON)

    is_active: Mapped[bool]   = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    project: Mapped["Project"] = relationship(back_populates="model_versions")

    def __repr__(self) -> str:
        return f"<ModelVersion type={self.model_type} tag={self.version_tag}>"


# ─────────────────────────────────────────────────────────────────────────────
# Audit log (append-only compliance record)
# ─────────────────────────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Immutable append-only audit log for industry compliance.
    Records every significant action: project open/save, run start/stop,
    parameter changes, model loads, exports.

    Never updated or deleted — only INSERTs.
    """
    __tablename__ = "audit_log"

    id: Mapped[int]              = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime]  = mapped_column(DateTime(timezone=True), default=_utcnow,
                                                  nullable=False, index=True)
    event_type: Mapped[str]      = mapped_column(String(64), nullable=False, index=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    project_name: Mapped[Optional[str]] = mapped_column(String(255))

    # What happened
    description: Mapped[str]     = mapped_column(Text, nullable=False)
    entity_type: Mapped[Optional[str]]  = mapped_column(String(64))  # 'run', 'model', 'project'
    entity_id: Mapped[Optional[str]]    = mapped_column(String(36))

    # Who / where
    username: Mapped[Optional[str]]  = mapped_column(String(128))
    hostname: Mapped[Optional[str]]  = mapped_column(String(128))
    process_id: Mapped[Optional[int]] = mapped_column(Integer)

    # Metadata payload
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    # Outcome
    success: Mapped[bool]            = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("ix_audit_event_project", "event_type", "project_id"),
        Index("ix_audit_timestamp",     "timestamp"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prevent UPDATE / DELETE on audit_log at the SQLAlchemy ORM level
# ─────────────────────────────────────────────────────────────────────────────

@event.listens_for(AuditLog, "before_update")
def _prevent_audit_update(mapper, connection, target):
    raise RuntimeError("AuditLog records are immutable — updates are not permitted.")
