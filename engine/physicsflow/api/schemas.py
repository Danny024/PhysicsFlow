"""
PhysicsFlow REST API — Pydantic request / response schemas.

All response schemas use model_config = ConfigDict(from_attributes=True)
so they can be constructed directly from SQLAlchemy ORM objects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Shared base
# ─────────────────────────────────────────────────────────────────────────────

class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    grpc_port: int
    rest_port: int
    db_backend: str          # "sqlite" | "postgresql"
    team_mode: bool
    ollama_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Projects
# ─────────────────────────────────────────────────────────────────────────────

class ProjectCreateRequest(BaseModel):
    name: str
    pfproj_path: str
    nx: Optional[int] = None
    ny: Optional[int] = None
    nz: Optional[int] = None
    n_wells: Optional[int] = None
    notes: Optional[str] = None


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None


class ProjectSchema(_ORM):
    id: str
    name: str
    pfproj_path: str
    created_at: datetime
    modified_at: datetime
    nx: Optional[int]
    ny: Optional[int]
    nz: Optional[int]
    n_wells: Optional[int]
    pino_trained: bool
    hm_completed: bool
    hm_converged: bool
    best_mismatch: Optional[float]
    notes: Optional[str]


class ProjectListResponse(BaseModel):
    projects: list[ProjectSchema]
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# Simulation / Training runs
# ─────────────────────────────────────────────────────────────────────────────

class SimulationRunRequest(BaseModel):
    project_id: str
    n_timesteps: int = 20
    use_surrogate: bool = True


class TrainingStartRequest(BaseModel):
    project_id: str
    epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-3
    w_pde: float = 1.0
    w_data: float = 1.0
    w_well: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 0.5
    device: str = "cuda"


class RunSchema(_ORM):
    id: str
    project_id: str
    run_type: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    loss_total: Optional[float]
    loss_pde: Optional[float]
    n_ensemble: Optional[int]
    error_message: Optional[str]


class RunListResponse(BaseModel):
    runs: list[RunSchema]
    total: int


class TrainingEpochSchema(_ORM):
    epoch: int
    loss_total: float
    loss_pde: Optional[float]
    loss_data: Optional[float]
    loss_well: Optional[float]
    recorded_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# History matching
# ─────────────────────────────────────────────────────────────────────────────

class HMStartRequest(BaseModel):
    project_id: str
    n_ensemble: int = 200
    max_iterations: int = 20
    localisation_radius: float = 12.0
    alpha_init: float = 10.0


class HMIterationSchema(_ORM):
    iteration: int
    mismatch: Optional[float]
    alpha: Optional[float]
    converged: bool
    eur_p10: Optional[float]
    eur_p50: Optional[float]
    eur_p90: Optional[float]
    recorded_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Well observations
# ─────────────────────────────────────────────────────────────────────────────

class WellObservationSchema(_ORM):
    well_name: str
    timestep: int
    wopr_obs: Optional[float]
    wopr_sim: Optional[float]
    wwpr_obs: Optional[float]
    wwpr_sim: Optional[float]
    wgpr_obs: Optional[float]
    wgpr_sim: Optional[float]
    bhp_obs: Optional[float]
    bhp_sim: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Model versions
# ─────────────────────────────────────────────────────────────────────────────

class ModelVersionSchema(_ORM):
    id: str
    project_id: str
    model_type: str
    version_tag: str
    file_path: str
    file_sha256: Optional[str]
    file_size_bytes: Optional[int]
    is_active: bool
    training_run_id: Optional[str]
    epochs_trained: Optional[int]
    loss_total: Optional[float]
    created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Audit log
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogSchema(_ORM):
    id: str
    event_type: str
    description: Optional[str]
    project_id: Optional[str]
    timestamp: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Agent / chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict] = []


# ─────────────────────────────────────────────────────────────────────────────
# tNavigator
# ─────────────────────────────────────────────────────────────────────────────

class tNavigatorImportResponse(BaseModel):
    project_id: str
    sim_path: str
    title: str = ""
    nx: int = 0
    ny: int = 0
    nz: int = 0
    n_active: int = 0
    n_wells: int = 0
    producers: list[str] = []
    injectors: list[str] = []
    n_timesteps: int = 0
    total_days: float = 0.0
    keywords_found: list[str] = []


class tNavigatorRunRequest(BaseModel):
    project_id: str
    sim_path: str
    n_timesteps: int = 30


class tNavigatorRunResponse(BaseModel):
    run_id: str
    status: str
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Generic responses
# ─────────────────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    message: str = ""
    data: Optional[dict[str, Any]] = None


class JobSubmittedResponse(BaseModel):
    run_id: str
    status: str = "queued"
    message: str = ""
