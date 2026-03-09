"""
PhysicsFlow engine configuration — all settings in one place.

Read from environment variables, a config file, or overridden programmatically.
No hardcoded paths or magic numbers anywhere in the codebase.
"""

from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class EngineConfig(BaseSettings):
    """
    Top-level engine configuration.

    Reads from environment variables prefixed with PHYSICSFLOW_.
    Also reads from a .env file in the engine working directory.
    """
    model_config = {"env_prefix": "PHYSICSFLOW_", "env_file": ".env", "extra": "ignore"}

    # ── Server ────────────────────────────────────────────────────────────────
    grpc_port: int = Field(default=50051, description="gRPC listen port")
    grpc_workers: int = Field(default=8, description="Thread pool size")
    log_level: str = Field(default="INFO")

    # ── Ollama / LLM ──────────────────────────────────────────────────────────
    ollama_host: str = Field(default="http://localhost:11434")
    default_llm_model: str = Field(default="phi3:mini",
                                    description="Default Ollama model for AI assistant")
    agent_max_tool_calls: int = Field(default=5)

    # ── Surrogate training (PINO) ─────────────────────────────────────────────
    default_mode: str = Field(default="pino", description="'pino' or 'fno'")
    default_relperm: str = Field(default="stone2", description="'stone2' or 'corey'")
    default_pde_method: int = Field(default=1, description="1=approximate, 2=extensive")
    default_epochs: int = Field(default=500)
    default_batch_size: int = Field(default=4)
    default_lr: float = Field(default=1e-3)
    default_w_pde: float = Field(default=1.0)
    default_w_data: float = Field(default=1.0)
    default_w_well: float = Field(default=1.0)
    default_w_ic: float = Field(default=1.0)
    default_w_bc: float = Field(default=0.5)

    # ── History matching (αREKI) ──────────────────────────────────────────────
    default_n_ensemble: int = Field(default=200)
    default_max_iterations: int = Field(default=20)
    default_localisation_radius: float = Field(default=12.0, description="Grid cells")
    default_alpha_init: float = Field(default=10.0)

    # ── Paths ─────────────────────────────────────────────────────────────────
    models_dir: Path = Field(default=Path("models"),
                             description="Directory to save/load trained models")
    projects_dir: Path = Field(default=Path("projects"))
    opm_flow_exe: Path = Field(default=Path("flow"),
                               description="Path to OPM FLOW binary")

    # ── GPU ───────────────────────────────────────────────────────────────────
    use_gpu: bool = Field(default=True)
    gpu_device: str = Field(default="cuda:0")
    jax_backend: str = Field(default="gpu", description="'gpu', 'cpu', or 'tpu'")

    # ── Grid defaults (overridden by Eclipse deck) ────────────────────────────
    default_nx: int = Field(default=46)
    default_ny: int = Field(default=112)
    default_nz: int = Field(default=22)
    default_dx: float = Field(default=50.0)
    default_dy: float = Field(default=50.0)
    default_dz: float = Field(default=20.0)

    def torch_device(self) -> str:
        """Return PyTorch device string."""
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    return self.gpu_device
            except ImportError:
                pass
        return "cpu"

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)


# Global singleton (loaded once at import time)
config = EngineConfig()
