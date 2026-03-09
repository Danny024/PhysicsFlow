"""
Reservoir context provider — shared state store for the LLM agent.

Acts as the single source of truth that the agent tools read from.
The gRPC services write to this store as simulations progress.
Thread-safe via a lock; designed for concurrent reads by multiple agent sessions.
"""

from __future__ import annotations
import threading
from typing import Any
import numpy as np


class ReservoirContextProvider:
    """
    Shared mutable state that bridges live simulation results to the LLM agent.

    Updated by gRPC service handlers as:
    - Training progresses (epoch loss values)
    - History matching iterates (mismatch, ensemble stats)
    - Simulations complete (field snapshots, well time series)
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._project_path: str | None = None
        self._project_meta: dict = {}

        # Simulation state
        self._simulation_state: dict | None = None
        self._well_results: dict[str, dict] = {}   # name → {wopr, wwpr, wgpr}
        self._time_days: list[float] = []

        # Field arrays (set after forward simulation completes)
        self._pressure_field: np.ndarray | None = None  # [Nx, Ny, Nz, Nt]
        self._sw_field:       np.ndarray | None = None
        self._sg_field:       np.ndarray | None = None
        self._grid_data:      dict[str, np.ndarray] | None = None

        # History matching state
        self._hm_history: list[dict] = []
        self._ensemble_stats: dict = {}
        self._per_well_mismatch: dict = {}
        self._overall_rmse: float = 0.0

        # Training state
        self._training_history: list[dict] = []

    # ── Project ───────────────────────────────────────────────────────────────

    def set_project(self, path: str, meta: dict | None = None) -> None:
        with self._lock:
            self._project_path = path
            self._project_meta = meta or {}

    def get_project_summary(self) -> str:
        """Return markdown-formatted project summary for LLM system prompt."""
        with self._lock:
            if self._project_path is None:
                return "No project loaded."

            meta = self._project_meta
            lines = [
                f"**Project:** {meta.get('field_name', 'Unknown')}",
                f"**Path:** {self._project_path}",
                f"**Grid:** {meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}",
                f"**Wells:** {meta.get('n_producers', '?')} producers, "
                f"{meta.get('n_injectors', '?')} injectors",
                f"**Sim period:** {meta.get('sim_days', '?')} days",
                f"**Surrogate:** {'Trained ✓' if meta.get('model_trained') else 'Not trained'}",
                f"**HM status:** {meta.get('hm_status', 'Not started')}",
            ]
            if self._hm_history:
                last = self._hm_history[-1]
                lines.append(
                    f"**HM progress:** iter {last['iteration']}, "
                    f"mismatch={last['data_mismatch']:.4f}"
                )
            return "\n".join(lines)

    def get_project_summary_dict(self) -> dict:
        with self._lock:
            return {
                "project_path": self._project_path,
                **self._project_meta,
                "hm_iterations_completed": len(self._hm_history),
                "training_epochs_completed": len(self._training_history),
                "has_simulation_results": self._pressure_field is not None,
            }

    # ── Write methods (called by gRPC services) ───────────────────────────────

    def update_simulation_state(self, state: dict) -> None:
        with self._lock:
            self._simulation_state = state

    def update_well_results(self, results: dict, time_days: list[float]) -> None:
        with self._lock:
            self._well_results = results
            self._time_days = time_days

    def update_field_arrays(
        self,
        pressure: np.ndarray,
        sw: np.ndarray,
        sg: np.ndarray,
    ) -> None:
        with self._lock:
            self._pressure_field = pressure
            self._sw_field = sw
            self._sg_field = sg

    def update_grid_data(self, grid_data: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._grid_data = grid_data

    def append_hm_iteration(self, metrics: dict) -> None:
        with self._lock:
            self._hm_history.append(metrics)
            self._overall_rmse = metrics.get("data_mismatch", self._overall_rmse)

    def update_ensemble_stats(self, stats: dict) -> None:
        with self._lock:
            self._ensemble_stats = stats

    def update_per_well_mismatch(self, mismatch: dict) -> None:
        with self._lock:
            self._per_well_mismatch = mismatch

    def reset_hm(self) -> None:
        with self._lock:
            self._hm_history.clear()
            self._ensemble_stats.clear()
            self._per_well_mismatch.clear()

    def append_training_epoch(self, metrics: dict) -> None:
        with self._lock:
            self._training_history.append(metrics)

    # ── Read properties ───────────────────────────────────────────────────────

    @property
    def simulation_state(self) -> dict | None:
        with self._lock:
            return self._simulation_state

    @property
    def well_results(self) -> dict:
        with self._lock:
            return dict(self._well_results)

    @property
    def time_days(self) -> list[float]:
        with self._lock:
            return list(self._time_days)

    @property
    def pressure_field(self) -> np.ndarray | None:
        with self._lock:
            return self._pressure_field

    @property
    def sw_field(self) -> np.ndarray | None:
        with self._lock:
            return self._sw_field

    @property
    def sg_field(self) -> np.ndarray | None:
        with self._lock:
            return self._sg_field

    @property
    def grid_data(self) -> dict | None:
        with self._lock:
            return self._grid_data

    @property
    def hm_history(self) -> list[dict]:
        with self._lock:
            return list(self._hm_history)

    @property
    def ensemble_stats(self) -> dict:
        with self._lock:
            return dict(self._ensemble_stats)

    @property
    def per_well_mismatch(self) -> dict:
        with self._lock:
            return dict(self._per_well_mismatch)

    @property
    def overall_rmse(self) -> float:
        with self._lock:
            return self._overall_rmse
