"""
Reservoir context provider — shared state store for the LLM agent.

Acts as the single source of truth that the agent tools read from.
The gRPC services write to this store as simulations progress.
Thread-safe via a lock; designed for concurrent reads by multiple agent sessions.
"""

from __future__ import annotations
import json
import math
import threading
from pathlib import Path
from typing import Any
import numpy as np

# Norne field canonical well lists (Equinor open dataset)
_NORNE_PRODUCERS: list[str] = [
    "B-1H", "B-2H", "B-3H", "B-4H", "B-5H",
    "C-1H", "C-2H", "C-3H", "C-4H", "C-4AH",
    "D-1H", "D-2H", "D-3AH", "D-4H",
    "E-1H", "E-2H", "E-3H", "E-4H", "E-4AH",
    "F-1H", "F-2H", "F-3H",
]
_NORNE_INJECTORS: list[str] = [
    "B-4BH", "C-3H",  "D-1CH", "D-3BH", "E-1H",
    "E-3AH", "E-3CH", "E-4BH", "F-4H",
]


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
            same_path = (self._project_path == path)
            self._project_path = path
            self._project_meta = meta or {}

        # Parse project file and seed baseline data if not already loaded
        if not same_path or not self._well_results:
            self._load_from_project_file(path)

    def _load_from_project_file(self, path: str) -> None:
        """
        Parse a .pfproj JSON file and populate the context with:
        - project metadata (field name, grid dims, well list)
        - synthetic baseline production profiles (if no real results yet)
        - synthetic per-well mismatch estimates
        """
        p = Path(path)
        if not p.exists() or p.suffix.lower() != ".pfproj":
            self._seed_norne_baseline()
            return

        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            self._seed_norne_baseline()
            return

        grid = doc.get("grid", {})
        nx = int(grid.get("nx", 46))
        ny = int(grid.get("ny", 112))
        nz = int(grid.get("nz", 22))

        # Well list from file, fallback to Norne defaults
        wells_raw = doc.get("wells", [])
        if isinstance(wells_raw, list) and wells_raw:
            producers  = [w.get("name", f"PROD-{i+1}") for i, w in enumerate(wells_raw)
                          if w.get("type", "PRODUCER").upper() != "INJECTOR"]
            injectors  = [w.get("name", f"INJ-{i+1}")  for i, w in enumerate(wells_raw)
                          if w.get("type", "PRODUCER").upper() == "INJECTOR"]
        else:
            producers, injectors = _NORNE_PRODUCERS, _NORNE_INJECTORS

        pvt = doc.get("pvt", {})
        pi  = float(pvt.get("initial_pressure_bar", 277.0))

        with self._lock:
            self._project_meta = {
                "field_name":   doc.get("name", p.stem),
                "nx": nx, "ny": ny, "nz": nz,
                "n_producers":  len(producers),
                "n_injectors":  len(injectors),
                "well_names":   producers + injectors,
                "producer_names": producers,
                "injector_names": injectors,
                "sim_days":     3650,
                "initial_pressure_bar": pi,
                "model_trained":  doc.get("model_paths", {}) != {},
                "hm_status":    "Not started",
            }

        # Only seed synthetic data if no real results have been written yet
        if not self._well_results:
            self._seed_well_profiles(producers, injectors)

    def _seed_norne_baseline(self) -> None:
        """Seed Norne defaults when no .pfproj file is available."""
        with self._lock:
            if not self._project_meta:
                self._project_meta = {
                    "field_name": "Norne Field (demo)",
                    "nx": 46, "ny": 112, "nz": 22,
                    "n_producers": len(_NORNE_PRODUCERS),
                    "n_injectors": len(_NORNE_INJECTORS),
                    "well_names":  _NORNE_PRODUCERS + _NORNE_INJECTORS,
                    "producer_names": _NORNE_PRODUCERS,
                    "injector_names": _NORNE_INJECTORS,
                    "sim_days": 3650,
                    "initial_pressure_bar": 277.0,
                    "model_trained": False,
                    "hm_status": "Not started",
                }
        if not self._well_results:
            self._seed_well_profiles(_NORNE_PRODUCERS, _NORNE_INJECTORS)

    def _seed_well_profiles(self, producers: list[str], injectors: list[str]) -> None:
        """
        Generate realistic Norne-scale synthetic production profiles so the agent
        has concrete data to reference before a simulation has run.
        Profiles are labelled source='synthetic_baseline' and represent expected
        (pre-history-match) well behaviour.
        """
        import random
        rng = random.Random(42)

        # 37 time steps: 0 to 3600 days at 100-day intervals
        n_steps  = 37
        dt       = 100.0
        time_days = [i * dt for i in range(n_steps)]

        # Per-producer reference peak rates (stb/day) and decline constants
        # Based on published Norne production data (Equinor open dataset)
        _peak_wopr = {
            "B-1H": 4200, "B-2H": 3800, "B-3H": 3500, "B-4H": 4000,
            "B-5H": 3200, "C-1H": 5100, "C-2H": 4800, "C-3H": 3700,
            "C-4H": 4300, "C-4AH": 4100, "D-1H": 3600, "D-2H": 3900,
            "D-3AH": 3400, "D-4H": 3100, "E-1H": 4500, "E-2H": 4200,
            "E-3H": 3800, "E-4H": 3500, "E-4AH": 3300, "F-1H": 3700,
            "F-2H": 3400, "F-3H": 3200,
        }
        _decline = 0.0004   # exponential decline per day ~ 14%/yr

        well_results: dict[str, dict] = {}
        per_well_mismatch: dict[str, dict] = {}

        # Producers
        for well in producers:
            peak    = _peak_wopr.get(well, rng.randint(2500, 5000))
            # Slightly perturbed decline for realism
            d       = _decline * (0.85 + rng.random() * 0.30)
            # Water breakthrough at random time (25–60% through sim)
            wb_step = int(n_steps * (0.25 + rng.random() * 0.35))
            wbfrac  = 0.05 + rng.random() * 0.45   # final water cut

            wopr, wwpr, wgpr = [], [], []
            for t in range(n_steps):
                # 3-month ramp-up then exponential decline
                ramp    = min(1.0, t / 3.0)
                q_oil   = peak * ramp * math.exp(-d * t * dt)
                q_oil   = max(q_oil, 0.0)
                # Water cut increases after breakthrough
                wcut    = 0.0
                if t > wb_step:
                    wcut = wbfrac * (1.0 - math.exp(-0.03 * (t - wb_step)))
                q_water = q_oil * wcut / max(1.0 - wcut, 0.01)
                q_gas   = q_oil * (500 + rng.random() * 200)   # GOR Mscf/STB
                wopr.append(round(q_oil,   1))
                wwpr.append(round(q_water, 1))
                wgpr.append(round(q_gas,   1))

            # Randomly assign above/below-expectation status for demo grounding
            # ~30% of wells underperform, ~20% overperform, rest on-target
            r = rng.random()
            if r < 0.30:
                status = "below_expectation"
                rmse_oil   = round(rng.uniform(0.18, 0.45), 3)
                rmse_water = round(rng.uniform(0.25, 0.60), 3)
            elif r < 0.50:
                status = "above_expectation"
                rmse_oil   = round(rng.uniform(0.05, 0.14), 3)
                rmse_water = round(rng.uniform(0.04, 0.12), 3)
            else:
                status = "on_target"
                rmse_oil   = round(rng.uniform(0.08, 0.18), 3)
                rmse_water = round(rng.uniform(0.06, 0.16), 3)

            well_results[well] = {
                "wopr": wopr, "wwpr": wwpr, "wgpr": wgpr,
                "source": "synthetic_baseline",
                "status": status,
            }
            per_well_mismatch[well] = {
                "oil":   rmse_oil,
                "water": rmse_water,
                "gas":   round(rng.uniform(0.04, 0.22), 3),
                "total": round((rmse_oil + rmse_water) / 2, 3),
                "status": status,
                "source": "synthetic_baseline",
            }

        # Injectors — water injection rate (WWIR)
        for well in injectors:
            rate = rng.randint(4000, 9000)
            wwir = [round(rate * min(1.0, t / 4.0), 0) for t in range(n_steps)]
            well_results[well] = {
                "wopr": [0.0] * n_steps,
                "wwpr": wwir,
                "wgpr": [0.0] * n_steps,
                "source": "synthetic_baseline",
                "status": "injector",
            }

        # Field totals
        field_wopr = [round(sum(well_results[w]["wopr"][t] for w in producers), 0)
                      for t in range(n_steps)]
        field_wwpr = [round(sum(well_results[w]["wwpr"][t] for w in producers), 0)
                      for t in range(n_steps)]
        well_results["FIELD"] = {
            "wopr": field_wopr, "wwpr": field_wwpr,
            "wgpr": [0.0] * n_steps,
            "source": "synthetic_baseline",
            "status": "field_total",
        }

        # Compute overall RMSE as mean of per-well totals
        overall_rmse = round(
            sum(v["total"] for v in per_well_mismatch.values()) / max(len(per_well_mismatch), 1), 3
        )

        with self._lock:
            self._well_results      = well_results
            self._time_days         = time_days
            self._per_well_mismatch = per_well_mismatch
            self._overall_rmse      = overall_rmse

    def get_project_summary(self) -> str:
        """Return markdown-formatted project summary for LLM system prompt."""
        with self._lock:
            if self._project_path is None and not self._project_meta:
                # No project loaded — seed Norne defaults so the agent is never empty
                self._seed_needed = True

        if getattr(self, "_seed_needed", False):
            self._seed_needed = False
            self._seed_norne_baseline()

        with self._lock:
            meta = self._project_meta
            path = self._project_path

            if not meta and path is None:
                return "No project loaded."

            lines = [
                f"**Project:** {meta.get('field_name', 'Unknown')}",
            ]
            if path:
                lines.append(f"**Path:** {path}")
            lines += [
                f"**Grid:** {meta.get('nx','?')}×{meta.get('ny','?')}×{meta.get('nz','?')} "
                f"({meta.get('nx',0)*meta.get('ny',0)*meta.get('nz',0):,} cells)",
                f"**Wells:** {meta.get('n_producers','?')} producers, "
                f"{meta.get('n_injectors','?')} injectors",
                f"**Sim period:** {meta.get('sim_days','?')} days",
                f"**Surrogate:** {'Trained ✓' if meta.get('model_trained') else 'Not trained'}",
                f"**HM status:** {meta.get('hm_status','Not started')}",
            ]

            # Well names
            prods = meta.get("producer_names", [])
            if prods:
                lines.append(f"**Producers:** {', '.join(prods)}")
            injs = meta.get("injector_names", [])
            if injs:
                lines.append(f"**Injectors:** {', '.join(injs)}")

            # Simulation results grounding note
            if self._well_results:
                source = next(iter(self._well_results.values())).get("source", "simulation")
                if source == "synthetic_baseline":
                    lines.append(
                        "**Data source:** synthetic_baseline — reference profiles "
                        "generated from Norne decline curves; run a simulation to replace "
                        "with actual engine results."
                    )
                else:
                    lines.append("**Data source:** live simulation results")

            if self._hm_history:
                last = self._hm_history[-1]
                lines.append(
                    f"**HM progress:** iter {last['iteration']}, "
                    f"mismatch={last['data_mismatch']:.4f}"
                )

            # Performance breakdown
            if self._per_well_mismatch:
                above = [w for w, v in self._per_well_mismatch.items()
                         if v.get("status") == "above_expectation"]
                below = [w for w, v in self._per_well_mismatch.items()
                         if v.get("status") == "below_expectation"]
                if above:
                    lines.append(f"**Above expectation:** {', '.join(above)}")
                if below:
                    lines.append(f"**Below expectation:** {', '.join(below)}")

            return "\n".join(lines)

    def get_project_summary_dict(self) -> dict:
        with self._lock:
            well_source = "none"
            if self._well_results:
                well_source = next(iter(self._well_results.values())).get(
                    "source", "simulation")
            above = [w for w, v in self._per_well_mismatch.items()
                     if v.get("status") == "above_expectation"]
            below = [w for w, v in self._per_well_mismatch.items()
                     if v.get("status") == "below_expectation"]
            return {
                "project_path": self._project_path,
                **self._project_meta,
                "hm_iterations_completed":   len(self._hm_history),
                "training_epochs_completed": len(self._training_history),
                "has_simulation_results":    self._pressure_field is not None,
                "has_well_profiles":         bool(self._well_results),
                "well_data_source":          well_source,
                "wells_above_expectation":   above,
                "wells_below_expectation":   below,
                "overall_rmse":              self._overall_rmse,
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
