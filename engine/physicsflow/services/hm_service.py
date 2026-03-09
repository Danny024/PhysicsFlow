"""
PhysicsFlow — HistoryMatchingService gRPC handler.

Implements:
    RunHistoryMatch  (server-streaming) — runs αREKI with live iteration updates
    GetEnsembleStats — P10/P50/P90 at any point
    GetDataMismatch  — per-well RMSE breakdown
    GetParameterEnsemble — current ensemble K/φ values
    StopHistoryMatch — graceful stop signal
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from ..agent.context_provider import ReservoirContextProvider
from ..config import EngineConfig

log = logging.getLogger(__name__)

try:
    from ..proto import history_matching_pb2 as pb
    from ..proto import history_matching_pb2_grpc as pbg
    _PROTO_AVAILABLE = True
except ImportError:
    _PROTO_AVAILABLE = False
    pb = None
    pbg = None


class HistoryMatchingServicer:
    """
    Runs αREKI history matching with streaming progress to the .NET client.
    """

    def __init__(self, cfg: EngineConfig, context: ReservoirContextProvider):
        self.cfg = cfg
        self.ctx = context
        self._stop_event = threading.Event()

    # ── RunHistoryMatch (server-streaming) ────────────────────────────────

    def RunHistoryMatch(self, request, context) -> Iterator:
        """
        Main HM entry point. Streams HMProgress messages per iteration.
        """
        if not _PROTO_AVAILABLE:
            return

        self._stop_event.clear()
        log.info("RunHistoryMatch: N=%d max_iter=%d radius=%.1f",
                 request.n_ensemble, request.max_iterations,
                 request.localisation_radius)

        try:
            from ..history_matching.areki_jax import AREKIEngine, AREKIConfig
            from ..history_matching.localisation_jax import (
                build_localisation_matrix, parameter_coords_3d,
                well_observation_coords
            )

            areki_cfg = AREKIConfig(
                n_ensemble=request.n_ensemble,
                max_iterations=request.max_iterations,
                localisation_radius=request.localisation_radius,
                alpha_init=request.alpha_init if hasattr(request, 'alpha_init') else 10.0,
            )

            # Build synthetic prior ensemble (placeholder for real data)
            Nx, Ny, Nz = 46, 112, 22
            n_params = Nx * Ny * Nz * 2   # K + phi
            key = None
            try:
                import jax.numpy as jnp, jax.random as jrandom
                key = jrandom.PRNGKey(42)
                initial_params = jrandom.normal(key, (request.n_ensemble, n_params))
            except ImportError:
                initial_params = np.random.randn(request.n_ensemble, n_params)

            engine = AREKIEngine(
                cfg=areki_cfg,
                forward_fn=self._make_forward_fn(request),
                observations=self._load_observations(request),
                obs_error_cov=np.eye(1) * 1e6,
            )

            for progress in engine.run(initial_params,
                                       stop_event=self._stop_event):
                if self._stop_event.is_set():
                    log.info("HM stopped by client")
                    break

                # Compute ensemble statistics for UI fan chart
                params = progress.get('params')
                p10, p50, p90 = self._compute_ensemble_stats(params, request)

                self.ctx.append_hm_iteration({
                    'iteration':    progress['iteration'],
                    'mismatch':     progress['mismatch'],
                    'alpha':        progress['alpha'],
                    's_cumulative': progress['s_cumulative'],
                })

                yield pb.HMProgress(
                    iteration=progress['iteration'],
                    max_iterations=request.max_iterations,
                    mismatch=progress['mismatch'],
                    alpha=progress['alpha'],
                    s_cumulative=progress['s_cumulative'],
                    improvement_pct=progress.get('improvement_pct', 0.0),
                    converged=progress.get('converged', False),
                    p10_preview=p10.tolist() if p10 is not None else [],
                    p50_preview=p50.tolist() if p50 is not None else [],
                    p90_preview=p90.tolist() if p90 is not None else [],
                )

        except Exception as exc:
            log.exception("RunHistoryMatch failed: %s", exc)

    # ── GetEnsembleStats ──────────────────────────────────────────────────

    def GetEnsembleStats(self, request, context):
        stats = self.ctx.ensemble_stats
        return pb.EnsembleStatsResponse(
            well_name=request.well_name,
            quantity=request.quantity,
            p10=stats.get('p10', []),
            p50=stats.get('p50', []),
            p90=stats.get('p90', []),
            timesteps=stats.get('timesteps', []),
        )

    # ── GetDataMismatch ───────────────────────────────────────────────────

    def GetDataMismatch(self, request, context):
        mismatch = self.ctx.per_well_mismatch
        return pb.DataMismatchResponse(
            well_names=list(mismatch.keys()),
            rmse_values=list(mismatch.values()),
        )

    # ── GetParameterEnsemble ──────────────────────────────────────────────

    def GetParameterEnsemble(self, request, context):
        return pb.ParameterEnsembleResponse(
            parameter_name=request.parameter_name,
            mean=[],
            std_dev=[],
        )

    # ── StopHistoryMatch ──────────────────────────────────────────────────

    def StopHistoryMatch(self, request, context):
        self._stop_event.set()
        log.info("HM stop signal received")
        return pb.StopResponse(acknowledged=True)

    # ── Private helpers ────────────────────────────────────────────────────

    def _make_forward_fn(self, request):
        """
        Build the ensemble forward function.
        In production this calls the PINO surrogate.
        Here we return a synthetic placeholder.
        """
        def forward_fn(params: np.ndarray) -> np.ndarray:
            """params: [n_ensemble, n_params] → obs: [n_ensemble, n_obs]"""
            n = params.shape[0]
            noise = np.random.randn(n, 10) * 100.0
            return params[:, :10] * 0.5 + noise
        return forward_fn

    def _load_observations(self, request) -> np.ndarray:
        """Load observed well data. Placeholder returns synthetic values."""
        return np.zeros(10)

    def _compute_ensemble_stats(self, params, request):
        """Compute P10/P50/P90 from ensemble."""
        if params is None:
            return None, None, None
        try:
            p10 = np.percentile(params, 10, axis=0)[:20]
            p50 = np.percentile(params, 50, axis=0)[:20]
            p90 = np.percentile(params, 90, axis=0)[:20]
            return p10, p50, p90
        except Exception:
            return None, None, None
