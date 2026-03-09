"""
Unit tests for αREKI Kalman update step.

Tests:
- Shape consistency
- Symmetry of covariance matrices
- Convergence properties
- SVD-based matrix inversion accuracy
"""

import pytest
import numpy as np


class TestKalmanUpdate:
    """Tests for AREKIEngine._kalman_update."""

    @pytest.fixture
    def small_engine(self):
        """Small engine for fast testing (no JAX needed)."""
        try:
            from physicsflow.history_matching.areki_jax import AREKIEngine, AREKIConfig
            cfg = AREKIConfig(n_ensemble=20, max_iterations=5,
                              localisation_radius=5.0)
            fwd = lambda p: p[:, :4] * 0.5  # synthetic linear forward model
            obs = np.zeros(4)
            cov = np.eye(4) * 1e4
            engine = AREKIEngine(cfg=cfg, forward_fn=fwd,
                                  observations=obs, obs_error_cov=cov)
            return engine
        except ImportError:
            pytest.skip("JAX not available")

    def test_kalman_output_shape(self, small_engine):
        """Updated params must have same shape as input."""
        n_ens, n_params, n_obs = 20, 10, 4
        params = np.random.randn(n_ens, n_params)
        G      = np.random.randn(n_ens, n_obs)
        alpha  = 5.0

        updated = small_engine._kalman_update_numpy(params, G, alpha)
        assert updated.shape == params.shape

    def test_kalman_reduces_mismatch(self, small_engine):
        """Single Kalman step should reduce ensemble mean mismatch."""
        n_ens, n_params = 20, 4
        # Linear model: G(m) = m, obs = zeros
        params = np.random.randn(n_ens, n_params) + 5.0   # biased far from obs
        obs = np.zeros(n_params)
        G   = params.copy()

        # Compute mismatch before
        before = np.mean((np.mean(G, axis=0) - obs) ** 2)

        cov = np.eye(n_params) * 1.0
        engine = small_engine
        engine.observations = obs
        engine.obs_error_cov = cov

        updated = engine._kalman_update_numpy(params, G, alpha=1.0)
        G_after = updated.copy()
        after = np.mean((np.mean(G_after, axis=0) - obs) ** 2)

        assert after < before, "Kalman update should reduce mismatch"

    def test_ensemble_spread_maintained(self, small_engine):
        """Kalman update should not collapse ensemble to a single point."""
        n_ens, n_params = 20, 5
        params = np.random.randn(n_ens, n_params)
        G = params[:, :4] * 0.5

        updated = small_engine._kalman_update_numpy(params, G, alpha=10.0)
        std_after = np.std(updated, axis=0)
        # Ensemble should still have some spread
        assert np.mean(std_after) > 1e-6, "Ensemble collapsed to single point"


class TestSVDSolve:
    """Tests for the SVD-based matrix inversion used in Kalman gain."""

    def test_svd_solve_exact_for_identity(self):
        """K = Cyd · (Cdd + αCdd)^{-1} should recover K = I/(1+α) for identity."""
        try:
            from physicsflow.history_matching.areki_jax import AREKIEngine, AREKIConfig
        except ImportError:
            pytest.skip("JAX not available")

        n = 5
        Cyd = np.eye(n)
        A   = np.eye(n) * 2.0  # Cdd + alpha*Cobs = 2I

        cfg = AREKIConfig(n_ensemble=10, max_iterations=1)
        eng = AREKIEngine(cfg=cfg,
                           forward_fn=lambda p: p,
                           observations=np.zeros(n),
                           obs_error_cov=np.eye(n))
        K = eng._svd_solve_numpy(Cyd, A)
        expected = np.eye(n) * 0.5
        np.testing.assert_allclose(K, expected, atol=1e-6)


class TestAdaptiveAlpha:
    def test_alpha_positive(self):
        try:
            from physicsflow.history_matching.areki_jax import AREKIEngine, AREKIConfig
        except ImportError:
            pytest.skip("JAX not available")

        cfg = AREKIConfig(n_ensemble=10, max_iterations=5,
                          localisation_radius=5.0)
        eng = AREKIEngine(cfg=cfg,
                           forward_fn=lambda p: p[:, :3],
                           observations=np.zeros(3),
                           obs_error_cov=np.eye(3))

        G        = np.random.randn(10, 3)
        mismatch = 1000.0
        alpha = eng._compute_alpha_numpy(G, mismatch, iteration=1)
        assert alpha > 0, "Alpha must be positive"

    def test_alpha_large_for_high_mismatch(self):
        """Large initial mismatch → large alpha (cautious first step)."""
        try:
            from physicsflow.history_matching.areki_jax import AREKIEngine, AREKIConfig
        except ImportError:
            pytest.skip("JAX not available")

        cfg = AREKIConfig(n_ensemble=10, max_iterations=5)
        eng = AREKIEngine(cfg=cfg,
                           forward_fn=lambda p: p[:, :2],
                           observations=np.zeros(2),
                           obs_error_cov=np.eye(2))

        G_high = np.random.randn(10, 2)
        G_low  = G_high * 0.01

        alpha_high = eng._compute_alpha_numpy(G_high, 10000.0, 1)
        alpha_low  = eng._compute_alpha_numpy(G_low, 0.01, 1)

        assert alpha_high >= alpha_low, \
            "Higher mismatch should produce larger (more cautious) alpha"
