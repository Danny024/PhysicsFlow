"""
Unit tests for Gaspari-Cohn localisation.

Tests:
- Boundary conditions (GC(0)=1, GC(r)=0 for r≥2c)
- Symmetry
- Positive-definiteness
- Localisation matrix shape
"""

import pytest
import numpy as np


class TestGaspariCohn:
    @pytest.fixture
    def gc_fn(self):
        try:
            from physicsflow.history_matching.localisation_jax import gaspari_cohn
            return gaspari_cohn
        except ImportError:
            pytest.skip("JAX not available")

    def test_zero_distance_returns_one(self, gc_fn):
        val = float(gc_fn(0.0, radius=10.0))
        assert abs(val - 1.0) < 1e-6, f"GC(0) = {val}, expected 1.0"

    def test_beyond_cutoff_returns_zero(self, gc_fn):
        """GC(d) = 0 for d ≥ 2c (twice the correlation radius)."""
        radius = 10.0
        val = float(gc_fn(2 * radius + 0.001, radius=radius))
        assert abs(val) < 1e-8, f"GC beyond cutoff = {val}, expected 0.0"

    def test_at_exact_cutoff_returns_zero(self, gc_fn):
        radius = 8.0
        val = float(gc_fn(2 * radius, radius=radius))
        assert abs(val) < 1e-6

    def test_monotone_decreasing(self, gc_fn):
        """GC should be monotone decreasing from 0 to 2c."""
        radius = 12.0
        distances = np.linspace(0, 2 * radius, 50)
        values = [float(gc_fn(d, radius)) for d in distances]
        diffs = np.diff(values)
        assert (diffs <= 1e-8).all(), "GC must be non-increasing"

    def test_positive_values(self, gc_fn):
        """GC values must be in [0, 1]."""
        radius = 10.0
        for d in np.linspace(0, 2.5 * radius, 100):
            val = float(gc_fn(d, radius))
            assert 0.0 <= val <= 1.0 + 1e-8, f"GC({d}) = {val} out of [0,1]"

    def test_symmetric(self, gc_fn):
        """GC is an even function: GC(d) = GC(-d)."""
        radius = 5.0
        for d in [1.0, 3.0, 5.0, 8.0]:
            assert abs(float(gc_fn(d, radius)) - float(gc_fn(-d, radius))) < 1e-8


class TestLocalisationMatrix:
    @pytest.fixture
    def build_fn(self):
        try:
            from physicsflow.history_matching.localisation_jax import build_localisation_matrix
            return build_localisation_matrix
        except ImportError:
            pytest.skip("JAX not available")

    def test_shape(self, build_fn):
        n_params, n_obs = 12, 6
        param_coords = np.random.rand(n_params, 3)
        obs_coords   = np.random.rand(n_obs, 3)
        L = np.array(build_fn(param_coords, obs_coords, radius=5.0))
        assert L.shape == (n_params, n_obs)

    def test_diagonal_near_one_for_collocated(self, build_fn):
        """When param and obs are at same location, L should be 1.0."""
        coords = np.array([[0.0, 0.0, 0.0]])
        L = np.array(build_fn(coords, coords, radius=10.0))
        assert abs(float(L[0, 0]) - 1.0) < 1e-6

    def test_far_cells_near_zero(self, build_fn):
        """Params far from observations should have L ≈ 0."""
        param_coords = np.array([[0.0, 0.0, 0.0]])
        obs_coords   = np.array([[100.0, 100.0, 100.0]])  # very far
        L = np.array(build_fn(param_coords, obs_coords, radius=5.0))
        assert abs(float(L[0, 0])) < 1e-6

    def test_values_in_range(self, build_fn):
        n_params, n_obs = 20, 8
        param_coords = np.random.rand(n_params, 3) * 50
        obs_coords   = np.random.rand(n_obs, 3) * 50
        L = np.array(build_fn(param_coords, obs_coords, radius=20.0))
        assert (L >= 0.0).all(), "Localisation weights must be non-negative"
        assert (L <= 1.0 + 1e-8).all(), "Localisation weights must be <= 1"


class TestParameterCoords3d:
    def test_shape(self):
        try:
            from physicsflow.history_matching.localisation_jax import parameter_coords_3d
        except ImportError:
            pytest.skip("JAX not available")

        nx, ny, nz = 4, 5, 3
        coords = np.array(parameter_coords_3d(nx, ny, nz))
        assert coords.shape == (nx * ny * nz, 3)

    def test_values_in_grid(self):
        try:
            from physicsflow.history_matching.localisation_jax import parameter_coords_3d
        except ImportError:
            pytest.skip("JAX not available")

        nx, ny, nz = 4, 5, 3
        coords = np.array(parameter_coords_3d(nx, ny, nz))
        assert coords[:, 0].min() >= 0
        assert coords[:, 0].max() <= nx - 1
        assert coords[:, 1].max() <= ny - 1
        assert coords[:, 2].max() <= nz - 1
