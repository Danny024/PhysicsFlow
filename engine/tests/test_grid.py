"""
Unit tests for ReservoirGrid.
"""

import pytest
import numpy as np
import torch

from physicsflow.core.grid import GridConfig, ReservoirGrid


@pytest.fixture
def small_grid():
    cfg = GridConfig(nx=4, ny=6, nz=3, dx=50.0, dy=50.0, dz=10.0, depth=1000.0)
    return ReservoirGrid(cfg)


@pytest.fixture
def norne_grid():
    cfg = GridConfig.norne()
    return ReservoirGrid(cfg)


class TestGridConfig:
    def test_norne_dimensions(self):
        cfg = GridConfig.norne()
        assert cfg.nx == 46
        assert cfg.ny == 112
        assert cfg.nz == 22

    def test_positive_dimensions_required(self):
        with pytest.raises((ValueError, Exception)):
            GridConfig(nx=-1, ny=10, nz=5)


class TestReservoirGrid:
    def test_n_cells(self, small_grid):
        assert small_grid.n_cells == 4 * 6 * 3

    def test_n_active_cells_leq_total(self, small_grid):
        assert small_grid.n_active_cells <= small_grid.n_cells

    def test_transmissibility_x_shape(self, small_grid):
        Tx = small_grid.transmissibility_x()
        assert Tx.shape == (small_grid.cfg.nx - 1, small_grid.cfg.ny, small_grid.cfg.nz)

    def test_transmissibility_y_shape(self, small_grid):
        Ty = small_grid.transmissibility_y()
        assert Ty.shape == (small_grid.cfg.nx, small_grid.cfg.ny - 1, small_grid.cfg.nz)

    def test_transmissibility_z_shape(self, small_grid):
        Tz = small_grid.transmissibility_z()
        assert Tz.shape == (small_grid.cfg.nx, small_grid.cfg.ny, small_grid.cfg.nz - 1)

    def test_transmissibility_positive(self, small_grid):
        Tx = small_grid.transmissibility_x()
        assert (Tx >= 0).all(), "Transmissibility must be non-negative"

    def test_to_torch_device_cpu(self, small_grid):
        tensors = small_grid.to_torch('cpu')
        assert isinstance(tensors, dict)
        for k, v in tensors.items():
            assert isinstance(v, torch.Tensor)
            assert v.device.type == 'cpu'

    def test_flatten_unflatten_roundtrip(self, small_grid):
        """Flatten then unflatten should reproduce original array."""
        arr = np.random.rand(small_grid.cfg.nx, small_grid.cfg.ny, small_grid.cfg.nz)
        flat    = small_grid.flatten(arr)
        restored = small_grid.unflatten(flat)
        np.testing.assert_array_almost_equal(arr, restored)

    def test_norne_grid_size(self, norne_grid):
        assert norne_grid.n_cells == 46 * 112 * 22
