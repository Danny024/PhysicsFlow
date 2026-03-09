"""
Unit tests for PhysicsFlow PVT correlations.

Tests validate:
- PVT property ranges against reservoir engineering expectations
- Monotonicity of correlations with pressure
- Unit conversion helpers
- PVTConfig defaults
"""

import pytest
import torch
import numpy as np

from physicsflow.core.pvt import BlackOilPVT, PVTConfig


@pytest.fixture
def pvt():
    cfg = PVTConfig.norne_defaults()
    return BlackOilPVT(cfg)


@pytest.fixture
def pressures():
    """Representative pressure range in Pa (50–500 bar)."""
    return torch.linspace(50e5, 500e5, 20)


class TestPVTConfig:
    def test_norne_defaults_exist(self):
        cfg = PVTConfig.norne_defaults()
        assert cfg is not None

    def test_norne_default_values_reasonable(self):
        cfg = PVTConfig.norne_defaults()
        assert 20.0 < cfg.api_gravity < 60.0, "API gravity should be 20–60"
        assert 0.5 < cfg.gas_gravity < 1.5, "Gas gravity should be 0.5–1.5"


class TestBlackOilPVT:
    def test_gas_viscosity_positive(self, pvt, pressures):
        mu_g = pvt.mu_g(pressures)
        assert (mu_g > 0).all(), "Gas viscosity must be positive"

    def test_gas_viscosity_range(self, pvt, pressures):
        mu_g = pvt.mu_g(pressures)
        # Reservoir gas: 0.01–0.05 cP = 1e-5 to 5e-5 Pa·s
        assert (mu_g < 1.0e-2).all(), "Gas viscosity too high (> 10 cP)"
        assert (mu_g > 1.0e-7).all(), "Gas viscosity too low"

    def test_solution_gor_increases_with_pressure(self, pvt, pressures):
        Rs = pvt.Rs(pressures)
        # GOR should be non-decreasing below bubble point
        diff = Rs[1:] - Rs[:-1]
        assert (diff >= -1e-3).all(), "GOR should not decrease with pressure (below bubble point)"

    def test_oil_fvf_positive(self, pvt, pressures):
        Bo = pvt.Bo(pressures)
        assert (Bo > 0).all(), "Oil FVF must be positive"
        assert (Bo >= 1.0).all(), "Oil FVF should be >= 1.0 (oil expands)"

    def test_gas_fvf_decreases_with_pressure(self, pvt, pressures):
        Bg = pvt.Bg(pressures)
        assert (Bg > 0).all(), "Gas FVF must be positive"
        # Higher pressure → gas compressed → smaller FVF
        diff = Bg[1:] - Bg[:-1]
        assert (diff <= 1e-8).all(), "Gas FVF should decrease with pressure"

    def test_oil_viscosity_positive(self, pvt, pressures):
        mu_o = pvt.mu_o(pressures)
        assert (mu_o > 0).all(), "Oil viscosity must be positive"

    def test_water_fvf_near_unity(self, pvt, pressures):
        Bw = pvt.Bw(pressures)
        assert (Bw > 0.9).all(), "Water FVF should be near 1.0"
        assert (Bw < 1.1).all(), "Water FVF should be near 1.0"

    def test_water_viscosity_range(self, pvt, pressures):
        mu_w = pvt.mu_w(pressures)
        # Water: 0.3–1.0 cP
        assert (mu_w > 0).all()
        assert (mu_w < 5e-3).all(), "Water viscosity should be < 5 cP"

    def test_all_properties_returns_dict(self, pvt, pressures):
        props = pvt.all_properties(pressures)
        expected_keys = {'mu_g', 'Rs', 'Bo', 'Bg', 'mu_o', 'mu_w', 'Bw'}
        assert expected_keys.issubset(set(props.keys()))

    def test_gradient_flows(self, pvt):
        """PVT ops must be differentiable for PINO training."""
        p = torch.linspace(100e5, 300e5, 5, requires_grad=True)
        Bo = pvt.Bo(p)
        Bo.sum().backward()
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    def test_batch_pressure(self, pvt):
        """PVT should work with batched 3D pressure tensors."""
        p = torch.rand(2, 10, 10, 5) * 300e5 + 50e5
        Bo = pvt.Bo(p)
        assert Bo.shape == p.shape
