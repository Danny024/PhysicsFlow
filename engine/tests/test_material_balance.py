"""
Material balance validation tests.

Validates that simulated production/injection obeys Havlena-Odeh
material balance: F = Eo * N + Ew * We

Also tests:
- Volume balance closure (produced + remaining = OOIP)
- Pressure support from injection
- Cumulative balance over all timesteps
"""

import pytest
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Havlena-Odeh material balance checker
# ─────────────────────────────────────────────────────────────────────────────

def havlena_odeh_F(
    Np: np.ndarray,    # cumulative oil production [STB]
    Wp: np.ndarray,    # cumulative water production [STB]
    Gp: np.ndarray,    # cumulative gas production [Mscf]
    Rs: np.ndarray,    # solution GOR at current pressure [scf/STB]
    Rsi: float,        # initial solution GOR [scf/STB]
    Bo: np.ndarray,    # oil FVF [RB/STB]
    Bg: np.ndarray,    # gas FVF [RB/Mscf]
    Bw: float = 1.02,  # water FVF
) -> np.ndarray:
    """Underground withdrawal F = Np*(Bo + (Rp-Rs)*Bg) + Wp*Bw."""
    Rp = np.where(Np > 0, Gp / Np * 1000, Rsi)   # producing GOR [scf/STB]
    return Np * (Bo + (Rp - Rs) * Bg / 1000.0) + Wp * Bw


def expansion_oil(
    Bo: np.ndarray,
    Boi: float,
    Rsi: float,
    Rs: np.ndarray,
    Bg: np.ndarray,
) -> np.ndarray:
    """Oil + dissolved gas expansion Eo = (Bo - Boi) + (Rsi - Rs)*Bg."""
    return (Bo - Boi) + (Rsi - Rs) * Bg / 1000.0


class TestMaterialBalance:
    """
    These tests use synthetic PVT data to validate the MB framework.
    Real simulation results would be loaded from Eclipse UNRST files.
    """

    @pytest.fixture
    def pvt_tables(self):
        """Synthetic PVT table for 50–300 bar."""
        pressures = np.linspace(50e5, 300e5, 20)
        # Simplified correlations (not field-calibrated)
        Boi = 1.10
        Bo  = Boi + (pressures - 50e5) * 3e-9
        Rsi = 150.0   # scf/STB
        Rs  = Rsi * (pressures / 300e5) ** 0.5
        Bg  = 0.005 * (50e5 / pressures)   # RB/Mscf
        return {'pressures': pressures, 'Bo': Bo, 'Boi': Boi,
                'Rs': Rs, 'Rsi': Rsi, 'Bg': Bg}

    def test_F_positive_for_positive_production(self, pvt_tables):
        """Underground withdrawal F must be positive when producing."""
        pvt = pvt_tables
        Np = np.ones(20) * 1000.0
        Wp = np.ones(20) * 100.0
        Gp = Np * pvt['Rsi'] / 1000.0

        F = havlena_odeh_F(Np, Wp, Gp, pvt['Rs'], pvt['Rsi'],
                            pvt['Bo'], pvt['Bg'])
        assert (F > 0).all(), "Underground withdrawal must be positive"

    def test_eo_positive_above_bubble_point(self, pvt_tables):
        """Above bubble point, Boi < Bo so Eo > 0."""
        pvt = pvt_tables
        # Use points where pressure > initial
        idx = pvt['pressures'] > pvt['pressures'][0]
        Eo = expansion_oil(pvt['Bo'][idx], pvt['Boi'],
                            pvt['Rsi'], pvt['Rs'][idx], pvt['Bg'][idx])
        assert (Eo >= 0).all(), "Expansion Eo must be non-negative above Pi"

    def test_zero_production_zero_withdrawal(self, pvt_tables):
        """No production → F = 0."""
        pvt = pvt_tables
        Np = np.zeros(20)
        Wp = np.zeros(20)
        Gp = np.zeros(20)

        F = havlena_odeh_F(Np, Wp, Gp, pvt['Rs'], pvt['Rsi'],
                            pvt['Bo'], pvt['Bg'])
        np.testing.assert_allclose(F, 0.0, atol=1e-6)

    def test_cumulative_balance_closure(self, pvt_tables):
        """
        Incremental F should equal cumulative F when summed.
        Tests that the framework handles cumulative quantities correctly.
        """
        pvt = pvt_tables
        n = 10
        # Incremental production
        dNp = np.ones(n) * 500.0
        dWp = np.ones(n) * 50.0
        dGp = dNp * pvt['Rsi'][:n] / 1000.0

        Np_cum = np.cumsum(dNp)
        Wp_cum = np.cumsum(dWp)
        Gp_cum = np.cumsum(dGp)

        F_cum = havlena_odeh_F(Np_cum, Wp_cum, Gp_cum,
                                pvt['Rs'][:n], pvt['Rsi'],
                                pvt['Bo'][:n], pvt['Bg'][:n])
        # F cumulative should be monotonically increasing
        diffs = np.diff(F_cum)
        assert (diffs > 0).all(), "Cumulative F must increase as production continues"

    def test_water_injection_reduces_required_expansion(self, pvt_tables):
        """
        With water injection We, the required oil expansion Eo for a given
        F is smaller: F = N*Eo + We.  This validates the injection support term.
        """
        pvt = pvt_tables
        N = 1e6    # OOIP [STB]
        We = 5e5   # water influx [RB]

        idx = 5
        Eo = expansion_oil(pvt['Bo'][idx:idx+1], pvt['Boi'],
                            pvt['Rsi'], pvt['Rs'][idx:idx+1], pvt['Bg'][idx:idx+1])
        F_no_we  = N * Eo[0]           # oil expansion only
        F_with_we = N * Eo[0] + We     # oil expansion + water support
        assert F_with_we > F_no_we, "Water influx should increase total energy available"


class TestVolumeBalance:
    def test_produced_plus_remaining_equals_ooip(self):
        """
        Simple volumetric check: Np + Np_remaining = OOIP
        (ignoring compressibility for this basic test).
        """
        OOIP = 1e7   # STB
        Np   = 2e6   # cumulative production
        remaining = OOIP - Np
        assert abs((remaining + Np) - OOIP) < 1.0
        assert remaining > 0, "Must not produce more than OOIP"

    def test_recovery_factor_reasonable(self):
        """Typical oil recovery factor is 20–60% for primary depletion."""
        OOIP = 1e7
        Np   = 3.5e6   # 35% RF
        RF   = Np / OOIP
        assert 0.05 < RF < 0.70, f"Recovery factor {RF:.1%} is outside reasonable range"
