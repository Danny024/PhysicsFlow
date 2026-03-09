"""
Unit tests for well model (Peacemann productivity index and rates).
"""

import pytest
import torch
import numpy as np

from physicsflow.core.wells import (
    WellConfig, WellType, Perforation, PeacemannWellModel,
    norne_default_wells, parse_compdat,
)


@pytest.fixture
def single_perf_well():
    return WellConfig(
        name='B-1H',
        well_type=WellType.PRODUCER,
        perforations=[Perforation(i=10, j=20, k=5,
                                  skin=0.0, wellbore_radius=0.108)],
        bhp_limit=150e5,
    )


@pytest.fixture
def peacemann():
    return PeacemannWellModel()


class TestPerforation:
    def test_perf_indices_non_negative(self, single_perf_well):
        for p in single_perf_well.perforations:
            assert p.i >= 0
            assert p.j >= 0
            assert p.k >= 0

    def test_wellbore_radius_positive(self, single_perf_well):
        for p in single_perf_well.perforations:
            assert p.wellbore_radius > 0


class TestPeacemannWellModel:
    def test_productivity_index_positive(self, peacemann, single_perf_well):
        perm = torch.ones(46, 112, 22) * 100.0   # 100 mD uniform
        J = peacemann.productivity_index(single_perf_well, perm)
        assert J > 0, "Productivity index must be positive"

    def test_productivity_index_scales_with_perm(self, peacemann, single_perf_well):
        perm_low  = torch.ones(46, 112, 22) * 10.0
        perm_high = torch.ones(46, 112, 22) * 1000.0
        J_low  = peacemann.productivity_index(single_perf_well, perm_low)
        J_high = peacemann.productivity_index(single_perf_well, perm_high)
        assert J_high > J_low, "Higher K → higher PI"

    def test_oil_rate_positive_for_producer(self, peacemann, single_perf_well):
        perm     = torch.ones(46, 112, 22) * 100.0
        pressure = torch.ones(46, 112, 22) * 250e5   # reservoir pressure 250 bar
        bhp      = 150e5                              # BHP 150 bar → drawdown 100 bar
        mu_o     = torch.ones(46, 112, 22) * 1.5e-3
        Bo       = torch.ones(46, 112, 22) * 1.2

        q_oil = peacemann.compute_oil_rates(
            single_perf_well, pressure, bhp, mu_o, Bo, perm
        )
        assert q_oil > 0, "Producer with positive drawdown must produce"

    def test_injection_rate_positive(self, peacemann):
        injector = WellConfig(
            name='E-3H',
            well_type=WellType.INJECTOR,
            perforations=[Perforation(i=5, j=40, k=8, skin=0.0, wellbore_radius=0.108)],
            bhp_limit=400e5,
        )
        perm     = torch.ones(46, 112, 22) * 200.0
        pressure = torch.ones(46, 112, 22) * 230e5
        bhp_inj  = 350e5   # injection BHP > reservoir → positive injection
        mu_w     = torch.ones(46, 112, 22) * 0.5e-3
        Bw       = torch.ones(46, 112, 22) * 1.02

        q_inj = peacemann.compute_injection_rates(
            injector, pressure, bhp_inj, mu_w, Bw, perm
        )
        assert q_inj > 0, "Injector with positive ΔP must inject"


class TestNorneDefaultWells:
    def test_returns_list(self):
        wells = norne_default_wells()
        assert isinstance(wells, list)
        assert len(wells) > 0

    def test_has_producers_and_injectors(self):
        wells = norne_default_wells()
        types = [w.well_type for w in wells]
        assert WellType.PRODUCER in types
        assert WellType.INJECTOR in types

    def test_all_perfs_in_grid(self):
        wells = norne_default_wells()
        for w in wells:
            for p in w.perforations:
                assert 0 <= p.i < 46
                assert 0 <= p.j < 112
                assert 0 <= p.k < 22

    def test_well_names_unique(self):
        wells = norne_default_wells()
        names = [w.name for w in wells]
        assert len(names) == len(set(names)), "Well names must be unique"


class TestParseCompdat:
    SAMPLE_COMPDAT = """
    COMPDAT
      'B-1H' 10 20 5 8 OPEN 1* 0.108 /
      'E-3H' 5 40 3 6 OPEN 1* 0.108 /
    /
    """

    def test_parse_returns_list(self):
        wells = parse_compdat(self.SAMPLE_COMPDAT)
        assert isinstance(wells, list)

    def test_parse_correct_count(self):
        wells = parse_compdat(self.SAMPLE_COMPDAT)
        # 2 wells × multiple perfs
        assert len(wells) >= 2

    def test_parse_zero_indexed(self):
        wells = parse_compdat(self.SAMPLE_COMPDAT)
        for w in wells:
            for p in w.perforations:
                # parse_compdat converts 1-based COMPDAT to 0-based
                assert p.i >= 0
                assert p.j >= 0
                assert p.k >= 0
