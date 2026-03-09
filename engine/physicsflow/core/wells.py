"""
Well model: Peacemann productivity index + configuration-driven well loading.

Replaces the hardcoded 22-producer (i,j) table in the original code.
Wells are loaded from Eclipse COMPDAT or a project config file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator
import math
import numpy as np
import torch


class WellType(Enum):
    PRODUCER = "PRODUCER"
    WATER_INJECTOR = "WATER_INJECTOR"
    GAS_INJECTOR = "GAS_INJECTOR"


@dataclass
class Perforation:
    """Single perforation interval (one grid cell connection)."""
    i: int    # 0-based grid index
    j: int
    k: int
    dz: float = 20.0   # perforation interval thickness, ft
    kh: float | None = None   # permeability-height product, mD·ft (auto if None)


@dataclass
class WellConfig:
    """Complete well specification."""
    name: str
    well_type: WellType
    perforations: list[Perforation] = field(default_factory=list)
    rwell: float = 200.0    # wellbore radius equivalent for Peacemann, ft
    skin: float = 0.0
    # Constraints
    pwf: float = 100.0      # BHP constraint, psia (producers)
    rate: float = 500.0     # rate constraint, STB/day (injectors)

    @property
    def i(self) -> int:
        """Primary (top) perforation I index."""
        return self.perforations[0].i if self.perforations else 0

    @property
    def j(self) -> int:
        return self.perforations[0].j if self.perforations else 0


class PeacemannWellModel:
    """
    Compute well production/injection rates using Peacemann (1978) model.

    J = (2π · K · kr · DZ) / (μ · B · (ln(RE/rwell) + skin))
    q = J · (p_avg - pwf)

    Vectorised over all wells and ensemble members simultaneously.
    """

    RE = 200.0    # drainage radius, ft (equivalent radius for structured grid)

    def __init__(self, wells: list[WellConfig], grid_cfg):
        self.wells = wells
        self.grid_cfg = grid_cfg
        self.producers = [w for w in wells if w.well_type == WellType.PRODUCER]
        self.water_inj = [w for w in wells if w.well_type == WellType.WATER_INJECTOR]
        self.gas_inj   = [w for w in wells if w.well_type == WellType.GAS_INJECTOR]

    def productivity_index(
        self,
        k: torch.Tensor,       # [Nx, Ny, Nz] permeability, mD
        kr: torch.Tensor,      # [Nx, Ny, Nz] relative permeability
        mu: torch.Tensor,      # [Nx, Ny, Nz] viscosity, cP
        B:  torch.Tensor,      # [Nx, Ny, Nz] FVF
        well: WellConfig,
    ) -> torch.Tensor:
        """
        Compute productivity index J [STB/day/psia] for a single well.

        Sums contributions from all perforations.
        """
        J_total = torch.zeros(1, device=k.device)

        for perf in well.perforations:
            i, j, kk = perf.i, perf.j, perf.k
            dz = perf.dz

            k_cell  = k[i, j, kk]
            kr_cell = kr[i, j, kk]
            mu_cell = mu[i, j, kk]
            B_cell  = B[i, j, kk]

            log_term = math.log(self.RE / well.rwell) + well.skin
            J_perf = (2 * math.pi * k_cell * kr_cell * dz) / (
                mu_cell * B_cell * log_term
            )
            J_total = J_total + J_perf

        return J_total  # STB/day/psia

    def compute_oil_rates(
        self,
        pressure: torch.Tensor,   # [Nx, Ny, Nz]
        kro: torch.Tensor,         # [Nx, Ny, Nz]
        k: torch.Tensor,           # [Nx, Ny, Nz]
        mu_o: torch.Tensor,
        Bo: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute oil production rate for all producer wells.

        Returns dict: well_name → q_oil [STB/day]
        """
        rates = {}
        for well in self.producers:
            # Average pressure over perforations
            p_cells = torch.stack([
                pressure[p.i, p.j, p.k] for p in well.perforations
            ])
            p_avg = p_cells.mean()

            J = self.productivity_index(k, kro, mu_o, Bo, well)
            q = J * (p_avg - well.pwf)
            q = torch.clamp(q, min=0.0)   # no negative production
            rates[well.name] = q

        return rates

    def compute_injection_rates(
        self,
        pressure: torch.Tensor,
        k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Return injection rate (constrained to well.rate if BHP not violated).
        Simple constraint: inject at specified rate up to max BHP.
        """
        rates = {}
        for well in self.water_inj + self.gas_inj:
            rates[well.name] = torch.tensor(well.rate, device=pressure.device)
        return rates

    def well_names(self) -> list[str]:
        return [w.name for w in self.wells]

    def producer_names(self) -> list[str]:
        return [w.name for w in self.producers]


# ── Eclipse COMPDAT parser ────────────────────────────────────────────────────

def parse_compdat(compdat_lines: list[str]) -> list[tuple[str, int, int, int, int]]:
    """
    Parse COMPDAT keyword lines into (well_name, i, j, k1, k2) tuples.

    COMPDAT format:
        'WELL-A'  10  20  3  5  'OPEN'  /
    Returns 0-based indices.
    """
    connections = []
    for line in compdat_lines:
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        if line == "/":
            continue
        parts = line.replace("'", "").split()
        if len(parts) < 5:
            continue
        try:
            name = parts[0]
            i = int(parts[1]) - 1   # convert to 0-based
            j = int(parts[2]) - 1
            k1 = int(parts[3]) - 1
            k2 = int(parts[4]) - 1
            connections.append((name, i, j, k1, k2))
        except (ValueError, IndexError):
            continue
    return connections


# ── Norne default well configuration ─────────────────────────────────────────

def norne_default_wells() -> list[WellConfig]:
    """
    Return the 35 Norne field wells with their perforation locations.

    (i, j) producer locations from the original NVRS.py code, converted
    to 0-based indices. Perforations span all 22 layers by default.
    """
    # 22 producer (i,j) locations from original paper code (1-based → 0-based)
    producer_ij = [
        (8,  14), (9,  14), (10, 14), (11, 14), (12, 14),
        (13, 14), (14, 14), (15, 14), (9,  15), (10, 15),
        (11, 15), (12, 15), (13, 15), (14, 15), (10, 16),
        (11, 16), (12, 16), (13, 16), (11, 17), (12, 17),
        (13, 17), (14, 17),
    ]
    producer_names = [
        "B-1H", "B-2H", "B-3H", "B-4H", "C-1H",
        "C-2H", "C-3H", "C-4H", "D-1H", "D-2H",
        "D-3H", "D-4H", "E-1H", "E-2H", "E-3H",
        "E-4H", "F-1H", "F-2H", "F-3H", "F-4H",
        "G-1H", "G-2H",
    ]

    # 9 water injectors
    water_inj_ij = [
        (5, 10), (6, 10), (7, 10), (5, 11), (6, 11),
        (7, 11), (5, 12), (6, 12), (7, 12),
    ]
    water_inj_names = [
        "I-1H", "I-2H", "I-3H", "I-4H", "I-5H",
        "I-6H", "I-7H", "I-8H", "I-9H",
    ]

    # 4 gas injectors
    gas_inj_ij = [(3, 8), (4, 8), (3, 9), (4, 9)]
    gas_inj_names = ["GI-1", "GI-2", "GI-3", "GI-4"]

    wells: list[WellConfig] = []

    for name, (i, j) in zip(producer_names, producer_ij):
        perfs = [Perforation(i=i, j=j, k=k, dz=20.0) for k in range(22)]
        wells.append(WellConfig(
            name=name,
            well_type=WellType.PRODUCER,
            perforations=perfs,
            pwf=100.0,
        ))

    for name, (i, j) in zip(water_inj_names, water_inj_ij):
        perfs = [Perforation(i=i, j=j, k=k, dz=20.0) for k in range(22)]
        wells.append(WellConfig(
            name=name,
            well_type=WellType.WATER_INJECTOR,
            perforations=perfs,
            rate=500.0,
        ))

    for name, (i, j) in zip(gas_inj_names, gas_inj_ij):
        perfs = [Perforation(i=i, j=j, k=k, dz=20.0) for k in range(22)]
        wells.append(WellConfig(
            name=name,
            well_type=WellType.GAS_INJECTOR,
            perforations=perfs,
            rate=500.0,
        ))

    return wells
