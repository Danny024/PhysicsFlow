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
    INJECTOR = "INJECTOR"
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
    skin: float = 0.0
    wellbore_radius: float = 0.108   # ft


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
    bhp_limit: float | None = None   # optional BHP limit (Pa or psia depending on context)

    @property
    def i(self) -> int:
        """Primary (top) perforation I index."""
        return self.perforations[0].i if self.perforations else 0

    @property
    def j(self) -> int:
        return self.perforations[0].j if self.perforations else 0

    def is_injector(self) -> bool:
        return self.well_type in (
            WellType.INJECTOR, WellType.WATER_INJECTOR, WellType.GAS_INJECTOR
        )


class PeacemannWellModel:
    """
    Compute well production/injection rates using Peacemann (1978) model.

    J = (2π · K · kr · DZ) / (μ · B · (ln(RE/rwell) + skin))
    q = J · (p_avg - pwf)

    Supports both single-well API (pass well to each method) and
    multi-well API (pass wells list to constructor for batch operations).
    """

    RE = 200.0    # drainage radius, ft

    def __init__(self, wells: list[WellConfig] | None = None, grid_cfg=None):
        self.wells = wells or []
        self.grid_cfg = grid_cfg
        self.producers = [w for w in self.wells if w.well_type == WellType.PRODUCER]
        self.water_inj = [w for w in self.wells
                          if w.well_type in (WellType.INJECTOR, WellType.WATER_INJECTOR)]
        self.gas_inj   = [w for w in self.wells if w.well_type == WellType.GAS_INJECTOR]

    def productivity_index(
        self,
        well: WellConfig,
        k: torch.Tensor,    # [Nx, Ny, Nz] permeability, mD
        kr: torch.Tensor | None = None,   # relative permeability (default 1.0)
        mu: torch.Tensor | None = None,   # viscosity (default 1.0 cP)
        B: torch.Tensor | None = None,    # FVF (default 1.0)
    ) -> torch.Tensor:
        """
        Compute productivity index J [STB/day/psia] for a single well.
        Sums contributions from all perforations.
        """
        J_total = torch.zeros(1, device=k.device)

        for perf in well.perforations:
            ii, jj, kk = perf.i, perf.j, perf.k
            dz = perf.dz

            k_cell  = k[ii, jj, kk]
            kr_cell = kr[ii, jj, kk] if kr is not None else torch.ones(1, device=k.device)[0]
            mu_cell = mu[ii, jj, kk] if mu is not None else torch.ones(1, device=k.device)[0]
            B_cell  = B[ii, jj, kk]  if B  is not None else torch.ones(1, device=k.device)[0]

            rw = perf.wellbore_radius if perf.wellbore_radius > 0 else well.rwell
            log_term = math.log(self.RE / rw) + perf.skin + well.skin
            J_perf = (2 * math.pi * k_cell * kr_cell * dz) / (
                mu_cell * B_cell * log_term
            )
            J_total = J_total + J_perf

        return J_total  # STB/day/psia

    def compute_oil_rates(
        self,
        well: WellConfig,
        pressure: torch.Tensor,   # [Nx, Ny, Nz]
        bhp: float,
        mu_o: torch.Tensor,
        Bo: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute oil production rate for a single producer well [STB/day].
        """
        p_cells = torch.stack([pressure[p.i, p.j, p.k] for p in well.perforations])
        p_avg = p_cells.mean()
        J = self.productivity_index(well, k, mu=mu_o, B=Bo)
        q = J * (p_avg - bhp)
        return torch.clamp(q, min=0.0)

    def compute_injection_rates(
        self,
        well: WellConfig,
        pressure: torch.Tensor,   # [Nx, Ny, Nz]
        bhp_inj: float,
        mu: torch.Tensor,
        B: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute injection rate for a single injector well [STB/day].
        Positive when bhp_inj > reservoir pressure.
        """
        p_cells = torch.stack([pressure[p.i, p.j, p.k] for p in well.perforations])
        p_avg = p_cells.mean()
        J = self.productivity_index(well, k, mu=mu, B=B)
        q = J * (bhp_inj - p_avg)
        return torch.clamp(q, min=0.0)

    # ── Batch helpers (used when wells list is passed to constructor) ──────────

    def compute_all_oil_rates(
        self,
        pressure: torch.Tensor,
        kro: torch.Tensor,
        k: torch.Tensor,
        mu_o: torch.Tensor,
        Bo: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Batch version: compute oil rates for all producers. Returns dict."""
        rates = {}
        for well in self.producers:
            p_cells = torch.stack([pressure[p.i, p.j, p.k] for p in well.perforations])
            p_avg = p_cells.mean()
            J = self.productivity_index(well, k, kro, mu_o, Bo)
            q = J * (p_avg - well.pwf)
            rates[well.name] = torch.clamp(q, min=0.0)
        return rates

    def compute_all_injection_rates(
        self,
        pressure: torch.Tensor,
        k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Batch version: injection rates for all injectors. Returns dict."""
        rates = {}
        for well in self.water_inj + self.gas_inj:
            rates[well.name] = torch.tensor(well.rate, device=pressure.device)
        return rates

    def well_names(self) -> list[str]:
        return [w.name for w in self.wells]

    def producer_names(self) -> list[str]:
        return [w.name for w in self.producers]


# ── Eclipse COMPDAT parser ────────────────────────────────────────────────────

def parse_compdat(compdat_input) -> list[WellConfig]:
    """
    Parse COMPDAT keyword lines into a list of WellConfig objects.

    Accepts either a multi-line string or a list of strings.

    COMPDAT format:
        'WELL-A'  10  20  3  5  'OPEN'  1*  0.108  /
    Returns WellConfig objects with 0-based perforation indices.
    """
    if isinstance(compdat_input, str):
        lines = compdat_input.split('\n')
    else:
        lines = list(compdat_input)

    well_perfs: dict[str, list[Perforation]] = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("--") or line in ("COMPDAT", "/"):
            continue
        # Remove trailing /
        if line.endswith("/"):
            line = line[:-1].strip()
        if not line:
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
            # Extract wellbore radius from field 8 if present and not '1*'
            rw = 0.108  # default, ft
            if len(parts) > 7 and parts[7] not in ('1*', '1'):
                try:
                    rw = float(parts[7])
                except ValueError:
                    pass
            if name not in well_perfs:
                well_perfs[name] = []
            for k in range(k1, k2 + 1):
                well_perfs[name].append(
                    Perforation(i=i, j=j, k=k, skin=0.0, wellbore_radius=rw)
                )
        except (ValueError, IndexError):
            continue

    return [
        WellConfig(name=name, well_type=WellType.PRODUCER, perforations=perfs)
        for name, perfs in well_perfs.items()
    ]


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
            well_type=WellType.INJECTOR,
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
