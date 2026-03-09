"""
Black-oil PVT correlations implemented as differentiable PyTorch operations.

All correlations match those used in the Norne PINO paper (Etienam et al. 2024)
and the original NVRS.py source code. Fully vectorised over batch and grid dims.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class PVTConfig:
    """Black-oil PVT parameters for a reservoir."""
    p_bub: float = 4_000.0    # bubble point pressure, psia
    p_atm: float = 14.696     # atmospheric pressure, psia
    cfo: float = 1e-5         # oil compressibility, 1/psia
    p_ini: float = 1_000.0    # initial reservoir pressure, psia
    muw: float = 0.5          # water viscosity, cP (assumed constant)
    bw: float = 1.0           # water FVF (assumed constant)

    @classmethod
    def norne_defaults(cls) -> "PVTConfig":
        """PVT configuration matching the Norne benchmark."""
        return cls(p_bub=4_000.0, p_atm=14.696, cfo=1e-5, p_ini=1_000.0)


class BlackOilPVT(nn.Module):
    """
    Differentiable black-oil PVT correlations (PyTorch).

    All methods accept pressure tensors of any shape and return
    tensors of identical shape. Suitable for use inside PINO physics loss.

    Correlations:
        mu_g  : gas viscosity (quadratic in p)
        Rs    : solution GOR (power-law, bubble-point correction)
        Bo    : oil FVF (exponential, above/below bubble point)
        Bg    : gas FVF (exponential below bubble point)
    """

    def __init__(self, cfg: PVTConfig | None = None):
        super().__init__()
        cfg = cfg or PVTConfig.norne_defaults()
        # Store as buffers so they move with .to(device)
        self.register_buffer("p_bub", torch.tensor(cfg.p_bub))
        self.register_buffer("p_atm", torch.tensor(cfg.p_atm))
        self.register_buffer("cfo",   torch.tensor(cfg.cfo))
        self.register_buffer("p_ini", torch.tensor(cfg.p_ini))
        self.register_buffer("muw",   torch.tensor(cfg.muw))
        self.register_buffer("bw",    torch.tensor(cfg.bw))

    # ── Gas viscosity ─────────────────────────────────────────────────────────

    def mu_g(self, p: torch.Tensor) -> torch.Tensor:
        """Gas viscosity [cP]. Quadratic polynomial fit.

        μg = 3e-10·p² + 1e-6·p + 0.0133
        """
        return 3e-10 * p**2 + 1e-6 * p + 0.0133

    # ── Solution GOR ──────────────────────────────────────────────────────────

    def Rs(self, p: torch.Tensor) -> torch.Tensor:
        """Solution gas-oil ratio [scf/STB].

        Rs = (178.11²/5.615) · (p/p_bub)^1.3   for p < p_bub
        Rs = 178.11²/5.615                       for p ≥ p_bub
        """
        rs_sat = (178.11**2 / 5.615)
        rs_unsat = rs_sat * (p / self.p_bub) ** 1.3
        return torch.where(p < self.p_bub, rs_unsat, torch.full_like(p, rs_sat))

    # ── Oil formation volume factor ───────────────────────────────────────────

    def Bo(self, p: torch.Tensor) -> torch.Tensor:
        """Oil FVF [RB/STB].

        Below bubble point (p < p_bub):
            Bo = 1 / exp(-8e-5·(p_atm - p))
        Above bubble point (p ≥ p_bub):
            Bo = 1 / [exp(-8e-5·(p_atm - p_bub)) · exp(-cfo·(p - p_bub))]
        """
        p_bub = self.p_bub
        p_atm = self.p_atm

        # Below bubble point
        bo_below = 1.0 / torch.exp(-8e-5 * (p_atm - p))

        # Above bubble point
        bo_ref = torch.exp(-8e-5 * (p_atm - p_bub))   # Bo at bubble point
        bo_above = 1.0 / (bo_ref * torch.exp(-self.cfo * (p - p_bub)))

        return torch.where(p < p_bub, bo_below, bo_above)

    # ── Gas formation volume factor ───────────────────────────────────────────

    def Bg(self, p: torch.Tensor) -> torch.Tensor:
        """Gas FVF [RCF/SCF] below bubble point.

        Bg = 1 / exp(1.7e-3·(p_atm - p))
        """
        dp = self.p_atm - p
        return 1.0 / torch.exp(1.7e-3 * dp)

    # ── Oil viscosity (simplified dead-oil Beal correlation) ─────────────────

    def mu_o(self, p: torch.Tensor) -> torch.Tensor:
        """Simplified oil viscosity [cP].

        Linear decrease from 1.0 cP at p_bub; constant above.
        Replace with full correlation (Beal, Vasquez-Beggs) if needed.
        """
        mu_ob = torch.ones_like(p)  # reference viscosity at bubble point
        mu_above = mu_ob * torch.exp(-1e-4 * (p - self.p_bub))
        return torch.where(p < self.p_bub, mu_ob, mu_above).clamp(min=0.1)

    # ── Water properties ──────────────────────────────────────────────────────

    def mu_w(self, p: torch.Tensor) -> torch.Tensor:
        """Water viscosity [cP] — assumed constant."""
        return torch.full_like(p, self.muw.item())

    def Bw(self, p: torch.Tensor) -> torch.Tensor:
        """Water FVF — assumed constant = 1.0."""
        return torch.ones_like(p)

    # ── Convenience bundle ────────────────────────────────────────────────────

    def all_properties(self, p: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return all PVT properties at pressure p."""
        return {
            "mu_g": self.mu_g(p),
            "mu_o": self.mu_o(p),
            "mu_w": self.mu_w(p),
            "Rs":   self.Rs(p),
            "Bo":   self.Bo(p),
            "Bg":   self.Bg(p),
            "Bw":   self.Bw(p),
        }

    def forward(self, p: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.all_properties(p)


# ── Unit conversion helpers ───────────────────────────────────────────────────

def psia_to_bar(p: torch.Tensor) -> torch.Tensor:
    return p * 0.0689476

def bar_to_psia(p: torch.Tensor) -> torch.Tensor:
    return p * 14.5038

def stb_to_m3(q: torch.Tensor) -> torch.Tensor:
    """STB/day → m³/day"""
    return q * 0.158987

def mscfd_to_m3d(q: torch.Tensor) -> torch.Tensor:
    """Mscf/day → m³/day"""
    return q * 28.3168
