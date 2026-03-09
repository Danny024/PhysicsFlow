"""
PhysicsFlow — Fourier Neural Operator (FNO) / Physics-Informed Neural Operator (PINO)
surrogate model for reservoir simulation.

Architecture:
    Input  : [K_log, phi, P_init, Sw_init] stacked over Nx×Ny×Nz grid
    Output : [P(t), Sw(t)] at each requested timestep

PINO loss components:
    L_data  — MSE against OPM FLOW snapshots
    L_pde   — Darcy flow residual (pressure equation)
    L_well  — Peacemann well rate mismatch
    L_ic    — Initial condition constraint
    L_bc    — No-flow boundary condition
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FNOConfig:
    """FNO architecture hyperparameters."""
    n_modes_x: int = 12        # Fourier modes retained in x
    n_modes_y: int = 12        # Fourier modes retained in y
    n_modes_z: int = 6         # Fourier modes retained in z (depth)
    d_model: int = 64          # Channel width throughout FNO
    n_layers: int = 4          # Number of Fourier layers
    in_channels: int = 6       # K_log, phi, P_init, Sw_init, x_norm, z_norm
    out_channels: int = 2      # P, Sw (per timestep)
    n_timesteps: int = 20      # Timesteps predicted simultaneously
    dropout: float = 0.0

    @classmethod
    def norne(cls) -> "FNOConfig":
        """Defaults tuned for the Norne 46×112×22 grid."""
        return cls(n_modes_x=12, n_modes_y=16, n_modes_z=6,
                   d_model=64, n_layers=4, n_timesteps=20)


@dataclass
class PINOLossConfig:
    """Weights for the composite PINO loss."""
    w_data: float = 1.0
    w_pde: float = 0.5
    w_well: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Convolution (core FNO building block)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralConv3d(nn.Module):
    """
    3-D Fourier integral operator: multiply low-frequency modes by
    learnable complex weights, then inverse-FFT back to physical space.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 modes_x: int, modes_y: int, modes_z: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mx = modes_x
        self.my = modes_y
        self.mz = modes_z

        scale = 1.0 / (in_ch * out_ch)
        # 8 octants of the 3-D FFT that we keep
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_x, modes_y, modes_z, dtype=torch.cfloat))
            for _ in range(4)
        ])

    def _mul(self, x_ft: Tensor, w: Tensor) -> Tensor:
        # x_ft : [B, in_ch, mx, my, mz]
        # w    : [in_ch, out_ch, mx, my, mz]
        return torch.einsum('bixyz,ioxyz->boxyz', x_ft, w)

    def forward(self, x: Tensor) -> Tensor:
        B, C, Nx, Ny, Nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(B, self.out_ch, Nx, Ny, Nz // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        mx, my, mz = self.mx, self.my, self.mz
        out_ft[:, :, :mx, :my, :mz]        = self._mul(x_ft[:, :, :mx, :my, :mz],        self.weights[0])
        out_ft[:, :, -mx:, :my, :mz]       = self._mul(x_ft[:, :, -mx:, :my, :mz],       self.weights[1])
        out_ft[:, :, :mx, -my:, :mz]       = self._mul(x_ft[:, :, :mx, -my:, :mz],       self.weights[2])
        out_ft[:, :, -mx:, -my:, :mz]      = self._mul(x_ft[:, :, -mx:, -my:, :mz],      self.weights[3])

        return torch.fft.irfftn(out_ft, s=(Nx, Ny, Nz), dim=(-3, -2, -1))


# ─────────────────────────────────────────────────────────────────────────────
# FNO Layer (spectral conv + pointwise residual)
# ─────────────────────────────────────────────────────────────────────────────

class FNOLayer3d(nn.Module):
    def __init__(self, d_model: int, modes_x: int, modes_y: int, modes_z: int,
                 dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv3d(d_model, d_model, modes_x, modes_y, modes_z)
        self.pointwise = nn.Conv3d(d_model, d_model, kernel_size=1)
        self.norm = nn.InstanceNorm3d(d_model, affine=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.norm(self.drop(self.spectral(x) + self.pointwise(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Full FNO Model
# ─────────────────────────────────────────────────────────────────────────────

class FNO3d(nn.Module):
    """
    3-D FNO surrogate.

    Input  x : [B, in_channels, Nx, Ny, Nz]
    Output   : [B, out_channels * n_timesteps, Nx, Ny, Nz]
    """

    def __init__(self, cfg: FNOConfig):
        super().__init__()
        self.cfg = cfg
        out_total = cfg.out_channels * cfg.n_timesteps

        # Lifting layer: in_channels → d_model
        self.lift = nn.Sequential(
            nn.Conv3d(cfg.in_channels, cfg.d_model * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(cfg.d_model * 2, cfg.d_model, kernel_size=1),
        )

        # FNO backbone
        self.layers = nn.ModuleList([
            FNOLayer3d(cfg.d_model, cfg.n_modes_x, cfg.n_modes_y, cfg.n_modes_z,
                       cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Projection layer: d_model → out_channels * n_timesteps
        self.project = nn.Sequential(
            nn.Conv3d(cfg.d_model, cfg.d_model * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(cfg.d_model * 2, out_total, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        return self.project(x)

    def predict(self, x: Tensor) -> Tensor:
        """
        Returns [B, n_timesteps, out_channels, Nx, Ny, Nz].
        """
        B = x.shape[0]
        out = self.forward(x)          # [B, T*C, Nx, Ny, Nz]
        T = self.cfg.n_timesteps
        C = self.cfg.out_channels
        Nx, Ny, Nz = out.shape[-3:]
        return out.view(B, T, C, Nx, Ny, Nz)


# ─────────────────────────────────────────────────────────────────────────────
# PINO Loss
# ─────────────────────────────────────────────────────────────────────────────

def _fd_gradient(field: Tensor, dim: int, dx: float) -> Tensor:
    """Central finite difference gradient along a spatial dimension."""
    # field: [B, Nx, Ny, Nz]
    return (torch.roll(field, -1, dim) - torch.roll(field, 1, dim)) / (2.0 * dx)


def darcy_pde_residual(
    pressure: Tensor,
    perm_log: Tensor,
    phi: Tensor,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    mu_o: float = 1.5e-3,       # Pa·s
    Bo: float = 1.2,
    ct: float = 1.0e-9,         # 1/Pa total compressibility
) -> Tensor:
    """
    Finite-difference residual of single-phase pressure equation:

        ∇·(K/μ ∇P) = φ ct (∂P/∂t)

    pressure : [B, T, Nx, Ny, Nz]  — predicted pressures (Pa)
    perm_log : [B, Nx, Ny, Nz]     — log10 permeability (mD)
    phi      : [B, Nx, Ny, Nz]     — porosity (fraction)
    Returns residual : [B, T-1, Nx, Ny, Nz]
    """
    K = 10.0 ** perm_log * 9.869e-16  # mD → m²

    # Time derivative (forward difference)
    dPdt = (pressure[:, 1:] - pressure[:, :-1]) / dt  # [B, T-1, Nx, Ny, Nz]

    # Spatial divergence term — computed at each timestep
    residuals = []
    for t in range(pressure.shape[1] - 1):
        P = pressure[:, t]    # [B, Nx, Ny, Nz]
        dPdx = _fd_gradient(P, -3, dx)
        dPdy = _fd_gradient(P, -2, dy)
        dPdz = _fd_gradient(P, -1, dz)

        Kx = _fd_gradient(K * dPdx / mu_o, -3, dx)
        Ky = _fd_gradient(K * dPdy / mu_o, -2, dy)
        Kz = _fd_gradient(K * dPdz / mu_o, -1, dz)

        div_flux = Kx + Ky + Kz
        rhs = phi * ct * dPdt[:, t]
        residuals.append(div_flux - rhs)

    return torch.stack(residuals, dim=1)   # [B, T-1, Nx, Ny, Nz]


class PINOLoss(nn.Module):
    """
    Composite PINO loss for reservoir surrogate training.
    """

    def __init__(self, cfg: PINOLossConfig, dx: float = 50.0, dy: float = 50.0,
                 dz: float = 20.0, dt: float = 30.0 * 86400):
        super().__init__()
        self.cfg = cfg
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt

    def forward(
        self,
        pred: Tensor,           # [B, T, C, Nx, Ny, Nz]  C=2 (P, Sw)
        target: Tensor,         # [B, T, C, Nx, Ny, Nz]
        perm_log: Tensor,       # [B, Nx, Ny, Nz]
        phi: Tensor,            # [B, Nx, Ny, Nz]
        well_rates_pred: Optional[Tensor] = None,   # [B, T, n_wells]
        well_rates_true: Optional[Tensor] = None,
        active_mask: Optional[Tensor] = None,       # [B, Nx, Ny, Nz] bool
    ) -> Tuple[Tensor, dict]:
        cfg = self.cfg

        # ── Data loss ─────────────────────────────────────────────────────────
        if active_mask is not None:
            m = active_mask.unsqueeze(1).unsqueeze(1).float()
            l_data = F.mse_loss(pred * m, target * m)
        else:
            l_data = F.mse_loss(pred, target)

        # ── PDE loss (pressure channel = index 0) ─────────────────────────────
        pressure_pred   = pred[:, :, 0]      # [B, T, Nx, Ny, Nz]
        residual        = darcy_pde_residual(pressure_pred, perm_log, phi,
                                             self.dt, self.dx, self.dy, self.dz)
        l_pde = (residual ** 2).mean()

        # ── IC loss — first timestep must match initial conditions ────────────
        l_ic = F.mse_loss(pred[:, 0], target[:, 0])

        # ── BC loss — no-flow: zero gradient at boundary faces ────────────────
        P = pred[:, :, 0]
        # x boundaries
        bc_x = ((P[:, :, 0]  - P[:, :, 1])  ** 2 +
                 (P[:, :, -1] - P[:, :, -2]) ** 2).mean()
        bc_y = ((P[:, :, :, 0]  - P[:, :, :, 1])  ** 2 +
                 (P[:, :, :, -1] - P[:, :, :, -2]) ** 2).mean()
        l_bc = bc_x + bc_y

        # ── Well rate loss ─────────────────────────────────────────────────────
        l_well = torch.zeros(1, device=pred.device)
        if well_rates_pred is not None and well_rates_true is not None:
            l_well = F.mse_loss(well_rates_pred, well_rates_true)

        total = (cfg.w_data * l_data +
                 cfg.w_pde  * l_pde  +
                 cfg.w_ic   * l_ic   +
                 cfg.w_bc   * l_bc   +
                 cfg.w_well * l_well)

        breakdown = {
            'data': l_data.item(),
            'pde':  l_pde.item(),
            'ic':   l_ic.item(),
            'bc':   l_bc.item(),
            'well': l_well.item(),
            'total': total.item(),
        }
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingState:
    epoch: int = 0
    best_loss: float = float('inf')
    history: List[dict] = field(default_factory=list)


def build_input_tensor(
    perm_log: Tensor,   # [B, Nx, Ny, Nz]
    phi: Tensor,        # [B, Nx, Ny, Nz]
    p_init: Tensor,     # [B, Nx, Ny, Nz]
    sw_init: Tensor,    # [B, Nx, Ny, Nz]
) -> Tensor:
    """
    Concatenate static inputs + normalised coordinate channels.
    Returns [B, 6, Nx, Ny, Nz].
    """
    B, Nx, Ny, Nz = perm_log.shape
    dev = perm_log.device

    # Normalised z-coordinate (depth proxy)
    z_idx = torch.linspace(0, 1, Nz, device=dev)
    z_norm = z_idx.view(1, 1, 1, Nz).expand(B, Nx, Ny, Nz)

    # Normalised x-coordinate (left-right proxy)
    x_idx = torch.linspace(0, 1, Nx, device=dev)
    x_norm = x_idx.view(1, Nx, 1, 1).expand(B, Nx, Ny, Nz)

    return torch.stack([perm_log, phi, p_init, sw_init, x_norm, z_norm], dim=1)


def train_one_epoch(
    model: FNO3d,
    optimizer: torch.optim.Optimizer,
    loss_fn: PINOLoss,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> dict:
    """Train for one epoch, return average loss breakdown."""
    model.train()
    totals: dict = {}
    n = 0

    for batch in dataloader:
        perm_log = batch['perm_log'].to(device)
        phi      = batch['phi'].to(device)
        p_init   = batch['p_init'].to(device)
        sw_init  = batch['sw_init'].to(device)
        target   = batch['target'].to(device)   # [B, T, 2, Nx, Ny, Nz]
        mask     = batch.get('mask')
        if mask is not None:
            mask = mask.to(device)

        x = build_input_tensor(perm_log, phi, p_init, sw_init)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast('cuda'):
                pred = model.predict(x)
                loss, breakdown = loss_fn(pred, target, perm_log, phi,
                                          active_mask=mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model.predict(x)
            loss, breakdown = loss_fn(pred, target, perm_log, phi,
                                      active_mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for k, v in breakdown.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def create_pino_model(cfg: Optional[FNOConfig] = None,
                      device: str = 'cpu') -> FNO3d:
    """Factory: create and move model to device."""
    if cfg is None:
        cfg = FNOConfig.norne()
    model = FNO3d(cfg).to(device)
    return model
