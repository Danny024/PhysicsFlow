"""
Reservoir grid object — generalised, configuration-driven.

Replaces the hardcoded 46×112×22 Norne dimensions scattered throughout
the original NVRS.py code. Accepts any Nx×Ny×Nz structured grid.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class GridConfig:
    """Grid geometry specification."""
    nx: int = 46
    ny: int = 112
    nz: int = 22
    dx: float = 50.0   # ft
    dy: float = 50.0   # ft
    dz: float = 20.0   # ft
    depth: float = 4_000.0   # reservoir depth, ft

    def __post_init__(self):
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError(
                f"Grid dimensions must be positive, got nx={self.nx}, "
                f"ny={self.ny}, nz={self.nz}"
            )

    @classmethod
    def norne(cls) -> "GridConfig":
        """Norne field benchmark grid (46×112×22, 50×50×20 ft cells)."""
        return cls(nx=46, ny=112, nz=22, dx=50.0, dy=50.0, dz=20.0, depth=4_000.0)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def cell_volume(self) -> float:
        """Single cell bulk volume [ft³]."""
        return self.dx * self.dy * self.dz

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)


class ReservoirGrid:
    """
    Structured reservoir grid with property arrays.

    Holds static properties (K, φ, NTG, actnum, fault multipliers)
    and provides helper methods for FVM/FNO operations.

    All arrays are stored as NumPy arrays; use .to_torch() to get
    PyTorch tensors on a specified device.
    """

    def __init__(self, cfg: GridConfig | None = None):
        self.cfg = cfg or GridConfig.norne()
        nx, ny, nz = self.cfg.nx, self.cfg.ny, self.cfg.nz

        # Static property arrays [Nx, Ny, Nz]
        self.actnum: np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)
        self.perm_x: np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)  # mD
        self.perm_y: np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)
        self.perm_z: np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)
        self.poro:   np.ndarray = np.full((nx, ny, nz), 0.2, dtype=np.float32)
        self.ntg:    np.ndarray = np.ones((nx, ny, nz), dtype=np.float32)

        # Fault transmissibility multipliers (53 values for Norne)
        self.fault_mult: np.ndarray = np.ones(53, dtype=np.float32)

        # Dynamic arrays (updated during simulation)
        self.pressure: np.ndarray | None = None   # [Nx, Ny, Nz, Nt]
        self.sw:       np.ndarray | None = None
        self.sg:       np.ndarray | None = None

    # ── Property loading ──────────────────────────────────────────────────────

    def set_permeability(self, kx: np.ndarray,
                         ky: np.ndarray | None = None,
                         kz: np.ndarray | None = None) -> None:
        """Set permeability [mD]. ky=kx, kz=0.1*kx if not provided."""
        self.perm_x = kx.astype(np.float32).reshape(self.cfg.shape)
        self.perm_y = (ky if ky is not None else kx).astype(np.float32).reshape(self.cfg.shape)
        self.perm_z = (kz if kz is not None else 0.1 * kx).astype(np.float32).reshape(self.cfg.shape)

    def set_porosity(self, poro: np.ndarray) -> None:
        self.poro = poro.astype(np.float32).reshape(self.cfg.shape)

    def set_ntg(self, ntg: np.ndarray) -> None:
        self.ntg = ntg.astype(np.float32).reshape(self.cfg.shape)

    def set_actnum(self, actnum: np.ndarray) -> None:
        self.actnum = actnum.astype(np.float32).reshape(self.cfg.shape)

    def set_fault_mult(self, mult: np.ndarray) -> None:
        self.fault_mult = mult.astype(np.float32)

    # ── Transmissibility ──────────────────────────────────────────────────────

    def transmissibility_x(self) -> np.ndarray:
        """Inter-cell transmissibility in X direction [mD·ft].

        Returns face-centred array of shape (nx-1, ny, nz) — one value
        per cell interface, using harmonic-mean permeability.
        """
        dx, dy, dz = self.cfg.dx, self.cfg.dy, self.cfg.dz
        k = self.perm_x * self.ntg
        k_l = k[:-1, :, :]   # left cell  [nx-1, ny, nz]
        k_r = k[1:,  :, :]   # right cell [nx-1, ny, nz]
        T = np.where(
            k_l + k_r > 0,
            2 * k_l * k_r / (k_l + k_r),
            0.0
        )
        return (T * (dy * dz) / dx).astype(np.float32)

    def transmissibility_y(self) -> np.ndarray:
        """Inter-cell transmissibility in Y direction, shape (nx, ny-1, nz)."""
        dx, dy, dz = self.cfg.dx, self.cfg.dy, self.cfg.dz
        k = self.perm_y * self.ntg
        k_l = k[:, :-1, :]
        k_r = k[:, 1:,  :]
        T = np.where(
            k_l + k_r > 0,
            2 * k_l * k_r / (k_l + k_r),
            0.0
        )
        return (T * (dx * dz) / dy).astype(np.float32)

    def transmissibility_z(self) -> np.ndarray:
        """Inter-cell transmissibility in Z direction, shape (nx, ny, nz-1)."""
        dx, dy, dz = self.cfg.dx, self.cfg.dy, self.cfg.dz
        k = self.perm_z * self.ntg
        k_l = k[:, :, :-1]
        k_r = k[:, :, 1: ]
        T = np.where(
            k_l + k_r > 0,
            2 * k_l * k_r / (k_l + k_r),
            0.0
        )
        return (T * (dx * dy) / dz).astype(np.float32)

    # ── Active cell helpers ───────────────────────────────────────────────────

    @property
    def n_cells(self) -> int:
        """Total number of grid cells (Nx × Ny × Nz)."""
        return self.cfg.n_cells

    @property
    def n_active(self) -> int:
        return int(self.actnum.sum())

    @property
    def n_active_cells(self) -> int:
        """Alias for n_active."""
        return self.n_active

    def active_mask(self) -> np.ndarray:
        return self.actnum > 0.5

    def flatten_active(self, field: np.ndarray) -> np.ndarray:
        """Extract active cells from [Nx, Ny, Nz] → [N_active]."""
        return field[self.active_mask()]

    def unflatten_active(self, vec: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Expand [N_active] → [Nx, Ny, Nz], filling inactive cells."""
        out = np.full(self.cfg.shape, fill, dtype=np.float32)
        out[self.active_mask()] = vec
        return out

    def flatten(self, field: np.ndarray) -> np.ndarray:
        """Alias for flatten_active: [Nx, Ny, Nz] → [N_active]."""
        return self.flatten_active(field)

    def unflatten(self, vec: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Alias for unflatten_active: [N_active] → [Nx, Ny, Nz]."""
        return self.unflatten_active(vec, fill)

    # ── PyTorch export ────────────────────────────────────────────────────────

    def to_torch(self, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """Export all static properties as PyTorch tensors."""
        return {
            "perm_x":    torch.from_numpy(self.perm_x).to(device),
            "perm_y":    torch.from_numpy(self.perm_y).to(device),
            "perm_z":    torch.from_numpy(self.perm_z).to(device),
            "poro":      torch.from_numpy(self.poro).to(device),
            "ntg":       torch.from_numpy(self.ntg).to(device),
            "actnum":    torch.from_numpy(self.actnum).to(device),
            "fault_mult":torch.from_numpy(self.fault_mult).to(device),
            "tx":        torch.from_numpy(self.transmissibility_x()).to(device),
            "ty":        torch.from_numpy(self.transmissibility_y()).to(device),
            "tz":        torch.from_numpy(self.transmissibility_z()).to(device),
        }

    # ── Statistics ────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, float]:
        """Quick property statistics for display."""
        active = self.active_mask()
        return {
            "n_cells_total": self.cfg.n_cells,
            "n_cells_active": self.n_active,
            "perm_min_mD":  float(self.perm_x[active].min()),
            "perm_max_mD":  float(self.perm_x[active].max()),
            "perm_mean_mD": float(self.perm_x[active].mean()),
            "poro_min":     float(self.poro[active].min()),
            "poro_max":     float(self.poro[active].max()),
            "poro_mean":    float(self.poro[active].mean()),
        }
