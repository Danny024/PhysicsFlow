"""
PhysicsFlow — VCAE + DDIM Prior Models for History Matching.

VCAE (Variational Convolutional Autoencoder):
    Encodes high-dimensional, non-Gaussian permeability fields K [Nx, Ny, Nz]
    into a compact Gaussian latent space z ∈ ℝ^d.
    αREKI operates on z rather than K directly, enabling:
        - Gaussian assumption of EKI to be valid in latent space
        - Geological structure preservation post-update
        - Dimensionality reduction: 46×112×22 → 256

DDIM (Denoising Diffusion Implicit Models) Decoder:
    Generates non-Gaussian K fields from updated latent z.
    Replaces the VAE decoder with a faster deterministic DDIM sampler.
    Conditioning: y_cond = z (continuous latent code)

Reference:
    Laloy et al. (2018) "Emulation of CPU-demanding reactive transport
    models: a comparison of Gaussian process and polynomial chaos expansion"
    + PhysicsFlow extensions for 3D reservoir fields.
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
class VCAEConfig:
    """Hyperparameters for the Variational Convolutional Autoencoder."""
    nx: int = 46
    ny: int = 112
    nz: int = 22
    latent_dim: int = 256
    base_channels: int = 32       # first conv layer channels
    n_encoder_layers: int = 4     # spatial downsampling steps
    kl_weight: float = 1e-4       # β-VAE weight on KL divergence
    dropout: float = 0.1

    @classmethod
    def norne(cls) -> "VCAEConfig":
        return cls(nx=46, ny=112, nz=22, latent_dim=256,
                   base_channels=32, n_encoder_layers=4)


@dataclass
class DDIMConfig:
    """Hyperparameters for the DDIM diffusion decoder."""
    latent_dim: int = 256
    n_timesteps: int = 1000        # Training diffusion timesteps
    n_inference_steps: int = 50    # DDIM inference steps (much fewer)
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = 'cosine'       # 'linear' or 'cosine'
    unet_channels: int = 64


# ─────────────────────────────────────────────────────────────────────────────
# VCAE — Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock3d(nn.Module):
    """3-D residual block with instance normalisation."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.SiLU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class VCAEEncoder(nn.Module):
    """
    Convolutional encoder: [B, 1, Nx, Ny, Nz] → (μ, log_σ²) ∈ ℝ^d
    """

    def __init__(self, cfg: VCAEConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        in_ch = 1
        out_ch = cfg.base_channels

        for i in range(cfg.n_encoder_layers):
            layers += [
                nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.SiLU(),
                ResBlock3d(out_ch, cfg.dropout),
            ]
            in_ch  = out_ch
            out_ch = min(out_ch * 2, 256)

        self.conv = nn.Sequential(*layers)
        # Compute flattened size after downsampling
        self._flat_size = self._compute_flat(cfg)
        self.mu_head     = nn.Linear(self._flat_size, cfg.latent_dim)
        self.logvar_head = nn.Linear(self._flat_size, cfg.latent_dim)

    def _compute_flat(self, cfg: VCAEConfig) -> int:
        """Forward pass of a dummy tensor to find flattened size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.nx, cfg.ny, cfg.nz)
            out   = self.conv(dummy)
        return int(out.numel())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (mu, logvar), each [B, latent_dim]."""
        h = self.conv(x).flatten(1)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterise(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """z = mu + ε·σ  where ε ~ N(0,I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


# ─────────────────────────────────────────────────────────────────────────────
# VCAE — Decoder
# ─────────────────────────────────────────────────────────────────────────────

class VCAEDecoder(nn.Module):
    """
    Convolutional decoder: z ∈ ℝ^d → [B, 1, Nx, Ny, Nz]
    """

    def __init__(self, cfg: VCAEConfig, encoder_flat_size: int,
                 encoder_spatial_shape: Tuple[int, int, int, int]):
        super().__init__()
        self.cfg = cfg
        self.spatial_shape = encoder_spatial_shape   # (C, Sx, Sy, Sz)

        self.fc = nn.Linear(cfg.latent_dim, encoder_flat_size)

        layers = []
        in_ch = encoder_spatial_shape[0]
        for i in range(cfg.n_encoder_layers):
            out_ch = max(in_ch // 2, cfg.base_channels)
            layers += [
                nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.SiLU(),
                ResBlock3d(out_ch, cfg.dropout),
            ]
            in_ch = out_ch

        layers += [nn.Conv3d(in_ch, 1, 3, padding=1)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        C, Sx, Sy, Sz = self.spatial_shape
        h = self.fc(z).view(-1, C, Sx, Sy, Sz)
        out = self.deconv(h)
        # Crop/pad to exact grid size
        return out[:, :, :self.cfg.nx, :self.cfg.ny, :self.cfg.nz]


# ─────────────────────────────────────────────────────────────────────────────
# Full VCAE
# ─────────────────────────────────────────────────────────────────────────────

class VCAE(nn.Module):
    """
    Variational Convolutional Autoencoder for reservoir permeability fields.

    Input  : log10(K) field [B, 1, Nx, Ny, Nz]
    Output : reconstructed log10(K) field [B, 1, Nx, Ny, Nz]
    Latent : μ, log_σ², z  each [B, latent_dim]
    """

    def __init__(self, cfg: Optional[VCAEConfig] = None):
        super().__init__()
        self.cfg = cfg or VCAEConfig.norne()
        self.encoder = VCAEEncoder(self.cfg)

        # Determine spatial shape after encoding
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.cfg.nx, self.cfg.ny, self.cfg.nz)
            enc_out = self.encoder.conv(dummy)
            enc_shape = enc_out.shape[1:]   # (C, Sx, Sy, Sz)

        self.decoder = VCAEDecoder(self.cfg,
                                   self.encoder._flat_size,
                                   enc_shape)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (z, mu, logvar)."""
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterise(mu, logvar)
        return z, mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (x_recon, mu, logvar)."""
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(self, x: Tensor, x_recon: Tensor, mu: Tensor,
             logvar: Tensor) -> Tuple[Tensor, dict]:
        """β-VAE loss = reconstruction MSE + β·KL."""
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.cfg.kl_weight * kl_loss
        return total, {'recon': recon_loss.item(),
                       'kl': kl_loss.item(),
                       'total': total.item()}


# ─────────────────────────────────────────────────────────────────────────────
# DDIM noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_beta_schedule(T: int, s: float = 0.008) -> Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f     = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0, 0.999).float()


def _linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> Tensor:
    return torch.linspace(beta_start, beta_end, T)


class DDIMScheduler:
    """
    Pre-computes DDIM noise schedule quantities.
    Used by both training (noise addition) and inference (denoising).
    """

    def __init__(self, cfg: DDIMConfig):
        self.cfg = cfg
        T = cfg.n_timesteps

        if cfg.schedule == 'cosine':
            betas = _cosine_beta_schedule(T)
        else:
            betas = _linear_beta_schedule(T, cfg.beta_start, cfg.beta_end)

        alphas       = 1.0 - betas
        alpha_cum    = torch.cumprod(alphas, dim=0)
        alpha_cum_prev = torch.cat([torch.ones(1), alpha_cum[:-1]])

        self.betas        = betas
        self.alpha_cum    = alpha_cum
        self.alpha_cum_prev = alpha_cum_prev
        self.sqrt_alpha_cum      = alpha_cum.sqrt()
        self.sqrt_one_minus_alpha = (1.0 - alpha_cum).sqrt()

        # DDIM sub-sequence (uniform spacing)
        step = max(1, T // cfg.n_inference_steps)
        self.ddim_timesteps = list(range(0, T, step))[::-1]

    def add_noise(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward process: q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t)I)
        Returns (noisy_x, noise).
        """
        sqrt_a   = self.sqrt_alpha_cum[t].to(x0.device).view(-1, *([1]*x0.ndim)[1:])
        sqrt_1ma = self.sqrt_one_minus_alpha[t].to(x0.device).view(-1, *([1]*x0.ndim)[1:])
        noise    = torch.randn_like(x0)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    def ddim_step(self, x_t: Tensor, pred_noise: Tensor,
                  t: int, t_prev: int, eta: float = 0.0) -> Tensor:
        """
        One DDIM denoising step (deterministic if eta=0).
        """
        dev = x_t.device
        ac_t    = self.alpha_cum[t].to(dev)
        ac_prev = self.alpha_cum[t_prev].to(dev) if t_prev >= 0 \
                  else torch.ones(1, device=dev)

        x0_pred = (x_t - (1 - ac_t).sqrt() * pred_noise) / ac_t.sqrt()
        x0_pred = x0_pred.clamp(-5, 5)

        sigma = eta * ((1 - ac_prev) / (1 - ac_t)).sqrt() * (1 - ac_t / ac_prev).sqrt()
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        x_prev = ac_prev.sqrt() * x0_pred + (1 - ac_prev - sigma**2).sqrt() * pred_noise + sigma * noise
        return x_prev


# ─────────────────────────────────────────────────────────────────────────────
# DDIM U-Net (conditioned on latent z)
# ─────────────────────────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                  nn.Linear(dim * 4, dim))

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) *
                         torch.arange(half, dtype=torch.float32, device=t.device) / half)
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)


class ConditionalResBlock3d(nn.Module):
    """3-D ResBlock with AdaGN conditioning on (time + latent)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)  # scale + shift
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # AdaGN conditioning
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class DDIMUNet3d(nn.Module):
    """
    Lightweight 3-D U-Net for DDIM denoising.
    Conditioned on timestep embedding + VCAE latent z.
    """

    def __init__(self, cfg: DDIMConfig):
        super().__init__()
        C = cfg.unet_channels
        cond_dim = cfg.latent_dim + C       # latent + time_emb

        self.time_emb = TimeEmbedding(C)
        self.cond_dim = cond_dim

        # Encoder path
        self.enc1 = ConditionalResBlock3d(1,      C,      cond_dim)
        self.down1 = nn.Conv3d(C,  C * 2, 4, stride=2, padding=1)
        self.enc2 = ConditionalResBlock3d(C * 2, C * 2, cond_dim)
        self.down2 = nn.Conv3d(C * 2, C * 4, 4, stride=2, padding=1)

        # Bottleneck
        self.mid = ConditionalResBlock3d(C * 4, C * 4, cond_dim)

        # Decoder path
        self.up2   = nn.ConvTranspose3d(C * 4, C * 2, 4, stride=2, padding=1)
        self.dec2  = ConditionalResBlock3d(C * 4, C * 2, cond_dim)  # skip
        self.up1   = nn.ConvTranspose3d(C * 2, C,     4, stride=2, padding=1)
        self.dec1  = ConditionalResBlock3d(C * 2, C,     cond_dim)  # skip
        self.out   = nn.Conv3d(C, 1, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor, z: Tensor) -> Tensor:
        """
        x : [B, 1, Nx, Ny, Nz]  — noisy field
        t : [B]                  — diffusion timestep indices
        z : [B, latent_dim]      — VCAE latent code
        Returns predicted noise [B, 1, Nx, Ny, Nz].
        """
        t_emb = self.time_emb(t)             # [B, C]
        cond  = torch.cat([z, t_emb], dim=1) # [B, cond_dim]

        e1 = self.enc1(x,       cond)
        e2 = self.enc2(self.down1(e1), cond)
        m  = self.mid(self.down2(e2),  cond)

        d2 = self.dec2(torch.cat([self.up2(m),  e2], dim=1), cond)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), cond)
        return self.out(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Full DDIM model
# ─────────────────────────────────────────────────────────────────────────────

class DDIMPrior(nn.Module):
    """
    DDIM prior model — generates permeability fields conditioned on latent z.

    Workflow in history matching:
        1. αREKI updates latent ensemble: z_i → z_i'
        2. DDIMPrior.sample(z_i') → K_i'  (decoded permeability field)
    """

    def __init__(self, cfg: Optional[DDIMConfig] = None):
        super().__init__()
        self.cfg = cfg or DDIMConfig()
        self.scheduler = DDIMScheduler(self.cfg)
        self.unet = DDIMUNet3d(self.cfg)

    def training_loss(self, x0: Tensor, z: Tensor) -> Tensor:
        """
        Compute DDIM training loss (MSE on predicted noise).

        x0 : [B, 1, Nx, Ny, Nz]   — clean log10(K) fields
        z  : [B, latent_dim]        — VCAE latent codes for conditioning
        """
        B = x0.shape[0]
        t = torch.randint(0, self.cfg.n_timesteps, (B,), device=x0.device)
        x_t, noise = self.scheduler.add_noise(x0, t)
        pred_noise  = self.unet(x_t, t, z)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, z: Tensor, nx: int, ny: int, nz: int,
               eta: float = 0.0) -> Tensor:
        """
        Generate a permeability field from latent z using DDIM.

        z   : [B, latent_dim]
        Returns [B, 1, nx, ny, nz] — log10(K) field.
        """
        B   = z.shape[0]
        dev = z.device
        x   = torch.randn(B, 1, nx, ny, nz, device=dev)

        timesteps = self.scheduler.ddim_timesteps
        for i, t_val in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=dev)
            pred_noise = self.unet(x, t_tensor, z)
            x = self.scheduler.ddim_step(x, pred_noise, t_val, t_prev, eta)

        return x

    @torch.no_grad()
    def sample_ensemble(self, z_ensemble: Tensor, nx: int, ny: int,
                        nz: int) -> Tensor:
        """
        Generate K-fields for an ensemble.
        z_ensemble : [N_ensemble, latent_dim]
        Returns    : [N_ensemble, 1, nx, ny, nz]
        """
        return self.sample(z_ensemble, nx, ny, nz)


# ─────────────────────────────────────────────────────────────────────────────
# Combined prior + encoder interface for αREKI
# ─────────────────────────────────────────────────────────────────────────────

class ReservoirPriorModel:
    """
    Facade combining VCAE encoder and DDIM decoder for use by αREKI engine.

    The αREKI engine sees only latent vectors z; this class handles the
    encoding/decoding between K-space and z-space.
    """

    def __init__(self, vcae: VCAE, ddim: DDIMPrior, device: str = 'cpu'):
        self.vcae   = vcae.to(device).eval()
        self.ddim   = ddim.to(device).eval()
        self.device = device

    def encode(self, k_log: 'np.ndarray') -> 'np.ndarray':
        """
        Encode a permeability field to latent z (deterministic, uses μ).
        k_log : [Nx, Ny, Nz]  log10 permeability
        Returns z : [latent_dim]
        """
        import numpy as np
        x = torch.tensor(k_log, dtype=torch.float32,
                          device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.vcae.encoder(x)
        return mu.squeeze(0).cpu().numpy()

    def decode(self, z: 'np.ndarray') -> 'np.ndarray':
        """
        Decode latent z to permeability field.
        z : [latent_dim]
        Returns k_log : [Nx, Ny, Nz]
        """
        import numpy as np
        z_t = torch.tensor(z, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        cfg = self.vcae.cfg
        with torch.no_grad():
            k = self.ddim.sample(z_t, cfg.nx, cfg.ny, cfg.nz)
        return k.squeeze(0).squeeze(0).cpu().numpy()

    def decode_ensemble(self, z_ensemble: 'np.ndarray') -> 'np.ndarray':
        """
        Decode a full ensemble in one GPU batch.
        z_ensemble : [N, latent_dim]
        Returns    : [N, Nx, Ny, Nz]
        """
        import numpy as np
        z_t = torch.tensor(z_ensemble, dtype=torch.float32, device=self.device)
        cfg = self.vcae.cfg
        with torch.no_grad():
            k = self.ddim.sample_ensemble(z_t, cfg.nx, cfg.ny, cfg.nz)
        return k.squeeze(1).cpu().numpy()
