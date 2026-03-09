"""
PhysicsFlow — PINO Pre-training on Norne Reference Dataset.

Generates a synthetic ensemble of perturbed K/phi fields from the Norne Eclipse
benchmark, builds a training dataset, and trains FNO3d with PINOLoss.

Usage (CLI):
    physicsflow-pretrain \\
        --deck path/to/NORNE_ATW2013.DATA \\
        --output-dir models/ \\
        --epochs 200 \\
        --ensemble 500 \\
        --lr 1e-3 \\
        --device cuda

Usage (Python API):
    from physicsflow.training.pretrain_norne import pretrain_norne, PretrainConfig
    cfg = PretrainConfig(deck_path="NORNE_ATW2013.DATA", epochs=100)
    checkpoint_path = pretrain_norne(cfg)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PretrainConfig:
    """Hyperparameter configuration for PINO pre-training on the Norne dataset."""
    deck_path:       Optional[str] = None     # path to NORNE_ATW2013.DATA (optional)
    output_dir:      str           = "models"
    epochs:          int           = 200
    ensemble_size:   int           = 500      # number of perturbed K/phi realisations
    batch_size:      int           = 4
    learning_rate:   float         = 1e-3
    weight_decay:    float         = 1e-4
    lr_decay_every:  int           = 50
    lr_decay_factor: float         = 0.5
    device:          str           = "cuda"   # 'cuda' or 'cpu'
    seed:            int           = 42
    log_every:       int           = 10
    save_every:      int           = 50

    # FNO architecture
    modes1:   int = 8
    modes2:   int = 8
    modes3:   int = 6
    width:    int = 32
    n_layers: int = 4

    # PINO loss weights
    w_data: float = 1.0
    w_pde:  float = 0.1
    w_ic:   float = 0.5
    w_bc:   float = 0.2
    w_well: float = 0.5

    # Grid dimensions (Norne 46×112×22)
    nx: int = 46
    ny: int = 112
    nz: int = 22

    n_timesteps: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_dataset(cfg: PretrainConfig) -> TensorDataset:
    """
    Build synthetic training dataset.

    If a valid Eclipse deck is provided, baseline K/phi are read from it.
    Otherwise synthetic lognormal Norne-like fields are used.

    Returns TensorDataset of:
        inputs  [N, 6, Nx, Ny, Nz]
        targets [N, T, 2, Nx, Ny, Nz]   (P and Sw for T timesteps)
    """
    logger.info("Building pre-training dataset (N={}, {}×{}×{})",
                cfg.ensemble_size, cfg.nx, cfg.ny, cfg.nz)

    rng = np.random.default_rng(cfg.seed)
    Nx, Ny, Nz, N, T = cfg.nx, cfg.ny, cfg.nz, cfg.ensemble_size, cfg.n_timesteps

    # Baseline K/phi
    if cfg.deck_path and Path(cfg.deck_path).exists():
        try:
            from ..io.eclipse_reader import EclipseReader
            reader = EclipseReader(cfg.deck_path)
            K_base  = reader.permeability().reshape(Nx, Ny, Nz)
            phi_base = reader.porosity().reshape(Nx, Ny, Nz)
            logger.info("Loaded baseline K/phi from Eclipse deck")
        except Exception as exc:
            logger.warning("Eclipse read failed ({}), using synthetic baseline", exc)
            K_base, phi_base = _synthetic_norne(rng, Nx, Ny, Nz)
    else:
        K_base, phi_base = _synthetic_norne(rng, Nx, Ny, Nz)

    # Coordinate channels (static, same for all ensemble members)
    x_c = np.broadcast_to(np.linspace(0, 1, Nx)[:, None, None], (Nx, Ny, Nz)).copy()
    z_c = np.broadcast_to(np.linspace(0, 1, Nz)[None, None, :], (Nx, Ny, Nz)).copy()

    inputs_list:  list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for _ in range(N):
        # Perturb K and phi with lognormal multiplicative noise
        K_i   = np.clip(K_base   * np.exp(rng.normal(0, 0.3, (Nx, Ny, Nz))), 0.1, 5000.0)
        phi_i = np.clip(phi_base * np.exp(rng.normal(0, 0.1, (Nx, Ny, Nz))), 0.05, 0.40)

        K_log_n = (np.log(K_i + 1e-6) - np.log(K_base + 1e-6).mean()) / \
                  (np.log(K_base + 1e-6).std() + 1e-8)
        phi_n   = (phi_i - 0.2) / 0.1
        P_init  = np.full((Nx, Ny, Nz), 277.0 / 400.0, dtype=np.float32)
        Sw_init = np.full((Nx, Ny, Nz), 0.20, dtype=np.float32)

        inp = np.stack([K_log_n, phi_n, P_init, Sw_init, x_c, z_c], axis=0).astype(np.float32)
        tgt = _synthetic_simulation(K_i, phi_i, P_init, Sw_init, T, rng)

        inputs_list.append(inp)
        targets_list.append(tgt)

    inputs  = torch.tensor(np.stack(inputs_list),  dtype=torch.float32)
    targets = torch.tensor(np.stack(targets_list), dtype=torch.float32)
    logger.info("Dataset: inputs {}, targets {}", tuple(inputs.shape), tuple(targets.shape))
    return TensorDataset(inputs, targets)


def _synthetic_norne(rng: np.random.Generator, Nx: int, Ny: int, Nz: int):
    K   = np.exp(rng.normal(4.5, 1.2, (Nx, Ny, Nz))).clip(0.1, 2000.0)
    phi = rng.beta(5, 15, (Nx, Ny, Nz)).clip(0.05, 0.35)
    return K, phi


def _synthetic_simulation(K, phi, P0, Sw0, T: int, rng) -> np.ndarray:
    """
    Fast analytical proxy simulation for dataset target generation.
    Simplified Buckley-Leverett pressure depletion + saturation front.
    Returns shape (T, 2, Nx, Ny, Nz): channel 0 = P, channel 1 = Sw.
    """
    Nx, Ny, Nz = K.shape
    target = np.zeros((T, 2, Nx, Ny, Nz), dtype=np.float32)
    kx = 0.5 * (K + np.roll(K, 1, axis=0))
    cx, cy = Nx // 2, Ny // 2

    for t in range(T):
        dt = (t + 1) / T
        dist = np.sqrt(
            ((np.arange(Nx)[:, None, None] - cx) ** 2 +
             (np.arange(Ny)[None, :, None] - cy) ** 2 +
             (np.arange(Nz)[None, None, :] * 2) ** 2).clip(1))
        P_t  = P0 - dt * 0.3 / (1.0 + 0.001 * kx) * (1.0 / dist.clip(1))
        Sw_t = np.clip(Sw0 + dt * 0.4 * phi * (kx / (kx.mean() + 1e-6)),
                       0.0, 1.0).astype(np.float32)
        Sw_t += rng.normal(0, 0.002, Sw_t.shape).astype(np.float32)
        target[t, 0] = P_t.astype(np.float32)
        target[t, 1] = Sw_t

    return target


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_norne(cfg: PretrainConfig) -> Path:
    """
    Pre-train FNO3d PINO surrogate on synthetic Norne ensemble data.

    Returns the path to the best saved checkpoint.
    """
    from ..surrogate.fno import FNO3d, FNOConfig, PINOLoss, PINOLossConfig
    from ..db.db_service import DatabaseService

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Pre-training device: {}", device)

    db = DatabaseService.instance()
    project_id = "norne_pretrain"
    run_id = db.start_run(project_id, "training",
                          config={"type": "pretrain_norne", **cfg.__dict__})
    db.audit("pretrain.started",
             f"PINO pre-training started: {cfg.epochs} epochs, N={cfg.ensemble_size}",
             project_id=project_id)

    fno_cfg = FNOConfig(
        n_modes_x=cfg.modes1, n_modes_y=cfg.modes2, n_modes_z=cfg.modes3,
        d_model=cfg.width, n_layers=cfg.n_layers,
        in_channels=6, out_channels=2, n_timesteps=cfg.n_timesteps,
    )
    model = FNO3d(fno_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("FNO3d parameters: {:,}", n_params)

    dataset = _build_dataset(cfg)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_decay_every, gamma=cfg.lr_decay_factor)
    criterion = PINOLoss(PINOLossConfig(
        w_data=cfg.w_data, w_pde=cfg.w_pde,
        w_ic=cfg.w_ic, w_bc=cfg.w_bc, w_well=cfg.w_well,
    ))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = out_dir / "pino_norne_pretrained.pt"

    logger.info("Starting pre-training for {} epochs", cfg.epochs)
    t0_total = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = ep_pde = ep_data = 0.0
        t0 = time.time()

        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            perm_log = inputs[:, 0]   # K_log channel
            phi_inp  = inputs[:, 1]   # phi channel
            loss, breakdown = criterion(preds, targets, perm_log, phi_inp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            ep_pde  += breakdown['pde']
            ep_data += breakdown['data']

        scheduler.step()
        nb  = len(loader)
        avg = ep_loss / nb; avg_pde = ep_pde / nb; avg_data = ep_data / nb

        db.record_epoch(run_id, epoch=epoch, loss_total=avg, loss_pde=avg_pde,
                        loss_data=avg_data, loss_well=0.0, loss_ic=0.0, loss_bc=0.0)

        if epoch % cfg.log_every == 0:
            logger.info("Epoch {:4d}/{} | total={:.6f} pde={:.6f} data={:.6f} | lr={:.2e} | {:.1f}s",
                        epoch, cfg.epochs, avg, avg_pde, avg_data,
                        scheduler.get_last_lr()[0], time.time() - t0)

        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "loss": avg, "config": cfg.__dict__}, best_path)

        if epoch % cfg.save_every == 0:
            ckpt = out_dir / f"pino_norne_epoch{epoch:04d}.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "loss": avg}, ckpt)

    logger.info("Pre-training done in {:.1f}s — best loss: {:.6f}", time.time() - t0_total, best_loss)
    db.complete_run(run_id, loss_total=best_loss)
    db.register_model(project_id=project_id, model_type="pino",
                      version_tag="norne_pretrained", file_path=str(best_path),
                      training_run_id=run_id,
                      epochs_trained=cfg.epochs, loss_total=best_loss,
                      notes=f"Pre-trained on Norne synthetic ensemble N={cfg.ensemble_size}")
    db.audit("pretrain.completed",
             f"PINO pre-training completed: best_loss={best_loss:.6f}",
             project_id=project_id)
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command("pretrain-norne")
@click.option("--deck",        default=None,    help="Path to NORNE_ATW2013.DATA Eclipse deck")
@click.option("--output-dir",  default="models",help="Directory to save checkpoints")
@click.option("--epochs",      default=200,     type=int)
@click.option("--ensemble",    default=500,     type=int)
@click.option("--batch-size",  default=4,       type=int)
@click.option("--lr",          default=1e-3,    type=float)
@click.option("--device",      default="cuda",  help="'cuda' or 'cpu'")
@click.option("--seed",        default=42,      type=int)
@click.option("--log-every",   default=10,      type=int)
@click.option("--save-every",  default=50,      type=int)
@click.option("--width",       default=32,      type=int)
@click.option("--modes",       default=8,       type=int)
@click.option("--modes-z",     default=6,       type=int)
@click.option("--w-pde",       default=0.1,     type=float)
@click.option("--w-data",      default=1.0,     type=float)
def main(deck, output_dir, epochs, ensemble, batch_size, lr, device,
         seed, log_every, save_every, width, modes, modes_z, w_pde, w_data):
    """Pre-train FNO3d PINO surrogate on the Norne reference dataset."""
    cfg = PretrainConfig(
        deck_path=deck, output_dir=output_dir, epochs=epochs,
        ensemble_size=ensemble, batch_size=batch_size, learning_rate=lr,
        device=device, seed=seed, log_every=log_every, save_every=save_every,
        width=width, modes1=modes, modes2=modes, modes3=modes_z,
        w_pde=w_pde, w_data=w_data,
    )
    pretrain_norne(cfg)


if __name__ == "__main__":
    main()
