"""
Gaspari-Cohn covariance localisation (JAX implementation).

Eliminates spurious long-range correlations in the Kalman gain matrix
by applying a smooth distance-based taper (Schur product).

Reference: Gaspari & Cohn (1999), Q. J. R. Meteorol. Soc.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np


def gaspari_cohn(distance: jnp.ndarray, radius: float) -> jnp.ndarray:
    """
    Gaspari-Cohn 5th-order piecewise rational function.

    Returns correlation weight in [0, 1]:
        = 1      at distance = 0
        = 0      at distance >= 2·radius
        smooth polynomial decay in between

    Parameters
    ----------
    distance : array of distances (any shape)
    radius   : localisation radius (half-width of support)
    """
    r = jnp.abs(distance) / radius   # normalised distance, support is [0, 2]
    r = jnp.clip(r, 0.0, 2.0)

    # Piece 1: 0 ≤ r ≤ 1
    p1 = (
        -0.25 * r**5
        + 0.5  * r**4
        + 0.625 * r**3
        - (5.0 / 3.0) * r**2
        + 1.0
    )

    # Piece 2: 1 < r ≤ 2
    p2 = (
        (1.0 / 12.0) * r**5
        - 0.5         * r**4
        + 0.625       * r**3
        + (5.0 / 3.0) * r**2
        - 5.0         * r
        + 4.0
        - (2.0 / 3.0) / r
    )

    result = jnp.where(r <= 1.0, p1, p2)
    result = jnp.where(r >= 2.0, 0.0, result)
    return jnp.clip(result, 0.0, 1.0)


def build_localisation_matrix(
    param_coords: np.ndarray,    # [N_params, 3]  (i, j, k) grid coords
    obs_coords: np.ndarray,      # [N_obs, 3]     (i, j, k) observation coords
    radius: float,               # localisation radius in grid cells
) -> np.ndarray:
    """
    Build the N_params × N_obs Gaspari-Cohn localisation matrix.

    Each entry L[p, o] = GC(distance(param_p, obs_o), radius).

    Parameters
    ----------
    param_coords : [N_params, 3] — grid cell (i,j,k) of each parameter
    obs_coords   : [N_obs, 3]   — grid cell (i,j,k) of each observation
                                   (typically the well perforation cells)
    radius       : localisation radius, grid cells

    Returns
    -------
    L : [N_params, N_obs] NumPy float32 array
    """
    # Pairwise distances via broadcasting
    # param_coords[:, None, :] → [N_params, 1, 3]
    # obs_coords[None, :, :]   → [1, N_obs, 3]
    diff = param_coords[:, None, :].astype(np.float32) - obs_coords[None, :, :].astype(np.float32)
    dist = np.sqrt((diff**2).sum(axis=-1))   # [N_params, N_obs]

    # Apply Gaspari-Cohn
    dist_j = jnp.array(dist)
    L_j = gaspari_cohn(dist_j, radius)
    return np.array(L_j, dtype=np.float32)


def well_observation_coords(wells, n_timesteps: int) -> np.ndarray:
    """
    Build observation coordinate array for all well time steps.

    For history matching, the 66 observations (22 wells × WOPR/WWPR/WGPR)
    at each time step are all associated with the well's (i, j) location.

    Returns [N_obs, 3] where N_obs = n_wells × 3 × n_timesteps.
    """
    coords = []
    for well in wells:
        if not well.perforations:
            continue
        i, j = well.perforations[0].i, well.perforations[0].j
        k_avg = np.mean([p.k for p in well.perforations])
        # Each of WOPR, WWPR, WGPR at each timestep maps to this cell
        for _ in range(3 * n_timesteps):
            coords.append([i, j, k_avg])
    return np.array(coords, dtype=np.float32)


def parameter_coords_3d(nx: int, ny: int, nz: int) -> np.ndarray:
    """
    Build [Nx*Ny*Nz, 3] coordinate array for a 3D grid.
    Used as param_coords in build_localisation_matrix.
    """
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    return np.stack([i_idx.ravel(), j_idx.ravel(), k_idx.ravel()], axis=1).astype(np.float32)
