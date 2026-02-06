"""
002-ModelEnsambles.py
=====================
Ensemble-based uncertainty estimation for 3-D gravity inversion using
Implicit Neural Representations (INR) with positional encoding (L = 2).

Approach
--------
Non-uniqueness is a fundamental property of potential-field inversion:
many different density models can explain the same gravity data equally
well.  To quantify how much the recovered model varies across equally
valid solutions we train **N_ENSEMBLE** independent INR networks, each
differing in:

  1. Random weight initialisation    – different starting points on the
     loss surface lead to different local minima.
  2. Different noise realisations    – each member sees the same true
     gravity contaminated with a fresh draw from the noise distribution,
     mimicking the real-world situation where the exact noise is unknown.
  3. (Optionally) shuffled mini-batch order or different learning-rate
     schedules could be added for further diversity.

After all members converge, we compute per-cell statistics:
  • **Ensemble mean**   – the best-estimate density model, combining
    information from all members.
  • **Ensemble std-dev** – pixel-wise uncertainty: high values flag
    regions where the data poorly constrain the model.
  • **Coefficient of variation (CV)** – std / |mean|, normalised
    uncertainty useful for comparing regions of different amplitude.

These maps reveal which parts of the model are robust (low σ) and which
are artefacts of the particular initialisation / noise draw (high σ).

References
----------
  • Mildenhall et al. (2020) – NeRF: positional encoding
  • Ensemble uncertainty in neural networks: Lakshminarayanan et al.
    (2017), "Simple and Scalable Predictive Uncertainty Estimation
    Using Deep Ensembles", NeurIPS.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt




# --- Grid / domain -----------------------------------------------------
DX       = 50.0       # cell size x (m)
DY       = 50.0       # cell size y (m)
DZ       = 50.0       # cell size z (m)
X_MAX    = 1000.0     # domain extent x (m)
Y_MAX    = 1000.0     # domain extent y (m)
Z_MAX    = 500.0      # domain extent z (m)

# --- Block model -------------------------------------------------------
RHO_BG   = 0.0        # background density contrast (kg/m³)
RHO_BLK  = 400.0      # block density contrast (kg/m³)

# --- Noise --------------------------------------------------------------
NOISE_LEVEL = 0.01     # fraction of gz_true std

# --- Training -----------------------------------------------------------
GAMMA    = 1.0         # data-term weight
EPOCHS   = 500         # epochs per ensemble member
LR       = 1e-2        # Adam learning rate

# --- INR network --------------------------------------------------------
HIDDEN       = 256     # hidden-layer width
DEPTH        = 4       # number of hidden layers
RHO_ABS_MAX  = 600.0   # tanh output scaling (kg/m³)

# --- Positional encoding (L = 2 octaves) --------------------------------
NUM_FREQS = 2

# --- Ensemble -----------------------------------------------------------
N_ENSEMBLE = 20        # number of independently trained members

# --- Member filtering ---------------------------------------------------
#   After training, optionally discard members that:
#     1. Failed to converge (RMS >> noise level)  →  RMS_FACTOR_MAX
#     2. Over-fitted the noise (RMS << noise level → χ² ≪ 1)  →  RMS_FACTOR_MIN
#     3. Produce a model far from the ensemble consensus
#        (spatial outlier)  →  SPATIAL_OUTLIER_ZSCORE
#
#   Set a threshold to None / np.inf to disable that filter.
RMS_FACTOR_MAX         = 3.0    # keep members with RMS < RMS_FACTOR_MAX × σ_noise (mGal)
RMS_FACTOR_MIN         = 0.1    # keep members with RMS > RMS_FACTOR_MIN × σ_noise (mGal)
SPATIAL_OUTLIER_ZSCORE = 3.0    # drop if >this fraction of cells lie >3σ from ensemble mean
SPATIAL_OUTLIER_FRAC   = 0.05   # fraction of cells that must be outliers to reject member

# --- Plotting -----------------------------------------------------------
CMAP       = 'turbo'
CMAP_STD   = 'magma'   # colormap for uncertainty maps
INV_VMAX   = 250       # fixed colourbar max for mean model



class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (NeRF-style, L frequency octaves).

    Maps each scalar coordinate p to:
        γ(p) = [p, sin(2⁰p), cos(2⁰p), … , sin(2^{L-1}p), cos(2^{L-1}p)]
    giving the network multiple spatial "rulers" at exponentially
    increasing resolution, counteracting the spectral bias of plain MLPs.
    """
    def __init__(self, num_freqs=2, include_input=True, input_dim=3):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))
        self.out_dim = (input_dim * (1 + 2 * num_freqs)
                        if include_input else input_dim * 2 * num_freqs)

    def forward(self, x):
        parts = [x] if self.include_input else []
        for f in self.freqs:
            parts += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(parts, dim=-1)


class DensityContrastINR(nn.Module):
    """MLP that maps normalised (x, y, z) → density contrast Δρ.

    Uses positional encoding → hidden layers (LeakyReLU) → tanh output
    scaled by rho_abs_max so the prediction is bounded.
    """
    def __init__(self, num_freqs=2, hidden=256, depth=4,
                 rho_abs_max=600.0):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=num_freqs)
        in_dim = self.pe.out_dim
        layers = [nn.Linear(in_dim, hidden), nn.LeakyReLU(0.01)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.01)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.rho_abs_max = float(rho_abs_max)

    def forward(self, x):
        return self.rho_abs_max * torch.tanh(self.net(self.pe(x)))


#-------------------------------------------------------------------------------------------------

def A_integral_torch(x, y, z):
    eps = 1e-20
    r = torch.sqrt(x**2 + y**2 + z**2).clamp_min(eps)
    return -(x * torch.log(torch.abs(y + r) + eps) +
             y * torch.log(torch.abs(x + r) + eps) -
             z * torch.atan2(x * y, z * r + eps))


@torch.inference_mode()
def construct_sensitivity_matrix_G(cell_grid, data_points, d1, d2, device):
    """Build the (Nobs × Ncells) forward-modelling kernel G."""
    Gamma = 6.67430e-11
    cx, cy, cz, czh = [cell_grid[:, i].unsqueeze(0) for i in range(4)]
    ox = data_points[:, 0].unsqueeze(1)
    oy = data_points[:, 1].unsqueeze(1)
    oz = data_points[:, 2].unsqueeze(1)
    x2, x1 = (cx + d1/2) - ox, (cx - d1/2) - ox
    y2, y1 = (cy + d2/2) - oy, (cy - d2/2) - oy
    z2, z1 = (cz + czh) - oz, (cz - czh) - oz
    A = (A_integral_torch(x2, y2, z2) - A_integral_torch(x2, y2, z1) -
         A_integral_torch(x2, y1, z2) + A_integral_torch(x2, y1, z1) -
         A_integral_torch(x1, y2, z2) + A_integral_torch(x1, y2, z1) +
         A_integral_torch(x1, y1, z2) - A_integral_torch(x1, y1, z1))
    return (Gamma * A).to(device)


def make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=0.0, rho_blk=400.0):
    """Create the true staircase-block density contrast model."""
    m = torch.full((Nx, Ny, Nz), rho_bg)
    for i in range(7):
        z_idx = 1 + i
        y_start, y_end = 11 - i, 16 - i
        x_start, x_end = 7, 13
        if 0 <= z_idx < Nz:
            ys, ye = max(0, y_start), min(Ny, y_end)
            xs, xe = max(0, x_start), min(Nx, x_end)
            m[xs:xe, ys:ye, z_idx] = rho_blk
    return m.view(-1), m


def get_block_boundaries(Nx, Ny, Nz):
    boundaries = []
    for i in range(7):
        z_idx = 1 + i
        y_start, y_end = 11 - i, 16 - i
        x_start, x_end = 7, 13
        if 0 <= z_idx < Nz:
            ys, ye = max(0, y_start), min(Ny, y_end)
            xs, xe = max(0, x_start), min(Nx, x_end)
            boundaries.append((xs, xe, ys, ye, z_idx))
    return boundaries


#--------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


#---------------------------------------------------------------------------------------

def train_one_member(member_id, coords_norm, G, gz_true, sigma_noise,
                     Nx, Ny, Nz, device, verbose=True):
    """Train one ensemble member end-to-end.

    Each member gets:
      • A unique random seed  →  different weight initialisation.
      • A fresh noise draw    →  different observed data.

    Returns
    -------
    m_inv : (Ncells,) tensor – recovered density contrast
    rms_gz : float           – final data-misfit RMS (mGal)
    history : dict           – training loss history
    """
    member_seed = 1000 * member_id + 7  # reproducible but diverse
    set_seed(member_seed)

    # Fresh noise realisation for this member
    noise = sigma_noise * torch.randn_like(gz_true)
    gz_obs = gz_true + noise
    Wd = 1.0 / sigma_noise

    model = DensityContrastINR(
        num_freqs=NUM_FREQS, hidden=HIDDEN, depth=DEPTH,
        rho_abs_max=RHO_ABS_MAX
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"total": [], "gravity": []}
    for ep in range(EPOCHS):
        opt.zero_grad()
        m_pred = model(coords_norm).view(-1)
        gz_pred = torch.matmul(G, m_pred.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        data_term = GAMMA * torch.mean((Wd * residual) ** 2)
        loss = data_term
        loss.backward()
        opt.step()
        history['gravity'].append(float(data_term.item()))
        history['total'].append(float(loss.item()))

    # Final evaluation
    with torch.no_grad():
        m_inv = model(coords_norm).view(-1)
        gz_final = (G @ m_inv.unsqueeze(1)).squeeze(1)
    rms_gz = float(torch.sqrt(torch.mean((gz_final - gz_obs)**2)).item()) * 1e5

    if verbose:
        print(f"  Member {member_id:2d}  |  final loss = {history['total'][-1]:.3e}"
              f"  |  RMS = {rms_gz:.4f} mGal  |  seed = {member_seed}")

    return m_inv, rms_gz, history


#---------------------------------------------------------------------------------------

def plot_ensemble_results(mean_3d, std_3d, cv_3d, tru_3d,
                          gz_obs, gz_pred_mean, obs,
                          x, y, z, dx, dy, dz,
                          block_boundaries, Nx, Ny, Nz):
    """Produce a comprehensive multi-panel figure.

    Row 0 : True model slices (XY, XZ, YZ)
    Row 1 : Ensemble mean slices with true-model boundary overlay
    Row 2 : Ensemble std-dev (uncertainty) slices
    Row 3 : Observed / predicted / residual gravity maps
    """
    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)

    x1d, y1d, z1d = x, y, z
    x_edge_min, x_edge_max = x1d[0] - dx/2, x1d[-1] + dx/2
    y_edge_min, y_edge_max = y1d[0] - dy/2, y1d[-1] + dy/2
    z_edge_min, z_edge_max = z1d[0] - dz/2, z1d[-1] + dz/2

    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]
    extent_xz = [x_edge_min, x_edge_max, z_edge_max, z_edge_min]
    extent_yz = [y_edge_min, y_edge_max, z_edge_max, z_edge_min]

    tru = tru_3d
    tru_max = tru.max()
    inv_max = INV_VMAX

    fig, axes = plt.subplots(4, 3, figsize=(17, 20))

    # ── Row 0: True model ──────────────────────────────────────────
    im = axes[0, 0].imshow(tru[:, :, iz].T, origin='lower', extent=extent_xy,
                           aspect='equal', vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 0].set_title(f"True Δρ XY @ z≈{z1d[iz]:.0f} m")
    fig.colorbar(im, ax=axes[0, 0], label='kg/m³', fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(tru[:, iy, :].T, origin='upper', extent=extent_xz,
                           aspect=1.0, vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 1].set_title(f"True Δρ XZ @ y≈{y1d[iy]:.0f} m")

    im = axes[0, 2].imshow(tru[ix, :, :].T, origin='upper', extent=extent_yz,
                           aspect=1.0, vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 2].set_title(f"True Δρ YZ @ x≈{x1d[ix]:.0f} m")

    # ── Row 1: Ensemble mean ───────────────────────────────────────
    im = axes[1, 0].imshow(mean_3d[:, :, iz].T, origin='lower', extent=extent_xy,
                           aspect='equal', vmin=0, vmax=inv_max, cmap=CMAP)
    axes[1, 0].set_title(f"Mean Δρ XY @ z≈{z1d[iz]:.0f} m")
    boundary_for_z = next((b for b in block_boundaries if b[4] == iz), None)
    if boundary_for_z:
        xs, xe, ys, ye, _ = boundary_for_z
        rect = plt.Rectangle((x1d[xs] - dx/2, y1d[ys] - dy/2),
                              (xe - xs) * dx, (ye - ys) * dy,
                              edgecolor='white', facecolor='none', linewidth=2)
        axes[1, 0].add_patch(rect)
    fig.colorbar(im, ax=axes[1, 0], label='kg/m³', fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(mean_3d[:, iy, :].T, origin='upper', extent=extent_xz,
                           aspect=1.0, vmin=0, vmax=inv_max, cmap=CMAP)
    axes[1, 1].set_title(f"Mean Δρ XZ @ y≈{y1d[iy]:.0f} m")
    # overlay XZ boundaries
    z_indices_in_slice, x_range = [], None
    for b in block_boundaries:
        bxs, bxe, bys, bye, bz = b
        if bys <= iy < bye:
            z_indices_in_slice.append(bz)
            if x_range is None:
                x_range = (bxs, bxe)
    if z_indices_in_slice and x_range:
        min_z, max_z = min(z_indices_in_slice), max(z_indices_in_slice)
        bxs, bxe = x_range
        rect = plt.Rectangle((x1d[bxs] - dx/2, z1d[min_z] - dz/2),
                              (bxe - bxs) * dx, (max_z - min_z + 1) * dz,
                              edgecolor='white', facecolor='none', linewidth=2)
        axes[1, 1].add_patch(rect)

    im = axes[1, 2].imshow(mean_3d[ix, :, :].T, origin='upper', extent=extent_yz,
                           aspect=1.0, vmin=0, vmax=inv_max, cmap=CMAP)
    axes[1, 2].set_title(f"Mean Δρ YZ @ x≈{x1d[ix]:.0f} m")
    for bxs, bxe, bys, bye, bz in block_boundaries:
        if bxs <= ix < bxe:
            rect = plt.Rectangle((y1d[bys] - dy/2, z1d[bz] - dz/2),
                                 (bye - bys) * dy, dz,
                                 edgecolor='white', facecolor='none', linewidth=2)
            axes[1, 2].add_patch(rect)

    # ── Row 2: Ensemble std-dev (uncertainty) ──────────────────────
    std_max = std_3d.max()

    im = axes[2, 0].imshow(std_3d[:, :, iz].T, origin='lower', extent=extent_xy,
                           aspect='equal', vmin=0, vmax=std_max, cmap=CMAP_STD)
    axes[2, 0].set_title(f"Std(Δρ) XY @ z≈{z1d[iz]:.0f} m")
    if boundary_for_z:
        xs, xe, ys, ye, _ = boundary_for_z
        rect = plt.Rectangle((x1d[xs] - dx/2, y1d[ys] - dy/2),
                              (xe - xs) * dx, (ye - ys) * dy,
                              edgecolor='cyan', facecolor='none', linewidth=2)
        axes[2, 0].add_patch(rect)
    fig.colorbar(im, ax=axes[2, 0], label='kg/m³', fraction=0.046, pad=0.04)

    im = axes[2, 1].imshow(std_3d[:, iy, :].T, origin='upper', extent=extent_xz,
                           aspect=1.0, vmin=0, vmax=std_max, cmap=CMAP_STD)
    axes[2, 1].set_title(f"Std(Δρ) XZ @ y≈{y1d[iy]:.0f} m")
    if z_indices_in_slice and x_range:
        min_z, max_z = min(z_indices_in_slice), max(z_indices_in_slice)
        bxs, bxe = x_range
        rect = plt.Rectangle((x1d[bxs] - dx/2, z1d[min_z] - dz/2),
                              (bxe - bxs) * dx, (max_z - min_z + 1) * dz,
                              edgecolor='cyan', facecolor='none', linewidth=2)
        axes[2, 1].add_patch(rect)

    im = axes[2, 2].imshow(std_3d[ix, :, :].T, origin='upper', extent=extent_yz,
                           aspect=1.0, vmin=0, vmax=std_max, cmap=CMAP_STD)
    axes[2, 2].set_title(f"Std(Δρ) YZ @ x≈{x1d[ix]:.0f} m")
    for bxs, bxe, bys, bye, bz in block_boundaries:
        if bxs <= ix < bxe:
            rect = plt.Rectangle((y1d[bys] - dy/2, z1d[bz] - dz/2),
                                 (bye - bys) * dy, dz,
                                 edgecolor='cyan', facecolor='none', linewidth=2)
            axes[2, 2].add_patch(rect)

    # ── Row 3: Gravity data (mean prediction) ─────────────────────
    def to_mgal(g):
        return 1e5 * g.detach().cpu().numpy()
    obs_mgal = to_mgal(gz_obs)
    pre_mgal = to_mgal(gz_pred_mean)
    res_mgal = obs_mgal - pre_mgal

    obs_x = obs[:, 0].cpu().numpy()
    obs_y = obs[:, 1].cpu().numpy()
    v = max(abs(obs_mgal).max(), abs(pre_mgal).max())

    sc = axes[3, 0].scatter(obs_x, obs_y, c=obs_mgal, s=80, cmap=CMAP,
                            vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[3, 0].set_title('Observed gz (mGal)')
    fig.colorbar(sc, ax=axes[3, 0], fraction=0.046, pad=0.04)

    sc = axes[3, 1].scatter(obs_x, obs_y, c=pre_mgal, s=80, cmap=CMAP,
                            vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[3, 1].set_title('Predicted gz (mean, mGal)')
    fig.colorbar(sc, ax=axes[3, 1], fraction=0.046, pad=0.04)

    rms_res = np.sqrt(np.mean(res_mgal**2))
    vmax_res = np.abs(res_mgal).max()
    sc = axes[3, 2].scatter(obs_x, obs_y, c=res_mgal, s=80, cmap=CMAP,
                            vmin=-vmax_res, vmax=vmax_res, marker='o',
                            edgecolors='none')
    axes[3, 2].set_title(f'Residual (RMS={rms_res:.3f} mGal)')
    fig.colorbar(sc, ax=axes[3, 2], fraction=0.046, pad=0.04)

    # Axis labels
    for r in range(3):
        axes[r, 0].set_xlabel('x (m)'); axes[r, 0].set_ylabel('y (m)')
        axes[r, 1].set_xlabel('x (m)'); axes[r, 1].set_ylabel('Depth (m)')
        axes[r, 2].set_xlabel('y (m)'); axes[r, 2].set_ylabel('Depth (m)')
    for ax in axes[3, :]:
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig('plots/Ensemble_main.png', dpi=300)
    plt.close(fig)
    print(f"  Saved plots/Ensemble_main.png")


def plot_cv_map(cv_3d, x, y, z, dx, dy, dz, block_boundaries,
                Nx, Ny, Nz):
    """Plot Coefficient-of-Variation (σ / |μ|) slices.

    CV highlights where the *relative* uncertainty is highest,
    irrespective of amplitude.  It is most informative inside the
    anomalous body (where mean ≠ 0).
    """
    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)
    x_edge_min, x_edge_max = x[0] - dx/2, x[-1] + dx/2
    y_edge_min, y_edge_max = y[0] - dy/2, y[-1] + dy/2
    z_edge_min, z_edge_max = z[0] - dz/2, z[-1] + dz/2
    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]
    extent_xz = [x_edge_min, x_edge_max, z_edge_max, z_edge_min]
    extent_yz = [y_edge_min, y_edge_max, z_edge_max, z_edge_min]

    cv_max = min(cv_3d.max(), 2.0)  # clip display for readability

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    im = axes[0].imshow(cv_3d[:, :, iz].T, origin='lower', extent=extent_xy,
                        aspect='equal', vmin=0, vmax=cv_max, cmap=CMAP_STD)
    axes[0].set_title(f"CV(Δρ) XY @ z≈{z[iz]:.0f} m")
    fig.colorbar(im, ax=axes[0], label='σ/|μ|', fraction=0.046, pad=0.04)

    im = axes[1].imshow(cv_3d[:, iy, :].T, origin='upper', extent=extent_xz,
                        aspect=1.0, vmin=0, vmax=cv_max, cmap=CMAP_STD)
    axes[1].set_title(f"CV(Δρ) XZ @ y≈{y[iy]:.0f} m")

    im = axes[2].imshow(cv_3d[ix, :, :].T, origin='upper', extent=extent_yz,
                        aspect=1.0, vmin=0, vmax=cv_max, cmap=CMAP_STD)
    axes[2].set_title(f"CV(Δρ) YZ @ x≈{x[ix]:.0f} m")

    for ax in axes:
        ax.set_xlabel('x (m)' if ax != axes[2] else 'y (m)')
        ax.set_ylabel('y (m)' if ax == axes[0] else 'Depth (m)')

    fig.tight_layout()
    fig.savefig('plots/Ensemble_CV.png', dpi=300)
    plt.close(fig)
    print(f"  Saved plots/Ensemble_CV.png")


def plot_member_spread(all_models_3d, x, y, z, dx, dy, dz,
                       Nx, Ny, Nz, n_show=None):
    """Show each ensemble member's XY slice to visualise spread."""
    n = all_models_3d.shape[0]
    if n_show is None:
        n_show = min(n, 10)
    iz = min(Nz - 1, 5)
    ncols = min(n_show, 5)
    nrows = int(np.ceil(n_show / ncols))

    x_edge_min, x_edge_max = x[0] - dx/2, x[-1] + dx/2
    y_edge_min, y_edge_max = y[0] - dy/2, y[-1] + dy/2
    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)
    for k in range(n_show):
        ax = axes[k // ncols, k % ncols]
        im = ax.imshow(all_models_3d[k, :, :, iz].T, origin='lower',
                       extent=extent_xy, aspect='equal', vmin=0,
                       vmax=INV_VMAX, cmap=CMAP)
        ax.set_title(f"Member {k}", fontsize=10)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
    # hide unused axes
    for k in range(n_show, nrows * ncols):
        axes[k // ncols, k % ncols].set_visible(False)

    fig.suptitle(f'Ensemble members – XY slice @ z≈{z[iz]:.0f} m',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig('plots/Ensemble_members.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plots/Ensemble_members.png")


def plot_loss_curves(all_histories):
    """Overlay training-loss curves for every ensemble member."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for k, h in enumerate(all_histories):
        ax.plot(h['total'], alpha=0.5, linewidth=0.8, label=f'M{k}')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss – all ensemble members')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    fig.tight_layout()
    fig.savefig('plots/Ensemble_loss.png', dpi=200)
    plt.close(fig)
    print(f"  Saved plots/Ensemble_loss.png")


# ══════════════════════════════════════════════════════════════════════
#                      ENSEMBLE FILTERING
# ══════════════════════════════════════════════════════════════════════

def filter_ensemble(all_models, all_rms, all_hist, sigma_noise_mgal):
    """Discard poorly-converged, over-fit, or outlier ensemble members.

    Filtering criteria (applied in order):

    1. **Convergence check** (RMS_FACTOR_MAX)
       Members whose final data-misfit RMS exceeds
       ``RMS_FACTOR_MAX × σ_noise`` did not converge adequately.
       They sit in a bad local minimum and would bias the ensemble
       mean / inflate variance with unphysical structure.

    2. **Over-fitting check** (RMS_FACTOR_MIN)
       Members whose RMS is *far below* the noise level (χ² ≪ 1)
       have fitted noise rather than signal.  Their models contain
       spurious small-scale artefacts that corrupt the mean and
       artificially inflate the std-dev.

    3. **Spatial outlier check** (SPATIAL_OUTLIER_ZSCORE / _FRAC)
       Even if a member's RMS is acceptable, its density model may
       be globally unusual (e.g. large artefacts in a region that
       all other members agree on).  We compute the preliminary
       ensemble mean & std from *all remaining* members, then flag
       any member where more than ``SPATIAL_OUTLIER_FRAC`` of cells
       deviate by more than ``SPATIAL_OUTLIER_ZSCORE × σ_cell``.

    Parameters
    ----------
    all_models : (N_ens, Ncells) ndarray
    all_rms    : list of float  (mGal per member)
    all_hist   : list of dict   (training histories)
    sigma_noise_mgal : float    (noise std in mGal)

    Returns
    -------
    filt_models, filt_rms, filt_hist, keep_mask
    """
    N = len(all_rms)
    keep = np.ones(N, dtype=bool)

    # ── 1. Convergence filter ──────────────────────────────────────
    if RMS_FACTOR_MAX is not None:
        rms_thresh = RMS_FACTOR_MAX * sigma_noise_mgal
        for k in range(N):
            if all_rms[k] > rms_thresh:
                keep[k] = False
        n_dropped = N - keep.sum()
        if n_dropped:
            print(f"  Filter 1 — convergence (RMS > {rms_thresh:.4f} mGal): "
                  f"dropped {n_dropped} member(s)")

    # ── 2. Over-fitting filter ─────────────────────────────────────
    if RMS_FACTOR_MIN is not None:
        rms_floor = RMS_FACTOR_MIN * sigma_noise_mgal
        n_before = keep.sum()
        for k in range(N):
            if keep[k] and all_rms[k] < rms_floor:
                keep[k] = False
        n_dropped = n_before - keep.sum()
        if n_dropped:
            print(f"  Filter 2 — over-fitting (RMS < {rms_floor:.4f} mGal): "
                  f"dropped {n_dropped} member(s)")

    # ── 3. Spatial outlier filter ──────────────────────────────────
    if (SPATIAL_OUTLIER_ZSCORE is not None
            and SPATIAL_OUTLIER_FRAC is not None
            and keep.sum() >= 3):  # need ≥3 members to compute σ
        idx_keep = np.where(keep)[0]
        subset = all_models[idx_keep]          # (N_kept, Ncells)
        mu  = subset.mean(axis=0)              # (Ncells,)
        sig = subset.std(axis=0, ddof=1)       # (Ncells,)
        sig = np.where(sig < 1e-12, 1e-12, sig)  # avoid /0

        n_before = keep.sum()
        Ncells = all_models.shape[1]
        for k in idx_keep:
            z_scores = np.abs(all_models[k] - mu) / sig
            frac_outlier = (z_scores > SPATIAL_OUTLIER_ZSCORE).sum() / Ncells
            if frac_outlier > SPATIAL_OUTLIER_FRAC:
                keep[k] = False
        n_dropped = n_before - keep.sum()
        if n_dropped:
            print(f"  Filter 3 — spatial outlier (>{SPATIAL_OUTLIER_FRAC*100:.0f}% "
                  f"cells >{SPATIAL_OUTLIER_ZSCORE:.1f}σ): "
                  f"dropped {n_dropped} member(s)")

    # ── Summary ────────────────────────────────────────────────────
    n_kept = int(keep.sum())
    print(f"  Ensemble filtering: kept {n_kept}/{N} members  "
          f"(dropped {N - n_kept})")
    if n_kept < 2:
        print("  ⚠  Fewer than 2 members remain — disabling filtering.")
        keep = np.ones(N, dtype=bool)
        n_kept = N

    idx = np.where(keep)[0]
    filt_models = all_models[idx]
    filt_rms    = [all_rms[k] for k in idx]
    filt_hist   = [all_hist[k] for k in idx]
    return filt_models, filt_rms, filt_hist, keep


# ══════════════════════════════════════════════════════════════════════
#                            MAIN
# ══════════════════════════════════════════════════════════════════════

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    # --- Build grid ------------------------------------------------
    dx, dy, dz = DX, DY, DZ
    x = np.arange(0.0, X_MAX + dx, dx)
    y = np.arange(0.0, Y_MAX + dy, dy)
    z = np.arange(0.0, Z_MAX + dz, dz)
    Nx, Ny, Nz = len(x), len(y), len(z)

    X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)

    c_mean = grid_coords.mean(axis=0, keepdims=True)
    c_std  = grid_coords.std(axis=0, keepdims=True)
    coords_norm = (grid_coords - c_mean) / (c_std + 1e-12)
    coords_norm = torch.tensor(coords_norm, dtype=torch.float32,
                               device=device, requires_grad=True)

    dz_half = dz / 2.0
    cell_grid = np.hstack([grid_coords,
                           np.full((grid_coords.shape[0], 1), dz_half)])
    cell_grid = torch.tensor(cell_grid, dtype=torch.float32, device=device)

    XX, YY = np.meshgrid(x, y, indexing='ij')
    obs = np.column_stack([XX.ravel(), YY.ravel(), -np.ones(XX.size)])
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # --- Sensitivity matrix (shared across all members) ------------
    print("Assembling sensitivity G …")
    t0 = time.time()
    G = construct_sensitivity_matrix_G(cell_grid, obs, dx, dy, device)
    G = G.clone().detach().requires_grad_(False)
    print(f"  G shape = {tuple(G.shape)}, time = {time.time() - t0:.2f}s")

    # --- True model & clean gravity --------------------------------
    rho_true_vec, rho_true_3d = make_block_model(
        Nx, Ny, Nz, dx, dy, dz, rho_bg=RHO_BG, rho_blk=RHO_BLK)
    rho_true_vec = rho_true_vec.to(device)

    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)
    sigma_noise = NOISE_LEVEL * gz_true.std()

    block_boundaries = get_block_boundaries(Nx, Ny, Nz)

    # --- Train ensemble --------------------------------------------
    print(f"\n{'═'*60}")
    print(f"  Training ensemble of {N_ENSEMBLE} members")
    print(f"  Encoding: positional (L = {NUM_FREQS})")
    print(f"  Epochs per member: {EPOCHS}")
    print(f"{'═'*60}\n")

    all_models  = []   # list of (Ncells,) tensors
    all_rms     = []
    all_hist    = []
    t_start = time.time()

    for k in range(N_ENSEMBLE):
        m_inv, rms_gz, hist = train_one_member(
            k, coords_norm, G, gz_true, sigma_noise,
            Nx, Ny, Nz, device, verbose=True)
        all_models.append(m_inv.cpu().numpy())
        all_rms.append(rms_gz)
        all_hist.append(hist)

    elapsed = time.time() - t_start
    print(f"\n  Total training time: {elapsed:.1f}s "
          f"({elapsed / N_ENSEMBLE:.1f}s per member)")

    # --- Filter ensemble members -----------------------------------
    all_models_raw = np.stack(all_models, axis=0)          # (N_ens, Ncells)
    sigma_noise_mgal = float(sigma_noise.item()) * 1e5     # convert to mGal

    print(f"\n{'─'*60}")
    print(f"  Filtering ensemble (σ_noise = {sigma_noise_mgal:.4f} mGal)")
    print(f"{'─'*60}")
    filt_models, filt_rms, filt_hist, keep_mask = filter_ensemble(
        all_models_raw, all_rms, all_hist, sigma_noise_mgal)
    N_kept = filt_models.shape[0]

    # Print per-member status
    print(f"\n  {'ID':>4s}  {'RMS (mGal)':>11s}  {'Status':>8s}")
    for k in range(N_ENSEMBLE):
        status = '  kept' if keep_mask[k] else 'dropped'
        print(f"  {k:4d}  {all_rms[k]:11.4f}  {status:>8s}")

    # --- Ensemble statistics (filtered) ----------------------------
    ens_mean = filt_models.mean(axis=0)                    # (Ncells,)
    ens_std  = filt_models.std(axis=0, ddof=1)             # (Ncells,)
    ens_cv   = ens_std / (np.abs(ens_mean) + 1e-12)        # coefficient of variation

    # Reshape to 3-D for plotting
    mean_3d = ens_mean.reshape(Nx, Ny, Nz)
    std_3d  = ens_std.reshape(Nx, Ny, Nz)
    cv_3d   = ens_cv.reshape(Nx, Ny, Nz)
    tru_3d  = rho_true_3d.cpu().numpy()
    all_3d  = filt_models.reshape(N_kept, Nx, Ny, Nz)

    # Mean prediction gravity
    with torch.no_grad():
        mean_tensor = torch.tensor(ens_mean, dtype=torch.float32, device=device)
        gz_pred_mean = (G @ mean_tensor.unsqueeze(1)).squeeze(1)

    # --- Summary statistics ----------------------------------------
    rms_rho = np.sqrt(np.mean((ens_mean - rho_true_vec.cpu().numpy())**2))
    rms_gz_mean = float(
        torch.sqrt(torch.mean(
            (gz_pred_mean - gz_true.to(device))**2)).item()) * 1e5
    mean_rms_members = np.mean(filt_rms)

    print(f"\n{'─'*60}")
    print(f"  Ensemble summary ({N_kept}/{N_ENSEMBLE} members after filtering)")
    print(f"{'─'*60}")
    print(f"  Mean member RMS data misfit : {mean_rms_members:.4f} mGal")
    print(f"  Ensemble-mean RMS misfit    : {rms_gz_mean:.4f} mGal")
    print(f"  Ensemble-mean Δρ error      : {rms_rho:.2f} kg/m³")
    print(f"  Max per-cell std(Δρ)        : {ens_std.max():.2f} kg/m³")
    print(f"  Mean per-cell std(Δρ)       : {ens_std.mean():.2f} kg/m³")
    print(f"  Median per-cell std(Δρ)     : {np.median(ens_std):.2f} kg/m³")
    print(f"{'─'*60}\n")

    # Use the first member's noise realisation for the observed-data
    # plot (all members see slightly different noise, but the
    # difference is negligible for display purposes).
    set_seed(1000 * 0 + 7)
    noise_display = sigma_noise * torch.randn_like(gz_true)
    gz_obs_display = gz_true + noise_display

    # --- Plots -----------------------------------------------------
    print("Generating plots …")
    plot_ensemble_results(
        mean_3d, std_3d, cv_3d, tru_3d,
        gz_obs_display, gz_pred_mean, obs,
        x, y, z, dx, dy, dz, block_boundaries, Nx, Ny, Nz)
    plot_cv_map(cv_3d, x, y, z, dx, dy, dz, block_boundaries, Nx, Ny, Nz)
    plot_member_spread(all_3d, x, y, z, dx, dy, dz, Nx, Ny, Nz)
    plot_loss_curves(all_hist)

    print("Done.")


if __name__ == '__main__':
    run()
