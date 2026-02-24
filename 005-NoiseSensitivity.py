"""
006-NoiseSensitivity.py
=======================
Noise-sensitivity study on the block model using positional-encoding INR.

Tests 4 noise types at 4 noise levels each, trains one INR per
combination, then summarises accuracy and data-fit quality.

Noise types
-----------
  1. Gaussian IID            -- standard assumption
  2. Laplace (heavy-tailed)  -- robust data scenarios
  3. Correlated Gaussian     -- spatially-smooth errors
  4. Outlier-contaminated    -- 10% of obs replaced by 5x noise

Noise levels
------------
  0.5%, 1%, 2%, 5% of std(gz_true)

Outputs
-------
  plots/Noise_block_results.png     -- recovered models (noise type x level)
  plots/Noise_block_heatmap.png     -- RMS rho heatmap (type x level)
  plots/Noise_block_metrics.txt     -- tabular summary
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

SEED = 42

DX, DY, DZ = 50.0, 50.0, 50.0
X_MAX, Y_MAX, Z_MAX = 1000.0, 1000.0, 500.0
RHO_BG, RHO_BLK = 0.0, 400.0

GAMMA  = 1.0
EPOCHS = 500
LR     = 1e-2
HIDDEN = 256
DEPTH  = 4
RHO_ABS_MAX = 600.0

NOISE_TYPES  = ['gaussian', 'laplace', 'correlated', 'outliers']
NOISE_LEVELS = [0.01, 0.03, 0.05]

CMAP = 'turbo'
VMAX = 250


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def A_integral_torch(x, y, z):
    eps = 1e-20
    r = torch.sqrt(x**2 + y**2 + z**2).clamp_min(eps)
    return -(x * torch.log(torch.abs(y + r) + eps) +
             y * torch.log(torch.abs(x + r) + eps) -
             z * torch.atan2(x * y, z * r + eps))


@torch.inference_mode()
def construct_sensitivity_matrix_G(cell_grid, data_points, d1, d2, device):
    Gamma = 6.67430e-11
    cx, cy, cz, czh = [cell_grid[:, i].unsqueeze(0) for i in range(4)]
    ox, oy, oz = [data_points[:, i].unsqueeze(1) for i in range(3)]
    x2, x1 = (cx + d1/2) - ox, (cx - d1/2) - ox
    y2, y1 = (cy + d2/2) - oy, (cy - d2/2) - oy
    z2, z1 = (cz + czh) - oz, (cz - czh) - oz
    A = (A_integral_torch(x2, y2, z2) - A_integral_torch(x2, y2, z1) -
         A_integral_torch(x2, y1, z2) + A_integral_torch(x2, y1, z1) -
         A_integral_torch(x1, y2, z2) + A_integral_torch(x1, y2, z1) +
         A_integral_torch(x1, y1, z2) - A_integral_torch(x1, y1, z1))
    return (Gamma * A).to(device)


def make_block_model(Nx, Ny, Nz, rho_bg=0.0, rho_blk=400.0):
    m = torch.full((Nx, Ny, Nz), rho_bg)
    for i in range(7):
        z_idx = 1 + i
        ys, ye = max(0, 11-i), min(Ny, 16-i)
        xs, xe = 7, min(Nx, 13)
        if 0 <= z_idx < Nz:
            m[xs:xe, ys:ye, z_idx] = rho_blk
    return m.view(-1), m


# ═══════════════════════════════════════════════════════════════════════
#  NOISE GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def add_noise(gz_true, noise_type, noise_frac, device):
    sigma = noise_frac * gz_true.std()
    Nobs = len(gz_true)

    if noise_type == 'gaussian':
        noise = sigma * torch.randn(Nobs, device=device)

    elif noise_type == 'laplace':
        noise = torch.distributions.Laplace(0.0, sigma / np.sqrt(2)).sample((Nobs,)).to(device)

    elif noise_type == 'correlated':
        raw = sigma * torch.randn(Nobs, device=device)
        raw_np = raw.cpu().numpy()
        k = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
        k /= k.sum()
        raw_np = np.convolve(raw_np, k, mode='same')
        noise = torch.tensor(raw_np, dtype=torch.float32, device=device)

    elif noise_type == 'outliers':
        noise = sigma * torch.randn(Nobs, device=device)
        n_out = max(1, int(0.10 * Nobs))
        idx = torch.randperm(Nobs)[:n_out]
        noise[idx] *= 5.0

    else:
        raise ValueError(noise_type)

    return gz_true + noise, sigma


# ═══════════════════════════════════════════════════════════════════════
#  ENCODING + MODEL
# ═══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=2, include_input=True, input_dim=3):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))
        self.out_dim = input_dim * (1 + 2*num_freqs) if include_input else input_dim * 2*num_freqs
    def forward(self, x):
        parts = [x] if self.include_input else []
        for f in self.freqs:
            parts += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(parts, dim=-1)


class DensityContrastINR(nn.Module):
    def __init__(self, hidden=256, depth=4, rho_abs_max=600.0, num_freqs=2):
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


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_one(coords_norm, G, gz_obs, sigma, device):
    Wd = 1.0 / sigma
    model = DensityContrastINR(hidden=HIDDEN, depth=DEPTH,
                               rho_abs_max=RHO_ABS_MAX).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    for ep in range(EPOCHS):
        opt.zero_grad()
        m_pred = model(coords_norm).view(-1)
        gz_pred = (G @ m_pred.unsqueeze(1)).squeeze(1)
        loss = GAMMA * torch.mean((Wd * (gz_pred - gz_obs)) ** 2)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))

    with torch.no_grad():
        m_inv = model(coords_norm).view(-1)
        gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)
    return m_inv, gz_pred, history


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def run():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    dx, dy, dz = DX, DY, DZ
    x = np.arange(0.0, X_MAX + dx, dx)
    y = np.arange(0.0, Y_MAX + dy, dy)
    z = np.arange(0.0, Z_MAX + dz, dz)
    Nx, Ny, Nz = len(x), len(y), len(z)

    X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)
    c_mean = grid_coords.mean(0, keepdims=True)
    c_std = grid_coords.std(0, keepdims=True)
    coords_norm = torch.tensor(
        (grid_coords - c_mean) / (c_std + 1e-12),
        dtype=torch.float32, device=device, requires_grad=True)

    cell_grid = torch.tensor(
        np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz/2)]),
        dtype=torch.float32, device=device)
    XX, YY = np.meshgrid(x, y, indexing='ij')
    obs = torch.tensor(
        np.column_stack([XX.ravel(), YY.ravel(), -np.ones(XX.size)]),
        dtype=torch.float32, device=device)

    print("Assembling sensitivity G ...")
    G = construct_sensitivity_matrix_G(cell_grid, obs, dx, dy, device)
    G = G.clone().detach().requires_grad_(False)

    rho_true_vec, _ = make_block_model(Nx, Ny, Nz)
    rho_true_vec = rho_true_vec.to(device)
    rho_true_np  = rho_true_vec.cpu().numpy()
    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)

    n_types  = len(NOISE_TYPES)
    n_levels = len(NOISE_LEVELS)

    rms_table = np.zeros((n_types, n_levels))
    models_3d = np.zeros((n_types, n_levels, Nx, Ny, Nz))

    for it, ntype in enumerate(NOISE_TYPES):
        for il, nlev in enumerate(NOISE_LEVELS):
            set_seed(SEED)  # reset so noise draw is reproducible
            gz_obs, sigma = add_noise(gz_true, ntype, nlev, device)
            print(f"\n  {ntype:12s} @ {nlev*100:.1f}%  (sigma={float(sigma):.4e})")

            m_inv, gz_pred, hist = train_one(coords_norm, G, gz_obs, sigma, device)
            m_np = m_inv.detach().cpu().numpy()
            rms = np.sqrt(np.mean((m_np - rho_true_np)**2))
            rms_table[it, il] = rms
            models_3d[it, il] = m_np.reshape(Nx, Ny, Nz)
            print(f"    final loss = {hist[-1]:.3e}  |  RMS rho = {rms:.2f}")

    # -------- metrics table --------
    lines = [f"{'Noise':14s}" + ''.join(f"  {int(nl*100):>3d}%" for nl in NOISE_LEVELS)]
    for it, ntype in enumerate(NOISE_TYPES):
        vals = ''.join(f"  {rms_table[it,il]:6.1f}" for il in range(n_levels))
        lines.append(f"{ntype:14s}{vals}")
    tbl = '\n'.join(lines)
    print(f"\n{tbl}")
    with open('plots/Noise_block_metrics.txt', 'w') as f:
        f.write(tbl)

    # -------- model slices figure (XY, XZ, YZ per noise config) --------
    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)
    ext_xy = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]
    ext_xz = [x[0]-dx/2, x[-1]+dx/2, z[-1]+dz/2, z[0]-dz/2]
    ext_yz = [y[0]-dy/2, y[-1]+dy/2, z[-1]+dz/2, z[0]-dz/2]

    ncols = n_levels * 3  # 3 slices per noise level
    fig, axes = plt.subplots(n_types, ncols,
                             figsize=(4*ncols, 4*n_types),
                             squeeze=False)
    for it in range(n_types):
        for il in range(n_levels):
            m3d = models_3d[it, il]
            col_base = il * 3

            # XY slice
            sl_xy = m3d[:, :, iz].T
            im = axes[it, col_base].imshow(sl_xy, origin='lower', extent=ext_xy,
                                           aspect='auto', vmin=0, vmax=VMAX, cmap=CMAP)
            axes[it, col_base].set_aspect(1.0)
            axes[it, col_base].set_xlabel('x (m)', fontsize=12, fontweight='bold')
            if il == 0:
                axes[it, col_base].set_ylabel(NOISE_TYPES[it], fontsize=13, fontweight='bold')

            # XZ slice
            sl_xz = m3d[:, iy, :].T
            axes[it, col_base+1].imshow(sl_xz, origin='upper', extent=ext_xz,
                                        aspect=1.0, vmin=0, vmax=VMAX, cmap=CMAP)
            axes[it, col_base+1].set_xlabel('x (m)', fontsize=12, fontweight='bold')

            # YZ slice
            sl_yz = m3d[ix, :, :].T
            axes[it, col_base+2].imshow(sl_yz, origin='upper', extent=ext_yz,
                                        aspect=1.0, vmin=0, vmax=VMAX, cmap=CMAP)
            axes[it, col_base+2].set_xlabel('y (m)', fontsize=12, fontweight='bold')

            # Per-panel title: noise type + level + slice orientation
            pct = f"{NOISE_LEVELS[il]*100:.1f}%"
            ntype = NOISE_TYPES[it]
            axes[it, col_base].set_title(f"{ntype} {pct} XY", fontsize=12, fontweight='bold')
            axes[it, col_base+1].set_title(f"{ntype} {pct} XZ", fontsize=12, fontweight='bold')
            axes[it, col_base+2].set_title(f"{ntype} {pct} YZ", fontsize=12, fontweight='bold')

    fig.suptitle(f'Noise sensitivity  |  block model  |  '
                 f'z={z[iz]:.0f} m, y={y[iy]:.0f} m, x={x[ix]:.0f} m',
                 fontsize=15, fontweight='bold')
    for ax in axes.ravel():
        ax.tick_params(labelsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('plots/Noise_block_results.png', dpi=300)
    plt.close(fig)
    print("  Saved plots/Noise_block_results.png")

    # -------- RMS heatmap --------
    fig2, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(rms_table, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels([f"{nl*100:.1f}%" for nl in NOISE_LEVELS])
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(NOISE_TYPES)
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Noise type')
    ax.set_title('RMS density error (kg/m\u00b3)')
    for it in range(n_types):
        for il in range(n_levels):
            ax.text(il, it, f"{rms_table[it,il]:.1f}",
                    ha='center', va='center', fontsize=10, color='black')
    fig2.colorbar(im, ax=ax, label='kg/m\u00b3')
    fig2.tight_layout()
    fig2.savefig('plots/Noise_block_heatmap.png', dpi=300)
    plt.close(fig2)
    print("  Saved plots/Noise_block_heatmap.png")

    print("\nDone.")


if __name__ == '__main__':
    run()
