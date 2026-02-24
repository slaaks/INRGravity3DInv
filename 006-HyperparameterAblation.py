"""
007-HyperparameterAblation.py
=============================
Systematic grid sweep over the main INR hyper-parameters to identify
sensitivity and optimal settings.

Sweep axes
----------
  Network width   : 64, 128, 256, 512
  Network depth   : 2, 3, 4, 5
  Encoding freqs  : L = 1, 2, 4, 8

Each (width, depth, L) triple is trained once on the noisy block model.
Results are stored in a 3-D array and presented as heatmaps and a
ranked table.

Outputs
-------
  plots/Ablation_heatmap_width_depth.png  -- heatmap(width, depth) per L
  plots/Ablation_heatmap_L_width.png      -- heatmap(L, width)   per depth
  plots/Ablation_pareto.png               -- RMS vs #params scatter
  plots/Ablation_metrics.txt              -- full table sorted by RMS
"""

import os
import time
import random
import itertools
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
NOISE_LEVEL = 0.01

GAMMA  = 1.0
EPOCHS = 500
LR     = 1e-2
RHO_ABS_MAX = 600.0

WIDTHS = [64, 128, 256]
DEPTHS = [2, 3, 4, 5]
L_FREQS = [1, 2, 4, 8]


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

        # Scale down the last layer so tanh starts in its linear regime,
        # preventing saturation for large hidden widths.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.rho_abs_max * torch.tanh(self.net(self.pe(x)))


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_one(hidden, depth, num_freqs, coords_norm, G, gz_obs, sigma, device):
    set_seed(SEED)
    Wd = 1.0 / sigma
    model = DensityContrastINR(hidden=hidden, depth=depth,
                               rho_abs_max=RHO_ABS_MAX,
                               num_freqs=num_freqs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        m_pred = model(coords_norm).view(-1)
        gz_pred = (G @ m_pred.unsqueeze(1)).squeeze(1)
        loss = GAMMA * torch.mean((Wd * (gz_pred - gz_obs)) ** 2)
        loss.backward()
        opt.step()
    elapsed = time.time() - t0
    final_loss = float(loss.item())

    with torch.no_grad():
        m_inv = model(coords_norm).view(-1)
        gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)

    return m_inv, gz_pred, n_params, elapsed, final_loss


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
    sigma = NOISE_LEVEL * gz_true.std()
    set_seed(SEED)
    gz_obs = gz_true + sigma * torch.randn_like(gz_true)

    # --- sweep ---
    combos = list(itertools.product(WIDTHS, DEPTHS, L_FREQS))
    records = []

    for idx, (w, d, L) in enumerate(combos):
        tag = f"w{w}_d{d}_L{L}"
        print(f"\n[{idx+1}/{len(combos)}] {tag}")
        m_inv, gz_pred, n_params, elapsed, final_loss = train_one(
            w, d, L, coords_norm, G, gz_obs, sigma, device)
        m_np = m_inv.detach().cpu().numpy()
        rms_rho = np.sqrt(np.mean((m_np - rho_true_np)**2))
        rms_gz  = float(torch.sqrt(torch.mean((gz_pred - gz_obs)**2)).item()) * 1e5

        records.append(dict(
            width=w, depth=d, L=L, n_params=n_params,
            rms_rho=rms_rho, rms_gz=rms_gz,
            time=elapsed, loss=final_loss))
        print(f"  params={n_params}  RMS_rho={rms_rho:.2f}  "
              f"RMS_gz={rms_gz:.4f}  time={elapsed:.1f}s")

    # --- metrics table ---
    records_sorted = sorted(records, key=lambda r: r['rms_rho'])
    header = (f"{'Width':>6s}  {'Depth':>5s}  {'L':>4s}  {'Params':>8s}  "
              f"{'RMS rho':>9s}  {'RMS gz':>9s}  {'Time(s)':>8s}")
    lines = [header, '-' * len(header)]
    for r in records_sorted:
        lines.append(f"{r['width']:6d}  {r['depth']:5d}  {r['L']:4d}  "
                     f"{r['n_params']:8d}  {r['rms_rho']:9.2f}  "
                     f"{r['rms_gz']:9.4f}  {r['time']:8.1f}")
    tbl = '\n'.join(lines)
    print(f"\n{tbl}")
    with open('plots/Ablation_metrics.txt', 'w') as f:
        f.write(tbl)

    # --- heatmap: width x depth for each L ---
    fig, axes = plt.subplots(1, len(L_FREQS),
                             figsize=(5*len(L_FREQS), 4), squeeze=False)
    rms_dict = {(r['width'], r['depth'], r['L']): r['rms_rho'] for r in records}
    for j, L_val in enumerate(L_FREQS):
        mat = np.zeros((len(WIDTHS), len(DEPTHS)))
        for iw, w in enumerate(WIDTHS):
            for id_, d in enumerate(DEPTHS):
                mat[iw, id_] = rms_dict.get((w, d, L_val), np.nan)
        im = axes[0, j].imshow(mat, cmap='YlOrRd', aspect='auto',
                               origin='lower')
        axes[0, j].set_xticks(range(len(DEPTHS)))
        axes[0, j].set_xticklabels(DEPTHS)
        axes[0, j].set_yticks(range(len(WIDTHS)))
        axes[0, j].set_yticklabels(WIDTHS)
        axes[0, j].set_xlabel('Depth')
        axes[0, j].set_ylabel('Width')
        axes[0, j].set_title(f'L = {L_val}')
        for iw in range(len(WIDTHS)):
            for id_ in range(len(DEPTHS)):
                axes[0, j].text(id_, iw, f"{mat[iw, id_]:.0f}",
                                ha='center', va='center', fontsize=8)
    fig.suptitle('RMS density error (kg/m\u00b3)  --  Width x Depth', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('plots/Ablation_heatmap_width_depth.png', dpi=300)
    plt.close(fig)
    print("  Saved plots/Ablation_heatmap_width_depth.png")

    # --- heatmap: L x width for each depth ---
    fig2, axes2 = plt.subplots(1, len(DEPTHS),
                               figsize=(5*len(DEPTHS), 4), squeeze=False)
    for j, d_val in enumerate(DEPTHS):
        mat = np.zeros((len(L_FREQS), len(WIDTHS)))
        for il, L_ in enumerate(L_FREQS):
            for iw, w in enumerate(WIDTHS):
                mat[il, iw] = rms_dict.get((w, d_val, L_), np.nan)
        im = axes2[0, j].imshow(mat, cmap='YlOrRd', aspect='auto',
                                origin='lower')
        axes2[0, j].set_xticks(range(len(WIDTHS)))
        axes2[0, j].set_xticklabels(WIDTHS)
        axes2[0, j].set_yticks(range(len(L_FREQS)))
        axes2[0, j].set_yticklabels(L_FREQS)
        axes2[0, j].set_xlabel('Width')
        axes2[0, j].set_ylabel('L (# freqs)')
        axes2[0, j].set_title(f'Depth = {d_val}')
        for il in range(len(L_FREQS)):
            for iw in range(len(WIDTHS)):
                axes2[0, j].text(iw, il, f"{mat[il, iw]:.0f}",
                                 ha='center', va='center', fontsize=8)
    fig2.suptitle('RMS density error (kg/m\u00b3)  --  L x Width', fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig('plots/Ablation_heatmap_L_width.png', dpi=300)
    plt.close(fig2)
    print("  Saved plots/Ablation_heatmap_L_width.png")

    # --- Pareto: RMS vs #params ---
    fig3, ax = plt.subplots(1, 1, figsize=(7, 5))
    for r in records:
        ax.scatter(r['n_params'], r['rms_rho'], s=50,
                   c=r['L'], cmap='viridis', vmin=min(L_FREQS),
                   vmax=max(L_FREQS), edgecolors='k', linewidths=0.3)
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(min(L_FREQS), max(L_FREQS)))
    fig3.colorbar(sm, ax=ax, label='L (encoding freqs)')
    ax.set_xlabel('# parameters')
    ax.set_ylabel('RMS density error (kg/m\u00b3)')
    ax.set_title('Pareto: accuracy vs. model size')
    ax.set_xscale('log')
    ax.grid(True, ls='--', alpha=0.3)
    fig3.tight_layout()
    fig3.savefig('plots/Ablation_pareto.png', dpi=300)
    plt.close(fig3)
    print("  Saved plots/Ablation_pareto.png")

    print("\nDone.")


if __name__ == '__main__':
    run()
