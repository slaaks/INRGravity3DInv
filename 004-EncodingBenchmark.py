"""
004-EncodingBenchmark.py
========================
Head-to-head comparison of spatial encoding strategies for INR-based
3-D gravity inversion on the staircase-block model.

Encodings tested
----------------
  1. None (plain MLP)          -- baseline showing spectral bias
  2. Positional (L=2)          -- sinusoidal Fourier features (NeRF)
  3. Gaussian random Fourier   -- Tancik et al. (2020)
  4. Multi-resolution hash     -- Muller et al. (2022) / Instant-NGP

All encodings use the same MLP backbone (width=256, depth=4) and the
same noisy data.  The script trains each, records convergence curves,
final metrics, and produces a comparison figure.

Outputs
-------
  plots/Encoding_comparison.png         -- model slices per encoding
  plots/Encoding_convergence.png        -- loss curves overlay
  plots/Encoding_comparison_metrics.txt -- numeric summary table
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
NOISE_LEVEL = 0.01

GAMMA  = 1.0
EPOCHS = 500
LR     = 1e-2
HIDDEN = 256
DEPTH  = 4
RHO_ABS_MAX = 600.0

CMAP = 'turbo'
INV_VMAX = 250

# Encoding-specific configs
ENCODING_LIST = [
    ('none',        {}),
    ('positional',  dict(num_freqs=2)),
    ('hash',        dict(n_levels=2, n_features_per_level=2,
                         log2_hashmap_size=17, base_resolution=4,
                         finest_resolution=128)),
]


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES  (same as 001/002)
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


def get_block_boundaries(Nx, Ny, Nz):
    boundaries = []
    for i in range(7):
        z_idx = 1 + i
        ys, ye = max(0, 11-i), min(Ny, 16-i)
        xs, xe = 7, min(Nx, 13)
        if 0 <= z_idx < Nz:
            boundaries.append((xs, xe, ys, ye, z_idx))
    return boundaries


# ═══════════════════════════════════════════════════════════════════════
#  ENCODING MODULES
# ═══════════════════════════════════════════════════════════════════════

class IdentityEncoding(nn.Module):
    """No encoding -- raw (x,y,z) passed to MLP."""
    def __init__(self, input_dim=3):
        super().__init__()
        self.out_dim = input_dim
    def forward(self, x):
        return x


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



class HashEncoding(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16,
                 finest_resolution=512, input_dim=3):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.input_dim = input_dim
        self.out_dim = n_levels * n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.growth_factor = (np.exp((np.log(finest_resolution) - np.log(base_resolution))
                                     / max(n_levels - 1, 1)) if n_levels > 1 else 1.0)
        self.base_resolution = base_resolution
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)])
        for t in self.hash_tables:
            nn.init.uniform_(t.weight, -1e-4, 1e-4)
        self.register_buffer('primes',
                             torch.tensor([1, 2654435761, 805459861], dtype=torch.long))

    def _hash(self, coords_int):
        result = torch.zeros(coords_int.shape[:-1], dtype=torch.long,
                             device=coords_int.device)
        for d in range(self.input_dim):
            result ^= coords_int[..., d] * self.primes[d]
        return result % self.hashmap_size

    def forward(self, x):
        x_min = x.min(0, keepdim=True).values
        x_max = x.max(0, keepdim=True).values
        x_s = (x - x_min) / (x_max - x_min + 1e-8)
        outputs = []
        for lv in range(self.n_levels):
            res = int(self.base_resolution * (self.growth_factor ** lv))
            xg = x_s * res
            xf = torch.floor(xg).long()
            xfr = xg - xf.float()
            corners = torch.stack([xf + torch.tensor([dx, dy, dz], device=x.device)
                                   for dz in (0,1) for dy in (0,1) for dx in (0,1)], dim=1)
            idx = self._hash(corners)
            feat = self.hash_tables[lv](idx)
            wx, wy, wz = xfr[:, 0:1], xfr[:, 1:2], xfr[:, 2:3]
            w = torch.stack([
                (1-wx)*(1-wy)*(1-wz), wx*(1-wy)*(1-wz),
                (1-wx)*wy*(1-wz),     wx*wy*(1-wz),
                (1-wx)*(1-wy)*wz,     wx*(1-wy)*wz,
                (1-wx)*wy*wz,         wx*wy*wz], dim=1)
            outputs.append((w * feat).sum(dim=1))
        return torch.cat(outputs, dim=-1)


def create_encoding(enc_type, **kwargs):
    if enc_type == 'none':
        return IdentityEncoding()
    elif enc_type == 'positional':
        return PositionalEncoding(**kwargs)
    elif enc_type == 'hash':
        return HashEncoding(**kwargs)
    else:
        raise ValueError(f"Unknown encoding: {enc_type}")


class DensityContrastINR(nn.Module):
    def __init__(self, encoding_type='positional', hidden=256, depth=4,
                 rho_abs_max=600.0, **enc_kwargs):
        super().__init__()
        self.pe = create_encoding(encoding_type, **enc_kwargs)
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

def train_one(enc_name, enc_kwargs, coords_norm, G, gz_true, sigma, device):
    # Re-seed so that the RNG sequence (noise draw -> weight init) is
    # identical to 001-EncodingComparisons.py for every encoding.
    set_seed(SEED)
    gz_obs = gz_true + sigma * torch.randn_like(gz_true)
    Wd = 1.0 / sigma
    model = DensityContrastINR(
        encoding_type=enc_name, hidden=HIDDEN, depth=DEPTH,
        rho_abs_max=RHO_ABS_MAX, **enc_kwargs
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())

    history = []
    t0 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        m_pred = model(coords_norm).view(-1)
        gz_pred = (G @ m_pred.unsqueeze(1)).squeeze(1)
        loss = GAMMA * torch.mean((Wd * (gz_pred - gz_obs)) ** 2)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if ep % 100 == 0 or ep == EPOCHS - 1:
            print(f"    [{enc_name:12s}] ep {ep:4d}  loss = {loss.item():.3e}")
    elapsed = time.time() - t0

    with torch.no_grad():
        m_inv = model(coords_norm).view(-1)
        gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)

    return m_inv, gz_pred, gz_obs, history, n_params, elapsed


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_results(results, tru_3d, x, y, z, dx, dy, dz,
                 block_boundaries, Nx, Ny, Nz):
    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)
    ext_xy = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]
    ext_xz = [x[0]-dx/2, x[-1]+dx/2, z[-1]+dz/2, z[0]-dz/2]
    ext_yz = [y[0]-dy/2, y[-1]+dy/2, z[-1]+dz/2, z[0]-dz/2]

    tru = tru_3d.cpu().numpy() if hasattr(tru_3d, 'cpu') else tru_3d
    n_enc = len(results)
    nrows = 1 + n_enc  # true + each encoding
    tru_max = float(np.max(tru))

    fig, axes = plt.subplots(nrows, 3, figsize=(16, 5 * nrows))

    # ---------- Row 0 : true model ----------
    for c, (sl, ext, title) in enumerate([
        (tru[:, :, iz].T, ext_xy, f"True \u0394\u03c1 XY @ z\u2248{z[iz]:.0f} m"),
        (tru[:, iy, :].T, ext_xz, f"True \u0394\u03c1 XZ @ y\u2248{y[iy]:.0f} m"),
        (tru[ix, :, :].T, ext_yz, f"True \u0394\u03c1 YZ @ x\u2248{x[ix]:.0f} m"),
    ]):
        origin = 'lower' if c == 0 else 'upper'
        im = axes[0, c].imshow(sl, origin=origin, extent=ext,
                               aspect='auto', vmin=0, vmax=tru_max, cmap=CMAP)
        axes[0, c].set_title(title)
        axes[0, c].set_aspect(1.0)
    fig.colorbar(im, ax=axes[0, 0], label='kg/m\u00b3', fraction=0.046, pad=0.04)

    # ---------- Rows 1..n : each encoding ----------
    boundary_for_z = next((b for b in block_boundaries if b[4] == iz), None)
    for row, (enc_name, info) in enumerate(results.items(), start=1):
        m3d = info['model'].detach().cpu().numpy().reshape(Nx, Ny, Nz)
        rms_rho = info['rms_rho']
        slices = [
            (m3d[:, :, iz].T, ext_xy),
            (m3d[:, iy, :].T, ext_xz),
            (m3d[ix, :, :].T, ext_yz),
        ]
        for c, (sl, ext) in enumerate(slices):
            origin = 'lower' if c == 0 else 'upper'
            im = axes[row, c].imshow(sl, origin=origin, extent=ext,
                                     aspect='auto', vmin=0, vmax=INV_VMAX,
                                     cmap=CMAP)
            axes[row, c].set_aspect(1.0)

        # titles
        axes[row, 0].set_title(
            f"{enc_name} \u0394\u03c1 XY @ z\u2248{z[iz]:.0f} m"
            f"  (RMS={rms_rho:.1f} kg/m\u00b3, params={info['n_params']//1000}k)")
        axes[row, 1].set_title(
            f"{enc_name} \u0394\u03c1 XZ @ y\u2248{y[iy]:.0f} m")
        axes[row, 2].set_title(
            f"{enc_name} \u0394\u03c1 YZ @ x\u2248{x[ix]:.0f} m")

        # per-row colorbar on the first panel
        fig.colorbar(im, ax=axes[row, 0], label='kg/m\u00b3',
                     fraction=0.046, pad=0.04)

        # block boundary overlay -- XY slice
        if boundary_for_z:
            xs, xe, ys, ye, _ = boundary_for_z
            rect = plt.Rectangle((x[xs]-dx/2, y[ys]-dy/2),
                                 (xe-xs)*dx, (ye-ys)*dy,
                                 edgecolor='white', facecolor='none', linewidth=2)
            axes[row, 0].add_patch(rect)

        # block boundary overlay -- XZ slice
        z_indices_in_slice = []
        x_range = None
        for b in block_boundaries:
            bxs, bxe, bys, bye, z_idx = b
            if bys <= iy < bye:
                z_indices_in_slice.append(z_idx)
                if x_range is None:
                    x_range = (bxs, bxe)
        if z_indices_in_slice and x_range:
            zmin_idx, zmax_idx = min(z_indices_in_slice), max(z_indices_in_slice)
            bxs, bxe = x_range
            rect = plt.Rectangle((x[bxs]-dx/2, z[zmin_idx]-dz/2),
                                 (bxe-bxs)*dx, (zmax_idx-zmin_idx+1)*dz,
                                 edgecolor='white', facecolor='none', linewidth=2)
            axes[row, 1].add_patch(rect)

        # block boundary overlay -- YZ slice
        for bxs, bxe, bys, bye, z_idx in block_boundaries:
            if bxs <= ix < bxe:
                rect = plt.Rectangle((y[bys]-dy/2, z[z_idx]-dz/2),
                                     (bye-bys)*dy, dz,
                                     edgecolor='white', facecolor='none', linewidth=2)
                axes[row, 2].add_patch(rect)

    for r in range(nrows):
        axes[r, 0].set_xlabel('x (m)'); axes[r, 0].set_ylabel('y (m)')
        axes[r, 1].set_xlabel('x (m)'); axes[r, 1].set_ylabel('Depth (m)')
        axes[r, 2].set_xlabel('y (m)'); axes[r, 2].set_ylabel('Depth (m)')

    fig.tight_layout()
    fig.savefig('plots/Encoding_comparison.png', dpi=300)
    plt.close(fig)
    print("  Saved plots/Encoding_comparison.png")

    # --- Convergence curves ---
    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, (enc_name, info) in enumerate(results.items()):
        ax.plot(info['history'], label=f"{enc_name} ({info['elapsed']:.1f}s)",
                color=colors[i % len(colors)], linewidth=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence by encoding strategy')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig('plots/Encoding_convergence.png', dpi=300)
    plt.close(fig2)
    print("  Saved plots/Encoding_convergence.png")


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

    rho_true_vec, rho_true_3d = make_block_model(Nx, Ny, Nz, rho_bg=RHO_BG, rho_blk=RHO_BLK)
    rho_true_vec = rho_true_vec.to(device)
    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)
    sigma = NOISE_LEVEL * gz_true.std()
    block_boundaries = get_block_boundaries(Nx, Ny, Nz)

    rho_true_np = rho_true_vec.cpu().numpy()
    results = {}

    for enc_name, enc_kwargs in ENCODING_LIST:
        print(f"\n{'='*60}")
        print(f"  Encoding: {enc_name}")
        print(f"{'='*60}")
        m_inv, gz_pred, gz_obs, hist, n_params, elapsed = train_one(
            enc_name, enc_kwargs, coords_norm, G, gz_true, sigma, device)

        m_np = m_inv.detach().cpu().numpy()
        rms_rho = np.sqrt(np.mean((m_np - rho_true_np)**2))
        rms_gz = float(torch.sqrt(torch.mean((gz_pred - gz_obs)**2)).item()) * 1e5

        results[enc_name] = dict(
            model=m_inv, gz_pred=gz_pred, history=hist,
            n_params=n_params, elapsed=elapsed,
            rms_rho=rms_rho, rms_gz=rms_gz)

        print(f"    Params: {n_params}  |  Time: {elapsed:.1f}s  |  "
              f"RMS rho: {rms_rho:.2f} kg/m3  |  RMS gz: {rms_gz:.4f} mGal")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"  {'Encoding':14s}  {'Params':>8s}  {'Time(s)':>8s}  "
          f"{'RMS rho':>10s}  {'RMS gz':>10s}")
    print(f"{'='*70}")
    lines = []
    for name, info in results.items():
        line = (f"  {name:14s}  {info['n_params']:8d}  {info['elapsed']:8.1f}  "
                f"{info['rms_rho']:10.2f}  {info['rms_gz']:10.4f}")
        print(line)
        lines.append(line)
    with open('plots/Encoding_comparison_metrics.txt', 'w') as f:
        f.write('\n'.join(lines))

    # --- Plots ---
    print("\nGenerating plots ...")
    plot_results(results, rho_true_3d, x, y, z, dx, dy, dz,
                 block_boundaries, Nx, Ny, Nz)
    print("Done.")


if __name__ == '__main__':
    run()
