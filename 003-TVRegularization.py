"""
003-TVRegularization.py
=======================
Compare three inversion strategies on the staircase-block model:

  1. INR with positional encoding (data-misfit only, implicit regularization)
  2. Classical CG with Tikhonov smallness + smoothness (L2 + smooth)
  3. Classical CG with Tikhonov smallness + Total Variation (L2 + TV)

TV regularization is the natural choice for blocky targets because it
penalizes the L1 norm of the model gradient, promoting piecewise-constant
solutions without smearing sharp edges (Rudin, Osher & Fatemi, 1992).

The TV term is non-smooth, so we use an IRLS (Iteratively Reweighted
Least Squares) approach: at each outer iteration the TV penalty is
approximated by a weighted L2 norm, whose weights depend on the current
model gradients, and the resulting quadratic sub-problem is solved by CG.

Outputs
-------
  plots/TV_comparison.png         – side-by-side model slices
  plots/TV_comparison_gravity.png – observed / predicted / residual maps
  plots/TV_comparison_metrics.txt – numeric summary
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

# Grid / domain
DX, DY, DZ = 50.0, 50.0, 50.0
X_MAX, Y_MAX, Z_MAX = 1000.0, 1000.0, 500.0

# Block model
RHO_BG, RHO_BLK = 0.0, 400.0

# Noise
NOISE_LEVEL = 0.01

# INR training
INR_EPOCHS = 500
INR_LR     = 1e-2
INR_HIDDEN = 256
INR_DEPTH  = 4
INR_NFREQS = 2
RHO_ABS_MAX = 600.0
GAMMA = 1.0

# Classical CG inversion
CG_MAX_ITER = 800
CG_TOL      = 5e-5

# L2 + Smoothness regularization
SMOOTH_CFG = dict(z0=50.0, beta=1.5, alpha_s=1e-2, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0)

# L2 + TV regularization  (IRLS)
TV_CFG = dict(z0=50.0, beta=1.5, alpha_s=1e-2, alpha_tv=0.5,
              irls_iters=15, irls_eps=1e-3)

# Plotting
CMAP = 'turbo'
VMAX = 250


# ═══════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        ys, ye = max(0, 11 - i), min(Ny, 16 - i)
        xs, xe = 7, min(Nx, 13)
        if 0 <= z_idx < Nz:
            m[xs:xe, ys:ye, z_idx] = rho_blk
    return m.view(-1), m


def get_block_boundaries(Nx, Ny, Nz):
    boundaries = []
    for i in range(7):
        z_idx = 1 + i
        ys, ye = max(0, 11 - i), min(Ny, 16 - i)
        xs, xe = 7, min(Nx, 13)
        if 0 <= z_idx < Nz:
            boundaries.append((xs, xe, ys, ye, z_idx))
    return boundaries


# ═══════════════════════════════════════════════════════════════════════
#  INR MODEL
# ═══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=2, include_input=True, input_dim=3):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))
        self.out_dim = input_dim * (1 + 2 * num_freqs) if include_input else input_dim * 2 * num_freqs

    def forward(self, x):
        parts = [x] if self.include_input else []
        for f in self.freqs:
            parts += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(parts, dim=-1)


class DensityContrastINR(nn.Module):
    def __init__(self, num_freqs=2, hidden=256, depth=4, rho_abs_max=600.0):
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
#  CLASSICAL INVERSION INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════

def idx_flat(i, j, k, Ny, Nz):
    return i * Ny * Nz + j * Nz + k


def build_grad_ops_sparse(Nx, Ny, Nz, dx, dy, dz, device):
    Ncells = Nx * Ny * Nz
    ops = []
    for dim, (Nd, dd, shift) in enumerate([
        (Nx, dx, lambda i, j, k: (i+1, j, k)),
        (Ny, dy, lambda i, j, k: (i, j+1, k)),
        (Nz, dz, lambda i, j, k: (i, j, k+1)),
    ]):
        rows, cols, vals = [], [], []
        sizes = [Nx, Ny, Nz]
        row = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    si, sj, sk = shift(i, j, k)
                    if si < Nx and sj < Ny and sk < Nz:
                        c0 = idx_flat(i, j, k, Ny, Nz)
                        c1 = idx_flat(si, sj, sk, Ny, Nz)
                        rows += [row, row]
                        cols += [c1, c0]
                        vals += [1.0/dd, -1.0/dd]
                        row += 1
        n_edges = row
        D = torch.sparse_coo_tensor(
            indices=torch.tensor([rows, cols], dtype=torch.long),
            values=torch.tensor(vals, dtype=torch.float32),
            size=(n_edges, Ncells), device=device
        ).coalesce()
        ops.append(D)
    return ops  # Dx, Dy, Dz


def depth_weights(grid_coords_t, z0, beta, normalize=True):
    z = grid_coords_t[:, 2]
    w = 1.0 / torch.pow(z + z0, beta)
    if normalize:
        w = w / (w.mean() + 1e-12)
    return w


def cg_solve(matvec, b, x0=None, max_iter=500, tol=1e-6, verbose=True):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    bnorm = torch.sqrt(torch.dot(b, b) + 1e-30)
    hist = [float(torch.sqrt(rs_old) / bnorm)]
    for it in range(1, max_iter + 1):
        Ap = matvec(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        rel_res = float(torch.sqrt(rs_new) / bnorm)
        hist.append(rel_res)
        if verbose and (it % 50 == 0 or rel_res < tol):
            print(f"    CG iter {it:4d}: rel_res = {rel_res:.3e}")
        if rel_res < tol:
            break
        p = r + (rs_new / (rs_old + 1e-30)) * p
        rs_old = rs_new
    return x, hist


# ═══════════════════════════════════════════════════════════════════════
#  INVERSION RUNNERS
# ═══════════════════════════════════════════════════════════════════════

def run_inr_inversion(coords_norm, G, gz_obs, sigma, device):
    """Train INR and return inverted model vector."""
    Wd = 1.0 / sigma
    model = DensityContrastINR(
        num_freqs=INR_NFREQS, hidden=INR_HIDDEN,
        depth=INR_DEPTH, rho_abs_max=RHO_ABS_MAX
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=INR_LR)
    for ep in range(INR_EPOCHS):
        opt.zero_grad()
        m_pred = model(coords_norm).view(-1)
        gz_pred = (G @ m_pred.unsqueeze(1)).squeeze(1)
        loss = GAMMA * torch.mean((Wd * (gz_pred - gz_obs)) ** 2)
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep == INR_EPOCHS - 1:
            print(f"    INR epoch {ep:4d}: loss = {loss.item():.3e}")

    with torch.no_grad():
        m_inv = model(coords_norm).view(-1)
        gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)
    return m_inv, gz_pred


def run_smooth_inversion(G, gz_obs, sigma, grid_coords, Nx, Ny, Nz,
                         dx, dy, dz, device):
    """Classical CG with Tikhonov smallness + L2-smoothness."""
    Wd = 1.0 / sigma
    Gw = Wd * G
    dw = Wd * gz_obs
    GT = Gw.T.contiguous()
    b = GT @ dw

    grid_t = torch.tensor(grid_coords, dtype=torch.float32, device=device)
    w2 = depth_weights(grid_t, SMOOTH_CFG['z0'], SMOOTH_CFG['beta']) ** 2
    Dx, Dy, Dz = build_grad_ops_sparse(Nx, Ny, Nz, dx, dy, dz, device)

    def matvec(m):
        out = GT @ (Gw @ m)
        out += SMOOTH_CFG['alpha_s']**2 * (w2 * m)
        for D, alpha in [(Dx, SMOOTH_CFG['alpha_x']),
                         (Dy, SMOOTH_CFG['alpha_y']),
                         (Dz, SMOOTH_CFG['alpha_z'])]:
            out += alpha**2 * torch.sparse.mm(D.T, torch.sparse.mm(D, m.unsqueeze(1))).squeeze(1)
        return out

    m_inv, _ = cg_solve(matvec, b, max_iter=CG_MAX_ITER, tol=CG_TOL, verbose=True)
    with torch.no_grad():
        gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)
    return m_inv, gz_pred


def run_tv_inversion(G, gz_obs, sigma, grid_coords, Nx, Ny, Nz,
                     dx, dy, dz, device):
    """Classical CG with Tikhonov smallness + Total Variation (IRLS)."""
    Wd = 1.0 / sigma
    Gw = Wd * G
    dw = Wd * gz_obs
    GT = Gw.T.contiguous()
    b = GT @ dw

    grid_t = torch.tensor(grid_coords, dtype=torch.float32, device=device)
    w2 = depth_weights(grid_t, TV_CFG['z0'], TV_CFG['beta']) ** 2
    Dx, Dy, Dz = build_grad_ops_sparse(Nx, Ny, Nz, dx, dy, dz, device)
    Ncells = Nx * Ny * Nz

    eps_tv = TV_CFG['irls_eps']
    alpha_tv = TV_CFG['alpha_tv']
    alpha_s = TV_CFG['alpha_s']

    m_current = torch.zeros(Ncells, dtype=torch.float32, device=device)

    for outer in range(TV_CFG['irls_iters']):
        # Compute IRLS weights from current model gradients
        with torch.no_grad():
            gx = torch.sparse.mm(Dx, m_current.unsqueeze(1)).squeeze(1)
            gy = torch.sparse.mm(Dy, m_current.unsqueeze(1)).squeeze(1)
            gz_grad = torch.sparse.mm(Dz, m_current.unsqueeze(1)).squeeze(1)

        # Per-edge weight: 1 / sqrt(grad^2 + eps^2) for each direction
        wx = 1.0 / torch.sqrt(gx**2 + eps_tv**2)
        wy = 1.0 / torch.sqrt(gy**2 + eps_tv**2)
        wz = 1.0 / torch.sqrt(gz_grad**2 + eps_tv**2)

        # Build sparse diagonal weight matrices
        Wx = torch.sparse_coo_tensor(
            indices=torch.stack([torch.arange(wx.shape[0], device=device)] * 2),
            values=wx, size=(wx.shape[0], wx.shape[0])
        ).coalesce()
        Wy = torch.sparse_coo_tensor(
            indices=torch.stack([torch.arange(wy.shape[0], device=device)] * 2),
            values=wy, size=(wy.shape[0], wy.shape[0])
        ).coalesce()
        Wz = torch.sparse_coo_tensor(
            indices=torch.stack([torch.arange(wz.shape[0], device=device)] * 2),
            values=wz, size=(wz.shape[0], wz.shape[0])
        ).coalesce()

        def matvec(m):
            out = GT @ (Gw @ m)
            out += alpha_s**2 * (w2 * m)
            out += alpha_tv**2 * torch.sparse.mm(
                Dx.T, torch.sparse.mm(Wx, torch.sparse.mm(Dx, m.unsqueeze(1)))
            ).squeeze(1)
            out += alpha_tv**2 * torch.sparse.mm(
                Dy.T, torch.sparse.mm(Wy, torch.sparse.mm(Dy, m.unsqueeze(1)))
            ).squeeze(1)
            out += alpha_tv**2 * torch.sparse.mm(
                Dz.T, torch.sparse.mm(Wz, torch.sparse.mm(Dz, m.unsqueeze(1)))
            ).squeeze(1)
            return out

        m_current, _ = cg_solve(matvec, b, x0=m_current,
                                max_iter=CG_MAX_ITER, tol=CG_TOL, verbose=False)
        with torch.no_grad():
            gz_pred_tmp = (G @ m_current.unsqueeze(1)).squeeze(1)
            rms_tmp = float(torch.sqrt(torch.mean((gz_pred_tmp - gz_obs)**2)).item()) * 1e5
        print(f"    IRLS outer {outer+1:2d}/{TV_CFG['irls_iters']}  "
              f"RMS = {rms_tmp:.4f} mGal")

    with torch.no_grad():
        gz_pred = (G @ m_current.unsqueeze(1)).squeeze(1)
    return m_current, gz_pred


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison(results, tru_3d, gz_obs, obs, x, y, z, dx, dy, dz,
                    block_boundaries, Nx, Ny, Nz):
    """Side-by-side model slices: True | INR | L2+Smooth | L2+TV."""

    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)
    x_e = [x[0] - dx/2, x[-1] + dx/2]
    y_e = [y[0] - dy/2, y[-1] + dy/2]
    z_e = [z[0] - dz/2, z[-1] + dz/2]
    ext_xy = [x_e[0], x_e[1], y_e[0], y_e[1]]
    ext_xz = [x_e[0], x_e[1], z_e[1], z_e[0]]
    ext_yz = [y_e[0], y_e[1], z_e[1], z_e[0]]

    tru = tru_3d.cpu().numpy() if hasattr(tru_3d, 'cpu') else tru_3d
    labels = ['True', 'INR (PE)', 'L2 + Smooth', 'L2 + TV']
    models = [tru]
    for key in ['inr', 'smooth', 'tv']:
        m = results[key]['model'].detach().cpu().numpy().reshape(Nx, Ny, Nz)
        models.append(m)

    boundary_for_z = next((b for b in block_boundaries if b[4] == iz), None)

    # --- Model slices (4 rows x 3 columns) ---
    fig, axes = plt.subplots(4, 3, figsize=(17, 22))
    for row, (label, m3d) in enumerate(zip(labels, models)):
        vmax_row = VMAX if row > 0 else float(np.max(tru))

        im = axes[row, 0].imshow(m3d[:, :, iz].T, origin='lower', extent=ext_xy,
                                 aspect='equal', vmin=0, vmax=vmax_row, cmap=CMAP)
        axes[row, 0].set_title(f"{label}  XY @ z={z[iz]:.0f} m")
        fig.colorbar(im, ax=axes[row, 0], label='kg/m\u00b3', fraction=0.046, pad=0.04)

        im = axes[row, 1].imshow(m3d[:, iy, :].T, origin='upper', extent=ext_xz,
                                 aspect=1.0, vmin=0, vmax=vmax_row, cmap=CMAP)
        axes[row, 1].set_title(f"{label}  XZ @ y={y[iy]:.0f} m")

        im = axes[row, 2].imshow(m3d[ix, :, :].T, origin='upper', extent=ext_yz,
                                 aspect=1.0, vmin=0, vmax=vmax_row, cmap=CMAP)
        axes[row, 2].set_title(f"{label}  YZ @ x={x[ix]:.0f} m")

        # Overlay block boundary on inverted models
        if row > 0 and boundary_for_z:
            xs, xe, ys, ye, _ = boundary_for_z
            rect = plt.Rectangle((x[xs] - dx/2, y[ys] - dy/2),
                                 (xe - xs) * dx, (ye - ys) * dy,
                                 ec='white', fc='none', lw=2)
            axes[row, 0].add_patch(rect)

        axes[row, 0].set_xlabel('x (m)'); axes[row, 0].set_ylabel('y (m)')
        axes[row, 1].set_xlabel('x (m)'); axes[row, 1].set_ylabel('Depth (m)')
        axes[row, 2].set_xlabel('y (m)'); axes[row, 2].set_ylabel('Depth (m)')

    fig.tight_layout()
    fig.savefig('plots/TV_comparison.png', dpi=300)
    plt.close(fig)
    print("  Saved plots/TV_comparison.png")

    # --- Gravity: Obs / Pred / Residual for each method ---
    obs_x = obs[:, 0].cpu().numpy()
    obs_y = obs[:, 1].cpu().numpy()
    obs_mgal = 1e5 * gz_obs.detach().cpu().numpy()

    fig2, axes2 = plt.subplots(3, 3, figsize=(17, 15))
    method_keys = ['inr', 'smooth', 'tv']
    method_labels = ['INR (PE)', 'L2 + Smooth', 'L2 + TV']
    for col, (key, mlabel) in enumerate(zip(method_keys, method_labels)):
        pre_mgal = 1e5 * results[key]['gz_pred'].detach().cpu().numpy()
        res_mgal = obs_mgal - pre_mgal
        v = max(abs(obs_mgal).max(), abs(pre_mgal).max())
        rms_res = np.sqrt(np.mean(res_mgal**2))

        sc = axes2[0, col].scatter(obs_x, obs_y, c=obs_mgal, s=60, cmap=CMAP,
                                   vmin=-v, vmax=v, edgecolors='none')
        axes2[0, col].set_title(f'Observed gz (mGal)')
        fig2.colorbar(sc, ax=axes2[0, col], fraction=0.046, pad=0.04)

        sc = axes2[1, col].scatter(obs_x, obs_y, c=pre_mgal, s=60, cmap=CMAP,
                                   vmin=-v, vmax=v, edgecolors='none')
        axes2[1, col].set_title(f'{mlabel} predicted')
        fig2.colorbar(sc, ax=axes2[1, col], fraction=0.046, pad=0.04)

        vr = np.abs(res_mgal).max()
        sc = axes2[2, col].scatter(obs_x, obs_y, c=res_mgal, s=60, cmap=CMAP,
                                   vmin=-vr, vmax=vr, edgecolors='none')
        axes2[2, col].set_title(f'{mlabel} residual (RMS={rms_res:.3f})')
        fig2.colorbar(sc, ax=axes2[2, col], fraction=0.046, pad=0.04)

    for ax in axes2.flat:
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal')

    fig2.tight_layout()
    fig2.savefig('plots/TV_comparison_gravity.png', dpi=300)
    plt.close(fig2)
    print("  Saved plots/TV_comparison_gravity.png")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def run():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    # --- Grid ---
    dx, dy, dz = DX, DY, DZ
    x = np.arange(0.0, X_MAX + dx, dx)
    y = np.arange(0.0, Y_MAX + dy, dy)
    z = np.arange(0.0, Z_MAX + dz, dz)
    Nx, Ny, Nz = len(x), len(y), len(z)

    X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)

    c_mean = grid_coords.mean(axis=0, keepdims=True)
    c_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = torch.tensor(
        (grid_coords - c_mean) / (c_std + 1e-12),
        dtype=torch.float32, device=device, requires_grad=True)

    dz_half = dz / 2.0
    cell_grid = torch.tensor(
        np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz_half)]),
        dtype=torch.float32, device=device)

    XX, YY = np.meshgrid(x, y, indexing='ij')
    obs = torch.tensor(
        np.column_stack([XX.ravel(), YY.ravel(), -np.ones(XX.size)]),
        dtype=torch.float32, device=device)

    # --- Sensitivity matrix ---
    print("Assembling sensitivity G ...")
    t0 = time.time()
    G = construct_sensitivity_matrix_G(cell_grid, obs, dx, dy, device)
    G = G.clone().detach().requires_grad_(False)
    print(f"  G shape = {tuple(G.shape)}, time = {time.time()-t0:.2f}s")

    # --- True model & data ---
    rho_true_vec, rho_true_3d = make_block_model(Nx, Ny, Nz, rho_bg=RHO_BG, rho_blk=RHO_BLK)
    rho_true_vec = rho_true_vec.to(device)
    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)
    sigma = NOISE_LEVEL * gz_true.std()
    gz_obs = gz_true + sigma * torch.randn_like(gz_true)
    block_boundaries = get_block_boundaries(Nx, Ny, Nz)

    results = {}

    # --- 1) INR inversion ---
    print("\n" + "=" * 60)
    print("  1) INR with positional encoding")
    print("=" * 60)
    set_seed(SEED)
    m_inr, gz_inr = run_inr_inversion(coords_norm, G, gz_obs, sigma, device)
    results['inr'] = {'model': m_inr, 'gz_pred': gz_inr}

    # --- 2) L2 + Smoothness ---
    print("\n" + "=" * 60)
    print("  2) Classical CG: L2 + Smoothness")
    print("=" * 60)
    m_sm, gz_sm = run_smooth_inversion(G, gz_obs, sigma, grid_coords,
                                       Nx, Ny, Nz, dx, dy, dz, device)
    results['smooth'] = {'model': m_sm, 'gz_pred': gz_sm}

    # --- 3) L2 + TV (IRLS) ---
    print("\n" + "=" * 60)
    print("  3) Classical CG: L2 + Total Variation (IRLS)")
    print("=" * 60)
    m_tv, gz_tv = run_tv_inversion(G, gz_obs, sigma, grid_coords,
                                   Nx, Ny, Nz, dx, dy, dz, device)
    results['tv'] = {'model': m_tv, 'gz_pred': gz_tv}

    # --- Metrics ---
    print("\n" + "-" * 60)
    print("  Summary")
    print("-" * 60)
    rho_true_np = rho_true_vec.cpu().numpy()
    lines = []
    for key, label in [('inr', 'INR (PE)'), ('smooth', 'L2+Smooth'), ('tv', 'L2+TV')]:
        m_np = results[key]['model'].detach().cpu().numpy()
        gz_p = results[key]['gz_pred']
        rms_rho = np.sqrt(np.mean((m_np - rho_true_np)**2))
        rms_gz = float(torch.sqrt(torch.mean((gz_p - gz_obs)**2)).item()) * 1e5
        line = f"  {label:14s}  |  RMS rho = {rms_rho:7.2f} kg/m3  |  RMS gz = {rms_gz:.4f} mGal"
        print(line)
        lines.append(line)

    with open('plots/TV_comparison_metrics.txt', 'w') as f:
        f.write('\n'.join(lines))

    # --- Plots ---
    print("\nGenerating plots ...")
    plot_comparison(results, rho_true_3d, gz_obs, obs, x, y, z,
                    dx, dy, dz, block_boundaries, Nx, Ny, Nz)
    print("Done.")


if __name__ == '__main__':
    run()
