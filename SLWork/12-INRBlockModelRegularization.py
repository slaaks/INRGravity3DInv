import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed = {seed}")

def A_integral_torch(x, y, z):
    eps = 1e-20
    r = torch.sqrt(x**2 + y**2 + z**2).clamp_min(eps)
    return -(x * torch.log(torch.abs(y + r) + eps) +
             y * torch.log(torch.abs(x + r) + eps) -
             z * torch.atan2(x * y, z * r + eps))

@torch.inference_mode()
def construct_sensitivity_matrix_G_torch(cell_grid, data_points, d1, d2, device):
    Gamma = 6.67430e-11
    cx = cell_grid[:, 0].unsqueeze(0)
    cy = cell_grid[:, 1].unsqueeze(0)
    cz = cell_grid[:, 2].unsqueeze(0)
    czh = cell_grid[:, 3].unsqueeze(0)
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

def generate_grf_torch(nx, ny, nz, dx, dy, dz, lam, nu, sigma, device):
    kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(nz, d=dz, device=device) * 2 * torch.pi
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k2 = Kx**2 + Ky**2 + Kz**2
    P = (k2 + (1/lam**2))**(-nu - 1.5)
    P[0, 0, 0] = 0
    noise = torch.randn(nx, ny, nz, dtype=torch.complex64, device=device)
    f = noise * torch.sqrt(P)
    m = torch.real(torch.fft.ifftn(f))
    m = (m - m.mean()) / (m.std() + 1e-9)
    return sigma * m

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=8, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))

    def forward(self, x):
        parts = [x] if self.include_input else []
        for f in self.freqs:
            parts += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(parts, dim=-1)

class DensityContrastINR(nn.Module):
    def __init__(self, nfreq=8, hidden=256, depth=5, rho_abs_max=600.0):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=nfreq, include_input=True)
        in_dim = 3 * (1 + 2 * nfreq)
        layers = []
        h = hidden
        layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.01)]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.LeakyReLU(0.01)]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)
        self.rho_abs_max = float(rho_abs_max)

    def forward(self, x):
        z = self.pe(x)
        out = self.net(z)
        return self.rho_abs_max * torch.tanh(out)
    
#---- Finite-difference gradient computatation for the regularizers ---
def compute_gradient(m, Nx, Ny, Nz, dx, dy, dz):
    m3 = m.reshape(Nx, Ny, Nz)

    gx = torch.empty_like(m3)
    gy = torch.empty_like(m3)
    gz = torch.empty_like(m3)

    #Central differences (interior)
    gx[1:-1, :, :] = (m3[2:, :, :] - m3[:-2, :, :]) / (2.0 * dx)
    gy[:, 1:-1, :] = (m3[:, 2:, :] - m3[:, :-2, :]) / (2.0 * dy)
    gz[:, :, 1:-1] = (m3[:, :, 2:] - m3[:, :, :-2]) / (2.0 * dz)

    #One-sided boundaries
    gx[0, :, :]  = (m3[1, :, :] - m3[0, :, :]) / dx
    gx[-1, :, :] = (m3[-1, :, :] - m3[-2, :, :]) / dx

    gy[:, 0, :]  = (m3[:, 1, :] - m3[:, 0, :]) / dy
    gy[:, -1, :] = (m3[:, -1, :] - m3[:, -2, :]) / dy

    gz[:, :, 0]  = (m3[:, :, 1] - m3[:, :, 0]) / dz
    gz[:, :, -1] = (m3[:, :, -1] - m3[:, :, -2]) / dz

    return gx, gy, gz

#--- Regularizer functions ---
#0th order Tikhonov
def tik0_loss(m, dx, dy, dz):
    cell_vol = dx * dy * dz
    return cell_vol * torch.mean(m**2)
#1st order Tikhonov
def tik1_loss(m, Nx, Ny, Nz, dx, dy, dz, wx=1.0, wy=1.0, wz=1.0):
    cell_vol = dx * dy * dz
    gx, gy, gz = compute_gradient(m, Nx, Ny, Nz, dx, dy, dz)
    gx, gy, gz = wx * gx, wy * gy, wz * gz
    return cell_vol * torch.mean(gx**2 + gy**2 + gz**2)
#Total variation (TV)
def tv_loss(m, Nx, Ny, Nz, dx, dy, dz, eps=1e-6, wx=1.0, wy=1.0, wz=1.0):
    cell_vol = dx * dy * dz
    gx, gy, gz = compute_gradient(m, Nx, Ny, Nz, dx, dy, dz)
    gx, gy, gz = wx * gx, wy * gy, wz * gz
    tv_vals = torch.sqrt(gx**2 + gy**2 + gz**2 + eps)
    return cell_vol * torch.mean(tv_vals)

def train_inr(model, opt, coords_norm, G, gz_obs, Wd,
              Nx, Ny, Nz, dx, dy, dz, cfg):

    history = {"total": [], "gravity": [], "tik0": [], "tik1": [], "tv": []}

    gamma = float(cfg.get('gamma', 1.0))

    #Regularization weights
    lam0  = float(cfg.get('tik0', 0.0)) #0th order
    lam1  = float(cfg.get('tik1', 0.0)) #1st order
    lam_tv = float(cfg.get('tv', 0.0)) #Total variation

    #Directional anisotrophy
    wx = float(cfg.get('wx', 1.0))
    wy = float(cfg.get('wy', 1.0))
    wz = float(cfg.get('wz', 1.0))
    #TV epsilon for stability
    tv_eps = float(cfg.get('tv_eps', 1e-6))

    device = next(model.parameters()).device

    for ep in range(cfg['epochs']):
        opt.zero_grad()

        m_pred = model(coords_norm).view(-1)

        gz_pred = torch.matmul(G, m_pred.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        data_term = gamma * torch.mean((Wd * residual) ** 2)

        #Regularization
        reg0 = lam0 * tik0_loss(m_pred, dx, dy, dz)
        reg1 = lam1 * tik1_loss(
            m_pred, Nx, Ny, Nz, dx, dy, dz,
            wx=wx, wy=wy, wz=wz
        )
        reg_tv = lam_tv * tv_loss(
            m_pred, Nx, Ny, Nz, dx, dy, dz,
            eps=tv_eps, wx=wx, wy=wy, wz=wz
        )
        loss = data_term + reg0 + reg1 + reg_tv

        loss.backward()
        opt.step()

        #Log
        history['gravity'].append(float(data_term.item()))
        history['total'].append(float(loss.item()))
        history['tik0'].append(float(reg0.item()))
        history['tik1'].append(float(reg1.item()))
        history['tv'].append(float(reg_tv.item()))

        if ep % 50 == 0 or ep == cfg['epochs'] - 1:
            print(
                f"Epoch {ep:4d} | data {history['gravity'][-1]:.3e} | "
                f"tik0 {history['tik0'][-1]:.3e} | tik1 {history['tik1'][-1]:.3e} | tv {history['tv'][-1]:.3e} | "
                f"total {history['total'][-1]:.3e}"
            )  
    return history

def make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=0.0, rho_blk=400.0):
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

def run(): 
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    dx = dy = 50.0
    dz = 50.0
    x = np.arange(0.0, 1000.0 + dx, dx)
    y = np.arange(0.0, 1000.0 + dy, dy)
    z = np.arange(0.0, 500.0 + dz, dz)
    Nx, Ny, Nz = len(x), len(y), len(z)

    Xc = x.astype(float)
    Yc = y.astype(float)
    Zc = z.astype(float)
    X3, Y3, Z3 = np.meshgrid(Xc, Yc, Zc, indexing='ij')
    grid_coords = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)

    c_mean = grid_coords.mean(axis=0, keepdims=True)
    c_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm_np = (grid_coords - c_mean) / (c_std + 1e-12)
    coords_norm = torch.tensor(coords_norm_np, dtype=torch.float32, device=device, requires_grad=False)

    dz_half = dz / 2.0
    cell_grid = np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz_half)])
    cell_grid = torch.tensor(cell_grid, dtype=torch.float32, device=device)

    XX, YY = np.meshgrid(x, y, indexing='ij')
    obs = np.column_stack([XX.ravel(), YY.ravel(), -np.ones(XX.size)])
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    print("Assembling sensitivity G ...")
    t0 = time.time()
    G = construct_sensitivity_matrix_G_torch(cell_grid, obs, dx, dy, device)
    G = G.clone().detach().requires_grad_(False)
    print(f"G shape = {tuple(G.shape)}, time = {time.time() - t0:.2f}s")

    rho_true_vec, rho_true_3d = make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=0.0, rho_blk=400.0)
    rho_true_vec = rho_true_vec.to(device)

    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)

    nl = 0.01
    sigma = nl * gz_true.std()
    noise = sigma * torch.randn_like(gz_true)
    gz_obs = gz_true + noise
    Wd = 1.0 / sigma
    #Regularizer configuration
    configs = {
        "unregularized": dict(gamma=1.0, epochs=300, lr=1e-2,
                              tik0=0.0, tik1=0.0, tv=0.0,
                              wx=1.0, wy=1.0, wz=1.0, tv_eps=1e-6),

        "tik0": dict(gamma=1.0, epochs=300, lr=1e-3,
                     tik0=1e-13, tik1=0.0, tv=0.0,
                     wx=1.0, wy=1.0, wz=1.0, tv_eps=1e-6),

        "tik1": dict(gamma=1.0, epochs=300, lr=1e-2,
                     tik0=0.0, tik1=8e-08, tv=0.0,
                     wx=1.0, wy=1.0, wz=1.0, tv_eps=1e-6),

        "tv":   dict(gamma=1.0, epochs=300, lr=1e-3,
                     tik0=0.0, tik1=0.0, tv=1e-7,
                     wx=1.0, wy=1.0, wz=1.0, tv_eps=1e-6)
    }

    results = {}

    for name, cfg in configs.items():
        print(f"\n--- Running inversion: {name} ---")

        model = DensityContrastINR(
            nfreq=2, hidden=256, depth=4, rho_abs_max=600.0
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

        hist = train_inr(
            model, opt, coords_norm, G, gz_obs, Wd,
            Nx, Ny, Nz, dx, dy, dz, cfg
        )

        with torch.no_grad():
            results[name] = model(coords_norm).view(Nx, Ny, Nz).detach().cpu().numpy()

    #Plot regularizers
    x1d = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
    y1d = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
    z1d = grid_coords[:, 2].reshape(Nx, Ny, Nz)[0, 0, :]

    x_edge_min, x_edge_max = x1d[0] - dx/2, x1d[-1] + dx/2
    y_edge_min, y_edge_max = y1d[0] - dy/2, y1d[-1] + dy/2
    z_edge_min, z_edge_max = z1d[0] - dz/2, z1d[-1] + dz/2

    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]
    extent_xz = [x_edge_min, x_edge_max, z_edge_max, z_edge_min]
    extent_yz = [y_edge_min, y_edge_max, z_edge_max, z_edge_min]

    slice_z = min(Nz - 1, 5)
    slice_y = Ny // 2
    slice_x = Nx // 2

    names = ["true", "unregularized", "tik0", "tik1", "tv"]
    models = [
        rho_true_3d.cpu().numpy(),
        results["unregularized"],
        results["tik0"],
        results["tik1"],
        results["tv"]
    ]

    vmin, vmax = 0, 250
    fig, axes = plt.subplots(3, 5, figsize=(28, 16))

    #XY
    for ax, name, m in zip(axes[0], names, models):
        im = ax.imshow(
            m[:, :, slice_z].T,
            origin="lower",
            extent=extent_xy,
            aspect="auto",
            cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{name} – XY @ z={z1d[slice_z]:.0f} m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    #XZ
    for ax, name, m in zip(axes[1], names, models):
        im = ax.imshow(
            m[:, slice_y, :].T,
            origin="upper",
            extent=extent_xz,
            aspect="auto",
            cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{name} – XZ @ y={y1d[slice_y]:.0f} m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Depth (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    #YZ
    for ax, name, m in zip(axes[2], names, models):
        im = ax.imshow(
            m[slice_x, :, :].T,
            origin="upper",
            extent=extent_yz,
            aspect="auto",
            cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{name} – YZ @ x={x1d[slice_x]:.0f} m")
        ax.set_xlabel("y (m)")
        ax.set_ylabel("Depth (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    #Distortion fix
    for row in axes:
        for ax in row:
            ax.set_aspect(1.0)

    fig.tight_layout()
    fig.savefig("plots/RegularizerComparison.png", dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    run()
