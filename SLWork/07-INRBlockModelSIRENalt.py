#Single-head SIREN

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

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / max(self.omega_0, 1e-8)
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 first_omega_0=30.0, hidden_omega_0=30.0, bias=True):
        super().__init__()
        layers = []
        layers.append(SineLayer(in_features, hidden_features, bias=bias, is_first=True, omega_0=first_omega_0))
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, bias=bias, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)

class DensityContrastSIREN(nn.Module):
    def __init__(self,
                 hidden=256,
                 rho_abs_max=600.0,
                 hidden_layers=3,
                 omega_0_first=30.0,
                 omega_0_hidden=30.0,
                 bias=True):
        super().__init__()
        self.rho_abs_max = float(rho_abs_max)

        self.backbone = Siren(
            in_features=3,
            hidden_features=hidden,
            hidden_layers=hidden_layers,
            first_omega_0=omega_0_first,
            hidden_omega_0=omega_0_hidden,
            bias=bias
        )
        self.head = nn.Linear(hidden, 1, bias=bias)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden) / max(omega_0_hidden, 1e-8)
            self.head.weight.uniform_(-bound, bound)
            if self.head.bias is not None:
                self.head.bias.uniform_(-bound, bound)

    def forward(self, x):
        h = self.backbone(x)
        raw = self.head(h)  #unconstrained
        return raw

def train_inr(model, opt, coords_scaled, G, gz_obs, Wd, Nx, Ny, Nz, dx, dy, dz, cfg, rho_abs_max):
    history = {"total": [], "gravity": []}
    for ep in range(cfg['epochs']):
        opt.zero_grad()
        
        raw_out = model(coords_scaled).view(-1)
        m_pred = (rho_abs_max * torch.tanh(raw_out))

        gz_pred = torch.matmul(G, m_pred.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        data_term = cfg['gamma'] * torch.mean((Wd * residual) ** 2)

        loss = data_term
        loss.backward()
        opt.step()
        history['gravity'].append(float(data_term.item()))
        history['total'].append(float(loss.item()))
        if ep % 50 == 0 or ep == cfg['epochs'] - 1:
            print(f"Epoch {ep:4d} | data {history['gravity'][-1]:.3e} | total {history['total'][-1]:.3e}")
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
    set_seed(42)
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

    ###Normalize to -1, 1 to avoid issues with SIREN layers
    mins = grid_coords.min(axis=0, keepdims=True)
    maxs = grid_coords.max(axis=0, keepdims=True)
    coords_scaled = 2.0 * (grid_coords - mins) / (maxs - mins + 1e-12) - 1.0
    coords_scaled = torch.tensor(coords_scaled, dtype=torch.float32, device=device)

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

    cfg = dict(gamma=1.0, epochs=300,lr=1e-4) #lr=1e-2 test SIREN with a lower lr
    
    rho_abs_max = 600.0
    model = DensityContrastSIREN(
        hidden=256,
        rho_abs_max=rho_abs_max,
        hidden_layers=3,
        omega_0_first=10,
        omega_0_hidden=15,
        bias=True
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=0) #weight_decay=1e-6
    
    hist = train_inr(model, opt, coords_scaled, G, gz_obs, Wd, Nx, Ny, Nz, dx, dy, dz, cfg, rho_abs_max)

    with torch.no_grad():
        raw_out_eval = model(coords_scaled).view(-1)
        m_inv = (rho_abs_max * torch.tanh(raw_out_eval))

        gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)

    def get_axes_coords():
        x1d = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
        y1d = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
        z1d = grid_coords[:, 2].reshape(Nx, Ny, Nz)[0, 0, :]
        return x1d, y1d, z1d

    x1d, y1d, z1d = get_axes_coords()
    block_boundaries = get_block_boundaries(Nx, Ny, Nz)

    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)
    tru = rho_true_3d.cpu().numpy()
    inv = m_inv.view(Nx, Ny, Nz).detach().cpu().numpy()

    tru_max = tru.max()
    inv_max = inv.max()

    fig1, axes = plt.subplots(3, 3, figsize=(16, 15))
    # 1D arrays of cell centers
    x1d, y1d, z1d = get_axes_coords()

    # cell-edge limits
    x_edge_min, x_edge_max = x1d[0] - dx/2, x1d[-1] + dx/2
    y_edge_min, y_edge_max = y1d[0] - dy/2, y1d[-1] + dy/2
    z_edge_min, z_edge_max = z1d[0] - dz/2, z1d[-1] + dz/2

    # use edges for all extents
    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]
    # for depth plots keep depth increasing downward by reversing z limits
    extent_xz = [x_edge_min, x_edge_max, z_edge_max, z_edge_min]
    extent_yz = [y_edge_min, y_edge_max, z_edge_max, z_edge_min]


    im = axes[0, 0].imshow(tru[:, :, iz].T, origin='lower', extent=extent_xy, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 0].set_title(f"True Δρ XY @ z≈{z1d[iz]:.0f} m")
    fig1.colorbar(im, ax=axes[0, 0], label='kg/m³', fraction=0.046, pad=0.04)
    im = axes[0, 1].imshow(tru[:, iy, :].T, origin='upper', extent=extent_xz, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 1].set_title(f"True Δρ XZ @ y≈{y1d[iy]:.0f} m")
    im = axes[0, 2].imshow(tru[ix, :, :].T, origin='upper', extent=extent_yz, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 2].set_title(f"True Δρ YZ @ x≈{x1d[ix]:.0f} m")

    im = axes[1, 0].imshow(inv[:, :, iz].T, origin='lower', extent=extent_xy, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 0].set_title(f"INR Δρ XY @ z≈{z1d[iz]:.0f} m")
    boundary_for_z = next((b for b in block_boundaries if b[4] == iz), None)
    if boundary_for_z:
        xs, xe, ys, ye, _ = boundary_for_z
        rect = plt.Rectangle((x[xs] - dx/2, y[ys] - dy/2),
                             (xe - xs) * dx,
                             (ye - ys) * dy,
                             edgecolor='white', facecolor='none', linewidth=2)
        axes[1, 0].add_patch(rect)
    fig1.colorbar(im, ax=axes[1, 0], label='kg/m³', fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(inv[:, iy, :].T, origin='upper', extent=extent_xz, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 1].set_title(f"INR Δρ XZ @ y≈{y1d[iy]:.0f} m")
    z_indices_in_slice = []
    x_range = None
    for b in block_boundaries:
        xs, xe, ys, ye, z_idx = b
        if ys <= iy < ye:
            z_indices_in_slice.append(z_idx)
            if x_range is None:
                x_range = (xs, xe)
    if z_indices_in_slice and x_range:
        min_z_idx, max_z_idx = min(z_indices_in_slice), max(z_indices_in_slice)
        xs, xe = x_range
        rect = plt.Rectangle((x[xs] - dx/2, z[min_z_idx] - dz/2),
                             (xe - xs) * dx,
                             (max_z_idx - min_z_idx + 1) * dz,
                             edgecolor='white', facecolor='none', linewidth=2)
        axes[1, 1].add_patch(rect)

    im = axes[1, 2].imshow(inv[ix, :, :].T, origin='upper', extent=extent_yz, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 2].set_title(f"INR Δρ YZ @ x≈{x1d[ix]:.0f} m")
    for xs, xe, ys, ye, z_idx in block_boundaries:
        if xs <= ix < xe:
            rect = plt.Rectangle((y[ys] - dy/2, z[z_idx] - dz/2),
                                 (ye - ys) * dy,
                                 dz,
                                 edgecolor='white', facecolor='none', linewidth=2)
            axes[1, 2].add_patch(rect)

    for ax in [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]:
        ax.set_aspect(1.0)
    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_aspect(1.0)

    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    for ax in [axes[0, 1], axes[1, 1]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Depth (m)')
    for ax in [axes[0, 2], axes[1, 2]]:
        ax.set_xlabel('y (m)')
        ax.set_ylabel('Depth (m)')

    def to_mgal(g):
        return 1e5 * g.detach().cpu().numpy()
    obs_mgal = to_mgal(gz_obs)
    pre_mgal = to_mgal(gz_pred)
    res_mgal = obs_mgal - pre_mgal

    obs_x = obs[:, 0].cpu().numpy()
    obs_y = obs[:, 1].cpu().numpy()

    v = max(abs(obs_mgal).max(), abs(pre_mgal).max())

    sc = axes[2, 0].scatter(obs_x, obs_y, c=obs_mgal, s=80, cmap='viridis', vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[2, 0].set_title('Observed gz (mGal)')
    fig1.colorbar(sc, ax=axes[2, 0], fraction=0.046, pad=0.04)

    sc = axes[2, 1].scatter(obs_x, obs_y, c=pre_mgal, s=80, cmap='viridis', vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[2, 1].set_title('Predicted gz (mGal)')
    fig1.colorbar(sc, ax=axes[2, 1], fraction=0.046, pad=0.04)

    vmax_res = np.abs(res_mgal).max()
    sc = axes[2, 2].scatter(obs_x, obs_y, c=res_mgal, s=80, cmap='viridis', vmin=-vmax_res, vmax=vmax_res, marker='o', edgecolors='none')
    axes[2, 2].set_title(f'Residual (RMS={np.sqrt(np.mean(res_mgal**2)):.3f} mGal)')
    fig1.colorbar(sc, ax=axes[2, 2], fraction=0.046, pad=0.04)

    for ax in [axes[2, 0], axes[2, 1], axes[2, 2]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

    fig1.tight_layout()
    fig1.tight_layout(rect=[0, 0.05, 1, 1])
    fig1.savefig('plots/INRBlockModel.png', dpi=300)
    plt.close(fig1)

    fig3, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(hist['gravity'], color='black')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig3.tight_layout()
    fig3.savefig('plots/INRBlockModel_loss.png', dpi=300)
    plt.close(fig3)

    rms_rho = torch.sqrt(torch.mean((m_inv - rho_true_vec.to(device)) ** 2)).item()
    rms_gz = torch.sqrt(torch.mean((gz_pred - gz_obs) ** 2)).item() * 1e5
    print(f"RMS density-contrast error ≈ {rms_rho:.2f} kg/m^3")
    print(f"RMS data misfit ≈ {rms_gz:.3f} mGal")

if __name__ == '__main__':
    run()
