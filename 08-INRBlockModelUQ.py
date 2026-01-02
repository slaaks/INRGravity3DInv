import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

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

def train_inr(model, opt, coords_norm, G, gz_obs, Wd, Nx, Ny, Nz, dx, dy, dz, cfg):
    history = {"total": [], "gravity": []}
    for ep in range(cfg['epochs']):
        opt.zero_grad()
        m_pred = model(coords_norm).view(-1)
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

####
def evaluate_model(model, coords_tensor, G_tensor):
    model.eval()
    with torch.no_grad():
        rho_pred = model(coords_tensor).view(-1)
        gz_pred = (G_tensor @ rho_pred.unsqueeze(1)).squeeze(1)
    return rho_pred.detach().cpu().numpy(), gz_pred.detach().cpu().numpy()

def evaluate_ensemble(models, coords_tensor, G_tensor):
    rho_preds, gz_preds = [], []
    for m in models:
        rho_p, gz_p = evaluate_model(m, coords_tensor, G_tensor)
        rho_preds.append(rho_p)
        gz_preds.append(gz_p)
    rho_preds = np.stack(rho_preds, axis=0)
    gz_preds  = np.stack(gz_preds, axis=0)
    return rho_preds.mean(0), rho_preds.std(0), gz_preds.mean(0), gz_preds.std(0)

def train_ensemble_inr(num_models, coords_norm, G, gz_obs, Wd,
                       Nx, Ny, Nz, dx, dy, dz, cfg, device):
    models = []
    base_seed = cfg.get('seed', 42)
    for k in range(num_models):
        seed_k = base_seed + k
        torch.manual_seed(seed_k); np.random.seed(seed_k); random.seed(seed_k)

        model = DensityContrastINR(
            nfreq=cfg.get('nfreq', 2),
            hidden=cfg.get('hidden', 256),
            depth=cfg.get('depth', 4),
            rho_abs_max=cfg.get('rho_abs_max', 600.0)
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-2))
        _ = train_inr(model, opt, coords_norm, G, gz_obs, Wd,
                      Nx, Ny, Nz, dx, dy, dz, cfg)
        models.append(model)
    return models

def plot_ensemble(
    rho_mean, rho_std, Nx, Ny, Nz, grid_coords, dx, dy, dz,
    save_path='plots/INR_ensemble.png',
    ix=None, iy=None, iz=None,
    cmap_mean='viridis', cmap_std='magma',
    obs=None,
    gz_mean_si=None,
    gz_std_si=None,
    noise_sigma_si=None,
    gravity_scatter_size=80,
    gravity_vmin=None,
    gravity_vmax=None,
    gravity_std_vmin=0.0,
    gravity_std_vmax=None
):

    #Slice indices
    ix = Nx // 2 if ix is None else int(ix)
    iy = Ny // 2 if iy is None else int(iy)
    iz = min(Nz - 1, 5) if iz is None else int(iz)

    #Reshape to 3D
    rho_mean_3d = rho_mean.reshape(Nx, Ny, Nz)
    rho_std_3d  = rho_std.reshape(Nx, Ny, Nz)

    #1D arrays of cell centers
    x1d = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
    y1d = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
    z1d = grid_coords[:, 2].reshape(Nx, Ny, Nz)[0, 0, :]

    #Cell-edge limits
    x_edge_min, x_edge_max = x1d[0] - dx/2, x1d[-1] + dx/2
    y_edge_min, y_edge_max = y1d[0] - dy/2, y1d[-1] + dy/2
    z_edge_min, z_edge_max = z1d[0] - dz/2, z1d[-1] + dz/2

    #Extents (depth increases downward in sections)
    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]
    extent_xz = [x_edge_min, x_edge_max, z_edge_max, z_edge_min]
    extent_yz = [y_edge_min, y_edge_max, z_edge_max, z_edge_min]

    #Block indeces for truth-rectangles
    def iter_blocks():
        for i in range(7):
            z_idx = 1 + i
            y_start, y_end = 11 - i, 16 - i
            x_start, x_end = 7, 13
            if 0 <= z_idx < Nz:
                ys, ye = max(0, y_start), min(Ny, y_end)
                xs, xe = max(0, x_start), min(Nx, x_end)
                if xs < xe and ys < ye:
                    yield xs, xe, ys, ye, z_idx

    nrows = 4
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 5 * nrows))

    #Density
    slices = [
        ('XY', rho_mean_3d[:, :, iz], rho_std_3d[:, :, iz], extent_xy, f'z≈{z1d[iz]:.0f} m'),
        ('XZ', rho_mean_3d[:, iy, :], rho_std_3d[:, iy, :], extent_xz, f'y≈{y1d[iy]:.0f} m'),
        ('YZ', rho_mean_3d[ix, :, :], rho_std_3d[ix, :, :], extent_yz, f'x≈{x1d[ix]:.0f} m'),
    ]

    for i, (label, mean_slice, std_slice, extent, coord) in enumerate(slices):
        origin = 'lower' if label == 'XY' else 'upper'

        im = axes[i, 0].imshow(mean_slice.T, origin=origin, extent=extent,
                               aspect='auto', cmap=cmap_mean)
        axes[i, 0].set_title(f"Mean Δρ {label} @ {coord}")
        fig.colorbar(im, ax=axes[i, 0], label='kg/m³', fraction=0.046, pad=0.04)

        #Truth overlay
        if label == 'XY':
            for xs, xe, ys, ye, z_idx in iter_blocks():
                if z_idx == iz:
                    x0 = x1d[xs] - dx/2; x1 = x1d[xe-1] + dx/2
                    y0 = y1d[ys] - dy/2; y1 = y1d[ye-1] + dy/2
                    rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                              fill=False, edgecolor='white', linewidth=2)
                    axes[i, 0].add_patch(rect)
        elif label == 'XZ':
            xs_list, xe_list, z_list = [], [], []
            for xs, xe, ys, ye, z_idx in iter_blocks():
                if ys <= iy < ye:
                    xs_list.append(xs); xe_list.append(xe); z_list.append(z_idx)
            if z_list:
                x_min_idx = min(xs_list); x_max_idx = max(xe_list)
                z_min_idx = min(z_list); z_max_idx = max(z_list)
                x0 = x1d[x_min_idx] - dx/2
                x1 = x1d[x_max_idx - 1] + dx/2
                z0 = z1d[z_min_idx] - dz/2
                z1 = z1d[z_max_idx] + dz/2
                rect = mpatches.Rectangle((x0, z0), x1 - x0, z1 - z0,
                                          fill=False, edgecolor='white', linewidth=2)
                axes[i, 0].add_patch(rect)
        else: #'YZ'
            for xs, xe, ys, ye, z_idx in iter_blocks():
                if xs <= ix < xe:
                    y0 = y1d[ys] - dy/2; y1 = y1d[ye-1] + dy/2
                    z0 = z1d[z_idx] - dz/2; z1 = z1d[z_idx] + dz/2
                    rect = mpatches.Rectangle((y0, z0), y1 - y0, z1 - z0,
                                              fill=False, edgecolor='white', linewidth=2)
                    axes[i, 0].add_patch(rect)

        im = axes[i, 1].imshow(std_slice.T, origin=origin, extent=extent,
                               aspect='auto', cmap=cmap_std)
        axes[i, 1].set_title(f"Std Δρ {label} @ {coord}")
        fig.colorbar(im, ax=axes[i, 1], label='kg/m³', fraction=0.046, pad=0.04)

        #Overlay on std panels too
        if label == 'XY':
            for xs, xe, ys, ye, z_idx in iter_blocks():
                if z_idx == iz:
                    x0 = x1d[xs] - dx/2; x1 = x1d[xe-1] + dx/2
                    y0 = y1d[ys] - dy/2; y1 = y1d[ye-1] + dy/2
                    rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                              fill=False, edgecolor='white', linewidth=2)
                    axes[i, 1].add_patch(rect)
        elif label == 'XZ':
            xs_list, xe_list, z_list = [], [], []
            for xs, xe, ys, ye, z_idx in iter_blocks():
                if ys <= iy < ye:
                    xs_list.append(xs); xe_list.append(xe); z_list.append(z_idx)
            if z_list:
                x_min_idx = min(xs_list); x_max_idx = max(xe_list)
                z_min_idx = min(z_list); z_max_idx = max(z_list)
                x0 = x1d[x_min_idx] - dx/2
                x1 = x1d[x_max_idx - 1] + dx/2
                z0 = z1d[z_min_idx] - dz/2
                z1 = z1d[z_max_idx] + dz/2
                rect = mpatches.Rectangle((x0, z0), x1 - x0, z1 - z0,
                                          fill=False, edgecolor='white', linewidth=2)
                axes[i, 1].add_patch(rect)
        else: #'YZ'
            for xs, xe, ys, ye, z_idx in iter_blocks():
                if xs <= ix < xe:
                    y0 = y1d[ys] - dy/2; y1 = y1d[ye-1] + dy/2
                    z0 = z1d[z_idx] - dz/2; z1 = z1d[z_idx] + dz/2
                    rect = mpatches.Rectangle((y0, z0), y1 - y0, z1 - z0,
                                              fill=False, edgecolor='white', linewidth=2)
                    axes[i, 1].add_patch(rect)

        #Axis labels
        if label == 'XY':
            axes[i, 0].set_xlabel('x (m)'); axes[i, 0].set_ylabel('y (m)')
            axes[i, 1].set_xlabel('x (m)'); axes[i, 1].set_ylabel('y (m)')
        elif label == 'XZ':
            axes[i, 0].set_xlabel('x (m)'); axes[i, 0].set_ylabel('Depth (m)')
            axes[i, 1].set_xlabel('x (m)'); axes[i, 1].set_ylabel('Depth (m)')
        else:  #YZ
            axes[i, 0].set_xlabel('y (m)'); axes[i, 0].set_ylabel('Depth (m)')
            axes[i, 1].set_xlabel('y (m)'); axes[i, 1].set_ylabel('Depth (m)')

        axes[i, 0].set_aspect(1.0)
        axes[i, 1].set_aspect(1.0)

    #Gravity row
    row = 3  #fourth row index

    #Coordinates of observation stations
    obs_x = obs[:, 0].detach().cpu().numpy()
    obs_y = obs[:, 1].detach().cpu().numpy()

    #Convert to mGal
    mean_mgal = 1e5 * gz_mean_si
    std_mgal  = 1e5 * gz_std_si

    #Symmetric color range for mean
    if gravity_vmin is None or gravity_vmax is None:
        vmax_sym = np.max(np.abs(mean_mgal))
        gravity_vmin = -vmax_sym if gravity_vmin is None else gravity_vmin
        gravity_vmax =  vmax_sym if gravity_vmax is None else gravity_vmax

    #upper bound for std if not set
    if gravity_std_vmax is None:
        gravity_std_vmax = float(np.percentile(std_mgal, 99))

    #Mean g_z (mGal)
    sc0 = axes[row, 0].scatter(
        obs_x, obs_y, c=mean_mgal, s=gravity_scatter_size,
        cmap='viridis', vmin=gravity_vmin, vmax=gravity_vmax,
        marker='o', edgecolors='none'
    )
    axes[row, 0].set_title('Ensemble mean g_z (mGal) @ surface')
    cb0 = fig.colorbar(sc0, ax=axes[row, 0], fraction=0.046, pad=0.04)
    cb0.set_label('mGal')
    axes[row, 0].set_xlabel('x (m)'); axes[row, 0].set_ylabel('y (m)')
    axes[row, 0].set_aspect('equal')

    #Std g_z (mGal)
    sc1 = axes[row, 1].scatter(
        obs_x, obs_y, c=std_mgal, s=gravity_scatter_size,
        cmap='magma', vmin=gravity_std_vmin, vmax=gravity_std_vmax,
        marker='o', edgecolors='none'
    )
    title = 'Ensemble std g_z (mGal) @ surface'
    if noise_sigma_si is not None:
        sigma_mgal = 1e5 * float(noise_sigma_si)
        title += f"(σ_noise ≈ {sigma_mgal:.3f} mGal)"
    axes[row, 1].set_title(title)
    cb1 = fig.colorbar(sc1, ax=axes[row, 1], fraction=0.046, pad=0.04)
    cb1.set_label('mGal')
    axes[row, 1].set_xlabel('x (m)'); axes[row, 1].set_ylabel('y (m)')
    axes[row, 1].set_aspect('equal')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

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

    c_mean = grid_coords.mean(axis=0, keepdims=True)
    c_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = (grid_coords - c_mean) / (c_std + 1e-12)
    coords_norm = torch.tensor(coords_norm, dtype=torch.float32, device=device, requires_grad=True)

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

    cfg = dict(gamma=1.0, epochs=300, lr=1e-2, seed=42, nfreq=2, hidden=256, depth=4, rho_abs_max=600.0)

    #Train ensemble
    num_models = 5
    models = train_ensemble_inr(num_models, coords_norm, G, gz_obs, Wd,
                            Nx, Ny, Nz, dx, dy, dz, cfg, device)

    rho_mean, rho_std, gz_mean_si, gz_std_si = evaluate_ensemble(models, coords_norm, G)

    #convert gravity to mGal
    gz_mean_mgal = 1e5 * gz_mean_si
    gz_std_mgal  = 1e5 * gz_std_si

    plot_ensemble(
        rho_mean=rho_mean, rho_std=rho_std,
        Nx=Nx, Ny=Ny, Nz=Nz,
        grid_coords=grid_coords, dx=dx, dy=dy, dz=dz,
        save_path='plots/INREnsemble.png',
        ix=Nx//2, iy=Ny//2, iz=min(Nz-1, 5),
        cmap_mean='viridis', cmap_std='magma',
        obs=obs,
        gz_mean_si=gz_mean_si,
        gz_std_si=gz_std_si,
        noise_sigma_si=sigma,
        gravity_scatter_size=80,
        gravity_vmin=None, gravity_vmax=None,
        gravity_std_vmin=0.0, gravity_std_vmax=None
    )

if __name__ == '__main__':
    run()
