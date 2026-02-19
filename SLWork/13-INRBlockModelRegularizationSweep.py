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

#@torch.inference_mode()
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
    
#Gradient helper function for regularizers
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


def tik0_loss(m, dx, dy, dz):
    cell_vol = dx * dy * dz
    return cell_vol * torch.mean(m**2)

def tik1_loss(m, Nx, Ny, Nz, dx, dy, dz, wx=1.0, wy=1.0, wz=1.0):
    cell_vol = dx * dy * dz
    gx, gy, gz = compute_gradient(m, Nx, Ny, Nz, dx, dy, dz)
    gx, gy, gz = wx * gx, wy * gy, wz * gz
    return cell_vol * torch.mean(gx**2 + gy**2 + gz**2)

def tv_loss(m, Nx, Ny, Nz, dx, dy, dz, eps=1e-6, wx=1.0, wy=1.0, wz=1.0):
    cell_vol = dx * dy * dz
    gx, gy, gz = compute_gradient(m, Nx, Ny, Nz, dx, dy, dz)
    gx, gy, gz = wx * gx, wy * gy, wz * gz
    tv_vals = torch.sqrt(gx**2 + gy**2 + gz**2 + eps)
    return cell_vol * torch.mean(tv_vals)

#Norms
def tik0_norm(m, dx, dy, dz):
    cell_vol = dx * dy * dz
    return torch.sqrt(cell_vol * torch.sum(m**2) + 1e-12)

def tik1_norm(m, Nx, Ny, Nz, dx, dy, dz, wx=1.0, wy=1.0, wz=1.0):
    cell_vol = dx * dy * dz
    gx, gy, gz = compute_gradient(m, Nx, Ny, Nz, dx, dy, dz)
    gx, gy, gz = wx * gx, wy * gy, wz * gz
    return torch.sqrt(cell_vol * torch.sum(gx**2 + gy**2 + gz**2) + 1e-12)

def tv_norm(m, Nx, Ny, Nz, dx, dy, dz, eps=1e-6, wx=1.0, wy=1.0, wz=1.0):
    return tv_loss(m, Nx, Ny, Nz, dx, dy, dz, eps=eps, wx=wx, wy=wy, wz=wz)

def train_inr(model, opt, coords_norm, G, gz_obs, Wd,
              Nx, Ny, Nz, dx, dy, dz, cfg):

    history = {"total": [], "gravity": [], "tik0": [], "tik1": [], "tv": []}

    gamma = float(cfg.get('gamma', 1.0))

    #Regularization weights
    lam0   = float(cfg.get('tik0', 0.0))  #0th order
    lam1   = float(cfg.get('tik1', 0.0))  #1st order
    lam_tv = float(cfg.get('tv', 0.0))    #total variation

    #Directional anisotropy
    wx = float(cfg.get('wx', 1.0))
    wy = float(cfg.get('wy', 1.0))
    wz = float(cfg.get('wz', 1.0))
    #TV eps for stability
    tv_eps = float(cfg.get('tv_eps', 1e-8))

    model.train()

    for ep in range(cfg['epochs']):
        opt.zero_grad()

        m_pred = model(coords_norm).view(-1)

        #Data term
        gz_pred = (G @ m_pred.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        data_term = gamma * torch.mean((Wd * residual) ** 2)

        #Regularization terms
        reg0 = lam0 * tik0_loss(m_pred, dx, dy, dz)
        reg1 = lam1 * tik1_loss(m_pred, Nx, Ny, Nz, dx, dy, dz, wx=wx, wy=wy, wz=wz)
        reg_tv = lam_tv * tv_loss(m_pred, Nx, Ny, Nz, dx, dy, dz,
                                  eps=tv_eps, wx=wx, wy=wy, wz=wz)

        loss = data_term + reg0 + reg1 + reg_tv
        loss.backward()
        opt.step()

        #Log
        history['gravity'].append(float(data_term.item()))
        history['tik0'].append(float(reg0.item()))
        history['tik1'].append(float(reg1.item()))
        history['tv'].append(float(reg_tv.item()))
        history['total'].append(float(loss.item()))

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

###
@torch.no_grad()
def misfit_norm(G, m, d_obs, Wd):
    r = (G @ m) - d_obs
    return torch.linalg.norm(Wd * r).item()
def run_single_inversion(model_ctor, coords_norm, G, d_obs, Wd,
                         Nx, Ny, Nz, dx, dy, dz,
                         lam, reg_type, epochs=300, lr=1e-3, device=None):

    if device is None:
        device = coords_norm.device

    #Create model
    model = model_ctor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    #Config
    cfg = dict(
        epochs=epochs,
        gamma=1.0,
        tik0 = lam if reg_type=="tik0" else 0.0,
        tik1 = lam if reg_type=="tik1" else 0.0,
        tv   = lam if reg_type=="tv"   else 0.0,
        wx=1.0, wy=1.0, wz=1.0,
        tv_eps=1e-8
    )

    train_inr(model, opt, coords_norm, G, d_obs, Wd,
              Nx, Ny, Nz, dx, dy, dz, cfg)

    with torch.no_grad():
        m = model(coords_norm).view(-1)

        mis = misfit_norm(G, m, d_obs, Wd)

        if reg_type == "tik0":
            reg = tik0_norm(m, dx, dy, dz).item()
        elif reg_type == "tik1":
            reg = tik1_norm(m, Nx, Ny, Nz, dx, dy, dz).item()
        elif reg_type == "tv":
            reg = tv_norm(m, Nx, Ny, Nz, dx, dy, dz).item()

    return mis, reg, model

#Helper function for lambda parameter sweep
def sweep_lambda(model_ctor, coords_norm, G, d_obs, Wd,
                 Nx, Ny, Nz, dx, dy, dz,
                 lambda_values, reg_type,
                 epochs=200, lr=1e-3,
                 device=None, warm_start=True,
                 wx=1.0, wy=1.0, wz=1.0, tv_eps=1e-8):

    if device is None:
        device = coords_norm.device

    misfits, regs, lam_out = [], [], []
    model_prev = None

    #Sweep from large to small
    for lam in sorted(lambda_values, reverse=True):
        model = model_prev if (warm_start and model_prev is not None) else model_ctor().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        cfg = dict(
            epochs = epochs,
            gamma  = 1.0,
            tik0   = lam if reg_type == "tik0" else 0.0,
            tik1   = lam if reg_type == "tik1" else 0.0,
            tv     = lam if reg_type == "tv"   else 0.0,
            wx=wx, wy=wy, wz=wz,
            tv_eps = tv_eps
        )

        train_inr(model, opt, coords_norm, G, d_obs, Wd,
                  Nx, Ny, Nz, dx, dy, dz, cfg)

        model_prev = model

        with torch.no_grad():
            m = model(coords_norm).view(-1)
            r = (G @ m) - d_obs
            mis = torch.linalg.norm(Wd * r).item()

            if reg_type == "tik0":
                reg = tik0_norm(m, dx, dy, dz).item()
            elif reg_type == "tik1":
                reg = tik1_norm(m, Nx, Ny, Nz, dx, dy, dz).item()
            else:
                reg = tv_norm(m, Nx, Ny, Nz, dx, dy, dz, eps=tv_eps, wx=wx, wy=wy, wz=wz).item()

        misfits.append(mis)
        regs.append(reg)
        lam_out.append(lam)

        print(f"λ={lam:.2e} | misfit={mis:.3e} | reg={reg:.3e}")

    return np.array(lam_out), np.array(misfits), np.array(regs)


def plot_lcurve(lams, misfits, regs, reg_type, title, filename):

    order = np.argsort(lams)
    lams, misfits, regs = np.asarray(lams)[order], np.asarray(misfits)[order], np.asarray(regs)[order]

    plt.figure(figsize=(6,5))
    plt.loglog(misfits, regs, "-o", markersize=4)

    plt.xlabel(r"Residual norm $\|g_{\mathrm{pred}} - g_{\mathrm{obs}}\|_2$")
    plt.ylabel(r"Solution norm $\|m\|_2$")

    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def run():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("plots", exist_ok=True)

    dx = dy = dz = 50.0
    x = np.arange(0, 1000+dx, dx)
    y = np.arange(0, 1000+dy, dy)
    z = np.arange(0,  500+dz, dz)
    Nx, Ny, Nz = len(x), len(y), len(z)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    c_mean = grid.mean(axis=0, keepdims=True)
    c_std  = grid.std(axis=0, keepdims=True)
    coords_norm = torch.tensor((grid - c_mean) / (c_std + 1e-12),
                               dtype=torch.float32, device=device)

    dz_half = dz / 2.0
    cell_grid = np.hstack([grid, np.full((grid.shape[0], 1), dz_half)])
    cell_grid = torch.tensor(cell_grid, dtype=torch.float32, device=device)

    XX, YY = np.meshgrid(x, y, indexing='ij')
    obs = torch.tensor(np.column_stack([XX.ravel(), YY.ravel(), -np.ones(XX.size)]),
                       dtype=torch.float32, device=device)

    print("Assembling sensitivity G ...")
    t0 = time.time()
    G = construct_sensitivity_matrix_G_torch(cell_grid, obs, dx, dy, device)
    G = G.detach()
    print(f"G shape = {tuple(G.shape)}, time = {time.time()-t0:.2f}s")

    rho_true_vec, _ = make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=0.0, rho_blk=400.0)
    rho_true_vec = rho_true_vec.to(device)
    with torch.no_grad():
        gz_true = G @ rho_true_vec

    noise_lvl = 0.01
    sigma = noise_lvl * gz_true.std()
    noise = sigma * torch.randn_like(gz_true)
    d_obs = gz_true + noise
    Wd = 1.0 / (sigma + 1e-12)

    def model_ctor(): #untrained INR model constructor
        return DensityContrastINR(nfreq=2, hidden=256, depth=4, rho_abs_max=600.0).to(device)

    lams_t0 = np.logspace(-7, -4, 10)  #Lambda ranges
    lams_t1 = np.logspace(-10, -6, 12)
    lams_tv = np.logspace(-5, -2, 12) 

    #Sweeps for different regularizers
    print("\n--- Sweep Tik0 ---")
    lam_t0, mis_t0, reg_t0 = sweep_lambda(
        model_ctor, coords_norm, G, d_obs, Wd,
        Nx, Ny, Nz, dx, dy, dz,
        lambda_values=lams_t0,
        reg_type="tik0",
        epochs=200, lr=1e-3,
        device=device, warm_start=True
    )
    plot_lcurve(lam_t0, mis_t0, reg_t0, reg_type="tik0",
                title="Tik0 L-curve", filename="plots/T0_curve.png")

    print("\n--- Sweep Tik1 ---")
    lam_t1, mis_t1, reg_t1 = sweep_lambda(
        model_ctor, coords_norm, G, d_obs, Wd,
        Nx, Ny, Nz, dx, dy, dz,
        lambda_values=lams_t1,
        reg_type="tik1",
        epochs=200, lr=1e-3,
        device=device, warm_start=True
    )
    plot_lcurve(lam_t1, mis_t1, reg_t1, reg_type="tik1",
                title="Tik1 L-curve", filename="plots/T1_curve.png")

    print("\n-- Sweep TV ---")
    lam_tv, mis_tv, reg_tv = sweep_lambda(
        model_ctor, coords_norm, G, d_obs, Wd,
        Nx, Ny, Nz, dx, dy, dz,
        lambda_values=lams_tv,
        reg_type="tv",
        epochs=200, lr=1e-4,
        device=device, warm_start=True
    )
    plot_lcurve(lam_tv, mis_tv, reg_tv, reg_type="tv",
                title="TV L-curve", filename="plots/TV_curve.png")

if __name__ == '__main__':
    run()
