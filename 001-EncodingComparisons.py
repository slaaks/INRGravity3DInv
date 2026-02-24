import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



# --- Random seed -------------------------------------------------------
SEED = 42 #10,42, 40, 41

# --- Grid / domain -----------------------------------------------------
DX = 50.0          # cell size in x (m)
DY = 50.0          # cell size in y (m)
DZ = 50.0          # cell size in z (m)
X_MAX = 1000.0     # domain extent in x (m)
Y_MAX = 1000.0     # domain extent in y (m)
Z_MAX = 500.0      # domain extent in z (m)

# --- Block model -------------------------------------------------------
RHO_BG  = 0.0      # background density contrast (kg/m³)
RHO_BLK = 400.0    # block density contrast (kg/m³)

# --- Noise level -------------------------------------------------------
NOISE_LEVEL = 0.01  # fraction of gz_true std

# --- Training / optimisation -------------------------------------------
GAMMA   = 1.0       # data-term weight
EPOCHS  = 500       # number of training epochs
LR      = 1e-2      # Adam learning rate

# --- INR network -------------------------------------------------------
HIDDEN       = 256   # hidden-layer width
DEPTH        = 4     # number of hidden layers
RHO_ABS_MAX  = 600.0 # tanh output scaling (kg/m³)

# --- Encoding strategy -------------------------------------------------
#   'positional'  – sinusoidal Fourier features  (NeRF)
#   'gaussian'    – random Fourier features      (Tancik et al., 2020)
#   'hash'        – multi-resolution hash tables  (Instant-NGP)
#   'triplane'    – three learnable 2-D feature planes
#   'combined'    – concatenation of multiple encodings
ENCODING_TYPE = 'hash'  # 'positional (2)', 'gaussian', 'hash', 'triplane', 'combined'

ENCODING_CONFIGS = {
    'positional': dict(num_freqs=2),
    'gaussian':   dict(num_freqs=64, sigma=4.0, include_input=True),
    'hash':       dict(n_levels=2, n_features_per_level=2,
                       log2_hashmap_size=17, base_resolution=4,
                       finest_resolution=128),
    'triplane':   dict(resolution=128, n_features=8),
    'combined':   dict(
        sub_encodings=['positional', 'triplane'],
        sub_kwargs=[dict(num_freqs=2),
                    dict(n_levels=8, n_features_per_level=2,
                         log2_hashmap_size=17, base_resolution=4,
                         finest_resolution=128)]),
}

# --- Plotting -----------------------------------------------------------
CMAP          = 'turbo'   # colormap for all plots
INV_VMAX      = 250       # fixed colorbar max for inverted model



class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (NeRF-style Fourier features).

    Intuition
    ---------
    Plain MLPs are biased toward learning low-frequency functions — they
    tend to produce overly smooth outputs and struggle to represent sharp
    edges or fine detail ("spectral bias").  Positional encoding fights
    this by lifting each input coordinate into a set of sine and cosine
    waves at *exponentially increasing* frequencies:

        γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), …]

    Think of it as giving the network a set of "rulers" at different
    scales: the lowest frequency captures the overall trend, while
    higher frequencies let the network resolve increasingly fine spatial
    variation.  The resulting feature vector is fixed (no learnable
    parameters in the encoding itself).

    Trade-offs
    ----------
    • Simple and deterministic — no extra learnable parameters.
    • The frequency ladder is fixed (powers of 2), so the spectrum can
      have gaps and may not cover all scales equally well.
    • Too few frequencies → overly smooth model; too many → potential
      noise fitting and slower convergence.
    • Works best when combined with a sufficiently deep/wide MLP.

    Args:
        num_freqs:      Number of frequency octaves (L).  Output grows
                        as input_dim × (1 + 2L) with include_input.
        include_input:  Whether to prepend the raw (x, y, z) values.
        input_dim:      Spatial dimensionality (default 3).
    """
    def __init__(self, num_freqs=8, include_input=True, input_dim=3):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))
        self.out_dim = input_dim * (1 + 2 * num_freqs) if include_input else input_dim * 2 * num_freqs

    def forward(self, x):
        parts = [x] if self.include_input else []
        for f in self.freqs:
            parts += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(parts, dim=-1)


class GaussianFourierEncoding(nn.Module):
    """Random Fourier Features (Tancik et al., 2020).

    Intuition
    ---------
    Instead of hand-picking frequencies on a power-of-2 ladder (as in
    positional encoding), this method draws a *random* matrix B from a
    Gaussian distribution N(0, σ²) and projects each input through it:

        γ(x) = [sin(2πBᵀx), cos(2πBᵀx)]

    Imagine throwing darts randomly across the frequency spectrum — you
    get a much more *uniform* coverage of spatial frequencies.  The key
    hyperparameter is σ (the standard deviation of B):
      • Small σ  → mostly low-frequency features → smooth reconstructions
      • Large σ  → high frequencies included → sharper edges but risk of
        over-fitting noise

    Because B is fixed at initialisation (not learned), the encoding is
    cheap and stable, but you must tune σ to match the expected spatial
    scale of features in your model.

    Trade-offs
    ----------
    • Broader, more uniform frequency coverage than positional encoding.
    • σ acts as a single dial controlling the resolution–smoothness
      trade-off; no need to choose individual frequency bands.
    • Still a fixed (non-adaptive) encoding — it cannot dynamically
      allocate more detail where the model is complex.
    • Excellent for geophysical inversions where the target property
      varies smoothly with occasional moderate contrasts.

    Args:
        input_dim:      Spatial dimensionality (default 3).
        num_freqs:      Number of random frequency components.
        sigma:          Std-dev of the Gaussian that generates B.
                        Controls the typical spatial wavelength:
                        larger σ → finer detail.
        include_input:  Whether to prepend the raw coordinates.
    """
    def __init__(self, input_dim=3, num_freqs=128, sigma=10.0, include_input=True):
        super().__init__()
        self.include_input = include_input
        B = torch.randn(input_dim, num_freqs) * sigma
        self.register_buffer('B', B)
        self.out_dim = 2 * num_freqs + (input_dim if include_input else 0)

    def forward(self, x):
        proj = x @ self.B
        parts = [x] if self.include_input else []
        parts += [torch.sin(2 * np.pi * proj),
                  torch.cos(2 * np.pi * proj)]
        return torch.cat(parts, dim=-1)


class HashEncoding(nn.Module):
    """Multi-resolution hash encoding (Müller et al., 2022 / Instant-NGP).

    Intuition
    ---------
    Imagine overlaying your 3-D domain with a stack of voxel grids, from
    very coarse (e.g. 4³) to very fine (e.g. 128³).  At each resolution
    level, every voxel stores a small learnable feature vector.  To
    encode a point, you look up its 8 surrounding voxel corners at every
    level, trilinearly interpolate, and concatenate across levels.

    The "hash" trick makes this memory-efficient: instead of allocating a
    full 3-D grid (which grows as O(R³)), each level maps voxel corners
    to a fixed-size hash table.  Collisions (two corners sharing one
    slot) are resolved implicitly by the gradient-based optimisation —
    the network learns to disentangle them.

    The result is an encoding that is:
    • **Adaptive**: features are *learned*, so detail concentrates where
      the data demands it (unlike fixed Fourier features).
    • **Multi-scale**: coarse levels capture large-scale trends; fine
      levels capture sharp boundaries and small anomalies.
    • **Memory-bounded**: hash table size is constant regardless of
      how fine the resolution is — crucial for large 3-D domains.

    Trade-offs
    ----------
    • Most flexible encoding — can represent both smooth and sharp models.
    • Hash collisions at fine levels can introduce small artifacts or a
      noise floor if the hash table is too small (increase
      log2_hashmap_size to mitigate).
    • Many hyperparameters to tune (n_levels, base/finest resolution,
      hash table size).
    • Learnable parameters mean more total parameters to optimise, and
      they may overfit noisy data without appropriate regularisation.

    Args:
        n_levels:               Number of resolution levels.  More levels
                                give smoother multi-scale interpolation.
        n_features_per_level:   Feature dimensions stored per hash entry.
                                Typically 2; larger values add capacity.
        log2_hashmap_size:      Log₂ of hash-table size per level.
                                2^19 ≈ 500 k entries is a common default.
        base_resolution:        Coarsest voxel-grid resolution (e.g. 4).
        finest_resolution:      Finest voxel-grid resolution (e.g. 128).
                                Intermediate levels are spaced
                                geometrically between base and finest.
        input_dim:              Spatial dimensionality (default 3).
    """
    def __init__(self, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16,
                 finest_resolution=512, input_dim=3):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.input_dim = input_dim
        self.out_dim = n_levels * n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size

        if n_levels > 1:
            self.growth_factor = np.exp(
                (np.log(finest_resolution) - np.log(base_resolution))
                / (n_levels - 1))
        else:
            self.growth_factor = 1.0
        self.base_resolution = base_resolution

        # Learnable hash tables (one per level)
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        for tbl in self.hash_tables:
            nn.init.uniform_(tbl.weight, -1e-4, 1e-4)

        # Large primes for the spatial hash
        self.register_buffer(
            'primes', torch.tensor([1, 2654435761, 805459861], dtype=torch.long))

    def _hash(self, coords_int):
        """Spatial hash: integer grid coords -> hash-table index."""
        result = torch.zeros(coords_int.shape[:-1],
                             dtype=torch.long, device=coords_int.device)
        for d in range(self.input_dim):
            result ^= coords_int[..., d] * self.primes[d]
        return result % self.hashmap_size

    def forward(self, x):
        # Normalise to [0, 1] using per-batch bounds
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        x_scaled = (x - x_min) / (x_max - x_min + 1e-8)

        outputs = []
        for level in range(self.n_levels):
            resolution = int(self.base_resolution * (self.growth_factor ** level))
            x_grid = x_scaled * resolution            # (N, 3)
            x_floor = torch.floor(x_grid).long()      # voxel origin
            x_frac  = x_grid - x_floor.float()        # interpolation weight

            # Eight voxel corners
            corners = []
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        corners.append(
                            x_floor + torch.tensor([dx, dy, dz], device=x.device))
            corners = torch.stack(corners, dim=1)      # (N, 8, 3)

            indices  = self._hash(corners)             # (N, 8)
            features = self.hash_tables[level](indices) # (N, 8, F)

            # Trilinear interpolation weights
            wx, wy, wz = (x_frac[:, 0:1],
                          x_frac[:, 1:2],
                          x_frac[:, 2:3])
            weights = torch.stack([
                (1-wx)*(1-wy)*(1-wz),  wx*(1-wy)*(1-wz),
                (1-wx)*   wy *(1-wz),  wx*   wy *(1-wz),
                (1-wx)*(1-wy)*   wz ,  wx*(1-wy)*   wz ,
                (1-wx)*   wy *   wz ,  wx*   wy *   wz ,
            ], dim=1)                                  # (N, 8, 1)

            outputs.append((weights * features).sum(dim=1))  # (N, F)

        return torch.cat(outputs, dim=-1)              # (N, n_levels*F)


class TriplaneEncoding(nn.Module):
    """Tri-plane encoding for 3D coordinates.

    Intuition
    ---------
    Instead of storing features in a full 3-D voxel grid (memory O(R³)),
    this encoding *factorises* the volume into three orthogonal 2-D
    learnable feature images — one per axis-aligned plane:

        plane_xy :  sees (x, y), ignores z
        plane_xz :  sees (x, z), ignores y
        plane_yz :  sees (y, z), ignores x

    For a query point (x, y, z) the encoding:
      1. Projects onto each plane (drops one coordinate).
      2. Bilinearly interpolates F features from the 2-D grid.
      3. Concatenates all three → output dim = 3 × F.

    Think of it like taking three "X-ray" views of the volume from the
    front, side, and top.  Each view captures 2-D spatial structure at
    the grid resolution.  The MLP that follows must combine these three
    partial views into a coherent 3-D prediction.

    Memory scales as 3 × F × R² rather than F × R³, making it far more
    efficient for high-resolution grids.  For R = 64, F = 16 this is
    ~49 k parameters vs ~4 M for a full voxel grid.

    Trade-offs
    ----------
    • Very memory-efficient for high-resolution spatial features.
    • Features are **axis-aligned**: each plane only encodes 2 of 3
      axes, so diagonal or spherical structures require the MLP to
      combine the three views — this can produce staircase artifacts
      aligned with the grid axes.
    • Bilinear interpolation within each plane smooths features
      between grid nodes, which can blur sharp density contrasts
      and underestimate amplitudes.
    • The resolution R controls the finest recoverable feature size:
      effective spatial sampling ≈ domain_extent / R.
    • Well-suited when the target model has dominant axis-aligned
      structure (e.g. layered geology, block models).

    Potential improvements
    ----------------------
    • Increase resolution to resolve finer boundaries.
    • Use **product** instead of concatenation (f_xy ⊙ f_xz ⊙ f_yz)
      to create true 3-D correlations and reduce axis-aligned bias.
    • Pair with positional encoding ('combined') for off-axis detail.

    Args:
        resolution:  Spatial resolution of each 2-D feature plane (R).
        n_features:  Number of feature channels per plane (F).
        input_dim:   Spatial dimensionality (default 3).
    """
    def __init__(self, resolution=64, n_features=16, input_dim=3):
        super().__init__()
        self.resolution = resolution
        self.n_features = n_features
        self.out_dim = 3 * n_features
        self.plane_xy = nn.Parameter(
            torch.randn(1, n_features, resolution, resolution) * 0.01)
        self.plane_xz = nn.Parameter(
            torch.randn(1, n_features, resolution, resolution) * 0.01)
        self.plane_yz = nn.Parameter(
            torch.randn(1, n_features, resolution, resolution) * 0.01)

    def forward(self, x):
        # Normalise each axis to [-1, 1] for grid_sample
        x_n = x.clone()
        for d in range(3):
            mn, mx = x[:, d].min(), x[:, d].max()
            x_n[:, d] = 2 * (x[:, d] - mn) / (mx - mn + 1e-8) - 1
        N = x.shape[0]
        xy = x_n[:, :2].view(1, N, 1, 2)
        xz = x_n[:, [0, 2]].view(1, N, 1, 2)
        yz = x_n[:, [1, 2]].view(1, N, 1, 2)
        f_xy = F.grid_sample(self.plane_xy, xy, align_corners=True,
                             mode='bilinear', padding_mode='border')
        f_xz = F.grid_sample(self.plane_xz, xz, align_corners=True,
                             mode='bilinear', padding_mode='border')
        f_yz = F.grid_sample(self.plane_yz, yz, align_corners=True,
                             mode='bilinear', padding_mode='border')
        f_xy = f_xy.squeeze(0).squeeze(-1).permute(1, 0)
        f_xz = f_xz.squeeze(0).squeeze(-1).permute(1, 0)
        f_yz = f_yz.squeeze(0).squeeze(-1).permute(1, 0)
        return torch.cat([f_xy, f_xz, f_yz], dim=-1)


class CombinedEncoding(nn.Module):
    """Concatenates outputs from several encoding strategies.

    Intuition
    ---------
    Each encoding has different strengths:
      • Positional / Gaussian — lightweight, fixed features that provide
        a reliable low-to-mid frequency baseline with no extra learnable
        parameters.
      • Hash — learnable, adaptive, multi-scale features that excel at
        sharp boundaries but add parameters and may overfit.
      • Triplane — memory-efficient learnable features, strong for
        axis-aligned structure but weaker for diagonal features.

    CombinedEncoding simply concatenates the outputs of two or more
    encodings, letting the MLP draw on the best qualities of each.
    For example, positional + hash gives stable low-frequency recovery
    from the positional branch and sharp-edge fidelity from the hash
    branch — the network learns which features to weight for each
    spatial location.

    Trade-offs
    ----------
    • Increases the input dimension to the MLP (sum of all sub-dim),
      which may require a wider first hidden layer.
    • More hyperparameters to tune (one set per sub-encoding).
    • Redundant frequency coverage is generally harmless — the MLP
      ignores features it does not need.
    • Particularly useful in geophysical inversion where smooth
      background + localised anomalies coexist.
    """
    def __init__(self, encodings):
        super().__init__()
        self.encodings = nn.ModuleList(encodings)
        self.out_dim = sum(e.out_dim for e in encodings)

    def forward(self, x):
        return torch.cat([enc(x) for enc in self.encodings], dim=-1)


def create_encoding(encoding_type, **kwargs):
    """Factory: build an encoding module by name.

    Supported types
    ---------------
    positional  – Sinusoidal Fourier features (Mildenhall et al., 2020
                  / NeRF).  Fixed frequencies on a power-of-2 ladder.
                  Simple, no learnable params; good general-purpose
                  baseline.  Tune `num_freqs` for resolution.

    gaussian    – Random Fourier Features (Tancik et al., 2020).  A
                  random projection matrix provides broad, uniform
                  frequency coverage.  Tune `sigma` to match the
                  spatial scale of target features.

    hash        – Multi-resolution hash tables (Müller et al., 2022 /
                  Instant-NGP).  Learnable feature grids at multiple
                  resolutions compressed via spatial hashing.  Most
                  flexible; best for sharp boundaries.  More params &
                  hyperparameters.

    triplane    – Three learnable 2-D feature planes (EG3D-style).
                  Memory-efficient O(R²) factorisation of 3-D space.
                  Strong for axis-aligned structure; may show staircase
                  artifacts for off-axis features.  Tune `resolution`.

    combined    – Concatenation of ≥2 encodings above.  Leverages
                  complementary strengths (e.g. positional for smooth
                  background + hash for sharp anomalies).
    """
    if encoding_type == 'positional':
        enc = PositionalEncoding(
            num_freqs=kwargs.get('num_freqs', 8),
            include_input=kwargs.get('include_input', True))
        return enc
    elif encoding_type == 'gaussian':
        return GaussianFourierEncoding(
            input_dim=kwargs.get('input_dim', 3),
            num_freqs=kwargs.get('num_freqs', 128),
            sigma=kwargs.get('sigma', 10.0))
    elif encoding_type == 'hash':
        return HashEncoding(
            n_levels=kwargs.get('n_levels', 16),
            n_features_per_level=kwargs.get('n_features_per_level', 2),
            log2_hashmap_size=kwargs.get('log2_hashmap_size', 19),
            base_resolution=kwargs.get('base_resolution', 16),
            finest_resolution=kwargs.get('finest_resolution', 512))
    elif encoding_type == 'triplane':
        return TriplaneEncoding(
            resolution=kwargs.get('resolution', 64),
            n_features=kwargs.get('n_features', 16))
    elif encoding_type == 'combined':
        sub_types  = kwargs.get('sub_encodings', ['positional', 'hash'])
        sub_kwargs = kwargs.get('sub_kwargs', [{}, {}])
        subs = [create_encoding(t, **kw) for t, kw in zip(sub_types, sub_kwargs)]
        return CombinedEncoding(subs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


class DensityContrastINR(nn.Module):
    """INR density-contrast model with pluggable spatial encoding."""
    def __init__(self, encoding_type='positional', hidden=256, depth=5,
                 rho_abs_max=600.0, **encoding_kwargs):
        super().__init__()
        self.pe = create_encoding(encoding_type, **encoding_kwargs)
        in_dim = self.pe.out_dim
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

# ──────────────────────────────────────────────────────────────────────
#                     UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

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
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    dx, dy, dz = DX, DY, DZ
    x = np.arange(0.0, X_MAX + dx, dx)
    y = np.arange(0.0, Y_MAX + dy, dy)
    z = np.arange(0.0, Z_MAX + dz, dz)
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

    rho_true_vec, rho_true_3d = make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=RHO_BG, rho_blk=RHO_BLK)
    rho_true_vec = rho_true_vec.to(device)

    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)

    sigma = NOISE_LEVEL * gz_true.std()
    noise = sigma * torch.randn_like(gz_true)
    gz_obs = gz_true + noise
    Wd = 1.0 / sigma

    cfg = dict(gamma=GAMMA, epochs=EPOCHS, lr=LR)

    print(f"\n▶  Encoding strategy: {ENCODING_TYPE}")
    model = DensityContrastINR(
        encoding_type=ENCODING_TYPE, hidden=HIDDEN, depth=DEPTH,
        rho_abs_max=RHO_ABS_MAX,
        **ENCODING_CONFIGS[ENCODING_TYPE]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    hist = train_inr(model, opt, coords_norm, G, gz_obs, Wd, Nx, Ny, Nz, dx, dy, dz, cfg)

    with torch.no_grad():
        m_inv = model(coords_norm).view(-1)
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
    inv_max = INV_VMAX

    fig1, axes = plt.subplots(3, 3, figsize=(16, 15))
    # --- replace your extent setup with this ---
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


    im = axes[0, 0].imshow(tru[:, :, iz].T, origin='lower', extent=extent_xy, aspect='auto', vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 0].set_title(f"True Δρ XY @ z≈{z1d[iz]:.0f} m")
    fig1.colorbar(im, ax=axes[0, 0], label='kg/m³', fraction=0.046, pad=0.04)
    im = axes[0, 1].imshow(tru[:, iy, :].T, origin='upper', extent=extent_xz, aspect='auto', vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 1].set_title(f"True Δρ XZ @ y≈{y1d[iy]:.0f} m")
    im = axes[0, 2].imshow(tru[ix, :, :].T, origin='upper', extent=extent_yz, aspect='auto', vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 2].set_title(f"True Δρ YZ @ x≈{x1d[ix]:.0f} m")

    im = axes[1, 0].imshow(inv[:, :, iz].T, origin='lower', extent=extent_xy, aspect='auto', vmin=0, vmax=inv_max, cmap=CMAP)
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

    im = axes[1, 1].imshow(inv[:, iy, :].T, origin='upper', extent=extent_xz, aspect='auto', vmin=0, vmax=inv_max, cmap=CMAP)
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

    im = axes[1, 2].imshow(inv[ix, :, :].T, origin='upper', extent=extent_yz, aspect='auto', vmin=0, vmax=inv_max, cmap=CMAP)
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

    sc = axes[2, 0].scatter(obs_x, obs_y, c=obs_mgal, s=80, cmap=CMAP, vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[2, 0].set_title('Observed gz (mGal)')
    fig1.colorbar(sc, ax=axes[2, 0], fraction=0.046, pad=0.04)

    sc = axes[2, 1].scatter(obs_x, obs_y, c=pre_mgal, s=80, cmap=CMAP, vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[2, 1].set_title('Predicted gz (mGal)')
    fig1.colorbar(sc, ax=axes[2, 1], fraction=0.046, pad=0.04)

    vmax_res = np.abs(res_mgal).max()
    sc = axes[2, 2].scatter(obs_x, obs_y, c=res_mgal, s=80, cmap=CMAP, vmin=-vmax_res, vmax=vmax_res, marker='o', edgecolors='none')
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
