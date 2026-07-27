"""PySCF-free Becke integration grids (numpy / cupy).

PySCF 없이 동일한 Becke grid를 생성한다.
Lebedev angular grid는 사전 캐시(`data/lebedev_grids.npz`)에서 로드.
GPU(cupy) 가속은 Becke partition에서 활용.

Usage:
    from grids import BeckeGrids
    grid = BeckeGrids(Z, coords_bohr).build(level=3)
    # grid.coords: (N, 3) in Bohr
    # grid.weights: (N,)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np


# ═════════════════════════════════════════════════════════════════════
# Constants (from PySCF, embedded for zero-dependency runtime)
# ═════════════════════════════════════════════════════════════════════

BOHR = 0.52917721092  # Angstrom per Bohr

# Bragg-Slater radii in Bohr, indexed by Z (0=dummy).
# Source: pyscf.data.radii.BRAGG
BRAGG_RADII = np.array([
    3.7794503594, 0.6614041436, 2.6456165744,  # dummy, H, He
    2.7401028806, 1.9842124308, 1.6062672059, 1.3228082872, 1.2283219810,  # Li-N
    1.1338356747, 0.9448630623, 2.8345891868,  # O, F, Ne
    3.4015070242, 2.8345891868, 2.3621576557, 2.0786987370, 1.8897261246,  # Na-P
    1.8897261246, 1.8897261246, 3.4015070242,  # S, Cl, Ar
    4.1573974740, 3.4015070242, 3.0235617993, 2.6456165744, 2.5511302682,  # K-V
    2.6456165744, 2.6456165744, 2.6456165744, 2.5511302682, 2.5511302682,  # Cr-Ni
    2.5511302682, 2.5511302682, 2.4566439619, 2.3621576557, 2.1731850432,  # Cu-As
    2.1731850432, 2.1731850432,  # Se, Br
    3.5904796367, 4.4408563927, 3.7794522491, 3.4015070242, 2.9290754931,  # Kr-Nb
    2.7401028806, 2.7401028806, 2.5511302682, 2.4566439619, 2.5511302682,  # Mo-Pd
    2.6456165744, 3.0235617993, 2.9290754931, 2.9290754931, 2.7401028806,  # Ag-Sb
    2.7401028806, 2.6456165744, 2.6456165744,  # Te, I, Xe
    3.9684248616, 4.9132879239, 4.0629111678, 3.6849659429,  # Cs-Ce
    3.4959933304, 3.4959933304, 3.4959933304, 3.4959933304, 3.4959933304,  # Pr-Eu
    3.4959933304, 3.4015070242, 3.3070207180, 3.3070207180, 3.3070207180,  # Gd-Ho
    3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,  # Er-Lu
    2.9290754931, 2.7401028806, 2.5511302682, 2.5511302682, 2.4566439619,  # Hf-Os
    2.5511302682, 2.5511302682, 2.5511302682, 2.8345891868,  # Ir-Hg
    3.5904796367, 3.4015070242, 3.0235617993, 3.5904796367, 2.7401028806,  # Tl-At
    3.9684248616, 3.4015070242, 4.0629111678, 3.6849659429,  # Rn-Ac
    3.4015070242, 3.4015070242, 3.3070207180, 3.3070207180, 3.3070207180,  # Th-Cm
    3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180, 3.3070207180,  # Bk-No
    3.3070207180,  # Lr (Z=103 → index 103, but array has 101 elements up to Fm=100)
])

# Treutler-Ahlrichs xi values, indexed by Z (0=dummy).
_TREUTLER_XI = np.array([
    1.0, 0.8, 0.9,
    1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,
    1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,
    1.5, 1.4, 1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1,
    1.1, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9,
    2.0, 1.7, 1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5,
    1.5, 1.3, 1.2, 1.2, 1.15, 1.15, 1.15,
    2.5, 2.2, 2.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    2.5, 2.1, 3.685,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
])

# Default grid sizes per period (rows) and level 0-6 (columns).
# Period: 0=dummy, 1=H/He, 2=Li-Ne, 3=Na-Ar, ... 9=superheavy
RAD_GRIDS = np.array([
    [10,  15,  20,  30,  35,  40,  50],
    [30,  40,  50,  60,  65,  70,  75],
    [40,  60,  65,  75,  80,  85,  90],
    [50,  75,  80,  90,  95, 100, 105],
    [60,  90,  95, 105, 110, 115, 120],
    [70, 105, 110, 120, 125, 130, 135],
    [80, 120, 125, 135, 140, 145, 150],
    [90, 135, 140, 150, 155, 160, 165],
    [100, 150, 155, 165, 170, 175, 180],
    [200, 200, 200, 200, 200, 200, 200],
])

# Angular order (Lebedev algebraic order) per period and level.
ANG_ORDER = np.array([
    [11, 15, 17, 17, 17, 17, 17],
    [17, 23, 23, 23, 23, 23, 23],
    [23, 29, 29, 29, 29, 29, 29],
    [29, 29, 35, 35, 35, 35, 35],
    [35, 41, 41, 41, 41, 41, 41],
    [41, 47, 47, 47, 47, 47, 47],
    [47, 53, 53, 53, 53, 53, 53],
    [53, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59],
    [65, 65, 65, 65, 65, 65, 65],
])

# Lebedev: algebraic order → number of points
LEBEDEV_ORDER = {
    0: 1, 3: 6, 5: 14, 7: 26, 9: 38, 11: 50, 13: 74, 15: 86,
    17: 110, 19: 146, 21: 170, 23: 194, 25: 230, 27: 266, 29: 302,
    31: 350, 35: 434, 41: 590, 47: 770, 53: 974, 59: 1202, 65: 1454,
    71: 1730, 77: 2030, 83: 2354, 89: 2702, 95: 3074, 101: 3470,
    107: 3890, 113: 4334, 119: 4802, 125: 5294, 131: 5810,
}
LEBEDEV_NGRID = np.array(list(LEBEDEV_ORDER.values()))

# Period lookup for Z
def _period(z: int) -> int:
    if z <= 2: return 1
    if z <= 10: return 2
    if z <= 18: return 3
    if z <= 36: return 4
    if z <= 54: return 5
    if z <= 86: return 6
    if z <= 118: return 7
    return 8


# ═════════════════════════════════════════════════════════════════════
# Lebedev angular grid (from cache)
# ═════════════════════════════════════════════════════════════════════

_LEBEDEV_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "lebedev_grids.npz"
_lebedev_data = None


def _load_lebedev_data():
    global _lebedev_data
    if _lebedev_data is None:
        _lebedev_data = np.load(str(_LEBEDEV_CACHE_PATH), allow_pickle=False)
    return _lebedev_data


@lru_cache(maxsize=50)
def lebedev_grid(n_points: int) -> np.ndarray:
    """Lebedev angular grid.

    Returns:
        (n_points, 4): [x, y, z, weight], weights sum to 1.
    """
    data = _load_lebedev_data()
    key = f"grid_{n_points}"
    if key not in data:
        raise ValueError(f"Lebedev grid with {n_points} points not in cache. "
                         f"Available: {[int(k.split('_')[1]) for k in data if k.startswith('grid_')]}")
    return data[key].copy()


# ═════════════════════════════════════════════════════════════════════
# Radial grid: Treutler-Ahlrichs M4
# ═════════════════════════════════════════════════════════════════════

def treutler_ahlrichs(n: int, charge: int) -> tuple[np.ndarray, np.ndarray]:
    """Treutler-Ahlrichs M4 radial grid (vectorized).

    Args:
        n: number of radial points
        charge: nuclear charge (atomic number Z)

    Returns:
        (r, dr): radial points (Bohr) and integration weights, shape (n,).
        Sorted from nucleus outward.
    """
    xi = _TREUTLER_XI[min(charge, len(_TREUTLER_XI) - 1)]
    step = np.pi / (n + 1)
    ln2 = xi / np.log(2.0)

    i = np.arange(1, n + 1, dtype=np.float64)
    angle = i * step
    x = np.cos(angle)

    log_term = np.log((1.0 - x) / 2.0)
    r = -ln2 * (1.0 + x) ** 0.6 * log_term
    dr = step * np.sin(angle) * ln2 * (1.0 + x) ** 0.6 * (
        -0.6 / (1.0 + x) * log_term + 1.0 / (1.0 - x)
    )

    return r[::-1].copy(), dr[::-1].copy()


# ═════════════════════════════════════════════════════════════════════
# Pruning
# ═════════════════════════════════════════════════════════════════════

def nwchem_prune(nuc: int, rads: np.ndarray, n_ang: int) -> np.ndarray:
    """NWChem pruning: angular grid sizes per radial shell.

    Args:
        nuc: atomic number
        rads: radial grid points (Bohr)
        n_ang: default angular grid size

    Returns:
        (n_rad,) int array of angular grid sizes per radial shell.
    """
    alphas = np.array([
        [0.25, 0.5, 1.0, 4.5],
        [0.1667, 0.5, 0.9, 3.5],
        [0.1, 0.4, 0.8, 2.5],
    ])

    leb_ngrid = LEBEDEV_NGRID[4:]  # starts at 38

    if n_ang < 50:
        return np.full(len(rads), n_ang, dtype=np.int64)
    elif n_ang == 50:
        leb_l = np.array([1, 2, 2, 2, 1])
    else:
        idx = int(np.where(leb_ngrid == n_ang)[0][0])
        leb_l = np.array([1, 3, idx - 1, idx, idx - 1])

    r_atom = BRAGG_RADII[min(nuc, len(BRAGG_RADII) - 1)] + 1e-200

    if nuc <= 2:
        alpha_row = alphas[0]
    elif nuc <= 10:
        alpha_row = alphas[1]
    else:
        alpha_row = alphas[2]

    place = ((rads / r_atom).reshape(-1, 1) > alpha_row).sum(axis=1)
    angs = leb_ngrid[leb_l[place]]
    return angs


# ═════════════════════════════════════════════════════════════════════
# Atomic grid generation
# ═════════════════════════════════════════════════════════════════════

def _default_rad_ang(nuc: int, level: int = 3) -> tuple[int, int]:
    """Default (n_rad, n_ang) for given nucleus and level."""
    # PySCF convention: RAD_GRIDS[level, period], ANG_ORDER[level, period]
    tab = np.array([2, 10, 18, 36, 54, 86, 118])
    period = int((nuc > tab).sum())
    lvl = min(level, RAD_GRIDS.shape[0] - 1)
    period = min(period, RAD_GRIDS.shape[1] - 1)
    n_rad = int(RAD_GRIDS[lvl, period])
    ang_order = int(ANG_ORDER[lvl, period])
    n_ang = LEBEDEV_ORDER.get(ang_order, 302)
    return n_rad, n_ang


def gen_atomic_grids(
    atomic_numbers: np.ndarray,
    level: int = 3,
    prune: Optional[str] = "nwchem",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-atom-type grids (coords relative to atom center, volumes).

    Returns:
        {Z: (coords [M, 3], weights [M])}, coords in Bohr.
    """
    unique_z = sorted(set(int(z) for z in atomic_numbers))
    result = {}

    for z in unique_z:
        n_rad, n_ang = _default_rad_ang(z, level)
        r, dr = treutler_ahlrichs(n_rad, z)
        rad_weight = 4.0 * np.pi * r ** 2 * dr

        # Pruning: angular grid size per radial shell
        if prune == "nwchem":
            angs = nwchem_prune(z, r, n_ang)
        else:
            angs = np.full(n_rad, n_ang, dtype=np.int64)

        # Group by angular grid size for efficiency
        coords_list = []
        weights_list = []
        unique_angs = sorted(set(angs.tolist()))

        for ang_n in unique_angs:
            idx = np.where(angs == ang_n)[0]
            ang_grid = lebedev_grid(ang_n)  # (ang_n, 4)
            ang_coords = ang_grid[:, :3]  # (ang_n, 3)
            ang_weights = ang_grid[:, 3]  # (ang_n,)

            # Process in batches of 12 radial shells (matching PySCF)
            for i0 in range(0, len(idx), 12):
                batch_idx = idx[i0:i0 + 12]
                # einsum('i,jk->jik', rad[batch], grid[:,:3]) → (ang_n, n_batch, 3)
                batch_coords = np.einsum('i,jk->jik', r[batch_idx], ang_coords).reshape(-1, 3)
                batch_weights = np.einsum('i,j->ji', rad_weight[batch_idx], ang_weights).ravel()
                coords_list.append(batch_coords)
                weights_list.append(batch_weights)

        result[z] = (np.vstack(coords_list), np.concatenate(weights_list))

    return result


# ═════════════════════════════════════════════════════════════════════
# Becke partition
# ═════════════════════════════════════════════════════════════════════

def _becke_switch(g, xp=np):
    """Becke switching function: 3 iterations of (3-g²)*g*0.5."""
    g = (3.0 - g * g) * g * 0.5
    g = (3.0 - g * g) * g * 0.5
    g = (3.0 - g * g) * g * 0.5
    return g


def _treutler_radii_adjust(atomic_numbers: np.ndarray) -> np.ndarray:
    """Compute pairwise radii adjustment matrix a[i,j].

    Returns:
        (n_atoms, n_atoms) array, clipped to [-0.5, 0.5].
    """
    z_clipped = np.minimum(atomic_numbers, len(BRAGG_RADII) - 1)
    rad = np.sqrt(BRAGG_RADII[z_clipped]) + 1e-200
    rr = rad[:, None] / rad[None, :]
    a = 0.25 * (rr.T - rr)
    return np.clip(a, -0.5, 0.5)


def becke_partition(
    atom_coords: np.ndarray,
    atomic_numbers: np.ndarray,
    atom_grids_tab: dict[int, tuple[np.ndarray, np.ndarray]],
    radii_adjust: bool = True,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Becke partition weights.

    전체 grid를 한번에 조립한 뒤 Becke partition을 적용.
    GPU 모드에서는 전체 연산을 GPU에서 수행 (CPU↔GPU 전송 최소화).

    Args:
        atom_coords: (n_atoms, 3) atom positions in Bohr
        atomic_numbers: (n_atoms,) atomic numbers
        atom_grids_tab: {Z: (coords, weights)} from gen_atomic_grids
        radii_adjust: use Treutler atomic radii adjustment
        use_gpu: use cupy for GPU acceleration

    Returns:
        (coords [N, 3], weights [N]) in Bohr
    """
    xp = np
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            pass

    n_atoms = len(atomic_numbers)

    # ── 1. Assemble all grid coords and volumes, track atom ownership ──
    all_coords_list = []
    all_vol_list = []
    atom_slices = []  # (start, end) for each atom
    offset = 0

    for ia in range(n_atoms):
        z = int(atomic_numbers[ia])
        atom_grid_coords, atom_grid_vol = atom_grids_tab[z]
        abs_coords = atom_grid_coords + atom_coords[ia]
        n_grid = len(atom_grid_vol)
        all_coords_list.append(abs_coords)
        all_vol_list.append(atom_grid_vol)
        atom_slices.append((offset, offset + n_grid))
        offset += n_grid

    all_coords = np.vstack(all_coords_list)  # (N_total, 3)
    all_vol = np.concatenate(all_vol_list)    # (N_total,)
    n_total = len(all_vol)

    # ── 2. Move to GPU if needed (single transfer) ──
    coords_xp = xp.asarray(all_coords) if xp is not np else all_coords
    vol_xp = xp.asarray(all_vol) if xp is not np else all_vol
    atom_coords_xp = xp.asarray(atom_coords) if xp is not np else atom_coords

    # ── 3. Atom-atom distances (small, stays on CPU is fine) ──
    atm_diff = atom_coords[:, None, :] - atom_coords[None, :, :]
    atm_dist = np.sqrt((atm_diff ** 2).sum(axis=2))  # (natm, natm)

    # Radii adjustment matrix
    a_mat = _treutler_radii_adjust(atomic_numbers) if radii_adjust else None

    # ── 4. Grid-to-atom distances: (natm, N_total) ──
    grid_dist = xp.sqrt(((coords_xp[None, :, :] - atom_coords_xp[:, None, :]) ** 2).sum(axis=2))

    # ── 5. Becke partition: all grid points at once ──
    pbecke = xp.ones((n_atoms, n_total), dtype=np.float64)

    for i in range(n_atoms):
        for j in range(i):
            g = (grid_dist[i] - grid_dist[j]) / atm_dist[i, j]

            if a_mat is not None:
                g = g + a_mat[i, j] * (1.0 - g * g)

            g = _becke_switch(g, xp)

            pbecke[i] *= 0.5 * (1.0 - g)
            pbecke[j] *= 0.5 * (1.0 + g)

    # ── 6. Normalize and apply per-atom volumes ──
    norm = pbecke.sum(axis=0)  # (N_total,)

    all_weights = xp.empty(n_total, dtype=np.float64)
    for ia in range(n_atoms):
        s, e = atom_slices[ia]
        all_weights[s:e] = vol_xp[s:e] * pbecke[ia, s:e] / norm[s:e]

    # ── 7. Transfer back ──
    if xp is not np:
        return xp.asnumpy(coords_xp), xp.asnumpy(all_weights)
    return all_coords, np.asarray(all_weights)


# ═════════════════════════════════════════════════════════════════════
# Main API
# ═════════════════════════════════════════════════════════════════════

class BeckeGrids:
    """PySCF-compatible Becke integration grid, no PySCF dependency.

    Example:
        grid = BeckeGrids(
            atomic_numbers=np.array([8, 1, 1]),
            atom_coords_bohr=np.array([[0,0,0], [0,1.43,1.11], [0,-1.43,1.11]]),
        )
        grid.build(level=3)
        # grid.coords: (N, 3) Bohr
        # grid.weights: (N,)
    """

    def __init__(
        self,
        atomic_numbers: np.ndarray,
        atom_coords_bohr: np.ndarray,
        use_gpu: bool = False,
    ):
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
        self.atom_coords = np.asarray(atom_coords_bohr, dtype=np.float64)
        self.use_gpu = use_gpu
        self.coords: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    def build(
        self,
        level: int = 3,
        prune: Optional[str] = "nwchem",
        radii_adjust: bool = True,
        alignment: int = 8,
    ) -> "BeckeGrids":
        """Build the grid.

        Args:
            alignment: pad grid to multiple of this (PySCF default=8, 0=no padding)

        Returns:
            self (for chaining)
        """
        atom_grids_tab = gen_atomic_grids(self.atomic_numbers, level=level, prune=prune)
        self.coords, self.weights = becke_partition(
            self.atom_coords,
            self.atomic_numbers,
            atom_grids_tab,
            radii_adjust=radii_adjust,
            use_gpu=self.use_gpu,
        )

        # Alignment padding (PySCF pads to multiples of 8)
        if alignment > 0:
            n = len(self.weights)
            remainder = n % alignment
            if remainder > 0:
                pad = alignment - remainder
                pad_coords = np.full((pad, 3), 1e-4)
                pad_weights = np.zeros(pad)
                self.coords = np.vstack([self.coords, pad_coords])
                self.weights = np.concatenate([self.weights, pad_weights])

        return self

    @property
    def size(self) -> int:
        return len(self.weights) if self.weights is not None else 0


def becke_grid_from_molecule(
    mol: "Molecule",
    level: int = 3,
    prune: Optional[str] = "nwchem",
    use_gpu: bool = False,
) -> "DensityGrid":
    """Molecule → DensityGrid (Angstrom), PySCF 불필요.

    기존 density.py의 becke_grid()를 대체.
    """
    from density import DensityGrid

    ANG_TO_BOHR = 1.0 / BOHR
    coords_bohr = mol.positions * ANG_TO_BOHR

    grid = BeckeGrids(mol.atomic_numbers, coords_bohr, use_gpu=use_gpu)
    grid.build(level=level, prune=prune)

    # Bohr → Angstrom
    coords_ang = grid.coords * BOHR

    return DensityGrid(
        coords=coords_ang,
        weights=grid.weights,
        rho=np.zeros(grid.size),
        grid_type="becke",
    )
