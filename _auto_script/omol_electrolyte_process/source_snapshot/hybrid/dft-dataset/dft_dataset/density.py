"""Electron density ρ(r) computation, storage, and sampling.

설계 원칙:
    - DensityGrid: numpy-only dataclass. 한번 계산하면 PySCF 없이 사용 가능.
    - compute_density: PySCF (CPU) 또는 gpu4pyscf (GPU)로 ρ(r) 계산.
    - Grid 생성: uniform_grid (numpy), becke_grid (PySCF).
    - Sampling/integration: numpy-only.

Usage:
    # PySCF로 계산 후 저장
    grid = becke_grid(pyscf_mol, level=3)
    density = compute_density(pyscf_mol, dm, grid)
    density.save_npz("rho.npz")

    # 이후 PySCF 없이 로드 + 사용
    density = DensityGrid.load_npz("rho.npz")
    pts, rho = density.sample(n=1000)
    n_elec = density.integrate()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ═════════════════════════════════════════════════════════════════════
# DensityGrid — PySCF 불필요 (numpy only)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class DensityGrid:
    """Electron density ρ(r) evaluated on a set of grid points.

    Attributes:
        coords: (N, 3) grid point coordinates, Angstrom
        weights: (N,) integration weights (Becke grid) or volume elements (uniform)
        rho: (N,) electron density values at each grid point
        grad_rho: (3, N) density gradient [∂ρ/∂x, ∂ρ/∂y, ∂ρ/∂z], optional
        grid_type: "becke" or "uniform"
    """
    coords: np.ndarray      # (N, 3), Angstrom
    weights: np.ndarray      # (N,)
    rho: np.ndarray          # (N,)
    grad_rho: Optional[np.ndarray] = None  # (3, N), optional
    grid_type: str = "becke"

    @property
    def n_points(self) -> int:
        return len(self.rho)

    # ── Integration ──────────────────────────────────────────────

    def integrate(self) -> float:
        """∫ρ(r)dr — should equal n_electrons."""
        return float(np.dot(self.rho, self.weights))

    def integrate_function(self, f: np.ndarray) -> float:
        """∫f(r)ρ(r)dr for arbitrary f values on the grid."""
        return float(np.dot(f * self.rho, self.weights))

    # ── Sampling ─────────────────────────────────────────────────

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        strategy: str = "density",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grid point를 sampling하여 (coords, rho) 반환.

        Args:
            n: number of points to sample
            seed: random seed
            strategy:
                "density" — ρ(r)에 비례하는 확률로 sampling (중요 영역 집중)
                "uniform" — 균등 sampling
                "weighted" — ρ(r)·w(r)에 비례 (적분 기여도 기준)

        Returns:
            (coords [n, 3], rho [n])
        """
        rng = np.random.default_rng(seed)
        n = min(n, self.n_points)

        if strategy == "density":
            prob = self.rho / self.rho.sum()
        elif strategy == "weighted":
            rho_w = self.rho * self.weights
            prob = rho_w / rho_w.sum()
        elif strategy == "uniform":
            prob = None
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use: density, uniform, weighted")

        idx = rng.choice(self.n_points, size=n, replace=False, p=prob)
        return self.coords[idx], self.rho[idx]

    # ── Statistics ───────────────────────────────────────────────

    def stats(self) -> dict:
        """Density 통계 요약."""
        return {
            "n_points": self.n_points,
            "n_electrons": self.integrate(),
            "rho_max": float(self.rho.max()),
            "rho_min": float(self.rho.min()),
            "rho_mean": float(self.rho.mean()),
            "grid_type": self.grid_type,
            "has_gradient": self.grad_rho is not None,
        }

    # ── I/O ──────────────────────────────────────────────────────

    def save_npz(self, path: str) -> None:
        data = {
            "coords": self.coords,
            "weights": self.weights,
            "rho": self.rho,
            "grid_type": np.array(self.grid_type),
        }
        if self.grad_rho is not None:
            data["grad_rho"] = self.grad_rho
        np.savez_compressed(path, **data)

    @classmethod
    def load_npz(cls, path: str) -> "DensityGrid":
        raw = np.load(path, allow_pickle=True)
        return cls(
            coords=raw["coords"],
            weights=raw["weights"],
            rho=raw["rho"],
            grad_rho=raw["grad_rho"] if "grad_rho" in raw else None,
            grid_type=str(raw["grid_type"]),
        )


# ═════════════════════════════════════════════════════════════════════
# Grid generators
# ═════════════════════════════════════════════════════════════════════

def uniform_grid(
    center: np.ndarray,
    extent: float = 5.0,
    n_per_dim: int = 50,
) -> DensityGrid:
    """Uniform cubic grid (PySCF 불필요).

    Args:
        center: (3,) grid center, Angstrom
        extent: half-width of cube, Angstrom
        n_per_dim: points per dimension

    Returns:
        DensityGrid with coords/weights set, rho=zeros (아직 계산 안 됨)
    """
    axes = [np.linspace(c - extent, c + extent, n_per_dim) for c in center]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    dv = (2 * extent / n_per_dim) ** 3
    weights = np.full(len(coords), dv)
    return DensityGrid(
        coords=coords,
        weights=weights,
        rho=np.zeros(len(coords)),
        grid_type="uniform",
    )


def becke_grid(pyscf_mol_or_Z=None, level: int = 3, prune="nwchem",
               *, atomic_numbers=None, positions_ang=None, use_gpu=False) -> DensityGrid:
    """Becke integration grid 생성.

    두 가지 방식으로 호출 가능:
        1. PySCF Mole: becke_grid(pyscf_mol, level=3)
        2. PySCF-free:  becke_grid(atomic_numbers=Z, positions_ang=pos, level=3)

    Args:
        pyscf_mol_or_Z: PySCF Mole object, 또는 None (keyword args 사용 시)
        level: grid density (0-9, 기본 3)
        prune: "nwchem" (기본) or None
        atomic_numbers: (N,) atomic numbers (PySCF-free 모드)
        positions_ang: (N,3) positions in Angstrom (PySCF-free 모드)
        use_gpu: GPU 가속 (PySCF-free 모드에서 Becke partition)

    Returns:
        DensityGrid with coords/weights set, rho=zeros
    """
    # PySCF-free path
    if atomic_numbers is not None or pyscf_mol_or_Z is None:
        from grids import BeckeGrids, BOHR
        if atomic_numbers is None:
            raise ValueError("atomic_numbers 필요")
        if positions_ang is None:
            raise ValueError("positions_ang 필요")
        coords_bohr = np.asarray(positions_ang) / BOHR
        grid = BeckeGrids(np.asarray(atomic_numbers), coords_bohr, use_gpu=use_gpu)
        grid.build(level=level, prune=prune)
        return DensityGrid(
            coords=grid.coords * BOHR,
            weights=grid.weights,
            rho=np.zeros(grid.size),
            grid_type="becke",
        )

    # PySCF path (legacy)
    from pyscf.dft import gen_grid

    grids = gen_grid.Grids(pyscf_mol_or_Z)
    grids.level = level
    if prune == "nwchem":
        grids.prune = gen_grid.nwchem_prune
    elif prune is None:
        grids.prune = None
    grids.build()

    BOHR_TO_ANG = 0.529177210903
    coords_ang = grids.coords * BOHR_TO_ANG

    return DensityGrid(
        coords=coords_ang,
        weights=grids.weights,
        rho=np.zeros(len(grids.weights)),
        grid_type="becke",
    )


# ═════════════════════════════════════════════════════════════════════
# Density computation (PySCF / gpu4pyscf)
# ═════════════════════════════════════════════════════════════════════

def compute_density(
    pyscf_mol,
    dm: np.ndarray,
    grid: DensityGrid,
    xctype: str = "LDA",
    use_gpu: bool = False,
) -> DensityGrid:
    """ρ(r)을 grid 위에서 계산.

    Args:
        pyscf_mol: PySCF Mole object
        dm: (nao, nao) density matrix
        grid: DensityGrid (coords/weights만 필요, rho는 덮어씀)
        xctype: "LDA" (ρ only) or "GGA" (ρ + ∇ρ)
        use_gpu: True면 gpu4pyscf 사용 (CUDA 필요)

    Returns:
        같은 grid에 rho (+ grad_rho for GGA) 채워서 반환
    """
    # Coords를 Bohr로 변환 (PySCF 내부 단위)
    ANG_TO_BOHR = 1.8897259886
    coords_bohr = grid.coords * ANG_TO_BOHR

    if use_gpu:
        rho = _compute_density_gpu(pyscf_mol, dm, coords_bohr, xctype)
    else:
        rho = _compute_density_cpu(pyscf_mol, dm, coords_bohr, xctype)

    if xctype == "LDA":
        grid.rho = rho
        grid.grad_rho = None
    else:  # GGA or mGGA
        grid.rho = rho[0]
        grid.grad_rho = rho[1:4]  # (3, N)

    return grid


def _compute_density_cpu(pyscf_mol, dm, coords_bohr, xctype):
    """CPU (PySCF) density evaluation."""
    from pyscf.dft import numint

    deriv = 0 if xctype == "LDA" else 1
    ao = numint.eval_ao(pyscf_mol, coords_bohr, deriv=deriv)
    return numint.eval_rho(pyscf_mol, ao, dm, xctype=xctype)


def _compute_density_gpu(pyscf_mol, dm, coords_bohr, xctype):
    """GPU (gpu4pyscf) density evaluation.

    Note: gpu4pyscf eval_rho expects (nao, ngrids) shape,
    but eval_ao returns (ngrids, nao) — transpose 필요.
    gpu4pyscf가 없으면 CPU fallback.
    """
    try:
        import cupy as cp
        from gpu4pyscf.dft import numint as gpu_numint

        coords_gpu = cp.asarray(coords_bohr)
        dm_gpu = cp.asarray(dm)

        deriv = 0 if xctype == "LDA" else 1
        ao = gpu_numint.eval_ao(pyscf_mol, coords_gpu, deriv=deriv)

        # gpu4pyscf eval_rho expects (nao, ngrids), eval_ao gives (ngrids, nao)
        if xctype == "LDA":
            ao_t = ao.T
        else:
            ao_t = cp.asarray([a.T for a in ao])

        rho = gpu_numint.eval_rho(pyscf_mol, ao_t, dm_gpu, xctype=xctype)
        return cp.asnumpy(rho)
    except ImportError:
        import warnings
        warnings.warn("gpu4pyscf not available, falling back to CPU")
        return _compute_density_cpu(pyscf_mol, dm, coords_bohr, xctype)


# ═════════════════════════════════════════════════════════════════════
# Density from Molecule (convenience)
# ═════════════════════════════════════════════════════════════════════

def density_from_molecule(
    mol: "Molecule",
    grid_type: str = "becke",
    level: int = 3,
    xctype: str = "LDA",
    use_gpu: bool = False,
    n_per_dim: int = 50,
    extent: float = 5.0,
) -> DensityGrid:
    """Molecule 객체에서 직접 density를 계산.

    PySCF Mole 객체를 내부에서 재구성하여 basis function 평가.

    Args:
        mol: Molecule dataclass (atomic_numbers, positions, density_matrix 필요)
        grid_type: "becke" or "uniform"
        level: Becke grid level (grid_type="becke" 시)
        xctype: "LDA" or "GGA"
        use_gpu: GPU 사용 여부
        n_per_dim: uniform grid일 때 축당 점 개수
        extent: uniform grid 반폭 (Angstrom)

    Returns:
        DensityGrid with rho computed
    """
    if mol.density_matrix is None:
        raise ValueError("density_matrix가 필요합니다. mol.compute_density_matrix() 먼저 호출하세요.")
    if mol.basis_info is None:
        raise ValueError("basis_info가 필요합니다.")

    from pyscf import gto

    # PySCF Mole 재구성
    atom_str = "; ".join(
        f"{z} {x:.10f} {y:.10f} {z_coord:.10f}"
        for z, (x, y, z_coord) in zip(mol.atomic_numbers, mol.positions)
    )
    pyscf_mol = gto.M(
        atom=atom_str,
        basis=mol.basis_info.name,
        charge=mol.charge,
        spin=mol.spin,
        unit="ang",
    )

    # Convention 확인: PySCF convention이어야 basis function과 일치
    if mol.convention != "pyscf":
        dm = mol.to_convention("pyscf").density_matrix
    else:
        dm = mol.density_matrix

    # Grid 생성
    if grid_type == "becke":
        grid = becke_grid(pyscf_mol, level=level)
    else:
        center = mol.positions.mean(axis=0)
        grid = uniform_grid(center, extent=extent, n_per_dim=n_per_dim)

    return compute_density(pyscf_mol, dm, grid, xctype=xctype, use_gpu=use_gpu)
