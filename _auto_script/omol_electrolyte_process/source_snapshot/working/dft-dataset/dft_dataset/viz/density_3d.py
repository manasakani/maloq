"""3D density and orbital grid computation using PySCF.

Ported from dft-viz/components/density_3d.py, adapted to use Molecule objects.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from molecule import Molecule

from .structure import ANG2BOHR, BOHR2ANG, ELEMENT_MAP


def _build_pyscf_mol(mol: "Molecule"):
    """Build a PySCF Mole from a Molecule. Positions are in Angstrom."""
    from pyscf import gto

    atom_list = []
    for z, pos in zip(mol.atomic_numbers, mol.positions):
        sym = ELEMENT_MAP.get(int(z), f"X{int(z)}")
        atom_list.append((sym, pos.tolist()))

    basis = mol.basis or "def2-svp"
    spin = getattr(mol, "spin", 0) or 0
    charge = getattr(mol, "charge", 0) or 0
    pyscf_mol = gto.M(
        atom=atom_list, basis=basis, unit="Angstrom",
        spin=spin, charge=charge,
    )
    pyscf_mol.build()
    return pyscf_mol


def compute_overlap_pyscf(mol: "Molecule") -> np.ndarray:
    """Compute overlap matrix S via PySCF from molecular geometry."""
    pyscf_mol = _build_pyscf_mol(mol)
    return pyscf_mol.intor("int1e_ovlp")


def compute_density_grid_3d(
    mol: "Molecule",
    resolution: int = 30,
    dm: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Compute total electron density on a 3D cuboid grid.

    Args:
        mol: Molecule with hamiltonian (and optionally overlap, density_matrix).
        resolution: grid points per dimension (30=fast, 50=medium, 80=high).
        dm: density matrix. If None, computed from mol.

    Returns:
        (field_3d, grid_meta) where field_3d is (nx, ny, nz) array.
    """
    from pyscf.tools.cubegen import Cube
    from pyscf.dft import numint

    pyscf_mol = _build_pyscf_mol(mol)

    if dm is None:
        dm = _get_density_matrix_pyscf(mol, pyscf_mol)

    cc = Cube(pyscf_mol, nx=resolution, ny=resolution, nz=resolution)
    coords = cc.get_coords()
    ngrids = len(coords)
    blksize = min(8000, ngrids)

    field = np.empty(ngrids)
    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ip0 + blksize, ngrids)
        ao = pyscf_mol.eval_gto("GTOval", coords[ip0:ip1])
        field[ip0:ip1] = numint.eval_rho(pyscf_mol, ao, dm)

    field_3d = field.reshape(cc.nx, cc.ny, cc.nz)
    grid_meta = _cube_to_meta(cc)
    return field_3d, grid_meta


def compute_orbital_grid(
    mol: "Molecule",
    orbital_idx: int,
    resolution: int = 30,
    coeffs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Compute a single orbital on a 3D grid.

    Args:
        mol: Molecule.
        orbital_idx: which MO to compute (0-indexed).
        resolution: grid points per dimension.
        coeffs: MO coefficient matrix (nao, nao). If None, computed from mol.

    Returns:
        (field_3d, grid_meta)
    """
    from pyscf.tools.cubegen import Cube

    pyscf_mol = _build_pyscf_mol(mol)

    if coeffs is None:
        coeffs = _get_mo_coeffs_pyscf(mol, pyscf_mol)

    coeff_vec = coeffs[:, orbital_idx]

    cc = Cube(pyscf_mol, nx=resolution, ny=resolution, nz=resolution)
    coords = cc.get_coords()
    ngrids = len(coords)
    blksize = min(8000, ngrids)

    field = np.empty(ngrids)
    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ip0 + blksize, ngrids)
        ao = pyscf_mol.eval_gto("GTOval", coords[ip0:ip1])
        field[ip0:ip1] = ao @ coeff_vec

    field_3d = field.reshape(cc.nx, cc.ny, cc.nz)
    grid_meta = _cube_to_meta(cc)
    return field_3d, grid_meta


def _get_density_matrix_pyscf(mol: "Molecule", pyscf_mol) -> np.ndarray:
    """Get density matrix in PySCF convention, computing if needed."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from solvers import solve_gen_eigh, build_density_matrix
    from conventions import build_reorder_indices, reorder_matrix

    H = mol.hamiltonian
    atoms = mol.atomic_numbers
    basis_info = mol.basis_info

    # Reorder H from current convention to PySCF if needed
    if basis_info and basis_info.convention != "pyscf":
        idx = build_reorder_indices(
            atoms, basis_info.atom_to_shells,
            basis_info.convention, "pyscf", basis_info.angular_type,
        )
        H = reorder_matrix(H, idx)

    S = pyscf_mol.intor("int1e_ovlp")
    eigvals, eigvecs = solve_gen_eigh(H, S)
    n_electrons = int(np.sum(atoms)) - mol.charge
    dm = build_density_matrix(eigvecs, n_electrons)
    return dm


def _get_mo_coeffs_pyscf(mol: "Molecule", pyscf_mol) -> np.ndarray:
    """Get MO coefficients in PySCF convention."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from solvers import solve_gen_eigh
    from conventions import build_reorder_indices, reorder_matrix

    H = mol.hamiltonian
    atoms = mol.atomic_numbers
    basis_info = mol.basis_info

    if basis_info and basis_info.convention != "pyscf":
        idx = build_reorder_indices(
            atoms, basis_info.atom_to_shells,
            basis_info.convention, "pyscf", basis_info.angular_type,
        )
        H = reorder_matrix(H, idx)

    S = pyscf_mol.intor("int1e_ovlp")
    _, eigvecs = solve_gen_eigh(H, S)
    return eigvecs


def _cube_to_meta(cc) -> dict:
    """Extract grid metadata from PySCF Cube object."""
    return {
        "nx": int(cc.nx),
        "ny": int(cc.ny),
        "nz": int(cc.nz),
        "boxorig": cc.boxorig.tolist(),
        "box": cc.box.tolist(),
    }
