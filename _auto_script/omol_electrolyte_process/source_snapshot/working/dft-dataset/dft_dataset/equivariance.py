"""Equivariance testing utilities.

SO(3) 회전에 대한 invariance/equivariance 검증 도구.
- Molecule 회전 (positions + orbital matrices)
- Wigner-D 행렬 계산 (real spherical harmonics, 임의 l)
- Invariance/equivariance test helpers

Usage:
    from equivariance import random_rotation, rotate_molecule, test_equivariance

    R = random_rotation()
    mol_rot = rotate_molecule(mol, R)
    test_equivariance(mol.hamiltonian, mol_rot.hamiltonian, mol, R)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .conventions import SHELL_TO_L, get_m_order


# ═════════════════════════════════════════════════════════════════════
# Rotation generation
# ═════════════════════════════════════════════════════════════════════

def random_rotation(seed: Optional[int] = None) -> np.ndarray:
    """Random SO(3) rotation matrix (uniform Haar measure).

    Returns:
        (3, 3) orthogonal matrix with det=+1.
    """
    rng = np.random.default_rng(seed)
    return Rotation.random(random_state=rng.integers(2**31)).as_matrix()


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Axis-angle → rotation matrix.

    Args:
        axis: (3,) rotation axis (will be normalized)
        angle: rotation angle in radians
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(angle * axis).as_matrix()


# ═════════════════════════════════════════════════════════════════════
# Wigner-D for real spherical harmonics
# ═════════════════════════════════════════════════════════════════════

def _real_sph_harm(l: int, coords: np.ndarray) -> np.ndarray:
    """Evaluate real spherical harmonics Y_l^m at given Cartesian coords.

    Args:
        l: angular momentum
        coords: (N, 3) Cartesian coordinates (will be normalized to unit sphere)

    Returns:
        (N, 2l+1) array, columns ordered as m = -l, -l+1, ..., +l
    """
    try:
        from scipy.special import sph_harm_y  # scipy >= 1.17
        def _ylm(m, l, theta, phi):
            return sph_harm_y(l, m, theta, phi)
    except ImportError:
        from scipy.special import sph_harm  # scipy < 1.17
        def _ylm(m, l, theta, phi):
            return sph_harm(m, l, phi, theta)

    # Normalize to unit sphere
    r = np.linalg.norm(coords, axis=1, keepdims=True)
    r = np.maximum(r, 1e-30)
    xyz = coords / r
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)

    N = len(coords)
    Y_real = np.zeros((N, 2 * l + 1))

    for idx, m in enumerate(range(-l, l + 1)):
        if m < 0:
            Y_complex = _ylm(abs(m), l, theta, phi)
            Y_real[:, idx] = np.sqrt(2) * np.imag(Y_complex) * (-1) ** m
        elif m == 0:
            Y_real[:, idx] = np.real(_ylm(0, l, theta, phi))
        else:
            Y_complex = _ylm(m, l, theta, phi)
            Y_real[:, idx] = np.sqrt(2) * np.real(Y_complex) * (-1) ** m

    return Y_real


def wigner_d_real(l: int, R: np.ndarray) -> np.ndarray:
    """Wigner-D matrix for real spherical harmonics under rotation R.

    Standard m-ordering: m = -l, -l+1, ..., +l (e3nn convention).

    Args:
        l: angular momentum quantum number
        R: (3, 3) rotation matrix

    Returns:
        (2l+1, 2l+1) matrix D such that Y_l(R·r) = D @ Y_l(r)
    """
    if l == 0:
        return np.array([[1.0]])

    n = 2 * l + 1

    # Generate n well-conditioned test directions on unit sphere
    # Use Fibonacci sphere for good distribution
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n) + 0.5
    theta = np.arccos(1 - 2 * i / (n + 1))
    phi = 2 * np.pi * i / golden
    test_dirs = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # Y_l at original and rotated directions
    Y_orig = _real_sph_harm(l, test_dirs)        # (n, 2l+1)
    Y_rot = _real_sph_harm(l, test_dirs @ R.T)   # (n, 2l+1)

    # D = Y_rot @ pinv(Y_orig)
    # Since Y_l(R·r) = D @ Y_l(r), and Y_rot[i,:] = Y_l(R·r_i) = D @ Y_orig[i,:]
    # → Y_rot = Y_orig @ D^T → D = (pinv(Y_orig) @ Y_rot)^T
    D = np.linalg.lstsq(Y_orig, Y_rot, rcond=None)[0].T

    return D


def wigner_d_convention(l: int, R: np.ndarray, convention: str = "e3nn") -> np.ndarray:
    """Wigner-D in specified m-ordering convention.

    Args:
        l: angular momentum
        R: (3, 3) rotation matrix
        convention: "e3nn" (m ascending) or "pyscf"

    Returns:
        (2l+1, 2l+1) matrix
    """
    D_std = wigner_d_real(l, R)  # standard m = -l..+l

    if convention == "e3nn":
        return D_std

    # Reorder from standard to target convention
    std_m = list(range(-l, l + 1))
    tgt_m = get_m_order(convention, l)

    # Permutation: tgt_m[i] → position of that m in std_m
    perm = [std_m.index(m) for m in tgt_m]
    return D_std[np.ix_(perm, perm)]


# ═════════════════════════════════════════════════════════════════════
# Full AO-basis rotation matrix
# ═════════════════════════════════════════════════════════════════════

def build_ao_rotation(
    atomic_numbers: np.ndarray,
    atom_to_shells: dict[int, str],
    R: np.ndarray,
    convention: str = "pyscf",
) -> np.ndarray:
    """Build block-diagonal Wigner-D for full AO basis.

    H_rot = D @ H_orig @ D^T

    Args:
        atomic_numbers: (N,) atomic numbers
        atom_to_shells: {Z: shell_string}, e.g. {1: "ssp", 8: "sssppd"}
        R: (3, 3) rotation matrix
        convention: m-ordering convention of the matrices

    Returns:
        (nao, nao) block-diagonal rotation matrix
    """
    blocks = []
    for z in atomic_numbers:
        shells = atom_to_shells[int(z)]
        for label in shells:
            l = SHELL_TO_L[label]
            D_l = wigner_d_convention(l, R, convention)
            blocks.append(D_l)

    return _block_diag(blocks)


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    """Block diagonal matrix from list of square blocks."""
    sizes = [b.shape[0] for b in blocks]
    n = sum(sizes)
    out = np.zeros((n, n), dtype=np.float64)
    offset = 0
    for b, s in zip(blocks, sizes):
        out[offset:offset + s, offset:offset + s] = b
        offset += s
    return out


# ═════════════════════════════════════════════════════════════════════
# Molecule rotation
# ═════════════════════════════════════════════════════════════════════

def rotate_positions(positions: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate atomic positions: pos_rot = pos @ R^T."""
    return positions @ R.T


def rotate_molecule(mol: "Molecule", R: np.ndarray) -> "Molecule":
    """Rotate a Molecule: positions + all orbital matrices.

    Positions are rotated geometrically.
    H, S, D, orbital_coeffs are rotated by Wigner-D:
        M_rot = D_full @ M_orig @ D_full^T

    Args:
        mol: source Molecule (unchanged)
        R: (3, 3) rotation matrix

    Returns:
        new Molecule with rotated geometry and matrices
    """
    from molecule import Molecule

    new_pos = rotate_positions(mol.positions, R)

    # Rotate matrices if basis_info available
    new_H = mol.hamiltonian
    new_S = mol.overlap
    new_D = mol.density_matrix
    new_C = mol.orbital_coeffs

    if mol.basis_info is not None and mol.basis_info.atom_to_shells:
        D_full = build_ao_rotation(
            mol.atomic_numbers,
            mol.basis_info.atom_to_shells,
            R,
            mol.basis_info.convention,
        )
        if mol.hamiltonian is not None:
            new_H = D_full @ mol.hamiltonian @ D_full.T
        if mol.overlap is not None:
            new_S = D_full @ mol.overlap @ D_full.T
        if mol.density_matrix is not None:
            new_D = D_full @ mol.density_matrix @ D_full.T
        if mol.orbital_coeffs is not None:
            new_C = D_full @ mol.orbital_coeffs

    # Forces rotate as vectors
    new_forces = mol.forces @ R.T if mol.forces is not None else None

    return Molecule(
        atomic_numbers=mol.atomic_numbers,
        positions=new_pos,
        mol_id=mol.mol_id,
        formula=mol.formula,
        smiles=mol.smiles,
        charge=mol.charge,
        spin=mol.spin,
        energy=mol.energy,  # invariant
        forces=new_forces,
        dipole=mol.dipole @ R.T if mol.dipole is not None else None,
        homo=mol.homo,  # invariant
        lumo=mol.lumo,  # invariant
        zpve=mol.zpve,
        U0=mol.U0,
        U=mol.U,
        H=mol.H,
        G=mol.G,
        Cv=mol.Cv,
        polarizability=mol.polarizability,
        hamiltonian=new_H,
        overlap=new_S,
        density_matrix=new_D,
        orbital_energies=mol.orbital_energies,  # invariant
        orbital_coeffs=new_C,
        basis_info=mol.basis_info,
        xc=mol.xc,
    )


# ═════════════════════════════════════════════════════════════════════
# Test helpers
# ═════════════════════════════════════════════════════════════════════

def test_invariance(
    val_orig: np.ndarray,
    val_rot: np.ndarray,
    name: str = "value",
    atol: float = 1e-10,
) -> bool:
    """Check scalar/vector invariance under rotation.

    Returns True if passed.
    """
    diff = np.max(np.abs(np.asarray(val_orig) - np.asarray(val_rot)))
    passed = diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_diff = {diff:.2e} (tol={atol:.0e})")
    return passed


def test_equivariance(
    M_orig: np.ndarray,
    M_rot: np.ndarray,
    mol: "Molecule",
    R: np.ndarray,
    name: str = "matrix",
    atol: float = 1e-10,
) -> bool:
    """Check matrix equivariance: M_rot ≈ D @ M_orig @ D^T.

    Args:
        M_orig: matrix at original geometry
        M_rot: matrix at rotated geometry
        mol: Molecule (for basis_info)
        R: rotation matrix used
        name: label for printing
        atol: tolerance

    Returns True if passed.
    """
    D_full = build_ao_rotation(
        mol.atomic_numbers,
        mol.basis_info.atom_to_shells,
        R,
        mol.basis_info.convention,
    )
    M_expected = D_full @ M_orig @ D_full.T
    diff = np.max(np.abs(M_rot - M_expected))
    passed = diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name} equivariance: max_diff = {diff:.2e} (tol={atol:.0e})")
    return passed
