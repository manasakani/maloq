"""End-to-end DFT property extraction pipeline.

Connects torch integrals, eigensolvers, and the Molecule dataclass
into a unified workflow for both PySCF reference and ML-predicted Hamiltonians.

Usage:
    from pipeline import properties_from_hamiltonian, compute_integrals_for_molecule

    # From ML-predicted H + torch-computed S
    props = properties_from_hamiltonian(H_pred, S, n_electrons=10)

    # Full workflow from Molecule
    mol = Molecule(atomic_numbers=Z, positions=pos, basis_info=...)
    mol = compute_integrals_for_molecule(mol, basis='def2-svp')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .solvers import (
    build_density_matrix,
    build_density_matrix_torch,
    solve_gen_eigh,
    solve_gen_eigh_torch,
)


# ── Core property extraction ────────────────────────────────────────

HA_TO_EV = 27.211386245988


def properties_from_hamiltonian(
    H: Tensor | np.ndarray,
    S: Tensor | np.ndarray,
    n_electrons: int | tuple[int, int],
    backend: str = "auto",
) -> dict:
    """Extract all electronic properties from a Hamiltonian matrix.

    Works with both numpy (single molecule) and torch (batched GPU).

    Args:
        H: Hamiltonian/Fock matrix.
            numpy: (nao, nao) or (2, nao, nao) for UKS.
            torch: (B, nao, nao) or (2, B, nao, nao) for UKS.
        S: Overlap matrix, same shape conventions.
        n_electrons: Total electrons (int), or (n_alpha, n_beta) for UKS.
        backend: "numpy", "torch", or "auto" (detect from input type).

    Returns:
        dict with keys:
            orbital_energies, orbital_coeffs, density_matrix,
            homo, lumo, gap_eV, band_energy
    """
    if backend == "auto":
        backend = "torch" if isinstance(H, Tensor) else "numpy"

    is_uks = isinstance(n_electrons, (tuple, list))

    if backend == "torch":
        return _props_torch(H, S, n_electrons, is_uks)
    else:
        return _props_numpy(H, S, n_electrons, is_uks)


def _props_numpy(H, S, n_electrons, is_uks):
    if is_uks:
        n_alpha, n_beta = n_electrons
        e_a, c_a = solve_gen_eigh(H[0], S)
        e_b, c_b = solve_gen_eigh(H[1], S)

        dm_a = build_density_matrix(c_a, n_alpha, occupation=1.0)
        dm_b = build_density_matrix(c_b, n_beta, occupation=1.0)

        homo_a = float(e_a[n_alpha - 1]) if n_alpha > 0 else -np.inf
        homo_b = float(e_b[n_beta - 1]) if n_beta > 0 else -np.inf
        homo = max(homo_a, homo_b)
        lumo = float(e_a[n_alpha]) if n_alpha < len(e_a) else np.inf

        return {
            "orbital_energies": np.stack([e_a, e_b]),
            "orbital_coeffs": np.stack([c_a, c_b]),
            "density_matrix": np.stack([dm_a, dm_b]),
            "homo": homo,
            "lumo": lumo,
            "gap_eV": (lumo - homo) * HA_TO_EV,
            "band_energy": float(np.trace(H[0] @ dm_a) + np.trace(H[1] @ dm_b)),
        }
    else:
        n_occ = n_electrons // 2
        energies, coeffs = solve_gen_eigh(H, S)
        dm = build_density_matrix(coeffs, n_electrons)

        homo = float(energies[n_occ - 1]) if n_occ > 0 else -np.inf
        lumo = float(energies[n_occ]) if n_occ < len(energies) else np.inf

        return {
            "orbital_energies": energies,
            "orbital_coeffs": coeffs,
            "density_matrix": dm,
            "homo": homo,
            "lumo": lumo,
            "gap_eV": (lumo - homo) * HA_TO_EV,
            "band_energy": float(np.trace(H @ dm)),
        }


def _props_torch(H, S, n_electrons, is_uks):
    if is_uks:
        n_alpha, n_beta = n_electrons
        e_a, c_a = solve_gen_eigh_torch(H[0], S)
        e_b, c_b = solve_gen_eigh_torch(H[1], S)

        dm_a = build_density_matrix_torch(c_a, n_alpha, occupation=1.0)
        dm_b = build_density_matrix_torch(c_b, n_beta, occupation=1.0)

        batched = e_a.dim() > 1
        if batched:
            homo_a = e_a[:, n_alpha - 1] if n_alpha > 0 else torch.full((e_a.shape[0],), -float("inf"))
            homo_b = e_b[:, n_beta - 1] if n_beta > 0 else torch.full((e_b.shape[0],), -float("inf"))
            homo = torch.maximum(homo_a, homo_b)
            lumo = e_a[:, n_alpha] if n_alpha < e_a.shape[-1] else torch.full_like(homo, float("inf"))
        else:
            homo = max(e_a[n_alpha - 1].item(), e_b[n_beta - 1].item()) if n_alpha > 0 and n_beta > 0 else -float("inf")
            lumo = e_a[n_alpha].item() if n_alpha < len(e_a) else float("inf")

        return {
            "orbital_energies": torch.stack([e_a, e_b]),
            "orbital_coeffs": torch.stack([c_a, c_b]),
            "density_matrix": torch.stack([dm_a, dm_b]),
            "homo": homo,
            "lumo": lumo,
            "gap_eV": (lumo - homo) * HA_TO_EV if not isinstance(lumo, Tensor) else (lumo - homo) * HA_TO_EV,
            "band_energy": torch.einsum("...ij,...ji->...", H[0], dm_a) + torch.einsum("...ij,...ji->...", H[1], dm_b),
        }
    else:
        n_occ = n_electrons // 2
        energies, coeffs = solve_gen_eigh_torch(H, S)
        dm = build_density_matrix_torch(coeffs, n_electrons)

        batched = energies.dim() > 1
        if batched:
            homo = energies[:, n_occ - 1]
            lumo = energies[:, n_occ]
        else:
            homo = energies[n_occ - 1].item()
            lumo = energies[n_occ].item()

        return {
            "orbital_energies": energies,
            "orbital_coeffs": coeffs,
            "density_matrix": dm,
            "homo": homo,
            "lumo": lumo,
            "gap_eV": (lumo - homo) * HA_TO_EV if not isinstance(lumo, Tensor) else (lumo - homo) * HA_TO_EV,
            "band_energy": torch.einsum("...ij,...ji->...", H, dm),
        }


# ── Molecule integration ────────────────────────────────────────────

def compute_integrals_for_molecule(
    mol: "Molecule",
    basis: str | None = None,
    xc: str | None = None,
    device: str = "cpu",
) -> "Molecule":
    """Compute 1-electron integrals and populate Molecule fields.

    Builds a PySCF Mole from the Molecule's geometry, computes S, T, V, hcore
    using the torch integrals engine, and stores them in the Molecule.

    Args:
        mol: Molecule with atomic_numbers and positions (in Ang or Bohr).
        basis: Basis set name (e.g. 'def2-svp'). Uses mol.basis if None.
        xc: XC functional name. Stored in mol.xc.
        device: Torch device for integral computation.

    Returns:
        The same Molecule with overlap, hamiltonian (hcore), and basis_info set.
    """
    from pyscf import gto
    from integrals import build_integrals
    from molecule import BasisInfo

    basis = basis or (mol.basis_info.name if mol.basis_info else None)
    if basis is None:
        raise ValueError("basis must be specified or present in mol.basis_info")

    # Build PySCF Mole
    unit = "Angstrom" if mol.length_unit == "Ang" else "Bohr"
    atoms = []
    for z, pos in zip(mol.atomic_numbers, mol.positions):
        sym = gto.elements.ELEMENTS[z]
        atoms.append(f"{sym} {pos[0]} {pos[1]} {pos[2]}")
    atom_str = "; ".join(atoms)

    pyscf_mol = gto.M(
        atom=atom_str, basis=basis, unit=unit,
        charge=mol.charge, spin=mol.spin, verbose=0,
    )

    # Compute integrals
    S, T, V, hcore = build_integrals(pyscf_mol, device=device)

    # Store in Molecule
    mol.overlap = S.cpu().numpy()
    mol.hamiltonian = hcore.cpu().numpy()

    if mol.basis_info is None or mol.basis_info.name != basis:
        try:
            mol.basis_info = BasisInfo.from_name(basis, convention="pyscf")
        except ValueError:
            mol.basis_info = BasisInfo(name=basis, convention="pyscf")

    if xc is not None:
        mol.xc = xc

    return mol


def run_dft_for_molecule(
    mol: "Molecule",
    basis: str | None = None,
    method: str = "rks",
    xc: str = "b3lyp",
    device: str = "cpu",
) -> "Molecule":
    """Run full DFT calculation and populate all Molecule fields.

    Uses torch-computed integrals + PySCF SCF.

    Args:
        mol: Molecule with atomic_numbers and positions.
        basis: Basis set name.
        method: 'rhf', 'uhf', 'rks', 'uks'.
        xc: XC functional.
        device: Torch device for integrals.

    Returns:
        Molecule with all electronic structure fields populated.
    """
    from pyscf import gto
    from integrals import run_scf
    from molecule import Molecule as MolCls, BasisInfo

    basis = basis or (mol.basis_info.name if mol.basis_info else None)
    if basis is None:
        raise ValueError("basis must be specified")

    unit = "Angstrom" if mol.length_unit == "Ang" else "Bohr"
    atoms = []
    for z, pos in zip(mol.atomic_numbers, mol.positions):
        sym = gto.elements.ELEMENTS[z]
        atoms.append(f"{sym} {pos[0]} {pos[1]} {pos[2]}")

    pyscf_mol = gto.M(
        atom="; ".join(atoms), basis=basis, unit=unit,
        charge=mol.charge, spin=mol.spin, verbose=0,
    )

    mf = run_scf(pyscf_mol, method=method, xc=xc, device=device)

    # Use existing from_pyscf and preserve original identifiers
    result = MolCls.from_pyscf(mf, mol_id=mol.mol_id)
    result.formula = mol.formula
    result.smiles = mol.smiles
    return result


def inject_hamiltonian(
    mol: "Molecule",
    H_pred: np.ndarray,
) -> dict:
    """Inject a predicted Hamiltonian into a Molecule and extract properties.

    Assumes mol.overlap exists. If not, call compute_integrals_for_molecule first.

    Args:
        mol: Molecule with overlap matrix set.
        H_pred: Predicted Hamiltonian (nao, nao) or (2, nao, nao) for UKS.

    Returns:
        dict with orbital_energies, density_matrix, homo, lumo, gap_eV, band_energy.
    """
    if mol.overlap is None:
        raise ValueError("mol.overlap is required. Call compute_integrals_for_molecule first.")

    if mol.unrestricted or (H_pred.ndim == 3 and H_pred.shape[0] == 2):
        n_alpha = (mol.num_electrons + mol.spin) // 2
        n_beta = (mol.num_electrons - mol.spin) // 2
        n_elec = (n_alpha, n_beta)
    else:
        n_elec = mol.num_electrons

    return properties_from_hamiltonian(H_pred, mol.overlap, n_elec)
