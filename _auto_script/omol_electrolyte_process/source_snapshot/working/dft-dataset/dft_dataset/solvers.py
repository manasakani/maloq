"""Eigenvalue solvers and density matrix construction for DFT.

Solves the generalized eigenvalue problem  HC = SCε  (Roothaan equation)
and builds density matrices from MO coefficients.

Provides both NumPy (single molecule) and PyTorch (batched, GPU, differentiable) paths.

Functions:
    solve_eigenproblem       — HC = SCε → orbital energies + MO coefficients (NumPy)
    solve_eigenproblem_torch — same, batched on GPU (PyTorch, differentiable)
    build_density_matrix       — MO coefficients → P  (NumPy)
    build_density_matrix_torch — same, batched (PyTorch)
    compare_hamiltonians     — H_pred vs H_true → MAE, RMSE, orbital/gap errors
    to_torch                 — Molecule → dict of PyTorch tensors
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.linalg as la


# ── Constants ────────────────────────────────────────────────────────

HA_TO_EV = 27.211386245988


# ── Eigenvalue solvers ───────────────────────────────────────────────

def solve_eigenproblem(
    H: np.ndarray,
    S: np.ndarray,
    method: str = "eigh",
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Roothaan equation  HC = SCε  (generalized eigenvalue problem).

    Transforms to standard form via S^{-1/2}, then diagonalizes.
    Eigenvalues with S < 1e-10 are discarded (linear dependency removal).

    Args:
        H: (nao, nao) Hamiltonian or Fock matrix (symmetric).
        S: (nao, nao) overlap matrix (symmetric positive-definite).
        method: "eigh" (default, numerically stable) or "cholesky" (faster).

    Returns:
        orbital_energies: (nao,) sorted in ascending order (Hartree).
        mo_coefficients:  (nao, nao) columns are MO vectors.

    Example:
        >>> energies, C = solve_eigenproblem(H, S)
        >>> n_occ = n_electrons // 2
        >>> homo, lumo = energies[n_occ - 1], energies[n_occ]
    """
    if method == "eigh":
        eigvals_s, U = np.linalg.eigh(S)
        mask = eigvals_s > 1e-10
        X = U[:, mask] / np.sqrt(eigvals_s[mask])
        H_orth = X.T @ H @ X
        energies, C_orth = np.linalg.eigh(H_orth)
        coefficients = X @ C_orth
    elif method == "cholesky":
        L = np.linalg.cholesky(S)
        L_inv = la.solve_triangular(L, np.eye(L.shape[0]), lower=True)
        H_transformed = L_inv @ H @ L_inv.T
        energies, C_prime = np.linalg.eigh(H_transformed)
        coefficients = L_inv.T @ C_prime
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'eigh' or 'cholesky'.")

    return energies, coefficients


def solve_eigenproblem_torch(
    H: "torch.Tensor",
    S: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Solve HC = SCε in PyTorch (batched, GPU, differentiable).

    Same algorithm as solve_eigenproblem but operates on torch tensors.
    Supports batched input for processing multiple molecules at once.

    Args:
        H: (nao, nao) or (batch, nao, nao) Hamiltonian/Fock matrix.
        S: (nao, nao) or (batch, nao, nao) overlap matrix.

    Returns:
        orbital_energies: (nao,) or (batch, nao).
        mo_coefficients:  (nao, nao) or (batch, nao, nao).

    Example:
        >>> # Single molecule
        >>> e, C = solve_eigenproblem_torch(H, S)
        >>> # Batch of 1000 molecules on GPU
        >>> e, C = solve_eigenproblem_torch(H_batch, S_batch)
    """
    import torch

    squeeze = H.dim() == 2
    if squeeze:
        H = H.unsqueeze(0)
        S = S.unsqueeze(0)

    eigvals_s, U = torch.linalg.eigh(S)
    eigvals_s = eigvals_s.clamp(min=1e-10)
    X = U / torch.sqrt(eigvals_s).unsqueeze(-2)

    H_orth = torch.bmm(torch.bmm(X.transpose(-1, -2), H), X)
    energies, C_orth = torch.linalg.eigh(H_orth)
    coefficients = torch.bmm(X, C_orth)

    if squeeze:
        energies = energies.squeeze(0)
        coefficients = coefficients.squeeze(0)

    return energies, coefficients


# ── Density matrix construction ──────────────────────────────────────

def build_density_matrix(
    mo_coefficients: np.ndarray,
    n_electrons: int,
    occupation: float = 2.0,
) -> np.ndarray:
    """Build density matrix from MO coefficients (NumPy).

    P = occupation × C_occ @ C_occ^T

    For RKS (closed-shell): occupation=2.0, n_electrons=total.
    For UKS (one spin channel): occupation=1.0, n_electrons=n_alpha or n_beta.

    Args:
        mo_coefficients: (nao, nao) MO coefficient matrix from solve_eigenproblem.
        n_electrons: Number of electrons to fill.
        occupation: Electrons per orbital (2.0 for RKS, 1.0 for UKS).

    Returns:
        P: (nao, nao) density matrix.

    Example:
        >>> P = build_density_matrix(C, n_electrons=10)  # RKS, 5 doubly-occupied
        >>> assert abs(np.trace(P @ S) - 10) < 1e-6
    """
    n_occ = int(n_electrons // occupation)
    C_occ = mo_coefficients[:, :n_occ]
    return occupation * (C_occ @ C_occ.T)


def build_density_matrix_torch(
    mo_coefficients: "torch.Tensor",
    n_electrons: int | tuple[int, int],
    occupation: float = 2.0,
) -> "torch.Tensor":
    """Build density matrix from MO coefficients (PyTorch, batched).

    Supports both RKS (single int) and UKS (tuple of alpha/beta counts).

    Args:
        mo_coefficients: (nao, nao), (batch, nao, nao), or
            (2, batch, nao, nao) / (2, nao, nao) for UKS.
        n_electrons: Total electrons (int) for RKS,
            or (n_alpha, n_beta) tuple for UKS.
        occupation: 2.0 for RKS, 1.0 for UKS per-spin.

    Returns:
        P: same shape as mo_coefficients.

    Example:
        >>> P = build_density_matrix_torch(C, n_electrons=10)          # RKS
        >>> P = build_density_matrix_torch(C, n_electrons=(5, 5))      # UKS
        >>> P = build_density_matrix_torch(C_batch, n_electrons=10)    # batched
    """
    import torch

    if isinstance(n_electrons, (tuple, list)):
        n_alpha, n_beta = n_electrons
        dm_a = build_density_matrix_torch(mo_coefficients[0], n_alpha, occupation=1.0)
        dm_b = build_density_matrix_torch(mo_coefficients[1], n_beta, occupation=1.0)
        return torch.stack([dm_a, dm_b])

    n_occ = int(n_electrons // occupation)
    C_occ = mo_coefficients[..., :, :n_occ]
    return occupation * (C_occ @ C_occ.transpose(-1, -2))


# ── Model evaluation ────────────────────────────────────────────────

def compare_hamiltonians(
    H_pred: np.ndarray,
    H_true: np.ndarray,
    S: Optional[np.ndarray] = None,
    n_electrons: int = 0,
) -> dict:
    """Compare predicted vs reference Hamiltonian matrices.

    Always computes element-wise errors (MAE, RMSE, max).
    If S and n_electrons are given, also computes orbital energy errors
    and HOMO-LUMO gap error.

    Args:
        H_pred: (nao, nao) predicted Hamiltonian.
        H_true: (nao, nao) reference Hamiltonian.
        S: (nao, nao) overlap matrix (optional, enables orbital comparison).
        n_electrons: Total electron count (required with S).

    Returns:
        dict with:
            mae, rmse, max_error          — element-wise (Hartree)
            mae_orbital_energies          — per-orbital MAE (Hartree), if S given
            gap_error_eV                  — HOMO-LUMO gap error (eV), if S given

    Example:
        >>> metrics = compare_hamiltonians(H_pred, H_ref, S=S, n_electrons=10)
        >>> print(f"H MAE: {metrics['mae']:.2e} Ha")
        >>> print(f"Gap error: {metrics['gap_error_eV']:.1f} meV")
    """
    diff = H_pred - H_true
    results = {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_error": float(np.max(np.abs(diff))),
    }

    if S is not None and n_electrons > 0:
        e_pred, _ = solve_eigenproblem(H_pred, S)
        e_true, _ = solve_eigenproblem(H_true, S)
        results["mae_orbital_energies"] = float(np.mean(np.abs(e_pred - e_true)))

        n_occ = n_electrons // 2
        gap_pred = (e_pred[n_occ] - e_pred[n_occ - 1]) * HA_TO_EV
        gap_true = (e_true[n_occ] - e_true[n_occ - 1]) * HA_TO_EV
        results["gap_error_eV"] = float(abs(gap_pred - gap_true))

    return results


# ── Torch conversion ────────────────────────────────────────────────

def to_torch(mol: "Molecule", device: str = "cpu") -> dict:
    """Convert a Molecule's arrays to PyTorch tensors.

    Converts atomic_numbers, positions, and any electronic structure matrices
    (hamiltonian, overlap, density_matrix, orbital_energies) that are present.

    Args:
        mol: Molecule object.
        device: Torch device ("cpu", "cuda", "cuda:0", ...).

    Returns:
        dict of tensors. Only non-None fields are included.

    Example:
        >>> tensors = to_torch(mol, device="cuda")
        >>> H = tensors["hamiltonian"]   # (nao, nao) on GPU
    """
    import torch

    result = {
        "atomic_numbers": torch.tensor(mol.atomic_numbers, dtype=torch.long, device=device),
        "positions": torch.tensor(mol.positions, dtype=torch.float64, device=device),
    }
    if mol.hamiltonian is not None:
        result["hamiltonian"] = torch.tensor(mol.hamiltonian, dtype=torch.float64, device=device)
    if mol.overlap is not None:
        result["overlap"] = torch.tensor(mol.overlap, dtype=torch.float64, device=device)
    if mol.density_matrix is not None:
        result["density_matrix"] = torch.tensor(mol.density_matrix, dtype=torch.float64, device=device)
    if mol.orbital_energies is not None:
        result["orbital_energies"] = torch.tensor(mol.orbital_energies, dtype=torch.float64, device=device)
    return result


# ── Backward-compatible aliases ──────────────────────────────────────

solve_gen_eigh = solve_eigenproblem
solve_gen_eigh_torch = solve_eigenproblem_torch
