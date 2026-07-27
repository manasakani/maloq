"""
Matrix Block ↔ Irreducible Representation Decomposition
========================================================

Decomposes a (2l1+1)×(2l2+1) matrix block (e.g., Hamiltonian orbital block)
into irreducible representation coefficients via Clebsch-Gordan decomposition,
and reconstructs back. Provably rotation-equivariant.

Math:
    H^{l1,l2} ∈ ℝ^{(2l1+1)×(2l2+1)}  transforms as  D^l1 H D^{l2,T}
    CG matrix C block-diagonalizes: C^T (D^l1 ⊗ D^l2) C = ⊕_L D^L
    Flatten:     h = C^T vec(H)         → irrep coefficients {h^L}
    Reconstruct: vec(H) = C h           → matrix block
    Equivariance: h'^L = D^L h^L        (each irrep rotates independently)

Usage:
    decomposer = IrrepDecomposer(l1=1, l2=2)  # p-d block
    coeffs = decomposer.flatten(H)             # {1: (3,), 2: (5,), 3: (7,)}
    H_rec = decomposer.reconstruct(coeffs)     # (3, 5) matrix

Dependencies: torch, numpy, scipy (no e3nn required)
"""
from __future__ import annotations

import torch
import numpy as np
from functools import lru_cache
from scipy.spatial.transform import Rotation as SciRotation


# =====================================================================
# Angular momentum generators for real spherical harmonics
# =====================================================================

@lru_cache(maxsize=32)
def _angular_momentum_generators(l: int) -> tuple[torch.Tensor, ...]:
    """Rotation generators Gx, Gy, Gz for real spherical harmonics of degree l.

    These are real antisymmetric matrices satisfying:
        D(R) = exp(θ n·G)
        -(Gx² + Gy² + Gz²) has eigenvalue l(l+1) (Casimir)

    Constructed by transforming complex angular momentum operators
    to the real spherical harmonic basis via the unitary U.
    """
    dim = 2 * l + 1
    if l == 0:
        z = torch.zeros(1, 1, dtype=torch.float64)
        return z, z, z

    # Complex basis operators
    Jz = torch.diag(torch.arange(-l, l + 1, dtype=torch.float64))
    Jp = torch.zeros(dim, dim, dtype=torch.float64)
    Jm = torch.zeros(dim, dim, dtype=torch.float64)
    for idx in range(dim):
        m = idx - l
        if m + 1 <= l:
            Jp[idx + 1, idx] = np.sqrt(l * (l + 1) - m * (m + 1))
        if m - 1 >= -l:
            Jm[idx - 1, idx] = np.sqrt(l * (l + 1) - m * (m - 1))
    Jx_c = (Jp + Jm) / 2
    Jy_c = (Jp - Jm) / (2j)

    # Complex-to-real unitary: Y_real = U · Y_complex
    U = _complex_to_real_unitary(l)

    # Generators: G = -iJ in real basis (real antisymmetric)
    Gx = (-1j * (U @ Jx_c.to(torch.complex128) @ U.conj().T)).real
    Gy = (-1j * (U @ Jy_c @ U.conj().T)).real
    Gz = (-1j * (U @ Jz.to(torch.complex128) @ U.conj().T)).real
    return Gx, Gy, Gz


@lru_cache(maxsize=16)
def _complex_to_real_unitary(l: int) -> torch.Tensor:
    """Unitary matrix U such that Y^l_real = U · Y^l_complex.

    Convention (Condon-Shortley phase, m ordering: -l, ..., +l):
        m > 0:  Y_m^real = ((-1)^m Y_m^complex + Y_{-m}^complex) / √2
        m = 0:  Y_0^real = Y_0^complex
        m < 0:  Y_m^real = i((-1)^{|m|+1} Y_{|m|}^complex + Y_{-|m|}^complex) / √2
    """
    dim = 2 * l + 1
    U = torch.zeros(dim, dim, dtype=torch.complex128)
    for idx in range(dim):
        m = idx - l
        if m > 0:
            U[idx, m + l] = ((-1) ** m) / np.sqrt(2)
            U[idx, -m + l] = 1.0 / np.sqrt(2)
        elif m == 0:
            U[idx, l] = 1.0
        else:
            am = abs(m)
            U[idx, am + l] = 1j * ((-1) ** (am + 1)) / np.sqrt(2)
            U[idx, -am + l] = 1j / np.sqrt(2)
    return U


def wigner_D(l: int, R: torch.Tensor) -> torch.Tensor:
    """Wigner-D matrix for real spherical harmonics of degree l.

    D(R) = exp(θ n·G) where R = rotation matrix, θ = angle, n = axis.
    """
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64)

    rotvec = SciRotation.from_matrix(R.detach().cpu().numpy()).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-15:
        return torch.eye(2 * l + 1, dtype=torch.float64)

    axis = rotvec / angle
    Gx, Gy, Gz = _angular_momentum_generators(l)
    Gn = axis[0] * Gx + axis[1] * Gy + axis[2] * Gz
    return torch.matrix_exp(angle * Gn)


# =====================================================================
# CG Change-of-Basis Matrix
# =====================================================================

@lru_cache(maxsize=128)
def _build_cg_matrix(l1: int, l2: int) -> tuple[torch.Tensor, list[int]]:
    """Build the CG change-of-basis matrix C for l1 ⊗ l2.

    C is orthogonal, shape (d1*d2, d1*d2), satisfying:
        C^T (D^l1 ⊗ D^l2) C = ⊕_{L} D^L

    Returns:
        C: orthogonal change-of-basis matrix
        L_list: list of irrep labels [|l1-l2|, ..., l1+l2]

    Method:
        1. Build total angular momentum Casimir J² in product basis
        2. Eigendecompose: eigenvalues L(L+1) identify each irrep
        3. Eigenvectors grouped by L form the columns of C
    """
    d1, d2 = 2 * l1 + 1, 2 * l2 + 1
    d_total = d1 * d2
    L_list = list(range(abs(l1 - l2), l1 + l2 + 1))

    # Trivial cases
    if l1 == 0 and l2 == 0:
        return torch.ones(1, 1, dtype=torch.float64), [0]
    if l1 == 0:
        return torch.eye(d2, dtype=torch.float64), [l2]
    if l2 == 0:
        return torch.eye(d1, dtype=torch.float64), [l1]

    I1 = torch.eye(d1, dtype=torch.float64)
    I2 = torch.eye(d2, dtype=torch.float64)

    # Total angular momentum generators in product basis
    G1x, G1y, G1z = _angular_momentum_generators(l1)
    G2x, G2y, G2z = _angular_momentum_generators(l2)
    Gx = torch.kron(G1x, I2) + torch.kron(I1, G2x)
    Gy = torch.kron(G1y, I2) + torch.kron(I1, G2y)
    Gz = torch.kron(G1z, I2) + torch.kron(I1, G2z)

    # Casimir operator: eigenvalues of -(Gx²+Gy²+Gz²) = L(L+1)
    G2 = Gx @ Gx + Gy @ Gy + Gz @ Gz
    casimir_eig, casimir_vec = torch.linalg.eigh(-G2)

    # Group eigenvectors by L (sorted by L)
    C = torch.zeros(d_total, d_total, dtype=torch.float64)
    col = 0
    for L in L_list:
        dim_L = 2 * L + 1
        target = L * (L + 1)

        # Find eigenvectors closest to this L(L+1) eigenvalue
        diffs = (casimir_eig - target).abs()
        _, sorted_idx = diffs.sort()
        indices = sorted_idx[:dim_L]

        C[:, col:col + dim_L] = casimir_vec[:, indices]
        col += dim_L

    return C, L_list


# =====================================================================
# IrrepDecomposer
# =====================================================================

class IrrepDecomposer:
    """CG-based matrix block ↔ irrep coefficient conversion.

    For a matrix block H ∈ ℝ^{(2l1+1)×(2l2+1)} transforming as
    H' = D^l1(R) H D^l2(R)^T under rotation R, this class decomposes
    H into irreducible representation coefficients h^L ∈ ℝ^{(2L+1)}
    for L = |l1-l2|, ..., l1+l2.

    The decomposition is:
        - Complete: reconstruct(flatten(H)) == H
        - Norm-preserving: ||H||²_F == Σ_L ||h^L||²
        - Rotation-equivariant: flatten(rotate(H)) == rotate_irreps(flatten(H))

    Args:
        l1: angular momentum of row indices (0=s, 1=p, 2=d, 3=f, 4=g)
        l2: angular momentum of column indices
    """

    def __init__(self, l1: int, l2: int):
        self.l1 = l1
        self.l2 = l2
        self.d1 = 2 * l1 + 1
        self.d2 = 2 * l2 + 1
        self._C, self._L_list = _build_cg_matrix(l1, l2)

        # Precompute block offsets for fast slicing
        self._offsets = {}
        offset = 0
        for L in self._L_list:
            dim_L = 2 * L + 1
            self._offsets[L] = (offset, offset + dim_L)
            offset += dim_L

    @property
    def L_list(self) -> list[int]:
        """List of irrep labels: [|l1-l2|, ..., l1+l2]."""
        return list(self._L_list)

    @property
    def cg_matrix(self) -> torch.Tensor:
        """The orthogonal CG change-of-basis matrix C."""
        return self._C

    # ------------------------------------------------------------------
    # Single sample
    # ------------------------------------------------------------------

    def flatten(self, H: torch.Tensor) -> dict[int, torch.Tensor]:
        """Decompose matrix block into irrep coefficients.

        Args:
            H: tensor of shape (2l1+1, 2l2+1)

        Returns:
            {L: tensor of shape (2L+1,)} for each L in L_list
        """
        h_vec = H.reshape(-1)
        h_irrep = self._C.T @ h_vec
        return {L: h_irrep[s:e].clone() for L, (s, e) in self._offsets.items()}

    def reconstruct(self, coeffs: dict[int, torch.Tensor]) -> torch.Tensor:
        """Reconstruct matrix block from irrep coefficients.

        Args:
            coeffs: {L: tensor of shape (2L+1,)}

        Returns:
            tensor of shape (2l1+1, 2l2+1)
        """
        h_irrep = torch.cat([coeffs[L] for L in self._L_list])
        h_vec = self._C @ h_irrep
        return h_vec.reshape(self.d1, self.d2)

    # ------------------------------------------------------------------
    # Batched
    # ------------------------------------------------------------------

    def flatten_batch(self, H: torch.Tensor) -> dict[int, torch.Tensor]:
        """Batch decomposition.

        Args:
            H: tensor of shape (B, 2l1+1, 2l2+1)

        Returns:
            {L: tensor of shape (B, 2L+1)}
        """
        B = H.shape[0]
        h_vec = H.reshape(B, -1)          # (B, d1*d2)
        h_irrep = h_vec @ self._C         # (B, d1*d2) — C^T applied from right
        return {L: h_irrep[:, s:e].clone() for L, (s, e) in self._offsets.items()}

    def reconstruct_batch(self, coeffs: dict[int, torch.Tensor]) -> torch.Tensor:
        """Batch reconstruction.

        Args:
            coeffs: {L: tensor of shape (B, 2L+1)}

        Returns:
            tensor of shape (B, 2l1+1, 2l2+1)
        """
        h_irrep = torch.cat([coeffs[L] for L in self._L_list], dim=-1)  # (B, d_total)
        h_vec = h_irrep @ self._C.T                                      # (B, d1*d2)
        return h_vec.reshape(-1, self.d1, self.d2)

    # ------------------------------------------------------------------
    # Rotation in irrep space
    # ------------------------------------------------------------------

    def rotate_matrix(self, H: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Rotate matrix block: H' = D^l1(R) H D^l2(R)^T.

        Args:
            H: (2l1+1, 2l2+1)  or  (B, 2l1+1, 2l2+1)
            R: (3, 3) rotation matrix
        """
        D1 = wigner_D(self.l1, R)
        D2 = wigner_D(self.l2, R)
        if H.dim() == 2:
            return D1 @ H @ D2.T
        else:
            return torch.einsum('ij,bjk,lk->bil', D1, H, D2)

    def rotate_irreps(self, coeffs: dict[int, torch.Tensor],
                      R: torch.Tensor) -> dict[int, torch.Tensor]:
        """Rotate irrep coefficients: h'^L = D^L(R) h^L.

        D^L is extracted from the CG matrix (not computed independently),
        guaranteeing basis consistency and exact equivariance.
        """
        D1 = wigner_D(self.l1, R)
        D2 = wigner_D(self.l2, R)
        Dkron = torch.kron(D1, D2)

        # Block-diagonalize: ⊕_L D^L = C^T (D^l1 ⊗ D^l2) C
        D_irrep = self._C.T @ Dkron @ self._C

        result = {}
        for L, (s, e) in self._offsets.items():
            D_L = D_irrep[s:e, s:e]
            h_L = coeffs[L]
            if h_L.dim() == 1:
                result[L] = D_L @ h_L
            else:  # batched: (B, 2L+1)
                result[L] = h_L @ D_L.T
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def irrep_norms(self, coeffs: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Compute ||h^L|| for each irrep (rotation invariant)."""
        return {L: h_L.norm(dim=-1) for L, h_L in coeffs.items()}

    def __repr__(self) -> str:
        irreps = " + ".join(f"(2×{L}+1)" for L in self._L_list)
        return (f"IrrepDecomposer(l1={self.l1}, l2={self.l2}, "
                f"({self.d1}×{self.d2}) → {irreps})")
