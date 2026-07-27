"""Pure-PyTorch 1-electron integrals via McMurchie-Davidson scheme.

Computes overlap (S), kinetic (T), nuclear attraction (V), and hcore = T + V
matrices identically to PySCF, but in PyTorch — enabling GPU acceleration
and batching over molecules with the same basis set.

Supports:
  - Angular momentum up to l=3 (f orbitals)
  - General contractions (NCTR > 1, e.g. cc-pVXZ)
  - Spherical and Cartesian basis
  - Single-molecule and batched-molecule modes
  - CPU and CUDA

API:
  build_integrals(mol)            → S, T, V, hcore  (single molecule)
  build_hcore_batch(mol, coords)  → hcore            (batched, fast grouped)

Reference:
  McMurchie, Davidson (1978) J. Comp. Phys. 26, 218-231.
  Taketa, Huzinaga, O-ohata (1966) J. Phys. Soc. Japan 21, 2313.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor


# ============================================================
# 1. Utilities
# ============================================================

def cart_ang(l: int) -> list[tuple[int, int, int]]:
    """Cartesian angular momentum indices (lx, ly, lz) in PySCF ordering."""
    out = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            out.append((lx, ly, l - lx - ly))
    return out


def ncart(l: int) -> int:
    """Number of Cartesian components for angular momentum l."""
    return (l + 1) * (l + 2) // 2


def _CINTcommon_fac_sp(l: int) -> float:
    """Normalization factor applied by libcint (CINTcommon_fac_sp).

    For s and p orbitals, libcint multiplies each Cartesian GTO by the
    solid-harmonic normalization factor sqrt((2l+1)/(4π)).  For l >= 2
    (d, f, ...) no additional factor is applied — the cart2sph transformation
    handles the conversion instead.
    """
    if l == 0:
        return 1.0 / math.sqrt(4.0 * math.pi)
    elif l == 1:
        return math.sqrt(3.0 / (4.0 * math.pi))
    else:
        return 1.0


def _cart2sph(mol) -> Tensor | None:
    """Get Cartesian-to-spherical transformation matrix, or None if Cartesian."""
    if mol.cart:
        return None
    c2s_np = mol.cart2sph_coeff()
    if hasattr(c2s_np, "toarray"):
        c2s_np = c2s_np.toarray()
    return np.asarray(c2s_np)


# ============================================================
# 2. Boys function
# ============================================================

def boys(n: int, x: Tensor) -> Tensor:
    """Boys function F_n(x) = integral_0^1 t^{2n} exp(-x t^2) dt.

    Uses the incomplete gamma function:
        F_n(x) = Gamma(n+1/2) P(n+1/2, x) / (2 x^{n+1/2})
    where P(a, x) = gammainc(a, x) is the regularized lower incomplete gamma.
    """
    a = n + 0.5
    gamma_a = math.gamma(a)

    small = x.abs() < 1e-14
    x_safe = torch.where(small, torch.ones_like(x), x)

    P = torch.special.gammainc(torch.full_like(x_safe, a), x_safe)
    F = (gamma_a * P) / (2.0 * x_safe.pow(a))

    return torch.where(small, torch.full_like(F, 1.0 / (2 * n + 1)), F)


# ============================================================
# 3. McMurchie-Davidson E-coefficients (1D)
# ============================================================

def compute_E(
    i_max: int, j_max: int,
    alpha: Tensor, beta: Tensor,
    A: Tensor, B: Tensor,
) -> Tensor:
    """1D McMurchie-Davidson expansion coefficients E^{ij}_t.

    Recurrence:
        E^{00}_0 = exp(-mu * (A-B)^2)
        E^{i+1,j}_t = (1/(2z)) E^{ij}_{t-1} + X_PA E^{ij}_t + (t+1) E^{ij}_{t+1}
        E^{i,j+1}_t = (1/(2z)) E^{ij}_{t-1} + X_PB E^{ij}_t + (t+1) E^{ij}_{t+1}

    Args:
        i_max, j_max: exclusive upper bounds for angular momentum indices.
        alpha, beta: (N,) primitive exponents.
        A, B: (N,) 1D center positions.

    Returns:
        E: (i_max, j_max, t_max, N) where t_max = i_max + j_max - 1.
    """
    N = alpha.shape[0]
    t_max = i_max + j_max - 1
    dtype, device = alpha.dtype, alpha.device

    zeta = alpha + beta
    oo2z = 0.5 / zeta
    mu = alpha * beta / zeta
    P = (alpha * A + beta * B) / zeta
    XPA = P - A
    XPB = P - B

    E = torch.zeros(i_max, j_max, t_max, N, dtype=dtype, device=device)
    E[0, 0, 0] = torch.exp(-mu * (A - B) ** 2)

    # Phase 1: build E[i, 0, t] for increasing i
    for i in range(i_max - 1):
        for t in range(i + 2):
            val = XPA * E[i, 0, t]
            if t > 0:
                val = val + oo2z * E[i, 0, t - 1]
            if t + 1 < t_max:
                val = val + (t + 1) * E[i, 0, t + 1]
            E[i + 1, 0, t] = val

    # Phase 2: build E[i, j, t] for increasing j
    for j in range(j_max - 1):
        for i in range(i_max):
            for t in range(min(i + j + 2, t_max)):
                val = XPB * E[i, j, t]
                if t > 0:
                    val = val + oo2z * E[i, j, t - 1]
                if t + 1 < t_max:
                    val = val + (t + 1) * E[i, j, t + 1]
                E[i, j + 1, t] = val

    return E


# ============================================================
# 4. Hermite Coulomb R-integrals
# ============================================================

def compute_R(
    t_max: int, u_max: int, v_max: int,
    n_boys: int,
    zeta: Tensor,
    RPC: Tensor,
) -> Tensor:
    """Hermite Coulomb integrals R_{tuv}(n=0).

    Recurrence:
        R_{000}(n) = (-2 zeta)^n F_n(zeta |RPC|^2)
        R_{t+1,u,v}(n) = t R_{t-1,u,v}(n+1) + X_PC R_{tuv}(n+1)
        (similarly for u, v)

    Args:
        t_max, u_max, v_max: exclusive upper bounds.
        n_boys: maximum Boys function order.
        zeta: (B,) combined exponents.
        RPC: (B, 3) vector P - C.

    Returns:
        R: (t_max, u_max, v_max, B) evaluated at n=0.
    """
    B = zeta.shape[0]
    dtype, device = zeta.dtype, zeta.device
    n_tot = n_boys + 1

    RPC_sq = (RPC ** 2).sum(dim=-1)
    x = zeta * RPC_sq

    R = torch.zeros(t_max, u_max, v_max, n_tot, B, dtype=dtype, device=device)

    for n in range(n_tot):
        R[0, 0, 0, n] = ((-2.0 * zeta) ** n) * boys(n, x)

    XPC, YPC, ZPC = RPC[:, 0], RPC[:, 1], RPC[:, 2]

    for t in range(t_max - 1):
        for n in range(n_tot - t - 1):
            val = XPC * R[t, 0, 0, n + 1]
            if t > 0:
                val = val + t * R[t - 1, 0, 0, n + 1]
            R[t + 1, 0, 0, n] = val

    for u in range(u_max - 1):
        for t in range(t_max):
            for n in range(n_tot - t - u - 1):
                val = YPC * R[t, u, 0, n + 1]
                if u > 0:
                    val = val + u * R[t, u - 1, 0, n + 1]
                R[t, u + 1, 0, n] = val

    for v in range(v_max - 1):
        for t in range(t_max):
            for u in range(u_max):
                for n in range(n_tot - t - u - v - 1):
                    val = ZPC * R[t, u, v, n + 1]
                    if v > 0:
                        val = val + v * R[t, u, v - 1, n + 1]
                    R[t, u, v + 1, n] = val

    return R[:, :, :, 0]


# ============================================================
# 5. Shell-pair integrals (supports general contractions)
# ============================================================

def shell_pair_integrals(
    la: int, lb: int,
    alpha_a: Tensor, alpha_b: Tensor,
    coeff_a: Tensor, coeff_b: Tensor,
    A: Tensor, B: Tensor,
    atom_coords: Tensor, atom_charges: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute overlap (S), kinetic (T), and nuclear attraction (V) blocks
    for one shell pair, in the Cartesian basis.

    Supports general contractions (NCTR > 1).

    Args:
        la, lb: angular momenta of shells a and b.
        alpha_a, alpha_b: (nprim_a,) and (nprim_b,) primitive exponents.
        coeff_a: (nprim_a, nctr_a) contraction coefficients.
        coeff_b: (nprim_b, nctr_b) contraction coefficients.
        A, B: (3,) shell centers in Bohr.
        atom_coords: (natm, 3) nuclear positions in Bohr.
        atom_charges: (natm,) nuclear charges.

    Returns:
        S, T, V: each (nctr_a*ncart_a, nctr_b*ncart_b) Cartesian integral blocks.
    """
    dtype, device = alpha_a.dtype, alpha_a.device
    ncart_a, ncart_b = ncart(la), ncart(lb)
    cart_a, cart_b = cart_ang(la), cart_ang(lb)

    if coeff_a.dim() == 1:
        coeff_a = coeff_a.unsqueeze(1)
    if coeff_b.dim() == 1:
        coeff_b = coeff_b.unsqueeze(1)
    nctr_a, nctr_b = coeff_a.shape[1], coeff_b.shape[1]

    na, nb = alpha_a.shape[0], alpha_b.shape[0]
    npairs = na * nb

    alpha = alpha_a.repeat_interleave(nb)
    beta = alpha_b.repeat(na)

    zeta = alpha + beta

    i_max = la + 1
    j_max = lb + 3

    Ex = compute_E(i_max, j_max, alpha, beta,
                   A[0].expand(npairs), B[0].expand(npairs))
    Ey = compute_E(i_max, j_max, alpha, beta,
                   A[1].expand(npairs), B[1].expand(npairs))
    Ez = compute_E(i_max, j_max, alpha, beta,
                   A[2].expand(npairs), B[2].expand(npairs))

    sqrt_pi_z = (math.pi / zeta).sqrt()
    pi32_z = sqrt_pi_z ** 3

    L_tot = la + lb
    r_max = L_tot + 1
    prefac = 2.0 * math.pi / zeta

    P = torch.stack(
        [(alpha * A[d] + beta * B[d]) / zeta for d in range(3)], dim=-1
    )

    R_data = []
    natm = atom_charges.shape[0]
    for ic in range(natm):
        C = atom_coords[ic]
        RPC = P - C.unsqueeze(0)
        R = compute_R(r_max, r_max, r_max, L_tot, zeta, RPC)
        R_data.append((-atom_charges[ic], R))

    fac_ab = _CINTcommon_fac_sp(la) * _CINTcommon_fac_sp(lb)

    nao_a = nctr_a * ncart_a
    nao_b = nctr_b * ncart_b
    S = torch.zeros(nao_a, nao_b, dtype=dtype, device=device)
    T = torch.zeros(nao_a, nao_b, dtype=dtype, device=device)
    V = torch.zeros(nao_a, nao_b, dtype=dtype, device=device)

    def _contract(prim_flat: Tensor) -> Tensor:
        prim_mat = prim_flat.reshape(na, nb)
        return coeff_a.T @ prim_mat @ coeff_b

    for m, (lxa, lya, lza) in enumerate(cart_a):
        for n, (lxb, lyb, lzb) in enumerate(cart_b):
            s_prim = pi32_z * Ex[lxa, lxb, 0] * Ey[lya, lyb, 0] * Ez[lza, lzb, 0]

            sx = sqrt_pi_z * Ex[lxa, lxb, 0]
            sy = sqrt_pi_z * Ey[lya, lyb, 0]
            sz = sqrt_pi_z * Ez[lza, lzb, 0]

            tx = beta * (2 * lxb + 1) * sx - 2.0 * beta * beta * sqrt_pi_z * Ex[lxa, lxb + 2, 0]
            if lxb >= 2:
                tx = tx - 0.5 * lxb * (lxb - 1) * sqrt_pi_z * Ex[lxa, lxb - 2, 0]
            ty = beta * (2 * lyb + 1) * sy - 2.0 * beta * beta * sqrt_pi_z * Ey[lya, lyb + 2, 0]
            if lyb >= 2:
                ty = ty - 0.5 * lyb * (lyb - 1) * sqrt_pi_z * Ey[lya, lyb - 2, 0]
            tz = beta * (2 * lzb + 1) * sz - 2.0 * beta * beta * sqrt_pi_z * Ez[lza, lzb + 2, 0]
            if lzb >= 2:
                tz = tz - 0.5 * lzb * (lzb - 1) * sqrt_pi_z * Ez[lza, lzb - 2, 0]

            t_prim = tx * sy * sz + sx * ty * sz + sx * sy * tz

            ex = Ex[lxa, lxb, :lxa + lxb + 1]
            ey = Ey[lya, lyb, :lya + lyb + 1]
            ez = Ez[lza, lzb, :lza + lzb + 1]

            v_prim = torch.zeros(npairs, dtype=dtype, device=device)
            for neg_Z, R in R_data:
                r = R[:lxa + lxb + 1, :lya + lyb + 1, :lza + lzb + 1]
                E_prod = (
                    ex[:, None, None, :]
                    * ey[None, :, None, :]
                    * ez[None, None, :, :]
                )
                v_prim += neg_Z * prefac * (E_prod * r).sum(dim=(0, 1, 2))

            S_c = _contract(s_prim) * fac_ab
            T_c = _contract(t_prim) * fac_ab
            V_c = _contract(v_prim) * fac_ab

            for ic in range(nctr_a):
                for jc in range(nctr_b):
                    S[ic * ncart_a + m, jc * ncart_b + n] = S_c[ic, jc]
                    T[ic * ncart_a + m, jc * ncart_b + n] = T_c[ic, jc]
                    V[ic * ncart_a + m, jc * ncart_b + n] = V_c[ic, jc]

    return S, T, V


# ============================================================
# 6. Single-molecule integral builder
# ============================================================

def build_integrals(
    mol,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build S, T, V, hcore for a PySCF Mole object.

    Computes integrals in the Cartesian basis, then transforms to spherical
    via PySCF's cart2sph_coeff.

    Returns:
        S, T, V, hcore: each (nao, nao) in PySCF's default (spherical) basis.
    """
    nbas = mol.nbas

    atom_coords = torch.tensor(mol.atom_coords(), dtype=dtype, device=device)
    atom_charges = torch.tensor(
        [mol.atom_charge(i) for i in range(mol.natm)], dtype=dtype, device=device
    )

    ao_loc = mol.ao_loc_nr(cart=True)
    nao_cart = ao_loc[-1]

    S_cart = torch.zeros(nao_cart, nao_cart, dtype=dtype, device=device)
    T_cart = torch.zeros(nao_cart, nao_cart, dtype=dtype, device=device)
    V_cart = torch.zeros(nao_cart, nao_cart, dtype=dtype, device=device)

    for i in range(nbas):
        li = mol.bas_angular(i)
        ai = mol.bas_atom(i)
        Ai = torch.tensor(mol.atom_coord(ai), dtype=dtype, device=device)
        alpha_i = torch.tensor(mol.bas_exp(i), dtype=dtype, device=device)
        coeff_i = torch.tensor(mol._libcint_ctr_coeff(i).copy(), dtype=dtype, device=device)
        si, ei = ao_loc[i], ao_loc[i + 1]

        for j in range(i, nbas):
            lj = mol.bas_angular(j)
            aj = mol.bas_atom(j)
            Aj = torch.tensor(mol.atom_coord(aj), dtype=dtype, device=device)
            alpha_j = torch.tensor(mol.bas_exp(j), dtype=dtype, device=device)
            coeff_j = torch.tensor(mol._libcint_ctr_coeff(j).copy(), dtype=dtype, device=device)
            sj, ej = ao_loc[j], ao_loc[j + 1]

            Sb, Tb, Vb = shell_pair_integrals(
                li, lj, alpha_i, alpha_j, coeff_i, coeff_j,
                Ai, Aj, atom_coords, atom_charges,
            )

            S_cart[si:ei, sj:ej] = Sb
            T_cart[si:ei, sj:ej] = Tb
            V_cart[si:ei, sj:ej] = Vb
            if i != j:
                S_cart[sj:ej, si:ei] = Sb.T
                T_cart[sj:ej, si:ei] = Tb.T
                V_cart[sj:ej, si:ei] = Vb.T

    c2s_np = _cart2sph(mol)
    if c2s_np is not None:
        c2s = torch.tensor(c2s_np, dtype=dtype, device=device)
        S = c2s.T @ S_cart @ c2s
        T = c2s.T @ T_cart @ c2s
        V = c2s.T @ V_cart @ c2s
    else:
        S, T, V = S_cart, T_cart, V_cart

    hcore = T + V
    return S, T, V, hcore


# ============================================================
# 7. Batched hcore builder (angular-momentum grouped, fast)
# ============================================================

def build_hcore_batch(
    mol,
    coords_batch: Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Build hcore for a batch of molecules sharing the same basis set.

    Groups all shell pairs by (la, lb) so compute_E / compute_R are called
    once per angular-momentum type, not once per shell pair.  This reduces
    Python-loop overhead from O(nbas^2) to O(n_types) ~ 6.

    Args:
        mol: PySCF Mole template (defines basis set).
        coords_batch: (nmol, natm, 3) atomic coordinates in Bohr.
        device: torch device.
        dtype: torch dtype.

    Returns:
        hcore_batch: (nmol, nao, nao) in PySCF's default basis.
    """
    nmol = coords_batch.shape[0]
    nbas = mol.nbas
    natm = mol.natm

    atom_charges = torch.tensor(
        [mol.atom_charge(i) for i in range(natm)], dtype=dtype, device=device
    )
    ao_loc = mol.ao_loc_nr(cart=True)
    nao_cart = ao_loc[-1]
    hcore_cart = torch.zeros(nmol, nao_cart, nao_cart, dtype=dtype, device=device)

    # --- Group shell pairs by (la, lb) ---
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i in range(nbas):
        la = mol.bas_angular(i)
        for j in range(i, nbas):
            lb = mol.bas_angular(j)
            groups[(la, lb)].append((i, j))

    # --- Process each angular-momentum group ---
    for (la, lb), shell_pairs in groups.items():
        K = len(shell_pairs)
        cart_a, cart_b = cart_ang(la), cart_ang(lb)

        # Collect primitive-pair data for ALL shell pairs in this group
        alpha_parts, beta_parts, cc_parts = [], [], []
        atom_i_list, atom_j_list = [], []
        pair_idx_list = []
        total_prims = 0

        for k, (i, j) in enumerate(shell_pairs):
            ai, aj = mol.bas_atom(i), mol.bas_atom(j)
            exps_i, exps_j = mol.bas_exp(i), mol.bas_exp(j)
            ci = mol._libcint_ctr_coeff(i)[:, 0]
            cj = mol._libcint_ctr_coeff(j)[:, 0]
            na, nb_ = len(exps_i), len(exps_j)
            np_ = na * nb_

            alpha_parts.append(np.repeat(exps_i, nb_))
            beta_parts.append(np.tile(exps_j, na))
            cc_parts.append(np.outer(ci, cj).ravel())
            atom_i_list.extend([ai] * np_)
            atom_j_list.extend([aj] * np_)
            pair_idx_list.extend([k] * np_)
            total_prims += np_

        alpha_base = torch.tensor(np.concatenate(alpha_parts), dtype=dtype, device=device)
        beta_base = torch.tensor(np.concatenate(beta_parts), dtype=dtype, device=device)
        cc_base = torch.tensor(np.concatenate(cc_parts), dtype=dtype, device=device)
        atom_i_idx = torch.tensor(atom_i_list, dtype=torch.long, device=device)
        atom_j_idx = torch.tensor(atom_j_list, dtype=torch.long, device=device)
        pair_idx = torch.tensor(pair_idx_list, dtype=torch.long, device=device)

        # Tile for nmol molecules
        total = nmol * total_prims
        alpha = alpha_base.repeat(nmol)
        beta = beta_base.repeat(nmol)
        cc = cc_base.repeat(nmol)

        # Centers per molecule
        Ai = coords_batch[:, atom_i_idx, :]  # (nmol, total_prims, 3)
        Aj = coords_batch[:, atom_j_idx, :]
        Ai_flat = Ai.reshape(-1, 3)
        Aj_flat = Aj.reshape(-1, 3)

        zeta = alpha + beta

        # ONE call to compute_E per dimension
        i_max = la + 1
        j_max = lb + 3
        Ex = compute_E(i_max, j_max, alpha, beta, Ai_flat[:, 0], Aj_flat[:, 0])
        Ey = compute_E(i_max, j_max, alpha, beta, Ai_flat[:, 1], Aj_flat[:, 1])
        Ez = compute_E(i_max, j_max, alpha, beta, Ai_flat[:, 2], Aj_flat[:, 2])

        sqrt_pi_z = (math.pi / zeta).sqrt()
        prefac = 2.0 * math.pi / zeta
        fac_ab = _CINTcommon_fac_sp(la) * _CINTcommon_fac_sp(lb)

        # Gaussian product center P
        P = torch.stack(
            [(alpha * Ai_flat[:, d] + beta * Aj_flat[:, d]) / zeta for d in range(3)],
            dim=-1,
        )

        # ONE call to compute_R per atom
        L_tot = la + lb
        r_max = L_tot + 1
        R_data = []
        for ic in range(natm):
            C = coords_batch[:, ic, :].repeat_interleave(total_prims, dim=0)
            RPC = P - C
            R = compute_R(r_max, r_max, r_max, L_tot, zeta, RPC)
            R_data.append((-atom_charges[ic], R))

        # Scatter indices
        row_bases = torch.tensor([ao_loc[i] for i, j in shell_pairs], dtype=torch.long, device=device)
        col_bases = torch.tensor([ao_loc[j] for i, j in shell_pairs], dtype=torch.long, device=device)
        is_offdiag = torch.tensor([i != j for i, j in shell_pairs], dtype=torch.bool, device=device)

        pair_idx_exp = pair_idx.unsqueeze(0).expand(nmol, -1)

        # Compute integrals for each Cartesian component pair
        for m, (lxa, lya, lza) in enumerate(cart_a):
            for n, (lxb, lyb, lzb) in enumerate(cart_b):
                sx = sqrt_pi_z * Ex[lxa, lxb, 0]
                sy = sqrt_pi_z * Ey[lya, lyb, 0]
                sz = sqrt_pi_z * Ez[lza, lzb, 0]

                tx = beta * (2 * lxb + 1) * sx - 2.0 * beta * beta * sqrt_pi_z * Ex[lxa, lxb + 2, 0]
                if lxb >= 2:
                    tx = tx - 0.5 * lxb * (lxb - 1) * sqrt_pi_z * Ex[lxa, lxb - 2, 0]
                ty = beta * (2 * lyb + 1) * sy - 2.0 * beta * beta * sqrt_pi_z * Ey[lya, lyb + 2, 0]
                if lyb >= 2:
                    ty = ty - 0.5 * lyb * (lyb - 1) * sqrt_pi_z * Ey[lya, lyb - 2, 0]
                tz = beta * (2 * lzb + 1) * sz - 2.0 * beta * beta * sqrt_pi_z * Ez[lza, lzb + 2, 0]
                if lzb >= 2:
                    tz = tz - 0.5 * lzb * (lzb - 1) * sqrt_pi_z * Ez[lza, lzb - 2, 0]

                t_prim = tx * sy * sz + sx * ty * sz + sx * sy * tz

                ex = Ex[lxa, lxb, :lxa + lxb + 1]
                ey = Ey[lya, lyb, :lya + lyb + 1]
                ez = Ez[lza, lzb, :lza + lzb + 1]
                v_prim = torch.zeros(total, dtype=dtype, device=device)
                for neg_Z, R in R_data:
                    r = R[:lxa + lxb + 1, :lya + lyb + 1, :lza + lzb + 1]
                    E_prod = ex[:, None, None, :] * ey[None, :, None, :] * ez[None, None, :, :]
                    v_prim += neg_Z * prefac * (E_prod * r).sum(dim=(0, 1, 2))

                weighted = (cc * (t_prim + v_prim) * fac_ab).reshape(nmol, total_prims)
                h_k = torch.zeros(nmol, K, dtype=dtype, device=device)
                h_k.scatter_add_(1, pair_idx_exp, weighted)

                rows = row_bases + m
                cols = col_bases + n
                hcore_cart[:, rows, cols] = h_k
                if is_offdiag.any():
                    od = is_offdiag
                    hcore_cart[:, cols[od], rows[od]] = h_k[:, od]

    # Cart → Sph
    c2s_np = _cart2sph(mol)
    if c2s_np is not None:
        c2s = torch.tensor(c2s_np, dtype=dtype, device=device)
        hcore = torch.einsum("ab,mbc,cd->mad", c2s.T, hcore_cart, c2s)
    else:
        hcore = hcore_cart

    return hcore


# ============================================================
# 8. Unified SCF interface
# ============================================================

_SCF_CLASSES = {
    "rhf": "pyscf.scf:RHF",
    "uhf": "pyscf.scf:UHF",
    "rks": "pyscf.dft:RKS",
    "uks": "pyscf.dft:UKS",
}


def _get_scf_class(method: str):
    """Import and return the PySCF SCF class for the given method name."""
    key = method.lower()
    if key not in _SCF_CLASSES:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_SCF_CLASSES)}")
    mod_path, cls_name = _SCF_CLASSES[key].split(":")
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def run_scf(
    mol,
    method: str = "uks",
    xc: str = "b3lyp",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    dm0=None,
    **scf_kwargs,
):
    """Run SCF using torch-computed hcore and overlap integrals.

    Builds S and hcore via PyTorch, injects them into PySCF's SCF solver,
    and runs the self-consistent field calculation.

    Args:
        mol: PySCF Mole object.
        method: 'rhf', 'uhf', 'rks', or 'uks'.
        xc: Exchange-correlation functional (only for 'rks'/'uks').
        device: Torch device for integral computation.
        dtype: Torch dtype for integrals.
        dm0: Initial density matrix. If None, uses PySCF default.
        **scf_kwargs: Extra keyword arguments passed to the SCF object
            (e.g. max_cycle=200, conv_tol=1e-12).

    Returns:
        mf: Converged PySCF SCF object with all standard attributes
            (e_tot, mo_coeff, mo_energy, mo_occ, make_rdm1(), ...).
    """
    S, T, V, hcore = build_integrals(mol, device=device, dtype=dtype)
    S_np = S.cpu().numpy()
    H_np = hcore.cpu().numpy()

    cls = _get_scf_class(method)
    mf = cls(mol)

    if method.lower() in ("rks", "uks"):
        mf.xc = xc

    for k, v in scf_kwargs.items():
        setattr(mf, k, v)

    mf.get_hcore = lambda *args: H_np
    mf.get_ovlp = lambda *args: S_np

    mf.kernel(dm0=dm0)
    return mf


def run_scf_batch(
    mol,
    coords_batch: Tensor,
    method: str = "uks",
    xc: str = "b3lyp",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    **scf_kwargs,
) -> list:
    """Run SCF for a batch of molecules with different geometries.

    Computes hcore in batch on GPU, then runs PySCF SCF for each molecule.
    The hcore computation is fast (GPU-batched); SCF iterations use PySCF's
    CPU routines for J/K/V_xc.

    Args:
        mol: PySCF Mole template (defines basis set).
        coords_batch: (nmol, natm, 3) atomic coordinates in Bohr.
        method: 'rhf', 'uhf', 'rks', or 'uks'.
        xc: Exchange-correlation functional.
        device: Torch device for batch hcore computation.
        dtype: Torch dtype.
        **scf_kwargs: Extra kwargs for SCF (max_cycle, conv_tol, ...).

    Returns:
        List of converged PySCF SCF objects, one per geometry.
    """
    S, _, _, _ = build_integrals(mol, device="cpu", dtype=dtype)
    S_np = S.numpy()

    hcore_all = build_hcore_batch(mol, coords_batch, device=device, dtype=dtype)
    hcore_all = hcore_all.cpu().numpy()  # (nmol, nao, nao)

    nmol = coords_batch.shape[0]
    coords_np = coords_batch.cpu().numpy() if isinstance(coords_batch, Tensor) else coords_batch

    cls = _get_scf_class(method)
    results = []

    for i in range(nmol):
        mol_i = mol.copy()
        mol_i.set_geom_(coords_np[i], unit="Bohr")
        mol_i.build()

        mf = cls(mol_i)
        if method.lower() in ("rks", "uks"):
            mf.xc = xc
        for k, v in scf_kwargs.items():
            setattr(mf, k, v)

        H_i = hcore_all[i]
        S_i = mol_i.intor("int1e_ovlp")  # S changes with geometry
        mf.get_hcore = lambda *args, _h=H_i: _h
        mf.get_ovlp = lambda *args, _s=S_i: _s

        mf.kernel()
        results.append(mf)

    return results
