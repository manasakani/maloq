"""GPU-accelerated Fock matrix construction.

Computes init Hamiltonian F = hcore + J + V_xc with GPU-accelerated
AO evaluation, density computation, and V_xc matrix assembly.
libxc runs on CPU for bit-identical XC functional results.

Usage:
    from fock import get_init_fock_gpu

    F = get_init_fock_gpu(mol, dm, xc="pbe", device="cuda")
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .integrals import cart_ang, ncart, _CINTcommon_fac_sp, _cart2sph


# ---------------------------------------------------------------------------
# Torch eval_ao: evaluate GTOs on arbitrary grid points
# ---------------------------------------------------------------------------

def eval_ao_torch(
    mol,
    coords: np.ndarray,
    deriv: int = 0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Evaluate AO basis functions on grid points, identical to PySCF eval_ao.

    Batched by angular momentum: all shells with the same l are processed
    in a single GPU kernel, minimizing launch overhead.

    Args:
        mol: PySCF Mole object.
        coords: (N_grid, 3) grid coordinates in Bohr.
        deriv: 0 = values only, 1 = values + gradient (for GGA).
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        deriv=0: (N_grid, nao) AO values.
        deriv=1: (4, N_grid, nao) = [value, d/dx, d/dy, d/dz].
    """
    dev = torch.device(device)
    ngrids = coords.shape[0]
    nbas = mol.nbas
    ao_loc = mol.ao_loc_nr(cart=True)
    nao_cart = ao_loc[-1]
    ncomp = 4 if deriv >= 1 else 1

    grid_t = torch.from_numpy(np.ascontiguousarray(coords)).pin_memory().to(dev, non_blocking=True)

    ao_cart = torch.zeros(ncomp, ngrids, nao_cart, dtype=dtype, device=dev)

    # Group shells by angular momentum for batched processing
    from collections import defaultdict
    l_groups: dict[int, list[int]] = defaultdict(list)
    for i in range(nbas):
        l_groups[mol.bas_angular(i)].append(i)

    for li, shell_ids in l_groups.items():
        ns = len(shell_ids)
        fac_sp = _CINTcommon_fac_sp(li)
        nc = ncart(li)
        cart_indices = cart_ang(li)

        # Gather all centers, exponents, coefficients for this l-group
        max_nprim = max(mol.bas_nprim(i) for i in shell_ids)
        centers = torch.zeros(ns, 3, dtype=dtype, device=dev)
        exps_pad = torch.zeros(ns, max_nprim, dtype=dtype, device=dev)
        coeff_pad = torch.zeros(ns, max_nprim, dtype=dtype, device=dev)
        ao_starts = []

        for idx, i in enumerate(shell_ids):
            ai = mol.bas_atom(i)
            centers[idx] = torch.tensor(mol.atom_coord(ai), dtype=dtype, device=dev)
            nprim = mol.bas_nprim(i)
            exps_pad[idx, :nprim] = torch.tensor(mol.bas_exp(i), dtype=dtype, device=dev)
            c = mol._libcint_ctr_coeff(i).copy().ravel()
            coeff_pad[idx, :nprim] = torch.tensor(c, dtype=dtype, device=dev)
            ao_starts.append(ao_loc[i])

        # dr: (ns, N_grid, 3)
        dr = grid_t.unsqueeze(0) - centers.unsqueeze(1)  # broadcast
        # r2: (ns, N_grid)
        r2 = (dr * dr).sum(dim=-1)

        # Radial: exp_vals (ns, N_grid, max_nprim)
        exp_vals = torch.exp(-exps_pad.unsqueeze(1) * r2.unsqueeze(2))
        # Contracted radial: (ns, N_grid)
        rad = (exp_vals * coeff_pad.unsqueeze(1)).sum(dim=-1)

        if deriv >= 1:
            # drad_factor = Σ_k α_k c_k exp(-α_k r²)
            drad_factor = (exp_vals * (exps_pad * coeff_pad).unsqueeze(1)).sum(dim=-1)  # (ns, N_grid)

        # x, y, z components: (ns, N_grid)
        xx, yy, zz = dr[..., 0], dr[..., 1], dr[..., 2]

        # Pre-compute column indices: (ns,) per Cartesian component
        col_indices = [
            torch.tensor([s + c_idx for s in ao_starts], dtype=torch.long, device=dev)
            for c_idx in range(nc)
        ]

        for c_idx, (lx, ly, lz) in enumerate(cart_indices):
            ang = torch.ones(ns, ngrids, dtype=dtype, device=dev)
            if lx > 0:
                ang = ang * xx.pow(lx)
            if ly > 0:
                ang = ang * yy.pow(ly)
            if lz > 0:
                ang = ang * zz.pow(lz)

            val = ang * rad * fac_sp  # (ns, N_grid)

            # Batch scatter: ao_cart[0, :, cols] = val.T
            cols = col_indices[c_idx]
            ao_cart[0, :, cols] = val.T  # (N_grid, ns)

            if deriv >= 1:
                for d in range(3):
                    if d == 0:
                        dang = (lx * xx.pow(max(lx-1, 0)) * yy.pow(ly) * zz.pow(lz)
                                if lx > 0 else torch.zeros_like(ang))
                    elif d == 1:
                        dang = (xx.pow(lx) * ly * yy.pow(max(ly-1, 0)) * zz.pow(lz)
                                if ly > 0 else torch.zeros_like(ang))
                    else:
                        dang = (xx.pow(lx) * yy.pow(ly) * lz * zz.pow(max(lz-1, 0))
                                if lz > 0 else torch.zeros_like(ang))

                    dval = (dang * rad + ang * (-2.0 * dr[..., d] * drad_factor)) * fac_sp
                    ao_cart[1 + d, :, cols] = dval.T

    # Cart → spherical
    c2s_np = _cart2sph(mol)
    if c2s_np is not None:
        c2s = torch.tensor(c2s_np, dtype=dtype, device=dev)
        ao_sph = torch.einsum("cgk,ks->cgs", ao_cart, c2s)
    else:
        ao_sph = ao_cart

    if deriv == 0:
        return ao_sph[0]
    return ao_sph


def get_vxc_gpu(
    mol,
    dm: np.ndarray,
    xc: str = "pbe",
    grids=None,
    device: str = "cuda",
    ao_eval: str = "auto",
) -> np.ndarray:
    """Compute V_xc matrix with GPU-accelerated density + assembly.

    Pipeline:
        1. eval_ao (PySCF CPU or torch GPU, auto-selected by nao)
        2. eval_rho on GPU (matmul)
        3. libxc on CPU (bit-identical XC)
        4. V_xc assembly on GPU (matmul)

    Args:
        mol: PySCF Mole object.
        dm: (nao, nao) density matrix.
        xc: XC functional name (e.g. "pbe", "b3lyp", "lda").
        grids: PySCF Grids object. Built from mol if None.
        device: Torch device for GPU matmuls.

    Returns:
        vxc: (nao, nao) V_xc matrix, numerically identical to PySCF.
    """
    from pyscf.dft import numint as ni, libxc, gen_grid

    # Build grids if needed
    if grids is None:
        grids = gen_grid.Grids(mol)
        grids.build()
    elif grids.coords is None:
        grids.build()

    coords = grids.coords   # (N_grid, 3)
    weights = grids.weights  # (N_grid,)

    # Determine XC type
    xctype = libxc.xc_type(xc)  # "LDA", "GGA", "MGGA"
    deriv = 1 if xctype == "GGA" else 0

    # --- Step 1: eval_ao ---
    dev = torch.device(device)

    # Auto-select: torch GPU for large molecules, PySCF CPU for small
    use_torch_ao = (ao_eval == "torch") or (ao_eval == "auto" and mol.nao >= 150)

    if use_torch_ao:
        ao_t = eval_ao_torch(mol, coords, deriv=deriv, device=device)
    else:
        ao_cpu = ni.eval_ao(mol, coords, deriv=deriv)
        ao_contig = np.ascontiguousarray(ao_cpu)
        ao_pin = torch.from_numpy(ao_contig).pin_memory()
        ao_t = ao_pin.to(dev, non_blocking=True)

    # --- Step 2: eval_rho on GPU ---
    dm_t = torch.as_tensor(dm, dtype=torch.float64, device=dev)

    if xctype == "LDA":
        # rho(r) = sum_ij D_ij phi_i(r) phi_j(r)
        c0 = ao_t @ dm_t                     # (N, nao)
        rho_t = (c0 * ao_t).sum(dim=-1)      # (N,)
        rho_np = rho_t.cpu().numpy()
    else:
        # GGA: ao_t has shape (4, N, nao) = [value, dx, dy, dz]
        phi = ao_t[0]       # (N, nao)
        dphi = ao_t[1:4]    # (3, N, nao)

        c0 = phi @ dm_t     # (N, nao)
        rho = (c0 * phi).sum(dim=-1)             # (N,)
        # grad_rho[d] = 2 * sum_ij D_ij dphi_d_i * phi_j
        drho = 2.0 * (c0.unsqueeze(0) * dphi).sum(dim=-1)  # (3, N)
        sigma = (drho * drho).sum(dim=0)          # (N,) = |∇ρ|²

        # libxc GGA input: (4, N_grid) = [rho, drho_x, drho_y, drho_z]
        rho_np = np.empty((4, len(weights)))
        rho_np[0] = rho.cpu().numpy()
        rho_np[1:4] = drho.cpu().numpy()

    # --- Step 3: libxc on CPU (bit-identical) ---
    exc, vxc_tuple = libxc.eval_xc(xc, rho_np)[:2]

    # --- Step 4: V_xc assembly on GPU ---
    w_t = torch.as_tensor(weights, dtype=torch.float64, device=dev)

    if xctype == "LDA":
        vrho = torch.as_tensor(vxc_tuple[0], dtype=torch.float64, device=dev)
        wv = w_t * vrho               # (N,)
        aow = ao_t * wv.unsqueeze(1)  # (N, nao)
        vxc_mat = ao_t.T @ aow        # (nao, nao)
    else:
        vrho = torch.as_tensor(vxc_tuple[0], dtype=torch.float64, device=dev)   # (N,)
        vsigma = torch.as_tensor(vxc_tuple[1], dtype=torch.float64, device=dev) # (N,)

        # GGA formula (PySCF numint.py lines 1152-1157):
        # wv[0] = 0.5 * vrho * w   (0.5 for symmetrization)
        # wv[d] = vsigma * 2 * drho[d] * w
        wv0 = w_t * vrho * 0.5                      # (N,)
        wv_sigma = w_t * vsigma * 2.0               # (N,)

        aow = phi * wv0.unsqueeze(1)                 # (N, nao)
        for d in range(3):
            aow = aow + dphi[d] * (wv_sigma * drho[d]).unsqueeze(1)

        vxc_mat = phi.T @ aow                        # (nao, nao)
        vxc_mat = vxc_mat + vxc_mat.T                # symmetrize

    return vxc_mat.cpu().numpy()


def get_init_fock_gpu(
    mol,
    dm: np.ndarray,
    xc: str = "pbe",
    density_fit: bool = True,
    mf=None,
    device: str = "cuda",
) -> np.ndarray:
    """Compute full init Fock matrix: hcore + J + V_xc.

    hcore and J use PySCF CPU (fast). V_xc uses GPU-accelerated matmuls.

    Args:
        mol: PySCF Mole object.
        dm: (nao, nao) density matrix.
        xc: XC functional name.
        density_fit: Use density fitting for J.
        mf: Pre-built PySCF mean-field object. If None, builds one.
            Reusing avoids grid rebuild overhead on repeated calls.
        device: Torch device.

    Returns:
        F: (nao, nao) Fock matrix = hcore + J + V_xc.
    """
    from pyscf import dft

    # hcore (fast, ~60ms)
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    # Build mf if not provided
    if mf is None:
        mf = dft.RKS(mol)
        if density_fit:
            mf = mf.density_fit()
        mf.xc = xc

    # J via density fitting (fast, ~130ms)
    j = mf.get_j(dm=dm)

    # V_xc (GPU-accelerated)
    vxc = get_vxc_gpu(mol, dm, xc=xc, grids=mf.grids, device=device)

    return hcore + j + vxc
