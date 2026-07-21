# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""
Fused Wigner D-matrix kernel supporting lmax up to 8.

Block packing strategy:
  lmax <= 3 : 1 group  [L=0,1,2,3]  (1+3+5+7=16, exactly full)
  lmax  = 4 : 2 groups [L=0,1,2,3] + [L=4]
  lmax  = 5 : 3 groups [L=0,1,2,3] + [L=4] + [L=5]
  lmax  = 6 : 4 groups [L=0,1,2,3] + [L=4] + [L=5] + [L=6]
  lmax  = 7 : 4 groups [L=0,7] + [L=1,6] + [L=2,5] + [L=3,4]  (zero waste!)
  lmax  = 8 : 4 groups (same as lmax=7) + 1 separate 32x32 for L=8

Key insight for lmax=7: 2*L_a+1 + 2*L_b+1 = 16 when L_a + L_b = 7.
For lmax<=6: keep L=0-3 together (fill one 16x16), then L=4+ as singles.

Single kernel launch computes Euler angles once, then processes
all 16x16 groups sequentially. A second kernel handles L=8 if needed.
"""

import random
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_CACHE = {}


def _compute_groups(lmax):
    """
    Fixed packing of L values into 16x16 and 32x32 blocks.

    lmax <= 3 : [0,1,2,3] subset as one group (exactly 16 for lmax=3)
    lmax  4-6 : [0,1,2,3] + individual [4], [5], [6] as needed
    lmax  = 7 : optimal pairs [0,7],[1,6],[2,5],[3,4] (zero waste)
    lmax  = 8 : same 4 pairs as lmax=7 + [8] in 32x32
    """
    if lmax <= 3:
        groups_16 = [list(range(lmax + 1))]
        groups_32 = []
    elif lmax <= 6:
        groups_16 = [[0, 1, 2, 3]] + [[l] for l in range(4, lmax + 1)]
        groups_32 = []
    elif lmax == 7:
        groups_16 = [[0, 7], [1, 6], [2, 5], [3, 4]]
        groups_32 = []
    else:  # lmax == 8
        groups_16 = [[0, 7], [1, 6], [2, 5], [3, 4]]
        groups_32 = [[8]]

    return groups_16, groups_32


def _l_output_start(l):
    """Output row/col start index for a given L: sum(2k+1 for k=0..l-1) = l^2."""
    return l * l


def _build_group_meta(lmax, device):
    """Build padded J matrices and structure metadata for groups up to lmax."""
    key = (lmax, device)
    if key in _CACHE:
        return _CACHE[key]

    groups_16, groups_32 = _compute_groups(lmax)
    out_dim = int((lmax + 1) ** 2)
    n16 = len(groups_16)

    # struct layout per group: [l_map(16), local_start(16), out_offset(16)] = 48 ints
    STRUCT_STRIDE = 48
    jd_stack = torch.zeros(max(n16, 1), 16, 16, device=device, dtype=torch.float32)
    struct_stack = torch.zeros(max(n16, 1), STRUCT_STRIDE, device=device, dtype=torch.int32)
    group_sizes = torch.zeros(max(n16, 1), device=device, dtype=torch.int32)

    for gi, l_list in enumerate(groups_16):
        l_map = []
        local_start = []
        out_offset = []
        pos = 0
        for l in l_list:
            sz = 2 * l + 1
            l_map.extend([l] * sz)
            local_start.extend([pos] * sz)
            abs_start = _l_output_start(l)
            out_offset.extend([abs_start + k for k in range(sz)])
            pos += sz
        grp_sz = pos
        group_sizes[gi] = grp_sz
        # Pad to 16
        l_map.extend([0] * (16 - len(l_map)))
        local_start.extend([0] * (16 - len(local_start)))
        out_offset.extend([0] * (16 - len(out_offset)))
        struct_stack[gi, :16] = torch.tensor(l_map, dtype=torch.int32)
        struct_stack[gi, 16:32] = torch.tensor(local_start, dtype=torch.int32)
        struct_stack[gi, 32:48] = torch.tensor(out_offset, dtype=torch.int32)

    result = {
        'out_dim': out_dim,
        'n16': n16,
        'groups_16': groups_16,
        'has_32': len(groups_32) > 0,
        'groups_32': groups_32,
        'jd_stack': jd_stack,
        'struct_stack': struct_stack,
        'group_sizes': group_sizes,
        'jd_ready': False,
    }

    if groups_32:
        l = groups_32[0][0]  # L=8
        sz = 2 * l + 1  # 17
        l_map_32 = [l] * sz + [0] * (32 - sz)
        start_map_32 = [0] * 32
        abs_start = _l_output_start(l)
        out_off_32 = [abs_start + k for k in range(sz)] + [0] * (32 - sz)
        struct_32 = torch.zeros(96, device=device, dtype=torch.int32)  # 32*3
        struct_32[:32] = torch.tensor(l_map_32, dtype=torch.int32)
        struct_32[32:64] = torch.tensor(start_map_32, dtype=torch.int32)
        struct_32[64:96] = torch.tensor(out_off_32, dtype=torch.int32)
        jd_32 = torch.zeros(32, 32, device=device, dtype=torch.float32)
        result['struct_32'] = struct_32
        result['jd_32'] = jd_32
        result['size_32'] = sz

    _CACHE[key] = result
    return result


def _fill_jd(meta, jd_list, lmax):
    """Fill J matrix data into pre-allocated buffers. Skipped after first call."""
    if meta['jd_ready']:
        return
    for gi, l_list in enumerate(meta['groups_16']):
        pos = 0
        for l in l_list:
            sz = 2 * l + 1
            meta['jd_stack'][gi, pos:pos+sz, pos:pos+sz] = jd_list[l].float()
            pos += sz

    for l_list in meta['groups_32']:
        l = l_list[0]
        sz = 2 * l + 1
        meta['jd_32'][:sz, :sz] = jd_list[l].float()
    meta['jd_ready'] = True


# ---------------------------------------------------------------------------
# Triton kernel: process one 16x16 group
# ---------------------------------------------------------------------------
@triton.jit
def _compute_wigner_block_16(
    jd_ptr, struct_ptr, output_ptr,
    alpha, beta, gamma,
    pid, out_dim: tl.constexpr, group_size,
):
    """Compute one 16x16 Wigner block and scatter-store to output."""
    BS: tl.constexpr = 16
    offs = tl.arange(0, BS)

    # Load structure: l_map(16), local_start(16), out_offset(16)
    l_val = tl.load(struct_ptr + offs)
    start_idx = tl.load(struct_ptr + BS + offs)
    out_off = tl.load(struct_ptr + 2 * BS + offs)

    # Load J matrix
    J = tl.load(jd_ptr + offs[:, None] * BS + offs[None, :]).to(tl.float32)

    # Compute z-rotation matrices
    local_i = offs - start_idx
    freq = (l_val - local_i).to(tl.float32)
    block_sz = 2 * l_val + 1
    anti_col = start_idx + (block_sz - 1 - local_i)

    is_diag = (offs[:, None] == offs[None, :])
    is_anti = (offs[None, :] == anti_col[:, None])
    valid = (offs[:, None] < group_size) & (offs[None, :] < group_size)

    freq_2d = freq[:, None]
    sin_a = tl.sin(freq_2d * alpha)
    cos_a = tl.cos(freq_2d * alpha)
    sin_b = tl.sin(freq_2d * beta)
    cos_b = tl.cos(freq_2d * beta)
    sin_c = tl.sin(freq_2d * gamma)
    cos_c = tl.cos(freq_2d * gamma)

    Xa = tl.zeros((BS, BS), dtype=tl.float32)
    Xb = tl.zeros((BS, BS), dtype=tl.float32)
    Xc = tl.zeros((BS, BS), dtype=tl.float32)

    Xa = tl.where(is_anti & valid, sin_a, Xa)
    Xb = tl.where(is_anti & valid, sin_b, Xb)
    Xc = tl.where(is_anti & valid, sin_c, Xc)
    Xa = tl.where(is_diag & valid, cos_a, Xa)
    Xb = tl.where(is_diag & valid, cos_b, Xb)
    Xc = tl.where(is_diag & valid, cos_c, Xc)

    # Xa @ J @ Xb @ J @ Xc
    T1 = tl.dot(Xa, J, allow_tf32=False)
    T2 = tl.dot(T1, Xb, allow_tf32=False)
    T3 = tl.dot(T2, J, allow_tf32=False)
    D = tl.dot(T3, Xc, allow_tf32=False)

    # Scatter-store using out_offset_map (handles non-contiguous L pairs)
    out_base = output_ptr + pid * out_dim * out_dim
    r = out_off[:, None]  # absolute output row for each block position
    c = out_off[None, :]  # absolute output col for each block position
    out_ptrs = out_base + r * out_dim + c
    store_mask = (offs[:, None] < group_size) & (offs[None, :] < group_size)
    tl.store(out_ptrs, D, mask=store_mask)


# ---------------------------------------------------------------------------
# Main 16x16 kernel: computes Euler angles once, iterates over groups
# ---------------------------------------------------------------------------
@triton.jit
def wigner_fused_kernel_16(
    edge_vec_ptr, jd_ptr, struct_ptr,
    group_sizes_ptr,
    output_ptr, seed,
    num_edges: tl.constexpr,
    out_dim: tl.constexpr,
    num_groups: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_edges:
        return

    # Load edge vector and compute Euler angles ONCE
    edge_x = tl.load(edge_vec_ptr + pid * 3 + 0)
    edge_y = tl.load(edge_vec_ptr + pid * 3 + 1)
    edge_z = tl.load(edge_vec_ptr + pid * 3 + 2)

    norm = tl.sqrt(edge_x * edge_x + edge_y * edge_y + edge_z * edge_z)
    norm = tl.maximum(norm, 1e-12)
    x = tl.minimum(tl.maximum(edge_x / norm, -1.0), 1.0)
    y = tl.minimum(tl.maximum(edge_y / norm, -1.0), 1.0)
    z = tl.minimum(tl.maximum(edge_z / norm, -1.0), 1.0)

    # Compute physical angles (matching torch backend)
    physical_alpha = libdevice.atan2(x, z)
    physical_beta = libdevice.acos(y)
    physical_gamma = tl.rand(seed, pid) * 6.283185307179586
        
    # Swap and negate to match torch convention: (-gamma, -beta, -alpha)
    alpha = -physical_gamma
    beta = -physical_beta
    gamma = -physical_alpha

    # Process each 16x16 group sequentially
    # struct_stride = 48 (l_map:16 + local_start:16 + out_offset:16)
    for gi in tl.static_range(num_groups):
        g_sz = tl.load(group_sizes_ptr + gi)
        _compute_wigner_block_16(
            jd_ptr + gi * 16 * 16,
            struct_ptr + gi * 48,
            output_ptr,
            alpha, beta, gamma,
            pid, out_dim, g_sz,
        )


# ---------------------------------------------------------------------------
# 32x32 kernel for L=8
# ---------------------------------------------------------------------------
@triton.jit
def wigner_fused_kernel_32(
    edge_vec_ptr, jd_ptr, struct_ptr,
    output_ptr, seed,
    num_edges: tl.constexpr,
    out_dim: tl.constexpr,
    group_size: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_edges:
        return

    BS: tl.constexpr = 32

    # Recompute Euler angles (same as 16x16 kernel)
    edge_x = tl.load(edge_vec_ptr + pid * 3 + 0)
    edge_y = tl.load(edge_vec_ptr + pid * 3 + 1)
    edge_z = tl.load(edge_vec_ptr + pid * 3 + 2)

    norm = tl.sqrt(edge_x * edge_x + edge_y * edge_y + edge_z * edge_z)
    norm = tl.maximum(norm, 1e-12)
    x = tl.minimum(tl.maximum(edge_x / norm, -1.0), 1.0)
    y = tl.minimum(tl.maximum(edge_y / norm, -1.0), 1.0)
    z = tl.minimum(tl.maximum(edge_z / norm, -1.0), 1.0)

    alpha = -libdevice.atan2(x, z)
    beta = -libdevice.acos(y)
    gamma = -tl.rand(seed, pid) * 6.283185307179586

    offs = tl.arange(0, BS)

    # Load structure: l_map(32), local_start(32), out_offset(32)
    l_val = tl.load(struct_ptr + offs)
    start_idx = tl.load(struct_ptr + BS + offs)
    out_off = tl.load(struct_ptr + 2 * BS + offs)

    # Load J matrix
    J = tl.load(jd_ptr + offs[:, None] * BS + offs[None, :]).to(tl.float32)

    # Compute z-rotation matrices
    local_i = offs - start_idx
    freq = (l_val - local_i).to(tl.float32)
    block_sz = 2 * l_val + 1
    anti_col = start_idx + (block_sz - 1 - local_i)

    is_diag = (offs[:, None] == offs[None, :])
    is_anti = (offs[None, :] == anti_col[:, None])
    valid = (offs[:, None] < group_size) & (offs[None, :] < group_size)

    freq_2d = freq[:, None]
    sin_a = tl.sin(freq_2d * alpha)
    cos_a = tl.cos(freq_2d * alpha)
    sin_b = tl.sin(freq_2d * beta)
    cos_b = tl.cos(freq_2d * beta)
    sin_c = tl.sin(freq_2d * gamma)
    cos_c = tl.cos(freq_2d * gamma)

    Xa = tl.zeros((BS, BS), dtype=tl.float32)
    Xb = tl.zeros((BS, BS), dtype=tl.float32)
    Xc = tl.zeros((BS, BS), dtype=tl.float32)

    Xa = tl.where(is_anti & valid, sin_a, Xa)
    Xb = tl.where(is_anti & valid, sin_b, Xb)
    Xc = tl.where(is_anti & valid, sin_c, Xc)
    Xa = tl.where(is_diag & valid, cos_a, Xa)
    Xb = tl.where(is_diag & valid, cos_b, Xb)
    Xc = tl.where(is_diag & valid, cos_c, Xc)

    T1 = tl.dot(Xa, J, allow_tf32=False)
    T2 = tl.dot(T1, Xb, allow_tf32=False)
    T3 = tl.dot(T2, J, allow_tf32=False)
    D = tl.dot(T3, Xc, allow_tf32=False)

    # Scatter-store using out_offset_map
    out_base = output_ptr + pid * out_dim * out_dim
    r = out_off[:, None]
    c = out_off[None, :]
    out_ptrs = out_base + r * out_dim + c
    store_mask = (offs[:, None] < group_size) & (offs[None, :] < group_size)
    tl.store(out_ptrs, D, mask=store_mask)


# ---------------------------------------------------------------------------
# Helper: extract Euler angles (for testing)
# ---------------------------------------------------------------------------
@triton.jit
def _extract_eulers_kernel(
    edge_vec_ptr, euler_ptr, seed,
    num_edges: tl.constexpr,
):
    """Extract the same Euler angles (alpha, beta, gamma) the main kernel computes."""
    pid = tl.program_id(0)
    if pid >= num_edges:
        return
    edge_x = tl.load(edge_vec_ptr + pid * 3 + 0)
    edge_y = tl.load(edge_vec_ptr + pid * 3 + 1)
    edge_z = tl.load(edge_vec_ptr + pid * 3 + 2)
    norm = tl.sqrt(edge_x * edge_x + edge_y * edge_y + edge_z * edge_z)
    norm = tl.maximum(norm, 1e-12)
    x = tl.minimum(tl.maximum(edge_x / norm, -1.0), 1.0)
    y = tl.minimum(tl.maximum(edge_y / norm, -1.0), 1.0)
    z = tl.minimum(tl.maximum(edge_z / norm, -1.0), 1.0)
    
    # Compute physical angles (matching torch backend)
    physical_alpha = libdevice.atan2(x, z)
    physical_beta = libdevice.acos(y)
    physical_gamma = tl.rand(seed, pid) * 6.283185307179586
        
    # Swap and negate to match torch convention: (-gamma, -beta, -alpha)
    alpha = -physical_gamma
    beta = -physical_beta
    gamma = -physical_alpha
    
    tl.store(euler_ptr + pid * 3 + 0, alpha)
    tl.store(euler_ptr + pid * 3 + 1, beta)
    tl.store(euler_ptr + pid * 3 + 2, gamma)


def extract_euler_angles(edge_distance_vec: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Extract the Euler angles that the fused kernel would compute internally.
    Returns [num_edges, 3] tensor of (alpha, beta, gamma).
    Useful for testing: feed these to PyTorch wigner_D for direct comparison.
    """
    num_edges = edge_distance_vec.shape[0]
    device = edge_distance_vec.device
    eulers = torch.empty(num_edges, 3, device=device, dtype=torch.float32)
    if num_edges > 0:
        _extract_eulers_kernel[(num_edges,)](
            edge_distance_vec.contiguous(), eulers, seed,
            num_edges=num_edges,
        )
    return eulers


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def edge_vec_to_wigner_fused(
    edge_distance_vec: torch.Tensor,
    Jd: list,
    lmax: int = 4,
    seed: int = None,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute block-diagonal Wigner D-matrix using fused Triton kernels.

    Supports lmax up to 8. Groups L values into 16x16 blocks where possible,
    with a separate 32x32 kernel for L=8.

    Args:
        edge_distance_vec: [num_edges, 3]
        Jd: List of per-L J matrices (from Jd.pt), length >= lmax+1
        lmax: Maximum angular momentum (0-8)
        seed: Random seed for gamma angle
        out: Optional pre-allocated output tensor [num_edges, out_dim, out_dim].
             Must be float32 and contiguous (stride must be row-major). If provided,
             the kernel writes only block-diagonal positions in-place; off-diagonal
             entries are never touched and must already be zero. If None, a new
             zeroed tensor is allocated.

    Returns:
        Block-diagonal Wigner D-matrix [num_edges, out_dim, out_dim]
        where out_dim = (lmax+1)^2
    """
    assert 0 <= lmax <= 8, f"lmax must be 0-8, got {lmax}"
    num_edges = edge_distance_vec.shape[0]
    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype

    meta = _build_group_meta(lmax, device)
    _fill_jd(meta, Jd, lmax)

    out_dim = meta['out_dim']
    n16 = meta['n16']

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    if out is not None:
        assert out.shape == (num_edges, out_dim, out_dim), (
            f"out shape {out.shape} != expected ({num_edges}, {out_dim}, {out_dim})"
        )
        assert out.is_contiguous(), "out must be contiguous (row-major stride)"
        wigner = out
        # Kernel writes only block-diagonal positions; off-diagonal entries are
        # never touched. Caller is responsible for ensuring they are zero.
    else:
        wigner = torch.zeros(num_edges, out_dim, out_dim, device=device, dtype=dtype)

    edge_distance_vec = edge_distance_vec.contiguous()
    grid = (num_edges,)

    # Launch 16x16 groups kernel
    if n16 > 0:
        wigner_fused_kernel_16[grid](
            edge_distance_vec,
            meta['jd_stack'],
            meta['struct_stack'],
            meta['group_sizes'],
            wigner,
            seed,
            num_edges=num_edges,
            out_dim=out_dim,
            num_groups=n16,
        )

    # Launch 32x32 kernel for L=8 if needed
    if meta['has_32']:
        wigner_fused_kernel_32[grid](
            edge_distance_vec,
            meta['jd_32'],
            meta['struct_32'],
            wigner,
            seed,
            num_edges=num_edges,
            out_dim=out_dim,
            group_size=meta['size_32'],
        )

    return wigner
