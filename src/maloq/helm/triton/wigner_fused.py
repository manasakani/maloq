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
import torch._dynamo  # noqa: F401  -- torch._dynamo.disable is used below
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

    # Packed (compact block-diagonal) addressing: for every position, where its
    # degree's block starts in the packed row, plus its own row within it. The
    # packed row is the concatenation of the (2l+1)^2 diagonal blocks in
    # ascending l -- the layout FlashSO2 consumes.
    pack_start = {}
    running = 0
    for l in range(lmax + 1):
        pack_start[l] = running
        running += (2 * l + 1) ** 2
    packed_dim = running

    pack_struct_stack = torch.zeros(max(n16, 1), 16, device=device, dtype=torch.int32)
    for gi, l_list in enumerate(groups_16):
        pack_base = []
        for l in l_list:
            sz = 2 * l + 1
            pack_base.extend([pack_start[l] + local * sz for local in range(sz)])
        pack_base.extend([0] * (16 - len(pack_base)))
        pack_struct_stack[gi] = torch.tensor(pack_base, dtype=torch.int32)

    result = {
        'out_dim': out_dim,
        'packed_dim': packed_dim,
        'pack_struct_stack': pack_struct_stack,
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
        pack_base_32 = [pack_start[l] + local * sz for local in range(sz)]
        pack_base_32.extend([0] * (32 - sz))
        result['struct_32'] = struct_32
        result['jd_32'] = jd_32
        result['size_32'] = sz
        result['pack_struct_32'] = torch.tensor(
            pack_base_32, device=device, dtype=torch.int32
        )

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
def _wigner_block_D(
    jd_ptr, struct_ptr,
    alpha, beta, gamma,
    group_size, BS: tl.constexpr,
):
    """Compute one BSxBS block-diagonal Wigner tile: D = Xa @ J @ Xb @ J @ Xc.

    Shared verbatim by the dense and packed kernels -- they differ only in
    where the tile is stored, never in how it is computed, so there is one
    copy of the algebra and the two layouts cannot drift apart numerically.

    Returns the tile plus the per-position degree and within-degree index the
    callers need to address their output.
    """

    offs = tl.arange(0, BS)

    # Load structure: l_map(BS), local_start(BS)  [out_offset follows, dense only]
    l_val = tl.load(struct_ptr + offs)
    start_idx = tl.load(struct_ptr + BS + offs)

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

    return D, l_val, local_i


@triton.jit
def _compute_wigner_block_dense(
    jd_ptr, struct_ptr, output_ptr,
    alpha, beta, gamma,
    pid, out_dim: tl.constexpr, group_size, BS: tl.constexpr,
):
    """Compute one BSxBS Wigner tile and scatter-store it into the dense matrix."""
    offs = tl.arange(0, BS)

    D, _l_val, _local_i = _wigner_block_D(
        jd_ptr, struct_ptr, alpha, beta, gamma, group_size, BS,
    )

    # Scatter-store using out_offset_map (handles non-contiguous L pairs)
    out_off = tl.load(struct_ptr + 2 * BS + offs)
    out_base = output_ptr + pid * out_dim * out_dim
    r = out_off[:, None]  # absolute output row for each block position
    c = out_off[None, :]  # absolute output col for each block position
    out_ptrs = out_base + r * out_dim + c
    store_mask = (offs[:, None] < group_size) & (offs[None, :] < group_size)
    tl.store(out_ptrs, D, mask=store_mask)


@triton.jit
def _store_wigner_block_packed(
    D, l_val, local_i, pack_struct_ptr,
    wigner_ptr, wigner_inv_ptr,
    pid, packed_dim: tl.constexpr, group_size, BS: tl.constexpr,
):
    """Store one tile in the compact block-diagonal layout, forward and inverse.

    The packed layout is the concatenation, over degrees in ascending order, of
    each (2l+1)x(2l+1) diagonal block in row-major order -- the same thing
    ``wigner.flatten(1).index_select(1, block_indices)`` produces, which is
    what FlashSO2 consumes.

    Off-diagonal positions of the tile (a group can hold two degrees, e.g.
    [L=0, L=7]) are exactly zero in the dense matrix and have no address in the
    packed one, so they are masked out rather than written anywhere.

    The inverse is the same tile stored with its within-block row and column
    swapped, which is the per-degree transpose. It costs a second store of a
    value already in registers, and saves the caller a dense
    ``transpose(1, 2).contiguous()`` -- a full extra read and write.
    """

    offs = tl.arange(0, BS)
    # pack_base[p] = start of degree(p)'s block in the packed row
    #                + local_row(p) * (2l+1)
    pack_base = tl.load(pack_struct_ptr + offs)

    same_degree = l_val[:, None] == l_val[None, :]
    valid = (offs[:, None] < group_size) & (offs[None, :] < group_size)
    store_mask = same_degree & valid

    base = pid * packed_dim
    fwd_ptrs = wigner_ptr + base + pack_base[:, None] + local_i[None, :]
    inv_ptrs = wigner_inv_ptr + base + pack_base[None, :] + local_i[:, None]
    tl.store(fwd_ptrs, D, mask=store_mask)
    tl.store(inv_ptrs, D, mask=store_mask)


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
        _compute_wigner_block_dense(
            jd_ptr + gi * 16 * 16,
            struct_ptr + gi * 48,
            output_ptr,
            alpha, beta, gamma,
            pid, out_dim, g_sz, 16,
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

    # Compute physical angles (matching torch backend)
    physical_alpha = libdevice.atan2(x, z)
    physical_beta = libdevice.acos(y)
    physical_gamma = tl.rand(seed, pid) * 6.283185307179586

    # Swap and negate to match torch convention: (-gamma, -beta, -alpha).
    # kernel_16 and _extract_eulers_kernel both do this; without the swap the
    # L=8 block is built from a different rotation than every other block.
    alpha = -physical_gamma
    beta = -physical_beta
    gamma = -physical_alpha

    _compute_wigner_block_dense(
        jd_ptr, struct_ptr, output_ptr,
        alpha, beta, gamma,
        pid, out_dim, group_size, BS,
    )


# ---------------------------------------------------------------------------
# Packed kernels: identical tiles, written straight into the compact layout
#
# The dense output is (lmax+1)^2 squared per edge but only sum (2l+1)^2 of it
# is non-zero -- 165 of 625 values at lmax=4, 285 of 1681 at lmax=6. Consumers
# that want the block-diagonal form (FlashSO2) had to allocate the dense tensor,
# gather from it, and transpose it for the inverse. These kernels write the
# compact form and its per-degree transpose directly, so none of that happens.
#
# Cost, measured with Nsight Compute at 65536 edges (GH200, medians of three
# launches) -- the packed kernel is neither faster nor slower per launch, it
# just asks far less of the memory system:
#
#   lmax  kernel                 dur us   SM %   DRAM %   read MB   write MB
#   4     wigner_fused_kernel_16  539.5  81.18     9.43      84.4      120.3
#   4     wigner_packed_kernel_16 546.5  80.61     2.78       0.8       60.3
#   6     wigner_fused_kernel_16 1141.7  70.56     9.15     174.2      246.2
#   6     wigner_packed_kernel_16 1143.0 71.39     4.62       0.9      211.9
#
# Both are compute-bound (62-81% of SM peak against 3-18% of DRAM peak), which
# is why halving the bytes does not halve the time. The near-total loss of read
# traffic is the write pattern: storing 16-wide runs into a 25-wide dense row
# leaves partial sectors that must be fetched before they can be written, while
# a packed block is contiguous.
# ---------------------------------------------------------------------------
@triton.jit
def wigner_packed_kernel_16(
    edge_vec_ptr, jd_ptr, struct_ptr, pack_struct_ptr,
    group_sizes_ptr,
    wigner_ptr, wigner_inv_ptr, seed,
    num_edges: tl.constexpr,
    packed_dim: tl.constexpr,
    num_groups: tl.constexpr,
):
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

    for gi in tl.static_range(num_groups):
        g_sz = tl.load(group_sizes_ptr + gi)
        D, l_val, local_i = _wigner_block_D(
            jd_ptr + gi * 16 * 16,
            struct_ptr + gi * 48,
            alpha, beta, gamma, g_sz, 16,
        )
        _store_wigner_block_packed(
            D, l_val, local_i,
            pack_struct_ptr + gi * 16,
            wigner_ptr, wigner_inv_ptr,
            pid, packed_dim, g_sz, 16,
        )


@triton.jit
def wigner_packed_kernel_32(
    edge_vec_ptr, jd_ptr, struct_ptr, pack_struct_ptr,
    wigner_ptr, wigner_inv_ptr, seed,
    num_edges: tl.constexpr,
    packed_dim: tl.constexpr,
    group_size: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_edges:
        return

    BS: tl.constexpr = 32

    edge_x = tl.load(edge_vec_ptr + pid * 3 + 0)
    edge_y = tl.load(edge_vec_ptr + pid * 3 + 1)
    edge_z = tl.load(edge_vec_ptr + pid * 3 + 2)

    norm = tl.sqrt(edge_x * edge_x + edge_y * edge_y + edge_z * edge_z)
    norm = tl.maximum(norm, 1e-12)
    x = tl.minimum(tl.maximum(edge_x / norm, -1.0), 1.0)
    y = tl.minimum(tl.maximum(edge_y / norm, -1.0), 1.0)
    z = tl.minimum(tl.maximum(edge_z / norm, -1.0), 1.0)

    physical_alpha = libdevice.atan2(x, z)
    physical_beta = libdevice.acos(y)
    physical_gamma = tl.rand(seed, pid) * 6.283185307179586

    alpha = -physical_gamma
    beta = -physical_beta
    gamma = -physical_alpha

    D, l_val, local_i = _wigner_block_D(
        jd_ptr, struct_ptr, alpha, beta, gamma, group_size, BS,
    )
    _store_wigner_block_packed(
        D, l_val, local_i, pack_struct_ptr,
        wigner_ptr, wigner_inv_ptr,
        pid, packed_dim, group_size, BS,
    )


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
@torch._dynamo.disable
def _draw_gamma_seed() -> int:
    """Draw the gamma-gauge seed in eager Python, never inside a traced graph.

    ``tl.rand(seed, pid)`` needs a Python ``int``. Dynamo rewrites
    ``random.randint`` into a tensor-valued RNG node, so tracing this call used
    to hand the kernel a tensor and abort Triton compilation outright -- the
    single hard failure blocking ``torch.compile`` on the backbone. Marking the
    draw ``disable``-d keeps it in eager Python and gives the kernel a real int.

    ``torch._dynamo.disable`` is inert outside compilation, so the eager path
    draws exactly the same way it always did: same RNG, same stream of seeds.
    Under compile it costs one graph break here rather than a crash.

    ``wigner_fused_op`` below removes even that break for callers that route
    through it; this stays the correct behaviour for any direct caller.

    The seed only picks the arbitrary gamma of the Wigner gauge, which the
    SO(2)-equivariant contraction removes, so it does not select a model.
    """
    return random.randint(0, 2**31 - 1)


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
        seed = _draw_gamma_seed()

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


def packed_wigner_width(lmax: int) -> int:
    """Width of one packed Wigner row: sum of (2l+1)^2 over degrees 0..lmax."""
    return sum((2 * l + 1) ** 2 for l in range(lmax + 1))


def edge_vec_to_wigner_packed(
    edge_distance_vec: torch.Tensor,
    Jd: list,
    lmax: int = 4,
    seed: int = None,
    out: torch.Tensor = None,
    out_inv: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Wigner D-matrix directly in the compact block-diagonal form.

    Same tiles, same Euler angles and same gamma gauge as
    ``edge_vec_to_wigner_fused`` -- for a given seed the packed output is
    bit-identical to gathering the diagonal blocks out of the dense one. What
    differs is that nothing dense is ever written: the kernel stores each tile
    straight into the packed row, and stores it a second time with its
    within-degree row and column swapped to produce the inverse, so the caller
    also skips a ``transpose(1, 2).contiguous()``.

    Args:
        edge_distance_vec: [num_edges, 3]
        Jd: List of per-L J matrices (from Jd.pt), length >= lmax+1
        lmax: Maximum angular momentum (0-8)
        seed: Random seed for the gamma gauge
        out / out_inv: Optional pre-allocated [num_edges, packed_dim] buffers.
            Every element is written, so they need not be zeroed -- unlike the
            dense path, where off-diagonal entries are left untouched.

    Returns:
        ``(wigner, wigner_inv)``, each [num_edges, packed_dim] with
        packed_dim = sum (2l+1)^2, holding the (2l+1)x(2l+1) diagonal blocks
        concatenated in ascending l, row-major within a block.
    """
    assert 0 <= lmax <= 8, f"lmax must be 0-8, got {lmax}"
    num_edges = edge_distance_vec.shape[0]
    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype

    meta = _build_group_meta(lmax, device)
    _fill_jd(meta, Jd, lmax)

    packed_dim = meta['packed_dim']
    n16 = meta['n16']

    if seed is None:
        seed = _draw_gamma_seed()

    def _buffer(buf, name):
        if buf is None:
            return torch.empty(num_edges, packed_dim, device=device, dtype=dtype)
        assert buf.shape == (num_edges, packed_dim), (
            f"{name} shape {buf.shape} != expected ({num_edges}, {packed_dim})"
        )
        assert buf.is_contiguous(), f"{name} must be contiguous (row-major stride)"
        return buf

    wigner = _buffer(out, "out")
    wigner_inv = _buffer(out_inv, "out_inv")

    edge_distance_vec = edge_distance_vec.contiguous()
    grid = (num_edges,)

    if n16 > 0:
        wigner_packed_kernel_16[grid](
            edge_distance_vec,
            meta['jd_stack'],
            meta['struct_stack'],
            meta['pack_struct_stack'],
            meta['group_sizes'],
            wigner,
            wigner_inv,
            seed,
            num_edges=num_edges,
            packed_dim=packed_dim,
            num_groups=n16,
        )

    if meta['has_32']:
        wigner_packed_kernel_32[grid](
            edge_distance_vec,
            meta['jd_32'],
            meta['struct_32'],
            meta['pack_struct_32'],
            wigner,
            wigner_inv,
            seed,
            num_edges=num_edges,
            packed_dim=packed_dim,
            group_size=meta['size_32'],
        )

    return wigner, wigner_inv


def wigner_fused_dense_and_packed(
    edge_distance_vec: torch.Tensor,
    Jd: list,
    lmax: int = 4,
    out: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense matrix and packed pair from a single gamma draw.

    The gamma gauge is arbitrary but must be the *same* arbitrary choice
    everywhere in a forward pass: the dense matrix rotates the edge-degree
    embedding while the packed pair rotates the message-passing blocks, and two
    independent draws would put them in different frames. Drawing once here is
    what makes that impossible to get wrong -- in particular under
    ``torch.compile``, where two separate custom ops would each draw their own.

    Returns ``(dense, packed, packed_inv)``.
    """

    seed = _draw_gamma_seed()
    dense = edge_vec_to_wigner_fused(
        edge_distance_vec, Jd, lmax=lmax, seed=seed, out=out
    )
    packed, packed_inv = edge_vec_to_wigner_packed(
        edge_distance_vec, Jd, lmax=lmax, seed=seed
    )
    return dense, packed, packed_inv


# ---------------------------------------------------------------------------
# torch.compile entry point
# ---------------------------------------------------------------------------
@torch.library.custom_op("maloq::wigner_fused", mutates_args=())
def wigner_fused_op(
    edge_distance_vec: torch.Tensor,
    jd: list[torch.Tensor],
    lmax: int,
) -> torch.Tensor:
    """``edge_vec_to_wigner_fused`` as a single opaque node for torch.compile.

    Calling the plain function under tracing breaks the graph, since the gamma
    seed must be drawn in eager Python (see ``_draw_gamma_seed``). Behind a
    custom op the draw, the metadata cache and the kernel launches are all
    interior to one node, so none of it is traced.

    No autograd kernel is registered, which matches eager: the Triton kernels
    write outside autograd, so the Wigner matrix already comes back with
    ``requires_grad=False``. There is no gradient here to lose.

    Allocates rather than taking the caller's persistent buffer -- a custom op
    must not return an alias of persistent state. float32 to match that
    buffer's dtype, so the traced path cannot diverge from eager if edge
    vectors are ever not float32.
    """
    out_dim = (lmax + 1) ** 2
    out = torch.zeros(
        edge_distance_vec.shape[0], out_dim, out_dim,
        device=edge_distance_vec.device, dtype=torch.float32,
    )
    return edge_vec_to_wigner_fused(edge_distance_vec, jd, lmax=lmax, out=out)


@wigner_fused_op.register_fake
def _wigner_fused_op_fake(
    edge_distance_vec: torch.Tensor,
    jd: list[torch.Tensor],
    lmax: int,
) -> torch.Tensor:
    """Shape/dtype only, so Dynamo can size the node without running Triton."""
    out_dim = (lmax + 1) ** 2
    return edge_distance_vec.new_empty(
        (edge_distance_vec.shape[0], out_dim, out_dim), dtype=torch.float32
    )



@torch.library.custom_op("maloq::wigner_fused_dense_and_packed", mutates_args=())
def wigner_fused_dense_and_packed_op(
    edge_distance_vec: torch.Tensor,
    jd: list[torch.Tensor],
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One traced node producing both layouts from one gamma draw.

    Two separate ops would draw two seeds and put the dense and packed
    rotations in different frames -- see ``wigner_fused_dense_and_packed``.
    """

    return wigner_fused_dense_and_packed(edge_distance_vec, jd, lmax=lmax)


@wigner_fused_dense_and_packed_op.register_fake
def _wigner_fused_dense_and_packed_op_fake(
    edge_distance_vec: torch.Tensor,
    jd: list[torch.Tensor],
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_edges = edge_distance_vec.shape[0]
    out_dim = (lmax + 1) ** 2
    width = packed_wigner_width(lmax)
    return (
        edge_distance_vec.new_empty((num_edges, out_dim, out_dim), dtype=torch.float32),
        edge_distance_vec.new_empty((num_edges, width), dtype=torch.float32),
        edge_distance_vec.new_empty((num_edges, width), dtype=torch.float32),
    )
