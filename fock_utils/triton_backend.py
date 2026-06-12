import torch
import triton
import triton.language as tl
import math
import time

def next_power_of_2(n):
    """Returns the next power of 2 for a given number, used to pad dimensions for Triton."""
    return 2 ** math.ceil(math.log2(n)) if n > 0 else 1

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=4)
    ],
    key=['num_edges', 'max_in_dim_key', 'max_h_dim_key'],
)
@triton.jit
def get_H_triton_block_kernel(
    net_out_ptr,
    wms_ptr,
    H_out_ptr,
    in_col_indices_ptr,
    in_starts_ptr,
    in_dims_ptr,
    h_starts_ptr,
    h_dims_ptr,
    wm_starts_ptr,
    num_edges,
    total_in_dim,
    total_h_dim,
    max_in_dim_key,
    max_h_dim_key,
    BLOCK_M: tl.constexpr,
    MAX_IN_DIM: tl.constexpr,
    MAX_H_DIM: tl.constexpr,
):
    """
    Compute one block product:
    net_out[:, in_start:in_start+in_dim] @ wms_block.T -> H_out[:, h_start:h_start+h_dim]
    """
    pid_m = tl.program_id(0)
    pid_block = tl.program_id(1)

    in_start = tl.load(in_starts_ptr + pid_block)
    in_dim = tl.load(in_dims_ptr + pid_block)
    h_start = tl.load(h_starts_ptr + pid_block)
    h_dim = tl.load(h_dims_ptr + pid_block)
    wm_start = tl.load(wm_starts_ptr + pid_block)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MAX_H_DIM)

    offs_k = tl.arange(0, MAX_IN_DIM)

    # Map local unsorted in-dimension indices to the input net_out column indices.
    col_idx = tl.load(in_col_indices_ptr + in_start + offs_k, mask=offs_k < in_dim, other=0)

    a_ptrs = net_out_ptr + offs_m[:, None] * total_in_dim + col_idx[None, :]
    a_mask = (offs_m[:, None] < num_edges) & (offs_k[None, :] < in_dim)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b_ptrs = wms_ptr + wm_start + offs_n[None, :] * in_dim + offs_k[:, None]
    b_mask = (offs_k[:, None] < in_dim) & (offs_n[None, :] < h_dim)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    # Element-wise multiply and sum over K dimension avoids tl.dot shape constraints.
    # acc = tl.sum(a[:, :, None] * b[None, :, :], axis=1)
    acc = tl.dot(a, b, allow_tf32=False)

    out_ptrs = H_out_ptr + offs_m[:, None] * total_h_dim + (h_start + offs_n[None, :])
    out_mask = (offs_m[:, None] < num_edges) & (offs_n[None, :] < h_dim)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=4)
    ],
    key=['num_edges', 'max_group_in_dim_key', 'max_group_h_dim_key'],
)
@triton.jit
def get_H_triton_grouped_kernel(
    net_out_ptr,
    grouped_wms_ptr,
    H_out_ptr,
    in_col_indices_ptr,
    group_in_starts_ptr,
    group_in_dims_ptr,
    group_h_starts_ptr,
    group_h_dims_ptr,
    group_wm_starts_ptr,
    num_edges,
    total_in_dim,
    total_h_dim,
    max_group_in_dim_key,
    max_group_h_dim_key,
    BLOCK_M: tl.constexpr,
    MAX_GROUP_IN_DIM: tl.constexpr,
    MAX_GROUP_H_DIM: tl.constexpr,
):
    """
    Compute one grouped block product:
    net_out[:, g_in_start:g_in_start+g_in_dim] @ grouped_wms.T -> H_out[:, g_h_start:g_h_start+g_h_dim]
    grouped_wms is a diagonal packing of multiple small per-block WMS matrices.
    """
    pid_m = tl.program_id(0)
    pid_group = tl.program_id(1)

    g_in_start = tl.load(group_in_starts_ptr + pid_group)
    g_in_dim = tl.load(group_in_dims_ptr + pid_group)
    g_h_start = tl.load(group_h_starts_ptr + pid_group)
    g_h_dim = tl.load(group_h_dims_ptr + pid_group)
    g_wm_start = tl.load(group_wm_starts_ptr + pid_group)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MAX_GROUP_H_DIM)
    offs_k = tl.arange(0, MAX_GROUP_IN_DIM)

    # Map local unsorted in-dimension indices to the input net_out column indices.
    col_idx = tl.load(in_col_indices_ptr + g_in_start + offs_k, mask=offs_k < g_in_dim, other=0)

    a_ptrs = net_out_ptr + offs_m[:, None] * total_in_dim + col_idx[None, :]
    a_mask = (offs_m[:, None] < num_edges) & (offs_k[None, :] < g_in_dim)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    b_ptrs = grouped_wms_ptr + g_wm_start + offs_n[None, :] * g_in_dim + offs_k[:, None]
    b_mask = (offs_k[:, None] < g_in_dim) & (offs_n[None, :] < g_h_dim)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc = tl.dot(a, b, allow_tf32=False)

    out_ptrs = H_out_ptr + offs_m[:, None] * total_h_dim + (g_h_start + offs_n[None, :])
    out_mask = (offs_m[:, None] < num_edges) & (offs_n[None, :] < g_h_dim)
    tl.store(out_ptrs, acc, mask=out_mask)


class TritonGetHFunction(torch.autograd.Function):
    """
    Standard PyTorch autograd wrapper to insert the Triton kernel into the computational graph.
    """
    @staticmethod
    def forward(ctx, net_out, decomp_obj):
        # Save shapes in case we need to return dummy gradients in backward
        ctx.net_out_shape = net_out.shape
        ctx.net_out_dtype = net_out.dtype
        ctx.net_out_device = net_out.device
        return decomp_obj._get_H_impl(net_out)

    @staticmethod
    def backward(ctx, grad_output):
        # DUMMY BACKWARD: returning zero gradients since we removed `get_net_out` 
        # (which is the mathematical inverse/backward of `get_H`).
        # This allows profiling/training pipelines to run fully without crashing, 
        # even if gradients are zeroed out here.
        grad_net_out = torch.zeros(ctx.net_out_shape, dtype=ctx.net_out_dtype, device=ctx.net_out_device)
        return grad_net_out, None


class TritonE3TensorDecomp:
    """
    Triton wrapper for the tensor decomposition process. 
    It computes the physical Fock (H) matrix from the GNN output (`net_out`) using Triton.
    """
    def __init__(
        self,
        decomp_obj,
        grouped_max_in_dim=32,
        grouped_max_h_dim=32,
        use_grouped_kernel=False,
    ):
        self._decomp_obj = decomp_obj  # Save a reference to the original PyTorch object for fallback
        self.device = decomp_obj.device
        self.dtype = decomp_obj.dtype
        self.num_blocks = len(decomp_obj.out_js_list)
        self.grouped_max_in_dim = int(grouped_max_in_dim)
        self.grouped_max_h_dim = int(grouped_max_h_dim)
        self.use_grouped_kernel = bool(use_grouped_kernel)
        
        # Load block boundaries metadata
        self.sort = decomp_obj.sort
        
        wms_flat_list = []
        wm_slices = [0]
        
        max_in_dim = 1
        max_h_dim = 1
        
        # Pre-process Wigner multipliers into a single flattened 1D tensor for easier Triton loading
        for i in range(self.num_blocks):
            in_dim = decomp_obj.in_slices[i+1] - decomp_obj.in_slices[i]
            h_dim = decomp_obj.H_slices[i+1] - decomp_obj.H_slices[i]
            
            # Find the maximum dimensions across all blocks to set compilation constraints
            max_in_dim = max(max_in_dim, in_dim)
            max_h_dim = max(max_h_dim, h_dim)
            
            wms_flat_list.append(decomp_obj.wms[i].clone().view(h_dim, in_dim).flatten())
            wm_slices.append(wm_slices[-1] + h_dim * in_dim)
            
        self.wms_flat = torch.cat(wms_flat_list).to(device=self.device, dtype=self.dtype)
        wm_slices_host = [int(x) for x in wm_slices[:-1]]

        # Precompute per-block metadata buffers once and keep them on device.
        in_starts_host = [int(x) for x in decomp_obj.in_slices[:-1]]
        in_dims_host = [
            int(decomp_obj.in_slices[i + 1] - decomp_obj.in_slices[i])
            for i in range(self.num_blocks)
        ]
        h_starts_host = [int(x) for x in decomp_obj.H_slices[:-1]]
        h_dims_host = [
            int(decomp_obj.H_slices[i + 1] - decomp_obj.H_slices[i])
            for i in range(self.num_blocks)
        ]

        self.in_starts = torch.tensor(in_starts_host, dtype=torch.int32, device=self.device)
        self.in_dims = torch.tensor(in_dims_host, dtype=torch.int32, device=self.device)
        self.h_starts = torch.tensor(h_starts_host, dtype=torch.int32, device=self.device)
        self.h_dims = torch.tensor(h_dims_host, dtype=torch.int32, device=self.device)
        self.wm_starts = torch.tensor(wm_slices_host, dtype=torch.int32, device=self.device)

        # Precompute mapping from unsorted feature indices (expected by WMS/in_slices)
        # to input net_out column indices. This avoids calling sort.inverse in get_H.
        total_in_dim = int(decomp_obj.in_slices[-1])
        if self.sort is not None:
            eye = torch.eye(total_in_dim, dtype=self.dtype, device=self.device)
            # Rows correspond to sorted basis vectors; columns correspond to unsorted output dims.
            inv_eye = self.sort.inverse(eye)
            # unsorted index -> input(sorted) index
            self.in_col_indices = torch.argmax(inv_eye, dim=0).to(dtype=torch.int32)
        else:
            self.in_col_indices = torch.arange(total_in_dim, dtype=torch.int32, device=self.device)

        # Build grouped metadata and grouped diagonal-packed WMS once.
        self._max_block_in_dim = int(max_in_dim)
        self._max_block_h_dim = int(max_h_dim)
        self._build_grouped_metadata(decomp_obj)
        
        self.total_h_dim_val = int(decomp_obj.H_slices[-1])
        
        # Saved for potential future autotuning/padding paths.
        self.MAX_IN_DIM = next_power_of_2(max_in_dim)
        self.MAX_H_DIM = next_power_of_2(max_h_dim)

    def _compute_group_kernel_dims(self, max_group_in, max_group_h):
        """Compute grouped kernel compile-time dimensions with power-of-two padding."""
        self.TARGET_GROUP_IN_DIM = max(16, next_power_of_2(max_group_in))
        self.TARGET_GROUP_H_DIM = next_power_of_2(max_group_h)
        return self.TARGET_GROUP_IN_DIM, self.TARGET_GROUP_H_DIM

    def _build_grouped_metadata(self, decomp_obj):
        """
        Build groups of consecutive decomposition blocks such that combined in/h dimensions
        stay within configured limits, and create diagonal-packed grouped WMS matrices.
        """
        groups = []
        i = 0
        while i < self.num_blocks:
            start = i
            in_sum = 0
            h_sum = 0
            while i < self.num_blocks:
                block_in = int(decomp_obj.in_slices[i + 1] - decomp_obj.in_slices[i])
                block_h = int(decomp_obj.H_slices[i + 1] - decomp_obj.H_slices[i])

                # Always include at least one block. Then greedily add while within bounds.
                if i == start:
                    in_sum += block_in
                    h_sum += block_h
                    i += 1
                    continue

                if in_sum + block_in <= self.grouped_max_in_dim and h_sum + block_h <= self.grouped_max_h_dim:
                    in_sum += block_in
                    h_sum += block_h
                    i += 1
                else:
                    break

            groups.append((start, i))

        grouped_wms_flat_list = []
        group_in_starts = []
        group_in_dims = []
        group_h_starts = []
        group_h_dims = []
        group_wm_starts = []

        wm_offset = 0
        max_group_in = 1
        max_group_h = 1

        for g_start, g_end in groups:
            g_in_start = int(decomp_obj.in_slices[g_start])
            g_in_end = int(decomp_obj.in_slices[g_end])
            g_h_start = int(decomp_obj.H_slices[g_start])
            g_h_end = int(decomp_obj.H_slices[g_end])

            g_in_dim = g_in_end - g_in_start
            g_h_dim = g_h_end - g_h_start

            diag_group = torch.zeros((g_h_dim, g_in_dim), dtype=self.dtype, device=self.device)
            local_h = 0
            local_in = 0
            for bi in range(g_start, g_end):
                b_in = int(decomp_obj.in_slices[bi + 1] - decomp_obj.in_slices[bi])
                b_h = int(decomp_obj.H_slices[bi + 1] - decomp_obj.H_slices[bi])
                b_wm = decomp_obj.wms[bi].reshape(b_h, b_in).to(device=self.device, dtype=self.dtype)
                diag_group[local_h:local_h + b_h, local_in:local_in + b_in] = b_wm
                local_h += b_h
                local_in += b_in

            grouped_wms_flat_list.append(diag_group.flatten())
            group_in_starts.append(g_in_start)
            group_in_dims.append(g_in_dim)
            group_h_starts.append(g_h_start)
            group_h_dims.append(g_h_dim)
            group_wm_starts.append(wm_offset)

            wm_offset += g_h_dim * g_in_dim
            max_group_in = max(max_group_in, g_in_dim)
            max_group_h = max(max_group_h, g_h_dim)

        self.group_ranges = groups
        self.num_groups = len(groups)
        self.grouped_wms_flat = torch.cat(grouped_wms_flat_list)
        self.group_in_starts = torch.tensor(group_in_starts, dtype=torch.int32, device=self.device)
        self.group_in_dims = torch.tensor(group_in_dims, dtype=torch.int32, device=self.device)
        self.group_h_starts = torch.tensor(group_h_starts, dtype=torch.int32, device=self.device)
        self.group_h_dims = torch.tensor(group_h_dims, dtype=torch.int32, device=self.device)
        self.group_wm_starts = torch.tensor(group_wm_starts, dtype=torch.int32, device=self.device)

        # Keep constexpr dimensions valid for tl.dot (K >= 16 for Triton's dot path).
        self.MAX_GROUP_IN_DIM, self.MAX_GROUP_H_DIM = self._compute_group_kernel_dims(
            max_group_in, max_group_h
        )

    def get_group_debug_info(self):
        """Return grouping metadata as a Python list of dictionaries."""
        info = []
        # Useful FLOP per edge for each group:
        # sum over constituent blocks of 2 * (block_h_dim * block_in_dim).
        useful_flop_values = []
        # Dense FLOP per edge actually computed by grouped dense tl.dot:
        # 2 * (grouped_h_dim * grouped_in_dim).
        dense_flop_values = []
        for gi, (g_start, g_end) in enumerate(self.group_ranges):
            in_dim = int(self.group_in_dims[gi].item())
            h_dim = int(self.group_h_dims[gi].item())
            dense_flop = 2 * in_dim * h_dim

            useful_flop = 0
            for bi in range(g_start, g_end):
                b_in = int(self.in_dims[bi].item())
                b_h = int(self.h_dims[bi].item())
                useful_flop += 2 * b_in * b_h

            useful_flop_values.append(useful_flop)
            dense_flop_values.append(dense_flop)
            info.append(
                {
                    'group_index': gi,
                    'block_start': g_start,
                    'block_end_exclusive': g_end,
                    'num_blocks': g_end - g_start,
                    'in_start': int(self.group_in_starts[gi].item()),
                    'in_dim': in_dim,
                    'h_start': int(self.group_h_starts[gi].item()),
                    'h_dim': h_dim,
                    'wm_start': int(self.group_wm_starts[gi].item()),
                    'useful_flop_per_edge': useful_flop,
                    'dense_flop_per_edge': dense_flop,
                }
            )

        if len(info) == 0:
            self.group_imbalance = {}
            return []

        num_blocks_values = [g['num_blocks'] for g in info]
        in_dim_values = [g['in_dim'] for g in info]
        h_dim_values = [g['h_dim'] for g in info]

        mean_blocks = float(sum(num_blocks_values)) / float(len(num_blocks_values))
        mean_in_dim = float(sum(in_dim_values)) / float(len(in_dim_values))
        mean_h_dim = float(sum(h_dim_values)) / float(len(h_dim_values))
        mean_useful_flop = (
            float(sum(useful_flop_values)) / float(len(useful_flop_values))
        )
        mean_dense_flop = (
            float(sum(dense_flop_values)) / float(len(dense_flop_values))
        )

        max_useful_flop = max(useful_flop_values)
        min_useful_flop = min(useful_flop_values)

        imbalance = {
            'num_groups': len(info),
            'num_blocks_min': min(num_blocks_values),
            'num_blocks_max': max(num_blocks_values),
            'num_blocks_mean': mean_blocks,
            'num_blocks_imbalance_ratio': (max(num_blocks_values) / mean_blocks) if mean_blocks > 0 else 0.0,
            'in_dim_min': min(in_dim_values),
            'in_dim_max': max(in_dim_values),
            'in_dim_mean': mean_in_dim,
            'h_dim_min': min(h_dim_values),
            'h_dim_max': max(h_dim_values),
            'h_dim_mean': mean_h_dim,
            'useful_flop_per_edge_min': min_useful_flop,
            'useful_flop_per_edge_max': max_useful_flop,
            'useful_flop_per_edge_mean': mean_useful_flop,
            'useful_flop_imbalance_ratio': (
                (max_useful_flop / mean_useful_flop) if mean_useful_flop > 0 else 0.0
            ),
            'useful_flop_max_over_min': (
                (max_useful_flop / min_useful_flop) if min_useful_flop > 0 else float('inf')
            ),
            'dense_flop_per_edge_mean': mean_dense_flop,
            'overall_useful_over_dense': (
                (float(sum(useful_flop_values)) / float(sum(dense_flop_values)))
                if sum(dense_flop_values) > 0
                else 0.0
            ),
        }

        for g in info:
            g['useful_flop_over_mean'] = (
                (g['useful_flop_per_edge'] / mean_useful_flop) if mean_useful_flop > 0 else 0.0
            )
            g['useful_over_dense'] = (
                (g['useful_flop_per_edge'] / g['dense_flop_per_edge']) if g['dense_flop_per_edge'] > 0 else 0.0
            )
            g['num_blocks_over_mean'] = (g['num_blocks'] / mean_blocks) if mean_blocks > 0 else 0.0

        self.group_imbalance = imbalance
        return info

    def print_group_debug_info(self, max_groups=None):
        """Pretty-print grouping metadata for quick debugging."""
        info = self.get_group_debug_info()
        imbalance = getattr(self, 'group_imbalance', {})
        if max_groups is None:
            max_groups = len(info)
        max_groups = int(max_groups)

        print("=== Triton Group Debug Info ===")
        print(
            f"num_blocks={self.num_blocks}, num_groups={self.num_groups}, "
            f"grouped_max_in_dim={self.grouped_max_in_dim}, grouped_max_h_dim={self.grouped_max_h_dim}"
        )
        print(
            f"MAX_GROUP_IN_DIM={self.MAX_GROUP_IN_DIM}, MAX_GROUP_H_DIM={self.MAX_GROUP_H_DIM}, "
            f"TARGET_GROUP_IN_DIM={getattr(self, 'TARGET_GROUP_IN_DIM', self.MAX_GROUP_IN_DIM)}, "
            f"TARGET_GROUP_H_DIM={getattr(self, 'TARGET_GROUP_H_DIM', self.MAX_GROUP_H_DIM)}"
        )
        if len(info) > 0:
            print(
                "imbalance: "
                f"num_blocks min/mean/max={imbalance['num_blocks_min']}/"
                f"{imbalance['num_blocks_mean']:.2f}/{imbalance['num_blocks_max']} "
                f"(max/mean={imbalance['num_blocks_imbalance_ratio']:.2f}x), "
                f"useful_flop/edge min/mean/max={imbalance['useful_flop_per_edge_min']}/"
                f"{imbalance['useful_flop_per_edge_mean']:.1f}/{imbalance['useful_flop_per_edge_max']} "
                f"(max/mean={imbalance['useful_flop_imbalance_ratio']:.2f}x, "
                f"max/min={imbalance['useful_flop_max_over_min']:.2f}x), "
                f"overall useful/dense={100.0 * imbalance['overall_useful_over_dense']:.1f}%"
            )

        shown = min(len(info), max_groups)
        for i in range(shown):
            g = info[i]
            print(
                f"group {g['group_index']:4d}: blocks [{g['block_start']:4d}, {g['block_end_exclusive']:4d}) "
                f"count={g['num_blocks']:3d} | "
                f"in_start={g['in_start']:6d}, in_dim={g['in_dim']:4d} | "
                f"h_start={g['h_start']:6d}, h_dim={g['h_dim']:4d} | "
                f"useful_flop/edge={g['useful_flop_per_edge']:7d} "
                f"({g['useful_flop_over_mean']:.2f}x mean) | "
                f"useful/dense={100.0 * g['useful_over_dense']:.1f}%"
            )

        if shown < len(info):
            print(f"... ({len(info) - shown} more groups)")

    def get_net_out(self, H):
        # WORKAROUND: Fall back to native PyTorch `get_net_out` from the wrapped object 
        # so target creation works during dataset loading
        return self._decomp_obj.get_net_out(H)

    def get_H(self, net_out):
        """
        Public function to get the H matrix. Uses autograd.Function to support PyTorch graphs.
        """
        # torch.cuda.synchronize()  
        # start_time = time.time()
        x = TritonGetHFunction.apply(net_out, self)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"Triton H computation time: {(end_time - start_time) * 1000:.2f} ms for {net_out.shape[0]} edges")
        return x

    def _get_H_impl(self, net_out):
        """
        Internal implementation of get_H.
        This version computes one block product per decomposition block,
        matching the blockwise matmul reference implementation.
        """
        num_edges = net_out.shape[0]
        total_in_dim = net_out.shape[1]
        total_h_dim = self.total_h_dim_val

        # Each block/group writes to a unique output slice, so no accumulation is needed.
        out = torch.empty((num_edges, total_h_dim), device=net_out.device, dtype=net_out.dtype)

        if self.use_grouped_kernel:
            grid = lambda meta: (
                triton.cdiv(num_edges, meta['BLOCK_M']),
                self.num_groups,
            )

            get_H_triton_grouped_kernel[grid](
                net_out,
                self.grouped_wms_flat,
                out,
                self.in_col_indices,
                self.group_in_starts,
                self.group_in_dims,
                self.group_h_starts,
                self.group_h_dims,
                self.group_wm_starts,
                num_edges,
                total_in_dim,
                total_h_dim,
                self.MAX_GROUP_IN_DIM,
                self.MAX_GROUP_H_DIM,
                MAX_GROUP_IN_DIM=self.MAX_GROUP_IN_DIM,
                MAX_GROUP_H_DIM=self.MAX_GROUP_H_DIM,
            )
        else:
            # 2D grid: edge tiles x decomposition blocks.
            grid = lambda meta: (
                triton.cdiv(num_edges, meta['BLOCK_M']),
                self.num_blocks,
            )

            get_H_triton_block_kernel[grid](
                net_out,
                self.wms_flat,
                out,
                self.in_col_indices,
                self.in_starts,
                self.in_dims,
                self.h_starts,
                self.h_dims,
                self.wm_starts,
                num_edges,
                total_in_dim,
                total_h_dim,
                self.MAX_IN_DIM,
                self.MAX_H_DIM,
                MAX_IN_DIM=self.MAX_IN_DIM,
                MAX_H_DIM=self.MAX_H_DIM,
            )
        return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=4)
    ],
    key=['num_edges'],
)
@triton.jit
def get_H_triton_grouped_balanced_kernel(
    net_out_ptr,
    balanced_wms_ptr,
    H_out_ptr,
    in_col_indices_ptr,
    group_block_counts_ptr,
    group_block_in_starts_ptr,
    group_block_in_dims_ptr,
    group_block_h_starts_ptr,
    group_block_h_dims_ptr,
    group_block_wm_starts_ptr,
    num_edges,
    total_in_dim,
    total_h_dim,
    BLOCK_M: tl.constexpr,
    MAX_BLOCKS_PER_GROUP: tl.constexpr,
    MAX_IN_DIM: tl.constexpr,
    MAX_H_DIM: tl.constexpr,
):
    """
    Compute one balanced group product where each group contains arbitrary (non-contiguous)
    original blocks. Each program handles one edge tile and one group, and loops through
    blocks assigned to that group.
    """
    pid_m = tl.program_id(0)
    pid_group = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MAX_H_DIM)
    offs_k = tl.arange(0, MAX_IN_DIM)

    group_count = tl.load(group_block_counts_ptr + pid_group)

    for slot in range(MAX_BLOCKS_PER_GROUP):
        active = slot < group_count
        flat_idx = pid_group * MAX_BLOCKS_PER_GROUP + slot

        in_start = tl.load(group_block_in_starts_ptr + flat_idx, mask=active, other=0)
        in_dim = tl.load(group_block_in_dims_ptr + flat_idx, mask=active, other=0)
        h_start = tl.load(group_block_h_starts_ptr + flat_idx, mask=active, other=0)
        h_dim = tl.load(group_block_h_dims_ptr + flat_idx, mask=active, other=0)
        wm_start = tl.load(group_block_wm_starts_ptr + flat_idx, mask=active, other=0)

        col_idx = tl.load(
            in_col_indices_ptr + in_start + offs_k,
            mask=active & (offs_k < in_dim),
            other=0,
        )

        a_ptrs = net_out_ptr + offs_m[:, None] * total_in_dim + col_idx[None, :]
        a_mask = active & (offs_m[:, None] < num_edges) & (offs_k[None, :] < in_dim)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = balanced_wms_ptr + wm_start + offs_n[None, :] * in_dim + offs_k[:, None]
        b_mask = active & (offs_k[:, None] < in_dim) & (offs_n[None, :] < h_dim)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, allow_tf32=False)

        out_ptrs = H_out_ptr + offs_m[:, None] * total_h_dim + (h_start + offs_n[None, :])
        out_mask = active & (offs_m[:, None] < num_edges) & (offs_n[None, :] < h_dim)
        tl.store(out_ptrs, acc, mask=out_mask)


class BalancedTritonE3TensorDecomp(TritonE3TensorDecomp):
    """
    Variant of TritonE3TensorDecomp that builds non-contiguous balanced groups.
    Groups are formed with first-fit-decreasing on per-block useful flop to reduce
    inter-group imbalance while still respecting grouped_max_in_dim/grouped_max_h_dim.
    """

    def _build_grouped_metadata(self, decomp_obj):
        block_in_dims = [
            int(decomp_obj.in_slices[i + 1] - decomp_obj.in_slices[i])
            for i in range(self.num_blocks)
        ]
        block_h_dims = [
            int(decomp_obj.H_slices[i + 1] - decomp_obj.H_slices[i])
            for i in range(self.num_blocks)
        ]
        block_work = [block_in_dims[i] * block_h_dims[i] for i in range(self.num_blocks)]

        # First-fit decreasing by useful work.
        order = sorted(range(self.num_blocks), key=lambda i: block_work[i], reverse=True)
        groups = []
        for bi in order:
            b_in = block_in_dims[bi]
            b_h = block_h_dims[bi]
            chosen = -1
            chosen_work = None
            for gi, g in enumerate(groups):
                if g['in_sum'] + b_in <= self.grouped_max_in_dim and g['h_sum'] + b_h <= self.grouped_max_h_dim:
                    if chosen == -1 or g['work_sum'] < chosen_work:
                        chosen = gi
                        chosen_work = g['work_sum']
            if chosen == -1:
                groups.append({'block_indices': [bi], 'in_sum': b_in, 'h_sum': b_h, 'work_sum': block_work[bi]})
            else:
                groups[chosen]['block_indices'].append(bi)
                groups[chosen]['in_sum'] += b_in
                groups[chosen]['h_sum'] += b_h
                groups[chosen]['work_sum'] += block_work[bi]

        # Sort blocks inside each group by output slice start for cleaner write locality/debug readability.
        self.group_block_indices = [
            sorted(g['block_indices'], key=lambda bi: int(decomp_obj.H_slices[bi]))
            for g in groups
        ]
        self.num_groups = len(self.group_block_indices)
        self.MAX_BLOCKS_PER_GROUP = max((len(x) for x in self.group_block_indices), default=1)

        # Build a contiguous, permuted grouped representation so kernel can perform one tl.dot per group.
        grouped_wms_flat_list = []
        group_in_starts_host = []
        group_in_dims_host = []
        group_h_starts_host = []
        group_h_dims_host = []
        group_wm_starts_host = []

        perm_in_unsorted = []
        perm_h_indices = []

        wm_offset = 0
        in_offset = 0
        h_offset = 0
        max_group_in = 1
        max_group_h = 1

        for blocks in self.group_block_indices:
            g_in_dim = sum(block_in_dims[bi] for bi in blocks)
            g_h_dim = sum(block_h_dims[bi] for bi in blocks)

            diag_group = torch.zeros((g_h_dim, g_in_dim), dtype=self.dtype, device=self.device)

            group_in_starts_host.append(in_offset)
            group_in_dims_host.append(g_in_dim)
            group_h_starts_host.append(h_offset)
            group_h_dims_host.append(g_h_dim)
            group_wm_starts_host.append(wm_offset)

            local_h = 0
            local_in = 0
            for bi in blocks:
                b_in = block_in_dims[bi]
                b_h = block_h_dims[bi]
                b_in_start = int(decomp_obj.in_slices[bi])
                b_h_start = int(decomp_obj.H_slices[bi])

                perm_in_unsorted.extend(range(b_in_start, b_in_start + b_in))
                perm_h_indices.extend(range(b_h_start, b_h_start + b_h))

                b_wm = decomp_obj.wms[bi].reshape(b_h, b_in).to(device=self.device, dtype=self.dtype)
                diag_group[local_h:local_h + b_h, local_in:local_in + b_in] = b_wm
                local_h += b_h
                local_in += b_in

            grouped_wms_flat_list.append(diag_group.flatten())

            wm_offset += g_h_dim * g_in_dim
            in_offset += g_in_dim
            h_offset += g_h_dim
            max_group_in = max(max_group_in, g_in_dim)
            max_group_h = max(max_group_h, g_h_dim)

        self.grouped_wms_flat = torch.cat(grouped_wms_flat_list) if grouped_wms_flat_list else torch.empty(0, device=self.device, dtype=self.dtype)
        self.group_in_starts = torch.tensor(group_in_starts_host, dtype=torch.int32, device=self.device)
        self.group_in_dims = torch.tensor(group_in_dims_host, dtype=torch.int32, device=self.device)
        self.group_h_starts = torch.tensor(group_h_starts_host, dtype=torch.int32, device=self.device)
        self.group_h_dims = torch.tensor(group_h_dims_host, dtype=torch.int32, device=self.device)
        self.group_wm_starts = torch.tensor(group_wm_starts_host, dtype=torch.int32, device=self.device)

        # Mapping for permuted grouped input domain -> sorted net_out column indices.
        perm_in_unsorted_t = torch.tensor(perm_in_unsorted, dtype=torch.int64, device=self.device)
        self.in_col_indices_balanced = self.in_col_indices[perm_in_unsorted_t].to(dtype=torch.int32)

        # Permuted output domain -> original output domain restore mapping.
        total_h_dim = int(decomp_obj.H_slices[-1])
        h_restore_indices = torch.empty((total_h_dim,), dtype=torch.int64, device=self.device)
        for perm_h, orig_h in enumerate(perm_h_indices):
            h_restore_indices[orig_h] = perm_h
        self.h_restore_indices = h_restore_indices

        # Keep placeholders for compatibility with old balanced kernel fields.
        self.group_block_counts = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.group_block_in_starts = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.group_block_in_dims = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.group_block_h_starts = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.group_block_h_dims = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.group_block_wm_starts = torch.empty((0,), dtype=torch.int32, device=self.device)

        self.group_ranges = [(-1, -1) for _ in range(self.num_groups)]
        self.MAX_GROUP_IN_DIM, self.MAX_GROUP_H_DIM = self._compute_group_kernel_dims(
            max_group_in, max_group_h
        )

    def get_group_debug_info(self):
        info = []
        useful_flop_values = []

        for gi, blocks in enumerate(self.group_block_indices):
            useful_flop = 0
            in_dim_sum = 0
            h_dim_sum = 0
            for bi in blocks:
                b_in = int(self.in_dims[bi].item())
                b_h = int(self.h_dims[bi].item())
                in_dim_sum += b_in
                h_dim_sum += b_h
                useful_flop += 2 * b_in * b_h

            useful_flop_values.append(useful_flop)
            info.append(
                {
                    'group_index': gi,
                    'block_indices': [int(bi) for bi in blocks],
                    'num_blocks': len(blocks),
                    'in_start': int(self.group_in_starts[gi].item()),
                    'h_start': int(self.group_h_starts[gi].item()),
                    'in_dim': in_dim_sum,
                    'h_dim': h_dim_sum,
                    'in_dim_sum': in_dim_sum,
                    'h_dim_sum': h_dim_sum,
                    'useful_flop_per_edge': useful_flop,
                }
            )

        if len(info) == 0:
            self.group_imbalance = {}
            return []

        mean_useful = float(sum(useful_flop_values)) / float(len(useful_flop_values))
        max_useful = max(useful_flop_values)
        min_useful = min(useful_flop_values)

        self.group_imbalance = {
            'num_groups': len(info),
            'useful_flop_per_edge_min': min_useful,
            'useful_flop_per_edge_max': max_useful,
            'useful_flop_per_edge_mean': mean_useful,
            'useful_flop_imbalance_ratio': (max_useful / mean_useful) if mean_useful > 0 else 0.0,
            'useful_flop_max_over_min': (max_useful / min_useful) if min_useful > 0 else float('inf'),
            'max_blocks_per_group': self.MAX_BLOCKS_PER_GROUP,
        }

        for g in info:
            g['useful_flop_over_mean'] = (
                (g['useful_flop_per_edge'] / mean_useful) if mean_useful > 0 else 0.0
            )
        return info

    def print_group_debug_info(self, max_groups=None):
        info = self.get_group_debug_info()
        imbalance = getattr(self, 'group_imbalance', {})
        if max_groups is None:
            max_groups = len(info)
        max_groups = int(max_groups)

        print("=== Triton Balanced Group Debug Info ===")
        print(
            f"num_blocks={self.num_blocks}, num_groups={self.num_groups}, "
            f"max_blocks_per_group={self.MAX_BLOCKS_PER_GROUP}, "
            f"grouped_max_in_dim={self.grouped_max_in_dim}, grouped_max_h_dim={self.grouped_max_h_dim}"
        )
        print(
            f"MAX_GROUP_IN_DIM={self.MAX_GROUP_IN_DIM}, MAX_GROUP_H_DIM={self.MAX_GROUP_H_DIM}, "
            f"TARGET_GROUP_IN_DIM={getattr(self, 'TARGET_GROUP_IN_DIM', self.MAX_GROUP_IN_DIM)}, "
            f"TARGET_GROUP_H_DIM={getattr(self, 'TARGET_GROUP_H_DIM', self.MAX_GROUP_H_DIM)}"
        )
        if len(info) > 0:
            print(
                f"imbalance useful_flop/edge min/mean/max="
                f"{imbalance['useful_flop_per_edge_min']}/"
                f"{imbalance['useful_flop_per_edge_mean']:.1f}/"
                f"{imbalance['useful_flop_per_edge_max']} "
                f"(max/mean={imbalance['useful_flop_imbalance_ratio']:.2f}x, "
                f"max/min={imbalance['useful_flop_max_over_min']:.2f}x)"
            )

        shown = min(len(info), max_groups)
        for i in range(shown):
            g = info[i]
            print(
                f"group {g['group_index']:4d}: count={g['num_blocks']:3d} | "
                f"in_start={g['in_start']:6d}, in_dim={g['in_dim']:4d} | "
                f"h_start={g['h_start']:6d}, h_dim={g['h_dim']:4d} | "
                f"useful_flop/edge={g['useful_flop_per_edge']:7d} "
                f"({g['useful_flop_over_mean']:.2f}x mean) | "
                f"blocks={g['block_indices']}"
            )
        if shown < len(info):
            print(f"... ({len(info) - shown} more groups)")

    def _get_H_impl(self, net_out):
        num_edges = net_out.shape[0]
        total_in_dim = net_out.shape[1]
        total_h_dim = self.total_h_dim_val
        out_perm = torch.empty((num_edges, total_h_dim), device=net_out.device, dtype=net_out.dtype)

        if self.use_grouped_kernel:
            grid = lambda meta: (
                triton.cdiv(num_edges, meta['BLOCK_M']),
                self.num_groups,
            )
            get_H_triton_grouped_kernel[grid](
                net_out,
                self.grouped_wms_flat,
                out_perm,
                self.in_col_indices_balanced,
                self.group_in_starts,
                self.group_in_dims,
                self.group_h_starts,
                self.group_h_dims,
                self.group_wm_starts,
                num_edges,
                total_in_dim,
                total_h_dim,
                self.MAX_GROUP_IN_DIM,
                self.MAX_GROUP_H_DIM,
                MAX_GROUP_IN_DIM=self.MAX_GROUP_IN_DIM,
                MAX_GROUP_H_DIM=self.MAX_GROUP_H_DIM,
            )
            # Restore original output column order.
            return out_perm[:, self.h_restore_indices]

        # Fall back to base non-grouped block kernel path.
        return super()._get_H_impl(net_out)


# --- FUNDAMENTAL UTILITY TO PREVENT GPU HANG IN TRITON ---
def next_power_of_2(n):
    """Returns the next power of 2 to avoid crashes on tl.arange"""
    return 1 << (int(n) - 1).bit_length()

_CACHE = {}

_AUTOTUNE_CONFIGS = []
for bs in [32, 64, 128, 256]:
    for gs in [1, 4, 8]:
        for warps in [2, 4, 8]:
            for stages in [2, 3, 4]:
                _AUTOTUNE_CONFIGS.append(triton.Config({'BLOCK_M': bs, 'GROUP_SIZE_M': gs}, num_warps=warps, num_stages=stages))

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=['num_edges', 'num_pid_n', 'max_in_dim_key', 'max_h_dim_key'],
)
@triton.jit
def get_H_triton_block_kernel_l2(
    net_out_ptr, wms_ptr, H_out_ptr,
    in_col_indices_ptr, in_starts_ptr, in_dims_ptr,
    h_starts_ptr, h_dims_ptr, wm_starts_ptr,
    num_edges, total_in_dim, total_h_dim, num_pid_n,
    max_in_dim_key, max_h_dim_key,
    BLOCK_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    MAX_IN_DIM: tl.constexpr, MAX_H_DIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(num_edges, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_block = (pid % num_pid_in_group) // group_size_m

    in_start = tl.load(in_starts_ptr + pid_block)
    in_dim = tl.load(in_dims_ptr + pid_block)
    h_start = tl.load(h_starts_ptr + pid_block)
    h_dim = tl.load(h_dims_ptr + pid_block)
    wm_start = tl.load(wm_starts_ptr + pid_block)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MAX_H_DIM)
    offs_k = tl.arange(0, MAX_IN_DIM)

    col_idx = tl.load(in_col_indices_ptr + in_start + offs_k, mask=offs_k < in_dim, other=0)

    a_ptrs = net_out_ptr + offs_m[:, None] * total_in_dim + col_idx[None, :]
    a_mask = (offs_m[:, None] < num_edges) & (offs_k[None, :] < in_dim)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    b_ptrs = wms_ptr + wm_start + offs_n[None, :] * MAX_IN_DIM + offs_k[:, None]
    b_mask = (offs_k[:, None] < in_dim) & (offs_n[None, :] < h_dim)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc = tl.dot(a, b, allow_tf32=False)

    out_ptrs = H_out_ptr + offs_m[:, None] * total_h_dim + (h_start + offs_n[None, :])
    out_mask = (offs_m[:, None] < num_edges) & (offs_n[None, :] < h_dim)
    tl.store(out_ptrs, acc, mask=out_mask)

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=['num_edges', 'num_pid_n', 'max_group_in_dim_key', 'max_group_h_dim_key'],
)
@triton.jit
def get_H_triton_grouped_kernel_l2(
    net_out_ptr, grouped_wms_ptr, H_out_ptr,
    in_col_indices_ptr, group_in_starts_ptr, group_in_dims_ptr,
    group_h_starts_ptr, group_h_dims_ptr, group_wm_starts_ptr,
    num_edges, total_in_dim, total_h_dim, num_pid_n,
    max_group_in_dim_key, max_group_h_dim_key,
    BLOCK_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    MAX_GROUP_IN_DIM: tl.constexpr, MAX_GROUP_H_DIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(num_edges, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_group = (pid % num_pid_in_group) // group_size_m

    g_in_start = tl.load(group_in_starts_ptr + pid_group)
    g_in_dim = tl.load(group_in_dims_ptr + pid_group)
    g_h_start = tl.load(group_h_starts_ptr + pid_group)
    g_h_dim = tl.load(group_h_dims_ptr + pid_group)
    g_wm_start = tl.load(group_wm_starts_ptr + pid_group)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MAX_GROUP_H_DIM)
    offs_k = tl.arange(0, MAX_GROUP_IN_DIM)

    col_idx = tl.load(in_col_indices_ptr + g_in_start + offs_k, mask=offs_k < g_in_dim, other=0)

    a_ptrs = net_out_ptr + offs_m[:, None] * total_in_dim + col_idx[None, :]
    a_mask = (offs_m[:, None] < num_edges) & (offs_k[None, :] < g_in_dim)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    b_ptrs = grouped_wms_ptr + g_wm_start + offs_n[None, :] * MAX_GROUP_IN_DIM + offs_k[:, None]
    b_mask = (offs_k[:, None] < g_in_dim) & (offs_n[None, :] < g_h_dim)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc = tl.dot(a, b, allow_tf32=False)

    out_ptrs = H_out_ptr + offs_m[:, None] * total_h_dim + (g_h_start + offs_n[None, :])
    out_mask = (offs_m[:, None] < num_edges) & (offs_n[None, :] < g_h_dim)
    tl.store(out_ptrs, acc, mask=out_mask)

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=['num_edges', 'num_pid_n', 'max_in_dim_key', 'max_h_dim_key'],
)
@triton.jit
def get_net_out_triton_block_kernel_l2(
    H_ptr, wms_ptr, net_out_ptr,
    in_col_indices_ptr, in_starts_ptr, in_dims_ptr,
    h_starts_ptr, h_dims_ptr, wm_starts_ptr,
    num_edges, total_in_dim, total_h_dim, num_pid_n,
    max_in_dim_key, max_h_dim_key,
    BLOCK_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    MAX_IN_DIM: tl.constexpr, MAX_H_DIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(num_edges, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_block = (pid % num_pid_in_group) // group_size_m

    in_start = tl.load(in_starts_ptr + pid_block)
    in_dim = tl.load(in_dims_ptr + pid_block)
    h_start = tl.load(h_starts_ptr + pid_block)
    h_dim = tl.load(h_dims_ptr + pid_block)
    wm_start = tl.load(wm_starts_ptr + pid_block)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, MAX_H_DIM)
    offs_n = tl.arange(0, MAX_IN_DIM)

    a_ptrs = H_ptr + offs_m[:, None] * total_h_dim + (h_start + offs_k[None, :])
    a_mask = (offs_m[:, None] < num_edges) & (offs_k[None, :] < h_dim)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    b_ptrs = wms_ptr + wm_start + offs_k[:, None] * MAX_IN_DIM + offs_n[None, :]
    b_mask = (offs_k[:, None] < h_dim) & (offs_n[None, :] < in_dim)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc = tl.dot(a, b, allow_tf32=False)

    col_idx = tl.load(in_col_indices_ptr + in_start + offs_n, mask=offs_n < in_dim, other=0)
    out_ptrs = net_out_ptr + offs_m[:, None] * total_in_dim + col_idx[None, :]
    out_mask = (offs_m[:, None] < num_edges) & (offs_n[None, :] < in_dim)
    
    # FIX: Use tl.store since in_col_indices map to unique locations
    tl.store(out_ptrs, acc, mask=out_mask)

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=['num_edges', 'num_pid_n', 'max_group_in_dim_key', 'max_group_h_dim_key'],
)
@triton.jit
def get_net_out_triton_grouped_kernel_l2(
    H_ptr, grouped_wms_ptr, net_out_ptr,
    in_col_indices_ptr, group_in_starts_ptr, group_in_dims_ptr,
    group_h_starts_ptr, group_h_dims_ptr, group_wm_starts_ptr,
    num_edges, total_in_dim, total_h_dim, num_pid_n,
    max_group_in_dim_key, max_group_h_dim_key,
    BLOCK_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    MAX_GROUP_IN_DIM: tl.constexpr, MAX_GROUP_H_DIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(num_edges, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_group = (pid % num_pid_in_group) // group_size_m

    g_in_start = tl.load(group_in_starts_ptr + pid_group)
    g_in_dim = tl.load(group_in_dims_ptr + pid_group)
    g_h_start = tl.load(group_h_starts_ptr + pid_group)
    g_h_dim = tl.load(group_h_dims_ptr + pid_group)
    g_wm_start = tl.load(group_wm_starts_ptr + pid_group)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, MAX_GROUP_H_DIM)
    offs_n = tl.arange(0, MAX_GROUP_IN_DIM)

    a_ptrs = H_ptr + offs_m[:, None] * total_h_dim + (g_h_start + offs_k[None, :])
    a_mask = (offs_m[:, None] < num_edges) & (offs_k[None, :] < g_h_dim)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    b_ptrs = grouped_wms_ptr + g_wm_start + offs_k[:, None] * MAX_GROUP_IN_DIM + offs_n[None, :]
    b_mask = (offs_k[:, None] < g_h_dim) & (offs_n[None, :] < g_in_dim)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc = tl.dot(a, b, allow_tf32=False)

    col_idx = tl.load(in_col_indices_ptr + g_in_start + offs_n, mask=offs_n < g_in_dim, other=0)
    out_ptrs = net_out_ptr + offs_m[:, None] * total_in_dim + col_idx[None, :]
    out_mask = (offs_m[:, None] < num_edges) & (offs_n[None, :] < g_in_dim)
    
    # FIX: Use tl.store since in_col_indices map to unique locations
    tl.store(out_ptrs, acc, mask=out_mask)


class TritonGetHFunctionL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, net_out, decomp_obj):
        ctx.decomp_obj = decomp_obj
        return decomp_obj._get_H_impl(net_out)

    @staticmethod
    def backward(ctx, grad_output):
        grad_net_out = ctx.decomp_obj._get_net_out_impl(grad_output)
        return grad_net_out, None


class TritonE3TensorDecompL2(TritonE3TensorDecomp):
    def __init__(self, *args, **kwargs):
        kwargs_super = kwargs.copy()
        kwargs_super.pop("MAX_IN_DIM", None)
        kwargs_super.pop("MAX_H_DIM", None)
        super().__init__(*args, **kwargs_super)
        
        decomp_obj = args[0] if len(args) > 0 else kwargs['decomp_obj']
        self.decomp_obj = decomp_obj
        
        # FIX: Include grouping sizes in cache key!
        cache_key = (
            tuple(self.decomp_obj.out_js_list), 
            getattr(self, 'use_grouped_kernel', False), 
            getattr(self, 'grouped_max_in_dim', 32),
            getattr(self, 'grouped_max_h_dim', 32),
            self.__class__.__name__
        )
        if cache_key in _CACHE:
            cached_data = _CACHE[cache_key]
            self.MAX_IN_DIM = cached_data['MAX_IN_DIM']
            self.MAX_H_DIM = cached_data['MAX_H_DIM']
            if hasattr(self, 'MAX_GROUP_IN_DIM'):
                self.MAX_GROUP_IN_DIM = cached_data.get('MAX_GROUP_IN_DIM', self.MAX_GROUP_IN_DIM)
            if hasattr(self, 'MAX_GROUP_H_DIM'):
                self.MAX_GROUP_H_DIM = cached_data.get('MAX_GROUP_H_DIM', self.MAX_GROUP_H_DIM)
            self.wms_flat = cached_data['wms_flat']
            self.wm_starts = cached_data['wm_starts']
            if self.use_grouped_kernel:
                self.grouped_wms_flat = cached_data['grouped_wms_flat']
                self.group_wm_starts = cached_data['group_wm_starts']
            return
            
        # FIX: Force power of 2 for Triton array dimensions
        if hasattr(self, 'MAX_IN_DIM'):
            self.MAX_IN_DIM = next_power_of_2(max(16, self.MAX_IN_DIM))
            self.MAX_H_DIM = next_power_of_2(max(16, self.MAX_H_DIM))
        else:
            self.MAX_IN_DIM = 16
            self.MAX_H_DIM = 16
            
        if hasattr(self, 'MAX_GROUP_IN_DIM'):
            self.MAX_GROUP_IN_DIM = next_power_of_2(max(16, self.MAX_GROUP_IN_DIM))
            self.MAX_GROUP_H_DIM = next_power_of_2(max(16, self.MAX_GROUP_H_DIM))

        # FIX: Re-pad wms_flat because MAX_IN_DIM was changed!
        import torch
        wms_padded_list = []
        for i in range(self.num_blocks):
            in_start = int(self.decomp_obj.in_slices[i])
            in_end = int(self.decomp_obj.in_slices[i+1])
            h_start = int(self.decomp_obj.H_slices[i])
            h_end = int(self.decomp_obj.H_slices[i+1])
            in_dim = in_end - in_start
            h_dim = h_end - h_start
            wm = self.decomp_obj.wms[i].clone().view(h_dim, in_dim)
            wm_padded = torch.zeros((self.MAX_H_DIM, self.MAX_IN_DIM), dtype=self.dtype, device=self.device)
            wm_padded[:h_dim, :in_dim] = wm
            wms_padded_list.append(wm_padded.flatten())
            
        self.wms_flat = torch.cat(wms_padded_list).to(device=self.device, dtype=self.dtype)
        self.wm_starts = torch.arange(self.num_blocks, dtype=torch.int32, device=self.device) * (self.MAX_H_DIM * self.MAX_IN_DIM)

        if self.use_grouped_kernel:
            grouped_wms_padded_list = []
            # Process blocks for each group
            for g_idx in range(self.num_groups):
                if hasattr(self, 'group_block_indices'):
                    blocks = self.group_block_indices[g_idx]
                else:
                    g_start, g_end = self.group_ranges[g_idx]
                    blocks = list(range(g_start, g_end))
                    
                in_dim = sum([int(self.decomp_obj.in_slices[b+1] - self.decomp_obj.in_slices[b]) for b in blocks])
                h_dim = sum([int(self.decomp_obj.H_slices[b+1] - self.decomp_obj.H_slices[b]) for b in blocks])
                
                g_wm = torch.zeros((h_dim, in_dim), dtype=self.dtype, device=self.device)
                in_offset = 0
                h_offset = 0
                for block_idx in blocks:
                    b_in = int(self.decomp_obj.in_slices[block_idx+1] - self.decomp_obj.in_slices[block_idx])
                    b_h = int(self.decomp_obj.H_slices[block_idx+1] - self.decomp_obj.H_slices[block_idx])
                    g_wm[h_offset:h_offset+b_h, in_offset:in_offset+b_in] = self.decomp_obj.wms[block_idx].view(b_h, b_in)
                    in_offset += b_in
                    h_offset += b_h
                    
                wm_padded = torch.zeros((self.MAX_GROUP_H_DIM, self.MAX_GROUP_IN_DIM), dtype=self.dtype, device=self.device)
                wm_padded[:h_dim, :in_dim] = g_wm
                grouped_wms_padded_list.append(wm_padded.flatten())
                
            self.grouped_wms_flat = torch.cat(grouped_wms_padded_list).to(device=self.device, dtype=self.dtype)
            self.group_wm_starts = torch.arange(self.num_groups, dtype=torch.int32, device=self.device) * (self.MAX_GROUP_H_DIM * self.MAX_GROUP_IN_DIM)

        cached_data = {
            'MAX_IN_DIM': self.MAX_IN_DIM,
            'MAX_H_DIM': self.MAX_H_DIM,
            'wms_flat': self.wms_flat,
            'wm_starts': self.wm_starts,
        }
        if hasattr(self, 'MAX_GROUP_IN_DIM'):
            cached_data['MAX_GROUP_IN_DIM'] = self.MAX_GROUP_IN_DIM
        if hasattr(self, 'MAX_GROUP_H_DIM'):
            cached_data['MAX_GROUP_H_DIM'] = self.MAX_GROUP_H_DIM
        if self.use_grouped_kernel:
            cached_data['grouped_wms_flat'] = self.grouped_wms_flat
            cached_data['group_wm_starts'] = self.group_wm_starts
        _CACHE[cache_key] = cached_data
        
    def get_H(self, net_out):
        return TritonGetHFunctionL2.apply(net_out, self)

    def get_net_out(self, H):
        return self._get_net_out_impl(H)

    def _get_H_impl(self, net_out):
        num_edges = net_out.shape[0]
        total_in_dim = net_out.shape[1]
        total_h_dim = self.total_h_dim_val
        out = torch.empty((num_edges, total_h_dim), device=net_out.device, dtype=net_out.dtype)

        if self.use_grouped_kernel:
            num_pid_n = self.num_groups
            grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_M']) * num_pid_n,)
            get_H_triton_grouped_kernel_l2[grid](
                net_out, self.grouped_wms_flat, out,
                self.in_col_indices, self.group_in_starts, self.group_in_dims,
                self.group_h_starts, self.group_h_dims, self.group_wm_starts,
                num_edges, total_in_dim, total_h_dim, num_pid_n,
                self.MAX_GROUP_IN_DIM, self.MAX_GROUP_H_DIM,
                MAX_GROUP_IN_DIM=self.MAX_GROUP_IN_DIM, MAX_GROUP_H_DIM=self.MAX_GROUP_H_DIM,
            )
        else:
            num_pid_n = self.num_blocks
            grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_M']) * num_pid_n,)
            get_H_triton_block_kernel_l2[grid](
                net_out, self.wms_flat, out,
                self.in_col_indices, self.in_starts, self.in_dims,
                self.h_starts, self.h_dims, self.wm_starts,
                num_edges, total_in_dim, total_h_dim, num_pid_n,
                self.MAX_IN_DIM, self.MAX_H_DIM,
                MAX_IN_DIM=self.MAX_IN_DIM, MAX_H_DIM=self.MAX_H_DIM,
            )
        return out

    def _get_net_out_impl(self, H):
        num_edges = H.shape[0]
        total_in_dim = int(self.decomp_obj.in_slices[-1])
        total_h_dim = self.total_h_dim_val
        out = torch.zeros((num_edges, total_in_dim), device=H.device, dtype=H.dtype)

        if self.use_grouped_kernel:
            num_pid_n = self.num_groups
            grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_M']) * num_pid_n,)
            get_net_out_triton_grouped_kernel_l2[grid](
                H, self.grouped_wms_flat, out,
                self.in_col_indices, self.group_in_starts, self.group_in_dims,
                self.group_h_starts, self.group_h_dims, self.group_wm_starts,
                num_edges, total_in_dim, total_h_dim, num_pid_n,
                self.MAX_GROUP_IN_DIM, self.MAX_GROUP_H_DIM,
                MAX_GROUP_IN_DIM=self.MAX_GROUP_IN_DIM, MAX_GROUP_H_DIM=self.MAX_GROUP_H_DIM,
            )
        else:
            num_pid_n = self.num_blocks
            grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_M']) * num_pid_n,)
            get_net_out_triton_block_kernel_l2[grid](
                H, self.wms_flat, out,
                self.in_col_indices, self.in_starts, self.in_dims,
                self.h_starts, self.h_dims, self.wm_starts,
                num_edges, total_in_dim, total_h_dim, num_pid_n,
                self.MAX_IN_DIM, self.MAX_H_DIM,
                MAX_IN_DIM=self.MAX_IN_DIM, MAX_H_DIM=self.MAX_H_DIM,
            )
        return out


class BalancedTritonE3TensorDecompL2(BalancedTritonE3TensorDecomp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        decomp_obj = args[0] if len(args) > 0 else kwargs['decomp_obj']
        self.decomp_obj = decomp_obj
        
        # FIX: Include grouping sizes in cache key!
        cache_key = (
            tuple(self.decomp_obj.out_js_list), 
            getattr(self, 'use_grouped_kernel', False), 
            getattr(self, 'grouped_max_in_dim', 32),
            getattr(self, 'grouped_max_h_dim', 32),
            self.__class__.__name__
        )
        if cache_key in _CACHE:
            cached_data = _CACHE[cache_key]
            self.MAX_IN_DIM = cached_data['MAX_IN_DIM']
            self.MAX_H_DIM = cached_data['MAX_H_DIM']
            if hasattr(self, 'MAX_GROUP_IN_DIM'):
                self.MAX_GROUP_IN_DIM = cached_data.get('MAX_GROUP_IN_DIM', self.MAX_GROUP_IN_DIM)
            if hasattr(self, 'MAX_GROUP_H_DIM'):
                self.MAX_GROUP_H_DIM = cached_data.get('MAX_GROUP_H_DIM', self.MAX_GROUP_H_DIM)
            self.wms_flat = cached_data['wms_flat']
            self.wm_starts = cached_data['wm_starts']
            if self.use_grouped_kernel:
                self.grouped_wms_flat = cached_data['grouped_wms_flat']
                self.group_wm_starts = cached_data['group_wm_starts']
            return
            
        # FIX: Force power of 2
        if hasattr(self, 'MAX_IN_DIM'):
            self.MAX_IN_DIM = next_power_of_2(max(16, self.MAX_IN_DIM))
            self.MAX_H_DIM = next_power_of_2(max(16, self.MAX_H_DIM))
        else:
            self.MAX_IN_DIM = 16
            self.MAX_H_DIM = 16

        if hasattr(self, 'MAX_GROUP_IN_DIM'):
            self.MAX_GROUP_IN_DIM = next_power_of_2(max(16, self.MAX_GROUP_IN_DIM))
            self.MAX_GROUP_H_DIM = next_power_of_2(max(16, self.MAX_GROUP_H_DIM))

        wms_padded_list = []
        for i in range(self.num_blocks):
            in_start = int(self.decomp_obj.in_slices[i])
            in_end = int(self.decomp_obj.in_slices[i+1])
            h_start = int(self.decomp_obj.H_slices[i])
            h_end = int(self.decomp_obj.H_slices[i+1])
            in_dim = in_end - in_start
            h_dim = h_end - h_start
            
            wm = self.decomp_obj.wms[i].clone().view(h_dim, in_dim)
            wm_padded = torch.zeros((self.MAX_H_DIM, self.MAX_IN_DIM), dtype=self.dtype, device=self.device)
            wm_padded[:h_dim, :in_dim] = wm
            wms_padded_list.append(wm_padded.flatten())
            
        self.wms_flat = torch.cat(wms_padded_list).to(device=self.device, dtype=self.dtype)
        self.wm_starts = torch.arange(self.num_blocks, dtype=torch.int32, device=self.device) * (self.MAX_H_DIM * self.MAX_IN_DIM)

        if self.use_grouped_kernel:
            grouped_wms_padded_list = []
            # Process blocks for each group
            for g_idx in range(self.num_groups):
                if hasattr(self, 'group_block_indices'):
                    blocks = self.group_block_indices[g_idx]
                else:
                    g_start, g_end = self.group_ranges[g_idx]
                    blocks = list(range(g_start, g_end))
                    
                in_dim = sum([int(self.decomp_obj.in_slices[b+1] - self.decomp_obj.in_slices[b]) for b in blocks])
                h_dim = sum([int(self.decomp_obj.H_slices[b+1] - self.decomp_obj.H_slices[b]) for b in blocks])
                
                g_wm = torch.zeros((h_dim, in_dim), dtype=self.dtype, device=self.device)
                in_offset = 0
                h_offset = 0
                for block_idx in blocks:
                    b_in = int(self.decomp_obj.in_slices[block_idx+1] - self.decomp_obj.in_slices[block_idx])
                    b_h = int(self.decomp_obj.H_slices[block_idx+1] - self.decomp_obj.H_slices[block_idx])
                    g_wm[h_offset:h_offset+b_h, in_offset:in_offset+b_in] = self.decomp_obj.wms[block_idx].view(b_h, b_in)
                    in_offset += b_in
                    h_offset += b_h
                wm_padded = torch.zeros((self.MAX_GROUP_H_DIM, self.MAX_GROUP_IN_DIM), dtype=self.dtype, device=self.device)
                wm_padded[:h_dim, :in_dim] = g_wm
                grouped_wms_padded_list.append(wm_padded.flatten())
                
            self.grouped_wms_flat = torch.cat(grouped_wms_padded_list).to(device=self.device, dtype=self.dtype)
            self.group_wm_starts = torch.arange(self.num_groups, dtype=torch.int32, device=self.device) * (self.MAX_GROUP_H_DIM * self.MAX_GROUP_IN_DIM)

        cached_data = {
            'MAX_IN_DIM': self.MAX_IN_DIM,
            'MAX_H_DIM': self.MAX_H_DIM,
            'wms_flat': self.wms_flat,
            'wm_starts': self.wm_starts,
        }
        if hasattr(self, 'MAX_GROUP_IN_DIM'):
            cached_data['MAX_GROUP_IN_DIM'] = self.MAX_GROUP_IN_DIM
        if hasattr(self, 'MAX_GROUP_H_DIM'):
            cached_data['MAX_GROUP_H_DIM'] = self.MAX_GROUP_H_DIM
        if self.use_grouped_kernel:
            cached_data['grouped_wms_flat'] = self.grouped_wms_flat
            cached_data['group_wm_starts'] = self.group_wm_starts
        _CACHE[cache_key] = cached_data

        
    def get_H(self, net_out):
        return TritonGetHFunctionL2.apply(net_out, self)

    def get_net_out(self, H):
        return self._get_net_out_impl(H)

    def _get_H_impl(self, net_out):
        num_edges = net_out.shape[0]
        total_in_dim = net_out.shape[1]
        total_h_dim = self.total_h_dim_val
        out_perm = torch.empty((num_edges, total_h_dim), device=net_out.device, dtype=net_out.dtype)

        if self.use_grouped_kernel:
            num_pid_n = self.num_groups
            grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_M']) * num_pid_n,)
            get_H_triton_grouped_kernel_l2[grid](
                net_out, self.grouped_wms_flat, out_perm,
                self.in_col_indices_balanced, self.group_in_starts, self.group_in_dims,
                self.group_h_starts, self.group_h_dims, self.group_wm_starts,
                num_edges, total_in_dim, total_h_dim, num_pid_n,
                self.MAX_GROUP_IN_DIM, self.MAX_GROUP_H_DIM,
                MAX_GROUP_IN_DIM=self.MAX_GROUP_IN_DIM, MAX_GROUP_H_DIM=self.MAX_GROUP_H_DIM,
            )
            return out_perm[:, self.h_restore_indices]

        return super()._get_H_impl(net_out)

    def _get_net_out_impl(self, H):
        num_edges = H.shape[0]
        total_in_dim = int(self.decomp_obj.in_slices[-1])
        total_h_dim = self.total_h_dim_val
        out = torch.zeros((num_edges, total_in_dim), device=H.device, dtype=H.dtype)

        if self.use_grouped_kernel:
            # FIX: use h_restore_indices.device instead of self.device
            if not hasattr(self, 'h_unrestore_indices'):
                unrestore = torch.empty_like(self.h_restore_indices)
                unrestore[self.h_restore_indices] = torch.arange(len(self.h_restore_indices), device=self.h_restore_indices.device)
                self.h_unrestore_indices = unrestore
                
            H_perm = H[:, self.h_unrestore_indices].contiguous()
            
            num_pid_n = self.num_groups
            grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_M']) * num_pid_n,)
            get_net_out_triton_grouped_kernel_l2[grid](
                H_perm, self.grouped_wms_flat, out,
                self.in_col_indices_balanced, self.group_in_starts, self.group_in_dims,
                self.group_h_starts, self.group_h_dims, self.group_wm_starts,
                num_edges, total_in_dim, total_h_dim, num_pid_n,
                self.MAX_GROUP_IN_DIM, self.MAX_GROUP_H_DIM,
                MAX_GROUP_IN_DIM=self.MAX_GROUP_IN_DIM, MAX_GROUP_H_DIM=self.MAX_GROUP_H_DIM,
            )
        else:
            num_pid_n = self.num_blocks
            grid = lambda meta: (triton.cdiv(num_edges, meta['BLOCK_M']) * num_pid_n,)
            get_net_out_triton_block_kernel_l2[grid](
                H, self.wms_flat, out,
                self.in_col_indices, self.in_starts, self.in_dims,
                self.h_starts, self.h_dims, self.wm_starts,
                num_edges, total_in_dim, total_h_dim, num_pid_n,
                self.MAX_IN_DIM, self.MAX_H_DIM,
                MAX_IN_DIM=self.MAX_IN_DIM, MAX_H_DIM=self.MAX_H_DIM,
            )
            
        return out