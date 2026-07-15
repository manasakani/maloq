import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import sys
import time
from typing import NamedTuple
import scipy.sparse as sparse

max_elements = 100
THREADS_PER_BLOCK = 128

def sparse_block_matrix_to_scipy_csr(sparse_matrix):
    """One-off conversion from the GPU block-list SparseBlockMatrix to a CPU
    scipy.sparse.csr_matrix, for writing to disk. Builds explicit per-element
    row/col indices -- fine for an I/O-time conversion, not meant for the
    training hot path."""
    rows = cp.asnumpy(sparse_matrix.fock_block_rows)
    cols = cp.asnumpy(sparse_matrix.fock_block_cols)
    offsets = cp.asnumpy(sparse_matrix.fock_block_offsets)
    data = cp.asnumpy(sparse_matrix.data)
    matrix_size = int(offsets[-1])

    row_idx_parts = []
    col_idx_parts = []
    for b in range(len(rows)):
        i, j = rows[b], cols[b]
        r0, r1 = offsets[i], offsets[i + 1]
        c0, c1 = offsets[j], offsets[j + 1]
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing='ij')
        row_idx_parts.append(rr.ravel())
        col_idx_parts.append(cc.ravel())

    row_idx = np.concatenate(row_idx_parts)
    col_idx = np.concatenate(col_idx_parts)

    coo = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(matrix_size, matrix_size))
    return coo.tocsr()

class SparseBlockMatrix(NamedTuple):
    data: cp.ndarray                # (total_elements,) float32 -- block values, row-major, back-to-back
    block_data_offsets: cp.ndarray  # (num_blocks + 1,) int64   -- start of block b in `data`
    fock_block_rows: cp.ndarray     # (num_blocks,) int32       -- atom index, row side of each block
    fock_block_cols: cp.ndarray     # (num_blocks,) int32       -- atom index, col side of each block
    fock_block_offsets: cp.ndarray  # (num_atoms + 1,) int32    -- same per-atom offsets as the dense path

_multiple_matrix2label_fp32 = cp.RawKernel(r'''
#define MAX_ELEMENTS 100

extern "C" __global__
void _multiple_matrix2label_fp32(
    const int num_structures,
    const int ** __restrict__  fock_block_rows,
    const int ** __restrict__  fock_block_cols,
    const int * __restrict__  fock_block_cumsum,
    const int ** __restrict__  fock_block_offsets,
    const int ** __restrict__  idx_to_atomic_number,
    const int ** __restrict__  orbital_templates,
    const int * __restrict__  orbital_template_lengths,
    float ** __restrict__  fock_matrices,
    const int * __restrict__ stride_fock_matrices_m,
    const int * __restrict__ stride_fock_matrices_n,
    float ** __restrict__  targets,
    const int * __restrict__ stride_targets_m,
    const int * __restrict__ stride_targets_n,
    const bool forward
){


    const int tbidx = blockIdx.x;

    // thread idx in thread block
    const int idx = threadIdx.x;

    const int tidx = tbidx * blockDim.x + idx;

    const int warp_idx = idx / 32;
    const int lane_idx = idx % 32;

    // NOTE: assume blockDim.x is multiple of 32
    const int warp_per_block = blockDim.x / 32;

    // get structure index
    // default to last structure
    int sidx = 0;
    for(int i = 0; i < num_structures; i++){
            // check if tbidx is in the range of structure i

            if(tbidx >= fock_block_cumsum[i] && tbidx < fock_block_cumsum[i+1]){
                sidx = i;
                break;
            }
    }
    const int bidx = tbidx - fock_block_cumsum[sidx];


    const int block_i = fock_block_rows[sidx][bidx];
    const int block_j = fock_block_cols[sidx][bidx];

    const int block_row_start = fock_block_offsets[sidx][block_i];
    const int block_row_end = fock_block_offsets[sidx][block_i + 1];
    const int block_col_start = fock_block_offsets[sidx][block_j];
    const int block_col_end = fock_block_offsets[sidx][block_j + 1];


    const int block_size_row = block_row_end - block_row_start;
    const int block_size_col = block_col_end - block_col_start;

    float *fock_matrix = fock_matrices[sidx];
    float *target = targets[sidx];

    const int stride_fock_matrice_m = stride_fock_matrices_m[sidx];
    const int stride_fock_matrice_n = stride_fock_matrices_n[sidx];
    const int stride_target_m = stride_targets_m[sidx];
    const int stride_target_n = stride_targets_n[sidx];

    float *block_target = target
        + bidx * stride_target_m;

    const int atomic_element_i = idx_to_atomic_number[sidx][block_i];
    const int atomic_element_j = idx_to_atomic_number[sidx][block_j];

    const int element_interaction_key = atomic_element_i * MAX_ELEMENTS + atomic_element_j;

    // TODO: think about the order (row vs column major)
    // load fock block into shared memory
    extern __shared__ float shared_interaction_block[];

    float *global_interaction_block = fock_matrix +
        block_row_start * stride_fock_matrice_m +
        block_col_start * stride_fock_matrice_n;

    if(true){
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            const int row = i / block_size_col;
            const int col = i % block_size_col;
            shared_interaction_block[row * block_size_col + col] = global_interaction_block[
                row * stride_fock_matrice_m +
                col * stride_fock_matrice_n
            ];
        }
        __syncthreads();
    }

    const int *orbital_template = orbital_templates[element_interaction_key];
    const int orbital_template_length = orbital_template_lengths[element_interaction_key];

    // let a warp handle a subblock
    for(int j = warp_idx; j < orbital_template_length; j += warp_per_block){
        const int subblock_row_start = orbital_template[j * 5 + 0];
        const int subblock_row_end = orbital_template[j * 5 + 1];
        const int subblock_col_start = orbital_template[j * 5 + 2];
        const int subblock_col_end = orbital_template[j * 5 + 3];
        const int output_slice_block_start = orbital_template[j * 5 + 4];

        const int subblock_size_row = subblock_row_end - subblock_row_start;
        const int subblock_size_col = subblock_col_end - subblock_col_start;

        float *shared_interaction_subblock = shared_interaction_block
             +
            subblock_row_start * block_size_col +
            subblock_col_start;

        // loop over elements in the subblock
        // using warp
        for(int k = lane_idx; k < subblock_size_row * subblock_size_col; k += 32){
            const int row = k / subblock_size_col;
            const int col = k % subblock_size_col;

            const int output_idx = (output_slice_block_start + k) * stride_target_n;

            if(forward){
                block_target[
                    output_idx
                ] = shared_interaction_subblock[
                    row * block_size_col +
                    col
                ];
            }
            else{
                shared_interaction_subblock[
                    row * block_size_col +
                    col
                ] = block_target[
                    output_idx
                ];

            }
        }

    }

    // store back to global memory in backward mode
    if(!forward){

        __syncthreads();
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            const int row = i / block_size_col;
            const int col = i % block_size_col;
            global_interaction_block[
                row * stride_fock_matrice_m +
                col * stride_fock_matrice_n
            ] = shared_interaction_block[row * block_size_col + col];

        }
    }

}
''',
    "_multiple_matrix2label_fp32",
)


_multiple_matrix2label_fp64 = cp.RawKernel(r'''
#define MAX_ELEMENTS 100

extern "C" __global__
void _multiple_matrix2label_fp64(
    const int num_structures,
    const int ** __restrict__  fock_block_rows,
    const int ** __restrict__  fock_block_cols,
    const int * __restrict__  fock_block_cumsum,
    const int ** __restrict__  fock_block_offsets,
    const int ** __restrict__  idx_to_atomic_number,
    const int ** __restrict__  orbital_templates,
    const int * __restrict__  orbital_template_lengths,
    double ** __restrict__  fock_matrices,
    const int * __restrict__ stride_fock_matrices_m,
    const int * __restrict__ stride_fock_matrices_n,
    double ** __restrict__  targets,
    const int * __restrict__ stride_targets_m,
    const int * __restrict__ stride_targets_n,
    const bool forward
){


    const int tbidx = blockIdx.x;

    // thread idx in thread block
    const int idx = threadIdx.x;

    const int tidx = tbidx * blockDim.x + idx;

    const int warp_idx = idx / 32;
    const int lane_idx = idx % 32;

    // NOTE: assume blockDim.x is multiple of 32
    const int warp_per_block = blockDim.x / 32;

    // get structure index
    // default to last structure
    int sidx = 0;
    for(int i = 0; i < num_structures; i++){
            // check if tbidx is in the range of structure i

            if(tbidx >= fock_block_cumsum[i] && tbidx < fock_block_cumsum[i+1]){
                sidx = i;
                break;
            }
    }
    const int bidx = tbidx - fock_block_cumsum[sidx];


    const int block_i = fock_block_rows[sidx][bidx];
    const int block_j = fock_block_cols[sidx][bidx];

    const int block_row_start = fock_block_offsets[sidx][block_i];
    const int block_row_end = fock_block_offsets[sidx][block_i + 1];
    const int block_col_start = fock_block_offsets[sidx][block_j];
    const int block_col_end = fock_block_offsets[sidx][block_j + 1];


    const int block_size_row = block_row_end - block_row_start;
    const int block_size_col = block_col_end - block_col_start;

    double *fock_matrix = fock_matrices[sidx];
    double *target = targets[sidx];

    const int stride_fock_matrice_m = stride_fock_matrices_m[sidx];
    const int stride_fock_matrice_n = stride_fock_matrices_n[sidx];
    const int stride_target_m = stride_targets_m[sidx];
    const int stride_target_n = stride_targets_n[sidx];

    double *block_target = target
        + bidx * stride_target_m;

    const int atomic_element_i = idx_to_atomic_number[sidx][block_i];
    const int atomic_element_j = idx_to_atomic_number[sidx][block_j];

    const int element_interaction_key = atomic_element_i * MAX_ELEMENTS + atomic_element_j;

    // TODO: think about the order (row vs column major)
    // load fock block into shared memory
    extern __shared__ double shared_interaction_block[];

    double *global_interaction_block = fock_matrix +
        block_row_start * stride_fock_matrice_m +
        block_col_start * stride_fock_matrice_n;

    if(true){
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            const int row = i / block_size_col;
            const int col = i % block_size_col;
            shared_interaction_block[row * block_size_col + col] = global_interaction_block[
                row * stride_fock_matrice_m +
                col * stride_fock_matrice_n
            ];
        }
        __syncthreads();
    }

    const int *orbital_template = orbital_templates[element_interaction_key];
    const int orbital_template_length = orbital_template_lengths[element_interaction_key];

    // let a warp handle a subblock
    for(int j = warp_idx; j < orbital_template_length; j += warp_per_block){
        const int subblock_row_start = orbital_template[j * 5 + 0];
        const int subblock_row_end = orbital_template[j * 5 + 1];
        const int subblock_col_start = orbital_template[j * 5 + 2];
        const int subblock_col_end = orbital_template[j * 5 + 3];
        const int output_slice_block_start = orbital_template[j * 5 + 4];

        const int subblock_size_row = subblock_row_end - subblock_row_start;
        const int subblock_size_col = subblock_col_end - subblock_col_start;

        double *shared_interaction_subblock = shared_interaction_block
             +
            subblock_row_start * block_size_col +
            subblock_col_start;

        // loop over elements in the subblock
        // using warp
        for(int k = lane_idx; k < subblock_size_row * subblock_size_col; k += 32){
            const int row = k / subblock_size_col;
            const int col = k % subblock_size_col;

            const int output_idx = (output_slice_block_start + k) * stride_target_n;

            if(forward){
                block_target[
                    output_idx
                ] = shared_interaction_subblock[
                    row * block_size_col +
                    col
                ];
            }
            else{
                shared_interaction_subblock[
                    row * block_size_col +
                    col
                ] = block_target[
                    output_idx
                ];

            }
        }

    }

    // store back to global memory in backward mode
    if(!forward){

        __syncthreads();
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            const int row = i / block_size_col;
            const int col = i % block_size_col;
            global_interaction_block[
                row * stride_fock_matrice_m +
                col * stride_fock_matrice_n
            ] = shared_interaction_block[row * block_size_col + col];

        }
    }

}
''',
    "_multiple_matrix2label_fp64",
)


def get_ptr(tensor):
    # check if numpy or cupy or torch
    if isinstance(tensor, np.ndarray):
        raise ValueError("Numpy arrays are not supported in this kernel.")
    elif isinstance(tensor, cp.ndarray):
        return tensor.data.ptr
    elif isinstance(tensor, torch.Tensor):
        raise ValueError("Torch tensors are not supported in this kernel.")
        # return tensor.data_ptr()
    else:
        raise ValueError("Unsupported tensor type")


def cupy_multiple_matrix2label(
    num_structures,
    orbital_templates,
    fock_block_offsets,
    idx_to_atomic_numbers,
    fock_block_rows,
    fock_block_cols,
    fock_matrices,
    targets,
    orbital_template_ptrs,
    forward,
):
    # Expect that fock_matrices is a cupy array !!!
    if not all(targets[0].dtype == t.dtype for t in targets):
        raise ValueError("All target tensors must have the same dtype")
    if not all(fock_matrices[0].dtype == f.dtype for f in fock_matrices):
        raise ValueError("All fock matrices must have the same dtype")
    if not all(targets[0].dtype == f.dtype for f in fock_matrices):
        raise ValueError("Target tensors and fock matrices must have the same dtype")

    if targets[0].dtype == cp.float32:
        float_size = 4
    elif targets[0].dtype == cp.float64:
        float_size = 8
    else:
        raise ValueError("Unsupported dtype for targets. Only float32 and float64 are supported.")

    targets = [cp.ascontiguousarray(t) for t in targets]
    fock_matrices = [cp.ascontiguousarray(f) for f in fock_matrices]

    # transform the inputs for cupy
    fock_matrices_ptrs = [get_ptr(fock_matrix) for fock_matrix in fock_matrices]
    fock_matrices_ptrs = cp.array(
        fock_matrices_ptrs, dtype=cp.uintp
    )
    fock_matrices_strides_m = cp.array(
        [fock_matrix.strides[0] // float_size for fock_matrix in fock_matrices],
        dtype=cp.int32
    )
    fock_matrices_strides_n = cp.array(
        [fock_matrix.strides[1] // float_size for fock_matrix in fock_matrices],
        dtype=cp.int32
    )

    target_ptrs = [get_ptr(t) for t in targets]
    target_ptrs = cp.array(
        target_ptrs, dtype=cp.uintp
    )
    target_strides_m = cp.array(
        [t.strides[0] // float_size for t in targets],
        dtype=cp.int32
    )
    target_strides_n = cp.array(
        [t.strides[1] // float_size for t in targets],
        dtype=cp.int32
    )

    fock_block_rows = [cp.array(fock_block_row, dtype=cp.int32) for fock_block_row in fock_block_rows]
    fock_block_cols = [cp.array(fock_block_col, dtype=cp.int32) for fock_block_col in fock_block_cols]
    fock_block_offsets = [cp.array(fock_block_offset, dtype=cp.int32) for fock_block_offset in fock_block_offsets]
    idx_to_atomic_numbers = [cp.array(idx_to_atomic_number, dtype=cp.int32) for idx_to_atomic_number in idx_to_atomic_numbers]

    fock_block_rows_ptrs = [get_ptr(fock_block_row) for fock_block_row in fock_block_rows]
    fock_block_rows_ptrs = cp.array(
        fock_block_rows_ptrs, dtype=cp.uintp
    )
    fock_block_cols_ptrs = [get_ptr(fock_block_col) for fock_block_col in fock_block_cols]
    fock_block_cols_ptrs = cp.array(
        fock_block_cols_ptrs, dtype=cp.uintp
    )
    fock_block_cumsum = cp.array(
        [len(fock_block_rows[i]) for i in range(num_structures)],
        dtype=cp.int32
    )
    fock_block_cumsum = cp.cumsum(fock_block_cumsum)
    fock_block_cumsum = cp.concatenate((cp.array([0], dtype=cp.int32), fock_block_cumsum), dtype=cp.int32)


    fock_block_offsets_ptrs = [get_ptr(fock_block_offset) for fock_block_offset in fock_block_offsets]
    fock_block_offsets_ptrs = cp.array(
        fock_block_offsets_ptrs, dtype=cp.uintp
    )
    idx_to_atomic_number_ptrs = [get_ptr(idx_to_atomic_number) for idx_to_atomic_number in idx_to_atomic_numbers]
    idx_to_atomic_number_ptrs = cp.array(
        idx_to_atomic_number_ptrs, dtype=cp.uintp
    )

    orbital_template_lengths = cp.array(
        [len(orbital_templates[i]) for i in range(len(orbital_templates))],
        dtype=cp.int32
    )

    blocks_per_grid = sum([len(fock_block_rows[i]) for i in range(num_structures)])

    max_block_size = 0
    for fock_block_offset in fock_block_offsets:
        block_sizes = np.diff(fock_block_offset.get())
        max_block_size = max(max_block_size, np.max(block_sizes))

    end_preprocess = time.perf_counter()
    cp.cuda.Stream.null.synchronize()

    start_kernel = time.perf_counter()
    if  targets[0].dtype == cp.float32:
        _multiple_matrix2label_fp32(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                cp.int32(num_structures),
                fock_block_rows_ptrs,
                fock_block_cols_ptrs,
                fock_block_cumsum,
                fock_block_offsets_ptrs,
                idx_to_atomic_number_ptrs,
                orbital_template_ptrs,
                orbital_template_lengths,
                fock_matrices_ptrs,
                fock_matrices_strides_m,
                fock_matrices_strides_n,
                target_ptrs,
                target_strides_m,
                target_strides_n,
                cp.bool_(forward)
            ),
            shared_mem = float_size * max_block_size**2,
        )
    elif targets[0].dtype == cp.float64:
        _multiple_matrix2label_fp64(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                cp.int32(num_structures),
                fock_block_rows_ptrs,
                fock_block_cols_ptrs,
                fock_block_cumsum,
                fock_block_offsets_ptrs,
                idx_to_atomic_number_ptrs,
                orbital_template_ptrs,
                orbital_template_lengths,
                fock_matrices_ptrs,
                fock_matrices_strides_m,
                fock_matrices_strides_n,
                target_ptrs,
                target_strides_m,
                target_strides_n,
                cp.bool_(forward)
            ),
            shared_mem = float_size * max_block_size**2,
        )

_single_matrix2label_fp32 = cp.RawKernel(r'''
#define MAX_ELEMENTS 100

extern "C" __global__
void _single_matrix2label_fp32(
    const int * __restrict__  fock_block_rows,
    const int * __restrict__  fock_block_cols,
    const int * __restrict__  fock_block_offsets,
    const int * __restrict__  idx_to_atomic_number,
    const int ** __restrict__  orbital_templates,
    const int * __restrict__  orbital_template_lengths,
    float * __restrict__  fock_matrix,
    const long long stride_fock_matrice_m,
    const long long stride_fock_matrice_n,
    float * __restrict__  target,
    const long long stride_target_m,
    const long long stride_target_n,
    const bool forward
){


    const int bidx = blockIdx.x;

    // thread idx in thread block
    const int idx = threadIdx.x;

    const int tidx = bidx * blockDim.x + idx;

    const int warp_idx = idx / 32;
    const int lane_idx = idx % 32;

    // NOTE: assume blockDim.x is multiple of 32
    const int warp_per_block = blockDim.x / 32;

    const int block_i = fock_block_rows[bidx];
    const int block_j = fock_block_cols[bidx];

    const int block_row_start = fock_block_offsets[block_i];
    const int block_row_end = fock_block_offsets[block_i + 1];
    const int block_col_start = fock_block_offsets[block_j];
    const int block_col_end = fock_block_offsets[block_j + 1];


    const int block_size_row = block_row_end - block_row_start;
    const int block_size_col = block_col_end - block_col_start;

    float *block_target = target
        + (long long)bidx * stride_target_m;

    const int atomic_element_i = idx_to_atomic_number[block_i];
    const int atomic_element_j = idx_to_atomic_number[block_j];

    const int element_interaction_key = atomic_element_i * MAX_ELEMENTS + atomic_element_j;

    // TODO: think about the order (row vs column major)
    // load fock block into shared memory
    extern __shared__ float shared_interaction_block[];

    float *global_interaction_block = fock_matrix +
        (long long)block_row_start * stride_fock_matrice_m +
        (long long)block_col_start * stride_fock_matrice_n;

    if(true){
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            const int row = i / block_size_col;
            const int col = i % block_size_col;
            shared_interaction_block[row * block_size_col + col] = global_interaction_block[
                (long long)row * stride_fock_matrice_m +
                (long long)col * stride_fock_matrice_n
            ];
        }
        __syncthreads();
    }

    const int *orbital_template = orbital_templates[element_interaction_key];
    const int orbital_template_length = orbital_template_lengths[element_interaction_key];

    // let a warp handle a subblock
    for(int j = warp_idx; j < orbital_template_length; j += warp_per_block){
        const int subblock_row_start = orbital_template[j * 5 + 0];
        const int subblock_row_end = orbital_template[j * 5 + 1];
        const int subblock_col_start = orbital_template[j * 5 + 2];
        const int subblock_col_end = orbital_template[j * 5 + 3];
        const int output_slice_block_start = orbital_template[j * 5 + 4];

        const int subblock_size_row = subblock_row_end - subblock_row_start;
        const int subblock_size_col = subblock_col_end - subblock_col_start;

        float *shared_interaction_subblock = shared_interaction_block
             +
            subblock_row_start * block_size_col +
            subblock_col_start;

        // loop over elements in the subblock
        // using warp
        for(int k = lane_idx; k < subblock_size_row * subblock_size_col; k += 32){
            const int row = k / subblock_size_col;
            const int col = k % subblock_size_col;

            const long long output_idx = (long long)(output_slice_block_start + k) * stride_target_n;

            if(forward){
                block_target[
                    output_idx
                ] = shared_interaction_subblock[
                    row * block_size_col +
                    col
                ];
            }
            else{
                shared_interaction_subblock[
                    row * block_size_col +
                    col
                ] = block_target[
                    output_idx
                ];

            }
        }

    }

    // store back to global memory in backward mode
    if(!forward){

        __syncthreads();
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            const int row = i / block_size_col;
            const int col = i % block_size_col;
            global_interaction_block[
                (long long)row * stride_fock_matrice_m +
                (long long)col * stride_fock_matrice_n
            ] = shared_interaction_block[row * block_size_col + col];

        }
    }

}
''',
    "_single_matrix2label_fp32",
)

_single_matrix2label_fp32_sparse = cp.RawKernel(r'''
#define MAX_ELEMENTS 100

extern "C" __global__
void _single_matrix2label_fp32_sparse(
    const int * __restrict__  fock_block_rows,
    const int * __restrict__  fock_block_cols,
    const int * __restrict__  fock_block_offsets,
    const long long * __restrict__  block_data_offsets,
    const int * __restrict__  idx_to_atomic_number,
    const int ** __restrict__  orbital_templates,
    const int * __restrict__  orbital_template_lengths,
    float * __restrict__  fock_matrix_sparse,
    float * __restrict__  target,
    const long long stride_target_m,
    const long long stride_target_n,
    const bool forward
){

    const int bidx = blockIdx.x;
    const int idx = threadIdx.x;

    const int warp_idx = idx / 32;
    const int lane_idx = idx % 32;
    const int warp_per_block = blockDim.x / 32;

    const int block_i = fock_block_rows[bidx];
    const int block_j = fock_block_cols[bidx];

    const int block_row_start = fock_block_offsets[block_i];
    const int block_row_end = fock_block_offsets[block_i + 1];
    const int block_col_start = fock_block_offsets[block_j];
    const int block_col_end = fock_block_offsets[block_j + 1];

    const int block_size_row = block_row_end - block_row_start;
    const int block_size_col = block_col_end - block_col_start;

    float *block_target = target
        + (long long)bidx * stride_target_m;

    const int atomic_element_i = idx_to_atomic_number[block_i];
    const int atomic_element_j = idx_to_atomic_number[block_j];

    const int element_interaction_key = atomic_element_i * MAX_ELEMENTS + atomic_element_j;

    extern __shared__ float shared_interaction_block[];

    // block b lives contiguously (row-major, block_size_row x block_size_col)
    // starting at block_data_offsets[b] -- no strides needed, unlike the dense
    // kernel which has to walk into a (matrix_size, matrix_size) buffer.
    float *global_interaction_block = fock_matrix_sparse + block_data_offsets[bidx];

    if(true){
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            shared_interaction_block[i] = global_interaction_block[i];
        }
        __syncthreads();
    }

    const int *orbital_template = orbital_templates[element_interaction_key];
    const int orbital_template_length = orbital_template_lengths[element_interaction_key];

    for(int j = warp_idx; j < orbital_template_length; j += warp_per_block){
        const int subblock_row_start = orbital_template[j * 5 + 0];
        const int subblock_row_end = orbital_template[j * 5 + 1];
        const int subblock_col_start = orbital_template[j * 5 + 2];
        const int subblock_col_end = orbital_template[j * 5 + 3];
        const int output_slice_block_start = orbital_template[j * 5 + 4];

        const int subblock_size_row = subblock_row_end - subblock_row_start;
        const int subblock_size_col = subblock_col_end - subblock_col_start;

        float *shared_interaction_subblock = shared_interaction_block
             +
            subblock_row_start * block_size_col +
            subblock_col_start;

        for(int k = lane_idx; k < subblock_size_row * subblock_size_col; k += 32){
            const int row = k / subblock_size_col;
            const int col = k % subblock_size_col;

            const long long output_idx = (long long)(output_slice_block_start + k) * stride_target_n;

            if(forward){
                block_target[
                    output_idx
                ] = shared_interaction_subblock[
                    row * block_size_col +
                    col
                ];
            }
            else{
                shared_interaction_subblock[
                    row * block_size_col +
                    col
                ] = block_target[
                    output_idx
                ];
            }
        }
    }

    if(!forward){
        __syncthreads();
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            global_interaction_block[i] = shared_interaction_block[i];
        }
    }
}
''',
    "_single_matrix2label_fp32_sparse",
)

_single_matrix2label_fp64 = cp.RawKernel(r'''
#define MAX_ELEMENTS 100

extern "C" __global__
void _single_matrix2label_fp64(
    const int * __restrict__  fock_block_rows,
    const int * __restrict__  fock_block_cols,
    const int * __restrict__  fock_block_offsets,
    const int * __restrict__  idx_to_atomic_number,
    const int ** __restrict__  orbital_templates,
    const int * __restrict__  orbital_template_lengths,
    float * __restrict__  fock_matrix,
    const int stride_fock_matrice_m,
    const int stride_fock_matrice_n,
    float * __restrict__  target,
    const int stride_target_m,
    const int stride_target_n,
    const bool forward
){


    const int bidx = blockIdx.x;

    // thread idx in thread block
    const int idx = threadIdx.x;

    const int tidx = bidx * blockDim.x + idx;

    const int warp_idx = idx / 32;
    const int lane_idx = idx % 32;

    // NOTE: assume blockDim.x is multiple of 32
    const int warp_per_block = blockDim.x / 32;

    const int block_i = fock_block_rows[bidx];
    const int block_j = fock_block_cols[bidx];

    const int block_row_start = fock_block_offsets[block_i];
    const int block_row_end = fock_block_offsets[block_i + 1];
    const int block_col_start = fock_block_offsets[block_j];
    const int block_col_end = fock_block_offsets[block_j + 1];


    const int block_size_row = block_row_end - block_row_start;
    const int block_size_col = block_col_end - block_col_start;

    float *block_target = target
        + bidx * stride_target_m;

    const int atomic_element_i = idx_to_atomic_number[block_i];
    const int atomic_element_j = idx_to_atomic_number[block_j];

    const int element_interaction_key = atomic_element_i * MAX_ELEMENTS + atomic_element_j;

    // TODO: think about the order (row vs column major)
    // load fock block into shared memory
    extern __shared__ float shared_interaction_block[];

    float *global_interaction_block = fock_matrix +
        block_row_start * stride_fock_matrice_m +
        block_col_start * stride_fock_matrice_n;

    if(true){
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            const int row = i / block_size_col;
            const int col = i % block_size_col;
            shared_interaction_block[row * block_size_col + col] = global_interaction_block[
                row * stride_fock_matrice_m +
                col * stride_fock_matrice_n
            ];
        }
        __syncthreads();
    }

    const int *orbital_template = orbital_templates[element_interaction_key];
    const int orbital_template_length = orbital_template_lengths[element_interaction_key];

    // let a warp handle a subblock
    for(int j = warp_idx; j < orbital_template_length; j += warp_per_block){
        const int subblock_row_start = orbital_template[j * 5 + 0];
        const int subblock_row_end = orbital_template[j * 5 + 1];
        const int subblock_col_start = orbital_template[j * 5 + 2];
        const int subblock_col_end = orbital_template[j * 5 + 3];
        const int output_slice_block_start = orbital_template[j * 5 + 4];

        const int subblock_size_row = subblock_row_end - subblock_row_start;
        const int subblock_size_col = subblock_col_end - subblock_col_start;

        float *shared_interaction_subblock = shared_interaction_block
             +
            subblock_row_start * block_size_col +
            subblock_col_start;

        // loop over elements in the subblock
        // using warp
        for(int k = lane_idx; k < subblock_size_row * subblock_size_col; k += 32){
            const int row = k / subblock_size_col;
            const int col = k % subblock_size_col;

            const int output_idx = (output_slice_block_start + k) * stride_target_n;

            if(forward){
                block_target[
                    output_idx
                ] = shared_interaction_subblock[
                    row * block_size_col +
                    col
                ];
            }
            else{
                shared_interaction_subblock[
                    row * block_size_col +
                    col
                ] = block_target[
                    output_idx
                ];

            }
        }

    }

    // store back to global memory in backward mode
    if(!forward){

        __syncthreads();
        for(int i = idx; i < block_size_row * block_size_col; i += blockDim.x){
            const int row = i / block_size_col;
            const int col = i % block_size_col;
            global_interaction_block[
                row * stride_fock_matrice_m +
                col * stride_fock_matrice_n
            ] = shared_interaction_block[row * block_size_col + col];

        }
    }

}
''',
    "_single_matrix2label_fp64",
)



def cupy_single_matrix2label(
    orbital_templates,
    fock_block_offsets,
    idx_to_atomic_number,
    fock_block_rows,
    fock_block_cols,
    fock_matrix,
    target,
    orbital_template_ptrs,
    forward
):
    assert target.dtype == fock_matrix.dtype

    if target.dtype == cp.float32:
        float_size = 4
    elif target.dtype == cp.float64:
        float_size = 8
    else:
        raise ValueError("Unsupported dtype for targets. Only float32 and float64 are supported.")



    target = cp.ascontiguousarray(target)
    fock_matrix = cp.ascontiguousarray(fock_matrix)

    target_stride_m = np.int64(target.strides[0] // float_size)
    target_stride_n = np.int64(target.strides[1] // float_size)

    fock_matrix_stride_m = np.int64(fock_matrix.strides[0] // float_size)
    fock_matrix_stride_n = np.int64(fock_matrix.strides[1] // float_size)


    fock_block_rows = cp.array(fock_block_rows, dtype=cp.int32)
    fock_block_cols = cp.array(fock_block_cols, dtype=cp.int32)
    fock_block_offsets = cp.array(fock_block_offsets, dtype=cp.int32)
    idx_to_atomic_number = cp.array(idx_to_atomic_number, dtype=cp.int32)

    orbital_template_lengths = cp.array(
        [len(orbital_templates[i]) for i in range(len(orbital_templates))],
        dtype=cp.int32
    )

    blocks_per_grid = len(fock_block_rows)

    max_block_size = np.max(np.diff(fock_block_offsets.get()))
    cp.cuda.Stream.null.synchronize()
    end_preprocess = time.perf_counter()
    start_kernel = time.perf_counter()

    if target.dtype == cp.float32:
        _single_matrix2label_fp32(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                fock_block_rows,
                fock_block_cols,
                fock_block_offsets,
                idx_to_atomic_number,
                orbital_template_ptrs,
                orbital_template_lengths,
                fock_matrix,
                fock_matrix_stride_m,
                fock_matrix_stride_n,
                target,
                target_stride_m,
                target_stride_n,
                cp.bool_(forward)
            ),
            shared_mem = float_size * max_block_size**2,
        )
    elif target.dtype == cp.float64:
        _single_matrix2label_fp64(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                fock_block_rows,
                fock_block_cols,
                fock_block_offsets,
                idx_to_atomic_number,
                orbital_template_ptrs,
                orbital_template_lengths,
                fock_matrix,
                fock_matrix_stride_m,
                fock_matrix_stride_n,
                target,
                target_stride_m,
                target_stride_n,
                cp.bool_(forward)
            ),
            shared_mem = float_size * max_block_size**2,
        )

def cupy_single_matrix2label_sparse(
    orbital_templates,
    fock_block_offsets,
    idx_to_atomic_number,
    fock_block_rows,
    fock_block_cols,
    target,
    orbital_template_ptrs,
    forward,
    sparse_matrix=None,       # a SparseBlockMatrix -- required when forward=True
    block_data_offsets=None,  # pass a cached one in if the block structure is unchanged since last call
):
    assert target.dtype == cp.float32, "sparse kernel only implements fp32"
    float_size = 4

    target = cp.ascontiguousarray(target)
    target_stride_m = np.int64(target.strides[0] // float_size)
    target_stride_n = np.int64(target.strides[1] // float_size)

    fock_block_rows = cp.asarray(fock_block_rows, dtype=cp.int32)
    fock_block_cols = cp.asarray(fock_block_cols, dtype=cp.int32)
    fock_block_offsets = cp.asarray(fock_block_offsets, dtype=cp.int32)
    idx_to_atomic_number = cp.asarray(idx_to_atomic_number, dtype=cp.int32)

    orbital_template_lengths = cp.array(
        [len(orbital_templates[i]) for i in range(len(orbital_templates))],
        dtype=cp.int32,
    )

    num_blocks = len(fock_block_rows)

    block_row_sizes = fock_block_offsets[fock_block_rows + 1] - fock_block_offsets[fock_block_rows]
    block_col_sizes = fock_block_offsets[fock_block_cols + 1] - fock_block_offsets[fock_block_cols]
    max_block_size = int(cp.maximum(block_row_sizes, block_col_sizes).max().get())

    if block_data_offsets is None:
        block_elem_counts = (block_row_sizes * block_col_sizes).astype(cp.int64)
        block_data_offsets = cp.concatenate([
            cp.zeros(1, dtype=cp.int64),
            cp.cumsum(block_elem_counts),
        ])

    total_elements = int(block_data_offsets[-1].get())

    if forward:
        assert sparse_matrix is not None, "sparse_matrix (with populated .data) is required when forward=True"
        data = cp.ascontiguousarray(sparse_matrix.data)
        assert data.size == total_elements, "sparse_matrix.data doesn't match the current block structure"
    else:
        data = cp.zeros((total_elements,), dtype=cp.float32)

    _single_matrix2label_fp32_sparse(
        (num_blocks,),
        (THREADS_PER_BLOCK,),
        (
            fock_block_rows,
            fock_block_cols,
            fock_block_offsets,
            block_data_offsets,
            idx_to_atomic_number,
            orbital_template_ptrs,
            orbital_template_lengths,
            data,
            target,
            target_stride_m,
            target_stride_n,
            cp.bool_(forward),
        ),
        shared_mem=float_size * max_block_size**2,
    )

    return SparseBlockMatrix(
        data=data,
        block_data_offsets=block_data_offsets,
        fock_block_rows=fock_block_rows,
        fock_block_cols=fock_block_cols,
        fock_block_offsets=fock_block_offsets,
    )

def numpy_single_matrix2label(
    orbital_template,
    fock_block_offsets,
    idx_to_atomic_number,
    fock_block_rows,
    fock_block_cols,
    fock_matrix,
    target,
    forward
):

    # iterates over atom pairs (i, j) within this molecule
    for idx in range(len(fock_block_rows)):
        i = fock_block_rows[idx].item()
        j = fock_block_cols[idx].item()

        # extract this atom pair subblock from the fock matrix:
        row_slice = slice(
            fock_block_offsets[i].item(), fock_block_offsets[i + 1].item()
        )
        col_slice = slice(
            fock_block_offsets[j].item(), fock_block_offsets[j + 1].item()
        )
        interaction_block = fock_matrix[row_slice, col_slice]

        # Get the template of all orbital interactions for this atom pair
        atomic_element_i = idx_to_atomic_number[i].item()
        atomic_element_j = idx_to_atomic_number[j].item()

        element_interaction_key = atomic_element_j + max_elements * atomic_element_i

        for row_slice_block, col_slice_block, output_slice_block in orbital_template[element_interaction_key]:

            if forward == True:
                # single orbital interaction (e.g. s-p, d-d, etc.)
                block = interaction_block[
                    row_slice_block,
                    col_slice_block
                ]

                target[idx][
                    output_slice_block
                ] = block.flatten()
            else:
                # single orbital interaction (e.g. s-p, d-d, etc.)
                fock_matrix[row_slice, col_slice][
                    row_slice_block,
                    col_slice_block
                ] = target[idx][
                    output_slice_block
                ].reshape(
                    row_slice_block.stop - row_slice_block.start,
                    col_slice_block.stop - col_slice_block.start
                )



def multiple_matrix2label(
    num_structures,
    orbital_template,
    fock_block_offsets,
    idx_to_atomic_number,
    fock_block_rows,
    fock_block_cols,
    fock_matrix,
    target,
    forward
):
    for k in range(num_structures):

        # iterates over atom pairs (i, j) within this molecule
        for idx in range(len(fock_block_rows[k])):
            i = fock_block_rows[k][idx]
            j = fock_block_cols[k][idx]

            # extract this atom pair subblock from the fock matrix:
            row_slice = slice(
                fock_block_offsets[k][i], fock_block_offsets[k][i + 1]
            )
            col_slice = slice(
                fock_block_offsets[k][j], fock_block_offsets[k][j + 1]
            )
            interaction_block = fock_matrix[k][row_slice, col_slice]

            # Get the template of all orbital interactions for this atom pair
            atomic_element_i = idx_to_atomic_number[i]
            atomic_element_j = idx_to_atomic_number[j]

            element_interaction_key = atomic_element_j + max_elements * atomic_element_i

            for row_slice_block, col_slice_block, output_slice_block in orbital_template[element_interaction_key]:

                    if forward == True:
                        # single orbital interaction (e.g. s-p, d-d, etc.)
                        block = interaction_block[
                            row_slice_block,
                            col_slice_block
                        ]

                        target[k][idx][
                            output_slice_block
                        ] = block.flatten()
                    else:
                        # single orbital interaction (e.g. s-p, d-d, etc.)
                        interaction_block[
                            row_slice_block,
                            col_slice_block
                        ] = target[k][idx][
                            output_slice_block
                        ].reshape(
                            row_slice_block.stop - row_slice_block.start,
                            col_slice_block.stop - col_slice_block.start
                        )


def get_orbital_template(equivariant_blocks, orbital_starts):
    """
    Clean up this function..
    """

    # make flat blocks a dictionary with the atom interactionas as keys
    flat_blocks_dict = {}
    for index_target, equivariant_block in enumerate(equivariant_blocks):
        for N_M_str, block_slice in equivariant_block.items():

            condition_numbers = tuple(map(int, N_M_str.split()))

            if condition_numbers not in flat_blocks_dict:
                flat_blocks_dict[condition_numbers] = []

            slice_out = slice(orbital_starts[index_target], orbital_starts[index_target + 1])
            slice_row = slice(block_slice[0], block_slice[1])
            slice_col = slice(block_slice[2], block_slice[3])
            flat_blocks_dict[condition_numbers].append((slice_row, slice_col, slice_out))

    max_elements = 100
    orbital_template = [[] for _ in range(max_elements**2)]

    for index_target, equivariant_block in enumerate(equivariant_blocks):
        for N_M_str, block_slice in equivariant_block.items():

            condition_numbers = tuple(map(int, N_M_str.split()))

            if condition_numbers not in flat_blocks_dict:
                flat_blocks_dict[condition_numbers] = []

            slice_out = slice(orbital_starts[index_target], orbital_starts[index_target + 1])
            slice_row = slice(block_slice[0], block_slice[1])
            slice_col = slice(block_slice[2], block_slice[3])

            orbital_template[
                condition_numbers[0]*max_elements + condition_numbers[1]
            ].append((slice_row, slice_col, slice_out))

    return orbital_template
