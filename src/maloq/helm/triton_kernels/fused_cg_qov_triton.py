"""One-launch QH9 CG-label to target-orbital QOV projection.

This is a deliberately specialized experimental operator for the QH9
def2-SVP padded atom basis: every final coupled label has 196 components and
decodes to one 14 x 14 atom-pair block.  The forward Triton kernel keeps that
local block in registers and contracts it directly into graph-level QOV; it
never writes a local or global AO density matrix to memory.

The compact sparse backward intentionally uses two kernels: the first writes
one 196-value edge-local block gradient, and the second applies a transpose-ELL
decoder.  This 16/32 MiB production-sized workspace is deterministic and was
substantially faster than forcing dynamic register gathers into one launch; it
is not a molecule-scale AO matrix.

Each decoder-bank entry follows the row-vector convention
``flat_padded_block = labels @ decoder[type]``.  Its autograd adjoint is
therefore ``grad_labels = grad_flat_block @ decoder[type].T``.  This is
intentionally not the MALOQ ``get_net_out`` inverse, which carries
degree-dependent ``2L+1`` factors.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in CPU-only installs
    triton = None
    tl = None


QH9_BLOCK_DIM = 14
QH9_NUM_COEFFICIENTS = QH9_BLOCK_DIM * QH9_BLOCK_DIM


@dataclass(frozen=True)
class SparseQH9DecoderBank:
    """Output-major ELL decoder and its exact Euclidean adjoint.

    ``values``/``indices`` list the input coefficients contributing to each
    decoded block element. ``transpose_values``/``transpose_indices`` list the
    decoded block elements contributing to each input coefficient. Sentinel
    index ``-1`` marks ELL padding.  The two layouts avoid global scatter in
    both the forward decoder and its label-gradient adjoint.
    """

    values: torch.Tensor
    indices: torch.Tensor
    transpose_values: torch.Tensor
    transpose_indices: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def num_decoders(self) -> int:
        return self.values.shape[0]

    @property
    def width(self) -> int:
        return self.values.shape[-1]

    @property
    def transpose_width(self) -> int:
        return self.transpose_values.shape[-1]


@torch.no_grad()
def compress_qh9_decoder_bank(
    decoder: torch.Tensor,
    *,
    zero_tolerance: float = 0.0,
) -> SparseQH9DecoderBank:
    """Convert a dense ``(T, 196, 196)`` decoder into paired ELL layouts.

    The dense convention is ``decoded[e, j] = sum_k labels[e, k] *
    decoder[type, k, j]``.  Entries are kept when their magnitude is strictly
    larger than ``zero_tolerance``.  The default keeps every floating-point
    nonzero so compression is algebraically lossless for the supplied tensor.
    """
    if decoder.ndim == 2:
        decoder = decoder.unsqueeze(0)
    if decoder.ndim != 3 or tuple(decoder.shape[-2:]) != (
        QH9_NUM_COEFFICIENTS,
        QH9_NUM_COEFFICIENTS,
    ):
        raise ValueError("decoder bank must have shape (T, 196, 196)")
    if decoder.dtype not in {torch.float32, torch.float64}:
        raise TypeError("QH9 decoder bank supports float32 and float64")
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")

    def _pack(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # rows: (T, output-row, candidate-column). Stable integer sorting keeps
        # the original dense summation order among retained coefficients.
        keep = rows.abs() > zero_tolerance
        max_nnz = max(1, int(keep.sum(dim=-1).max().item()))
        width = 1 << (max_nnz - 1).bit_length()
        candidates = torch.arange(
            QH9_NUM_COEFFICIENTS,
            device=rows.device,
            dtype=torch.int32,
        ).view(1, 1, -1)
        sentinel = torch.full_like(candidates, QH9_NUM_COEFFICIENTS)
        packed_indices = torch.where(keep, candidates, sentinel)
        packed_indices = packed_indices.sort(dim=-1, stable=True).values[..., :width]
        valid = packed_indices < QH9_NUM_COEFFICIENTS
        safe_indices = packed_indices.clamp_max(QH9_NUM_COEFFICIENTS - 1).long()
        packed_values = rows.gather(-1, safe_indices)
        packed_values = torch.where(valid, packed_values, 0.0).contiguous()
        packed_indices = torch.where(valid, packed_indices, -1).to(torch.int32)
        return packed_values, packed_indices.contiguous()

    dense = decoder.detach().contiguous()
    values, indices = _pack(dense.transpose(1, 2).contiguous())
    transpose_values, transpose_indices = _pack(dense)
    return SparseQH9DecoderBank(
        values=values,
        indices=indices,
        transpose_values=transpose_values,
        transpose_indices=transpose_indices,
    )


if triton is not None:

    @triton.jit
    def _decode_sparse_qh9_block(
        labels_ptr,
        decoder_values_ptr,
        decoder_indices_ptr,
        edge,
        decoder_index,
        labels_stride_e,
        decoder_stride_t,
        decoder_stride_o,
        BLOCK_DIM: tl.constexpr,
        BLOCK_ORBITAL: tl.constexpr,
        ELL_WIDTH: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        offs_b = tl.arange(0, BLOCK_ORBITAL)
        offs_n = tl.arange(0, ELL_WIDTH)
        if USE_FP64:
            block = tl.zeros((BLOCK_ORBITAL, BLOCK_ORBITAL), dtype=tl.float64)
        else:
            block = tl.zeros((BLOCK_ORBITAL, BLOCK_ORBITAL), dtype=tl.float32)
        for row in tl.static_range(0, BLOCK_DIM):
            output_index = row * BLOCK_DIM + offs_b
            ell_ptrs = (
                decoder_index * decoder_stride_t
                + output_index[:, None] * decoder_stride_o
                + offs_n[None, :]
            )
            input_index = tl.load(
                decoder_indices_ptr + ell_ptrs,
                mask=offs_b[:, None] < BLOCK_DIM,
                other=-1,
            )
            value = tl.load(
                decoder_values_ptr + ell_ptrs,
                mask=(offs_b[:, None] < BLOCK_DIM) & (input_index >= 0),
                other=0.0,
            )
            coefficient = tl.load(
                labels_ptr + edge * labels_stride_e + input_index,
                mask=(offs_b[:, None] < BLOCK_DIM) & (input_index >= 0),
                other=0.0,
            )
            if USE_FP64:
                value = value.to(tl.float64)
                coefficient = coefficient.to(tl.float64)
            else:
                value = value.to(tl.float32)
                coefficient = coefficient.to(tl.float32)
            decoded_row = tl.sum(value * coefficient, axis=1)
            block += tl.where(
                offs_b[:, None] == row,
                decoded_row[None, :],
                0.0,
            )
        return block

    @triton.jit
    def _cg_labels_to_qov_deterministic_fwd_kernel(
        labels_ptr,
        decoder_ptr,
        decoder_index_ptr,
        f_occ_ptr,
        f_virt_ptr,
        graph_edge_offsets_ptr,
        graph_edge_order_ptr,
        row_ao_ptr,
        col_ao_ptr,
        row_valid_ptr,
        col_valid_ptr,
        n_occ_ptr,
        n_virt_ptr,
        qov_anchor_ptr,
        qov_ptr,
        labels_stride_e,
        decoder_stride_t,
        decoder_stride_k,
        f_occ_stride_g,
        f_occ_stride_ao,
        f_virt_stride_g,
        f_virt_stride_ao,
        qov_stride_g,
        qov_stride_o,
        BLOCK_DIM: tl.constexpr,
        NUM_COEFF: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_ORBITAL: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_VIRT: tl.constexpr,
        MAX_OCC: tl.constexpr,
        MAX_VIRT: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        graph = tl.program_id(0)
        occ_start = tl.program_id(1) * BLOCK_OCC
        offs_o = occ_start + tl.arange(0, BLOCK_OCC)
        offs_v = tl.arange(0, BLOCK_VIRT)
        n_occ = tl.load(n_occ_ptr + graph)
        n_virt = tl.load(n_virt_ptr + graph)
        output_mask = (offs_o[:, None] < MAX_OCC) & (offs_v[None, :] < MAX_VIRT)
        active_mask = (offs_o[:, None] < n_occ) & (offs_v[None, :] < n_virt)
        q_tile = tl.load(
            qov_anchor_ptr
            + graph * qov_stride_g
            + offs_o[:, None] * qov_stride_o
            + offs_v[None, :],
            mask=output_mask,
            other=0.0,
        )
        if USE_FP64:
            q_tile = q_tile.to(tl.float64)
        else:
            q_tile = q_tile.to(tl.float32)

        edge_start = tl.load(graph_edge_offsets_ptr + graph)
        edge_end = tl.load(graph_edge_offsets_ptr + graph + 1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_b = tl.arange(0, BLOCK_ORBITAL)
        for edge_position in tl.range(edge_start, edge_end):
            edge = tl.load(graph_edge_order_ptr + edge_position)
            decoder_index = tl.load(decoder_index_ptr + edge)
            coeff = tl.load(
                labels_ptr + edge * labels_stride_e + offs_k,
                mask=offs_k < NUM_COEFF,
                other=0.0,
            )
            if USE_FP64:
                coeff = coeff.to(tl.float64)
                block = tl.zeros(
                    (BLOCK_ORBITAL, BLOCK_ORBITAL),
                    dtype=tl.float64,
                )
            else:
                coeff = coeff.to(tl.float32)
                block = tl.zeros(
                    (BLOCK_ORBITAL, BLOCK_ORBITAL),
                    dtype=tl.float32,
                )
            for row in tl.static_range(0, BLOCK_DIM):
                flat_col = row * BLOCK_DIM + offs_b
                decoder = tl.load(
                    decoder_ptr
                    + decoder_index * decoder_stride_t
                    + offs_k[:, None] * decoder_stride_k
                    + flat_col[None, :],
                    mask=(offs_k[:, None] < NUM_COEFF)
                    & (offs_b[None, :] < BLOCK_DIM),
                    other=0.0,
                )
                if USE_FP64:
                    decoder = decoder.to(tl.float64)
                else:
                    decoder = decoder.to(tl.float32)
                decoded_row = tl.sum(coeff[:, None] * decoder, axis=0)
                block += tl.where(
                    offs_b[:, None] == row,
                    decoded_row[None, :],
                    0.0,
                )

            row_ids = tl.load(
                row_ao_ptr + edge * BLOCK_DIM + offs_b,
                mask=offs_b < BLOCK_DIM,
                other=0,
            )
            col_ids = tl.load(
                col_ao_ptr + edge * BLOCK_DIM + offs_b,
                mask=offs_b < BLOCK_DIM,
                other=0,
            )
            row_valid = tl.load(
                row_valid_ptr + edge * BLOCK_DIM + offs_b,
                mask=offs_b < BLOCK_DIM,
                other=0,
            ).to(tl.int1)
            col_valid = tl.load(
                col_valid_ptr + edge * BLOCK_DIM + offs_b,
                mask=offs_b < BLOCK_DIM,
                other=0,
            ).to(tl.int1)
            left = tl.load(
                f_occ_ptr
                + graph * f_occ_stride_g
                + row_ids[:, None] * f_occ_stride_ao
                + offs_o[None, :],
                mask=row_valid[:, None]
                & (offs_b[:, None] < BLOCK_DIM)
                & (offs_o[None, :] < n_occ),
                other=0.0,
            )
            right = tl.load(
                f_virt_ptr
                + graph * f_virt_stride_g
                + col_ids[:, None] * f_virt_stride_ao
                + offs_v[None, :],
                mask=col_valid[:, None]
                & (offs_b[:, None] < BLOCK_DIM)
                & (offs_v[None, :] < n_virt),
                other=0.0,
            )
            if USE_FP64:
                left = left.to(tl.float64)
                right = right.to(tl.float64)
            else:
                left = left.to(tl.float32)
                right = right.to(tl.float32)
            left_block = tl.sum(
                left[:, :, None] * block[:, None, :],
                axis=0,
            )
            contribution = tl.sum(
                left_block[:, :, None] * right[None, :, :],
                axis=1,
            )
            if SYMMETRIZE:
                left_transposed = tl.load(
                    f_occ_ptr
                    + graph * f_occ_stride_g
                    + col_ids[:, None] * f_occ_stride_ao
                    + offs_o[None, :],
                    mask=col_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_o[None, :] < n_occ),
                    other=0.0,
                )
                right_transposed = tl.load(
                    f_virt_ptr
                    + graph * f_virt_stride_g
                    + row_ids[:, None] * f_virt_stride_ao
                    + offs_v[None, :],
                    mask=row_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_v[None, :] < n_virt),
                    other=0.0,
                )
                if USE_FP64:
                    left_transposed = left_transposed.to(tl.float64)
                    right_transposed = right_transposed.to(tl.float64)
                else:
                    left_transposed = left_transposed.to(tl.float32)
                    right_transposed = right_transposed.to(tl.float32)
                left_block_transposed = tl.sum(
                    left_transposed[:, :, None]
                    * tl.trans(block)[:, None, :],
                    axis=0,
                )
                contribution += tl.sum(
                    left_block_transposed[:, :, None]
                    * right_transposed[None, :, :],
                    axis=1,
                )
                contribution *= 0.25
            else:
                contribution *= 0.5
            q_tile += tl.where(active_mask, contribution, 0.0)

        tl.store(
            qov_ptr
            + graph * qov_stride_g
            + offs_o[:, None] * qov_stride_o
            + offs_v[None, :],
            q_tile,
            mask=output_mask,
        )

    @triton.jit
    def _cg_labels_to_qov_fwd_kernel(
        labels_ptr,
        decoder_ptr,
        decoder_index_ptr,
        f_occ_ptr,
        f_virt_ptr,
        edge_graph_ptr,
        row_ao_ptr,
        col_ao_ptr,
        row_valid_ptr,
        col_valid_ptr,
        n_occ_ptr,
        n_virt_ptr,
        qov_ptr,
        labels_stride_e,
        decoder_stride_t,
        decoder_stride_k,
        f_occ_stride_g,
        f_occ_stride_ao,
        f_virt_stride_g,
        f_virt_stride_ao,
        qov_stride_g,
        qov_stride_o,
        BLOCK_DIM: tl.constexpr,
        NUM_COEFF: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_ORBITAL: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_VIRT: tl.constexpr,
        MAX_OCC: tl.constexpr,
        MAX_VIRT: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        edge = tl.program_id(0)
        graph = tl.load(edge_graph_ptr + edge)
        decoder_index = tl.load(decoder_index_ptr + edge)

        offs_k = tl.arange(0, BLOCK_K)
        offs_b = tl.arange(0, BLOCK_ORBITAL)
        coeff = tl.load(
            labels_ptr + edge * labels_stride_e + offs_k,
            mask=offs_k < NUM_COEFF,
            other=0.0,
        )
        if USE_FP64:
            coeff = coeff.to(tl.float64)
            block = tl.zeros((BLOCK_ORBITAL, BLOCK_ORBITAL), dtype=tl.float64)
        else:
            coeff = coeff.to(tl.float32)
            block = tl.zeros((BLOCK_ORBITAL, BLOCK_ORBITAL), dtype=tl.float32)

        # Decode all 196 coupled coefficients once per edge.  Each decoder row
        # is reduced independently, then inserted into the 16 x 16 register tile.
        for row in tl.static_range(0, BLOCK_DIM):
            flat_col = row * BLOCK_DIM + offs_b
            decoder = tl.load(
                decoder_ptr
                + decoder_index * decoder_stride_t
                + offs_k[:, None] * decoder_stride_k
                + flat_col[None, :],
                mask=(offs_k[:, None] < NUM_COEFF)
                & (offs_b[None, :] < BLOCK_DIM),
                other=0.0,
            )
            if USE_FP64:
                decoder = decoder.to(tl.float64)
            else:
                decoder = decoder.to(tl.float32)
            decoded_row = tl.sum(coeff[:, None] * decoder, axis=0)
            row_mask = offs_b[:, None] == row
            block += tl.where(row_mask, decoded_row[None, :], 0.0)

        row_ids = tl.load(
            row_ao_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        )
        col_ids = tl.load(
            col_ao_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        )
        row_valid = tl.load(
            row_valid_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        ).to(tl.int1)
        col_valid = tl.load(
            col_valid_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        ).to(tl.int1)
        n_occ = tl.load(n_occ_ptr + graph)
        n_virt = tl.load(n_virt_ptr + graph)

        # One program owns one atom pair and walks every QOV output tile.  The
        # only global reduction is the final atomic add over atom pairs.
        for occ_start in tl.static_range(0, MAX_OCC, BLOCK_OCC):
            offs_o = occ_start + tl.arange(0, BLOCK_OCC)
            left = tl.load(
                f_occ_ptr
                + graph * f_occ_stride_g
                + row_ids[:, None] * f_occ_stride_ao
                + offs_o[None, :],
                mask=row_valid[:, None]
                & (offs_b[:, None] < BLOCK_DIM)
                & (offs_o[None, :] < n_occ),
                other=0.0,
            )
            if USE_FP64:
                left = left.to(tl.float64)
            else:
                left = left.to(tl.float32)

            # (occ, col-AO) = F_occ(row-AO)^T @ decoded_block.
            left_block = tl.sum(
                left[:, :, None] * block[:, None, :],
                axis=0,
            )
            if SYMMETRIZE:
                left_transposed = tl.load(
                    f_occ_ptr
                    + graph * f_occ_stride_g
                    + col_ids[:, None] * f_occ_stride_ao
                    + offs_o[None, :],
                    mask=col_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_o[None, :] < n_occ),
                    other=0.0,
                )
                if USE_FP64:
                    left_transposed = left_transposed.to(tl.float64)
                else:
                    left_transposed = left_transposed.to(tl.float32)
                left_block_transposed = tl.sum(
                    left_transposed[:, :, None]
                    * tl.trans(block)[:, None, :],
                    axis=0,
                )

            for virt_start in tl.static_range(0, MAX_VIRT, BLOCK_VIRT):
                offs_v = virt_start + tl.arange(0, BLOCK_VIRT)
                right = tl.load(
                    f_virt_ptr
                    + graph * f_virt_stride_g
                    + col_ids[:, None] * f_virt_stride_ao
                    + offs_v[None, :],
                    mask=col_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_v[None, :] < n_virt),
                    other=0.0,
                )
                if USE_FP64:
                    right = right.to(tl.float64)
                else:
                    right = right.to(tl.float32)
                q_tile = tl.sum(
                    left_block[:, :, None] * right[None, :, :],
                    axis=1,
                )
                if SYMMETRIZE:
                    right_transposed = tl.load(
                        f_virt_ptr
                        + graph * f_virt_stride_g
                        + row_ids[:, None] * f_virt_stride_ao
                        + offs_v[None, :],
                        mask=row_valid[:, None]
                        & (offs_b[:, None] < BLOCK_DIM)
                        & (offs_v[None, :] < n_virt),
                        other=0.0,
                    )
                    if USE_FP64:
                        right_transposed = right_transposed.to(tl.float64)
                    else:
                        right_transposed = right_transposed.to(tl.float32)
                    q_tile += tl.sum(
                        left_block_transposed[:, :, None]
                        * right_transposed[None, :, :],
                        axis=1,
                    )
                    q_tile *= 0.25
                else:
                    q_tile *= 0.5
                q_ptrs = (
                    qov_ptr
                    + graph * qov_stride_g
                    + offs_o[:, None] * qov_stride_o
                    + offs_v[None, :]
                )
                tl.atomic_add(
                    q_ptrs,
                    q_tile,
                    mask=(offs_o[:, None] < n_occ)
                    & (offs_v[None, :] < n_virt),
                )

    @triton.jit
    def _cg_labels_to_qov_bwd_kernel(
        grad_qov_ptr,
        decoder_ptr,
        decoder_index_ptr,
        f_occ_ptr,
        f_virt_ptr,
        edge_graph_ptr,
        row_ao_ptr,
        col_ao_ptr,
        row_valid_ptr,
        col_valid_ptr,
        n_occ_ptr,
        n_virt_ptr,
        grad_labels_ptr,
        decoder_stride_t,
        decoder_stride_k,
        f_occ_stride_g,
        f_occ_stride_ao,
        f_virt_stride_g,
        f_virt_stride_ao,
        grad_qov_stride_g,
        grad_qov_stride_o,
        grad_labels_stride_e,
        BLOCK_DIM: tl.constexpr,
        NUM_COEFF: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_ORBITAL: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_VIRT: tl.constexpr,
        MAX_OCC: tl.constexpr,
        MAX_VIRT: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        edge = tl.program_id(0)
        graph = tl.load(edge_graph_ptr + edge)
        decoder_index = tl.load(decoder_index_ptr + edge)
        offs_b = tl.arange(0, BLOCK_ORBITAL)
        row_ids = tl.load(
            row_ao_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        )
        col_ids = tl.load(
            col_ao_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        )
        row_valid = tl.load(
            row_valid_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        ).to(tl.int1)
        col_valid = tl.load(
            col_valid_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        ).to(tl.int1)
        n_occ = tl.load(n_occ_ptr + graph)
        n_virt = tl.load(n_virt_ptr + graph)

        if USE_FP64:
            grad_block = tl.zeros(
                (BLOCK_ORBITAL, BLOCK_ORBITAL),
                dtype=tl.float64,
            )
        else:
            grad_block = tl.zeros(
                (BLOCK_ORBITAL, BLOCK_ORBITAL),
                dtype=tl.float32,
            )

        # Reconstruct dL/dD_AB = 0.5 F_occ,A (dL/dQOV) F_virt,B^T.
        for occ_start in tl.static_range(0, MAX_OCC, BLOCK_OCC):
            offs_o = occ_start + tl.arange(0, BLOCK_OCC)
            left = tl.load(
                f_occ_ptr
                + graph * f_occ_stride_g
                + row_ids[:, None] * f_occ_stride_ao
                + offs_o[None, :],
                mask=row_valid[:, None]
                & (offs_b[:, None] < BLOCK_DIM)
                & (offs_o[None, :] < n_occ),
                other=0.0,
            )
            if USE_FP64:
                left = left.to(tl.float64)
            else:
                left = left.to(tl.float32)
            for virt_start in tl.static_range(0, MAX_VIRT, BLOCK_VIRT):
                offs_v = virt_start + tl.arange(0, BLOCK_VIRT)
                right = tl.load(
                    f_virt_ptr
                    + graph * f_virt_stride_g
                    + col_ids[:, None] * f_virt_stride_ao
                    + offs_v[None, :],
                    mask=col_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_v[None, :] < n_virt),
                    other=0.0,
                )
                grad_q = tl.load(
                    grad_qov_ptr
                    + graph * grad_qov_stride_g
                    + offs_o[:, None] * grad_qov_stride_o
                    + offs_v[None, :],
                    mask=(offs_o[:, None] < n_occ)
                    & (offs_v[None, :] < n_virt),
                    other=0.0,
                )
                if USE_FP64:
                    right = right.to(tl.float64)
                    grad_q = grad_q.to(tl.float64)
                else:
                    right = right.to(tl.float32)
                    grad_q = grad_q.to(tl.float32)
                left_grad_q = tl.sum(
                    left[:, :, None] * grad_q[None, :, :],
                    axis=1,
                )
                direct_grad = tl.sum(
                    left_grad_q[:, None, :] * right[None, :, :],
                    axis=2,
                )
                if SYMMETRIZE:
                    virt_row = tl.load(
                        f_virt_ptr
                        + graph * f_virt_stride_g
                        + row_ids[:, None] * f_virt_stride_ao
                        + offs_v[None, :],
                        mask=row_valid[:, None]
                        & (offs_b[:, None] < BLOCK_DIM)
                        & (offs_v[None, :] < n_virt),
                        other=0.0,
                    )
                    occ_col = tl.load(
                        f_occ_ptr
                        + graph * f_occ_stride_g
                        + col_ids[:, None] * f_occ_stride_ao
                        + offs_o[None, :],
                        mask=col_valid[:, None]
                        & (offs_b[:, None] < BLOCK_DIM)
                        & (offs_o[None, :] < n_occ),
                        other=0.0,
                    )
                    if USE_FP64:
                        virt_row = virt_row.to(tl.float64)
                        occ_col = occ_col.to(tl.float64)
                    else:
                        virt_row = virt_row.to(tl.float32)
                        occ_col = occ_col.to(tl.float32)
                    virt_row_grad = tl.sum(
                        virt_row[:, None, :] * grad_q[None, :, :],
                        axis=2,
                    )
                    transpose_grad = tl.sum(
                        virt_row_grad[:, None, :] * occ_col[None, :, :],
                        axis=2,
                    )
                    grad_block += 0.25 * (direct_grad + transpose_grad)
                else:
                    grad_block += 0.5 * direct_grad

        # Apply the decoder adjoint.  Do not substitute the MALOQ inverse here.
        offs_k = tl.arange(0, BLOCK_K)
        if USE_FP64:
            grad_coeff = tl.zeros((BLOCK_K,), dtype=tl.float64)
        else:
            grad_coeff = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for row in tl.static_range(0, BLOCK_DIM):
            grad_row = tl.sum(
                tl.where(offs_b[:, None] == row, grad_block, 0.0),
                axis=0,
            )
            flat_col = row * BLOCK_DIM + offs_b
            decoder = tl.load(
                decoder_ptr
                + decoder_index * decoder_stride_t
                + offs_k[:, None] * decoder_stride_k
                + flat_col[None, :],
                mask=(offs_k[:, None] < NUM_COEFF)
                & (offs_b[None, :] < BLOCK_DIM),
                other=0.0,
            )
            if USE_FP64:
                decoder = decoder.to(tl.float64)
            else:
                decoder = decoder.to(tl.float32)
            grad_coeff += tl.sum(decoder * grad_row[None, :], axis=1)
        tl.store(
            grad_labels_ptr + edge * grad_labels_stride_e + offs_k,
            grad_coeff,
            mask=offs_k < NUM_COEFF,
        )

    @triton.jit
    def _sparse_cg_labels_to_qov_deterministic_fwd_kernel(
        labels_ptr,
        decoder_values_ptr,
        decoder_indices_ptr,
        decoder_index_ptr,
        f_occ_ptr,
        f_virt_ptr,
        graph_edge_offsets_ptr,
        graph_edge_order_ptr,
        row_ao_ptr,
        col_ao_ptr,
        row_valid_ptr,
        col_valid_ptr,
        n_occ_ptr,
        n_virt_ptr,
        qov_anchor_ptr,
        qov_ptr,
        labels_stride_e,
        decoder_stride_t,
        decoder_stride_o,
        f_occ_stride_g,
        f_occ_stride_ao,
        f_virt_stride_g,
        f_virt_stride_ao,
        qov_stride_g,
        qov_stride_o,
        output_occ,
        output_virt,
        BLOCK_DIM: tl.constexpr,
        BLOCK_ORBITAL: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_VIRT: tl.constexpr,
        ELL_WIDTH: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        graph = tl.program_id(0)
        occ_start = tl.program_id(1) * BLOCK_OCC
        offs_o = occ_start + tl.arange(0, BLOCK_OCC)
        offs_v = tl.arange(0, BLOCK_VIRT)
        n_occ = tl.load(n_occ_ptr + graph)
        n_virt = tl.load(n_virt_ptr + graph)
        output_mask = (offs_o[:, None] < output_occ) & (
            offs_v[None, :] < output_virt
        )
        active_mask = (offs_o[:, None] < n_occ) & (offs_v[None, :] < n_virt)
        q_tile = tl.load(
            qov_anchor_ptr
            + graph * qov_stride_g
            + offs_o[:, None] * qov_stride_o
            + offs_v[None, :],
            mask=output_mask,
            other=0.0,
        )
        if USE_FP64:
            q_tile = q_tile.to(tl.float64)
        else:
            q_tile = q_tile.to(tl.float32)

        edge_start = tl.load(graph_edge_offsets_ptr + graph)
        edge_end = tl.load(graph_edge_offsets_ptr + graph + 1)
        offs_b = tl.arange(0, BLOCK_ORBITAL)
        for edge_position in tl.range(edge_start, edge_end):
            edge = tl.load(graph_edge_order_ptr + edge_position)
            decoder_index = tl.load(decoder_index_ptr + edge)
            block = _decode_sparse_qh9_block(
                labels_ptr,
                decoder_values_ptr,
                decoder_indices_ptr,
                edge,
                decoder_index,
                labels_stride_e,
                decoder_stride_t,
                decoder_stride_o,
                BLOCK_DIM,
                BLOCK_ORBITAL,
                ELL_WIDTH,
                USE_FP64,
            )
            row_ids = tl.load(
                row_ao_ptr + edge * BLOCK_DIM + offs_b,
                mask=offs_b < BLOCK_DIM,
                other=0,
            )
            col_ids = tl.load(
                col_ao_ptr + edge * BLOCK_DIM + offs_b,
                mask=offs_b < BLOCK_DIM,
                other=0,
            )
            row_valid = tl.load(
                row_valid_ptr + edge * BLOCK_DIM + offs_b,
                mask=offs_b < BLOCK_DIM,
                other=0,
            ).to(tl.int1)
            col_valid = tl.load(
                col_valid_ptr + edge * BLOCK_DIM + offs_b,
                mask=offs_b < BLOCK_DIM,
                other=0,
            ).to(tl.int1)
            left = tl.load(
                f_occ_ptr
                + graph * f_occ_stride_g
                + row_ids[:, None] * f_occ_stride_ao
                + offs_o[None, :],
                mask=row_valid[:, None]
                & (offs_b[:, None] < BLOCK_DIM)
                & (offs_o[None, :] < n_occ),
                other=0.0,
            )
            right = tl.load(
                f_virt_ptr
                + graph * f_virt_stride_g
                + col_ids[:, None] * f_virt_stride_ao
                + offs_v[None, :],
                mask=col_valid[:, None]
                & (offs_b[:, None] < BLOCK_DIM)
                & (offs_v[None, :] < n_virt),
                other=0.0,
            )
            if USE_FP64:
                left = left.to(tl.float64)
                right = right.to(tl.float64)
            else:
                left = left.to(tl.float32)
                right = right.to(tl.float32)
            if USE_FP64:
                left_block = tl.sum(
                    left[:, :, None] * block[:, None, :],
                    axis=0,
                )
                contribution = tl.sum(
                    left_block[:, :, None] * right[None, :, :],
                    axis=1,
                )
            else:
                left_block = tl.dot(
                    tl.trans(left),
                    block,
                    input_precision="ieee",
                )
                contribution = tl.dot(
                    left_block,
                    right,
                    input_precision="ieee",
                )
            if SYMMETRIZE:
                left_transposed = tl.load(
                    f_occ_ptr
                    + graph * f_occ_stride_g
                    + col_ids[:, None] * f_occ_stride_ao
                    + offs_o[None, :],
                    mask=col_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_o[None, :] < n_occ),
                    other=0.0,
                )
                right_transposed = tl.load(
                    f_virt_ptr
                    + graph * f_virt_stride_g
                    + row_ids[:, None] * f_virt_stride_ao
                    + offs_v[None, :],
                    mask=row_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_v[None, :] < n_virt),
                    other=0.0,
                )
                if USE_FP64:
                    left_transposed = left_transposed.to(tl.float64)
                    right_transposed = right_transposed.to(tl.float64)
                else:
                    left_transposed = left_transposed.to(tl.float32)
                    right_transposed = right_transposed.to(tl.float32)
                if USE_FP64:
                    left_block_transposed = tl.sum(
                        left_transposed[:, :, None]
                        * tl.trans(block)[:, None, :],
                        axis=0,
                    )
                    contribution += tl.sum(
                        left_block_transposed[:, :, None]
                        * right_transposed[None, :, :],
                        axis=1,
                    )
                else:
                    left_block_transposed = tl.dot(
                        tl.trans(left_transposed),
                        tl.trans(block),
                        input_precision="ieee",
                    )
                    contribution += tl.dot(
                        left_block_transposed,
                        right_transposed,
                        input_precision="ieee",
                    )
                contribution *= 0.25
            else:
                contribution *= 0.5
            q_tile += tl.where(active_mask, contribution, 0.0)

        tl.store(
            qov_ptr
            + graph * qov_stride_g
            + offs_o[:, None] * qov_stride_o
            + offs_v[None, :],
            q_tile,
            mask=output_mask,
        )

    @triton.jit
    def _sparse_cg_labels_to_qov_fwd_kernel(
        labels_ptr,
        decoder_values_ptr,
        decoder_indices_ptr,
        decoder_index_ptr,
        f_occ_ptr,
        f_virt_ptr,
        edge_graph_ptr,
        row_ao_ptr,
        col_ao_ptr,
        row_valid_ptr,
        col_valid_ptr,
        n_occ_ptr,
        n_virt_ptr,
        qov_ptr,
        labels_stride_e,
        decoder_stride_t,
        decoder_stride_o,
        f_occ_stride_g,
        f_occ_stride_ao,
        f_virt_stride_g,
        f_virt_stride_ao,
        qov_stride_g,
        qov_stride_o,
        PADDED_OCC: tl.constexpr,
        PADDED_VIRT: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        BLOCK_ORBITAL: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_VIRT: tl.constexpr,
        ELL_WIDTH: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        edge = tl.program_id(0)
        graph = tl.load(edge_graph_ptr + edge)
        decoder_index = tl.load(decoder_index_ptr + edge)
        block = _decode_sparse_qh9_block(
            labels_ptr,
            decoder_values_ptr,
            decoder_indices_ptr,
            edge,
            decoder_index,
            labels_stride_e,
            decoder_stride_t,
            decoder_stride_o,
            BLOCK_DIM,
            BLOCK_ORBITAL,
            ELL_WIDTH,
            USE_FP64,
        )
        offs_b = tl.arange(0, BLOCK_ORBITAL)
        row_ids = tl.load(
            row_ao_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        )
        col_ids = tl.load(
            col_ao_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        )
        row_valid = tl.load(
            row_valid_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        ).to(tl.int1)
        col_valid = tl.load(
            col_valid_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        ).to(tl.int1)
        n_occ = tl.load(n_occ_ptr + graph)
        n_virt = tl.load(n_virt_ptr + graph)

        for occ_start in tl.static_range(0, PADDED_OCC, BLOCK_OCC):
            offs_o = occ_start + tl.arange(0, BLOCK_OCC)
            left = tl.load(
                f_occ_ptr
                + graph * f_occ_stride_g
                + row_ids[:, None] * f_occ_stride_ao
                + offs_o[None, :],
                mask=row_valid[:, None]
                & (offs_b[:, None] < BLOCK_DIM)
                & (offs_o[None, :] < n_occ),
                other=0.0,
            )
            if USE_FP64:
                left = left.to(tl.float64)
            else:
                left = left.to(tl.float32)
            if USE_FP64:
                left_block = tl.sum(
                    left[:, :, None] * block[:, None, :],
                    axis=0,
                )
            else:
                left_block = tl.dot(
                    tl.trans(left),
                    block,
                    input_precision="ieee",
                )
            if SYMMETRIZE:
                left_transposed = tl.load(
                    f_occ_ptr
                    + graph * f_occ_stride_g
                    + col_ids[:, None] * f_occ_stride_ao
                    + offs_o[None, :],
                    mask=col_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_o[None, :] < n_occ),
                    other=0.0,
                )
                if USE_FP64:
                    left_transposed = left_transposed.to(tl.float64)
                else:
                    left_transposed = left_transposed.to(tl.float32)
                if USE_FP64:
                    left_block_transposed = tl.sum(
                        left_transposed[:, :, None]
                        * tl.trans(block)[:, None, :],
                        axis=0,
                    )
                else:
                    left_block_transposed = tl.dot(
                        tl.trans(left_transposed),
                        tl.trans(block),
                        input_precision="ieee",
                    )
            for virt_start in tl.static_range(0, PADDED_VIRT, BLOCK_VIRT):
                offs_v = virt_start + tl.arange(0, BLOCK_VIRT)
                right = tl.load(
                    f_virt_ptr
                    + graph * f_virt_stride_g
                    + col_ids[:, None] * f_virt_stride_ao
                    + offs_v[None, :],
                    mask=col_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_v[None, :] < n_virt),
                    other=0.0,
                )
                if USE_FP64:
                    right = right.to(tl.float64)
                else:
                    right = right.to(tl.float32)
                if USE_FP64:
                    q_tile = tl.sum(
                        left_block[:, :, None] * right[None, :, :],
                        axis=1,
                    )
                else:
                    q_tile = tl.dot(
                        left_block,
                        right,
                        input_precision="ieee",
                    )
                if SYMMETRIZE:
                    right_transposed = tl.load(
                        f_virt_ptr
                        + graph * f_virt_stride_g
                        + row_ids[:, None] * f_virt_stride_ao
                        + offs_v[None, :],
                        mask=row_valid[:, None]
                        & (offs_b[:, None] < BLOCK_DIM)
                        & (offs_v[None, :] < n_virt),
                        other=0.0,
                    )
                    if USE_FP64:
                        right_transposed = right_transposed.to(tl.float64)
                    else:
                        right_transposed = right_transposed.to(tl.float32)
                    if USE_FP64:
                        q_tile += tl.sum(
                            left_block_transposed[:, :, None]
                            * right_transposed[None, :, :],
                            axis=1,
                        )
                    else:
                        q_tile += tl.dot(
                            left_block_transposed,
                            right_transposed,
                            input_precision="ieee",
                        )
                    q_tile *= 0.25
                else:
                    q_tile *= 0.5
                tl.atomic_add(
                    qov_ptr
                    + graph * qov_stride_g
                    + offs_o[:, None] * qov_stride_o
                    + offs_v[None, :],
                    q_tile,
                    mask=(offs_o[:, None] < n_occ)
                    & (offs_v[None, :] < n_virt),
                )

    @triton.jit
    def _sparse_qov_to_block_grad_kernel(
        grad_qov_ptr,
        f_occ_ptr,
        f_virt_ptr,
        edge_graph_ptr,
        row_ao_ptr,
        col_ao_ptr,
        row_valid_ptr,
        col_valid_ptr,
        n_occ_ptr,
        n_virt_ptr,
        grad_blocks_ptr,
        f_occ_stride_g,
        f_occ_stride_ao,
        f_virt_stride_g,
        f_virt_stride_ao,
        grad_qov_stride_g,
        grad_qov_stride_o,
        grad_blocks_stride_e,
        PADDED_OCC: tl.constexpr,
        PADDED_VIRT: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        BLOCK_ORBITAL: tl.constexpr,
        BLOCK_OCC: tl.constexpr,
        BLOCK_VIRT: tl.constexpr,
        SYMMETRIZE: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        edge = tl.program_id(0)
        graph = tl.load(edge_graph_ptr + edge)
        offs_b = tl.arange(0, BLOCK_ORBITAL)
        row_ids = tl.load(
            row_ao_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        )
        col_ids = tl.load(
            col_ao_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        )
        row_valid = tl.load(
            row_valid_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        ).to(tl.int1)
        col_valid = tl.load(
            col_valid_ptr + edge * BLOCK_DIM + offs_b,
            mask=offs_b < BLOCK_DIM,
            other=0,
        ).to(tl.int1)
        n_occ = tl.load(n_occ_ptr + graph)
        n_virt = tl.load(n_virt_ptr + graph)
        if USE_FP64:
            grad_block = tl.zeros(
                (BLOCK_ORBITAL, BLOCK_ORBITAL),
                dtype=tl.float64,
            )
        else:
            grad_block = tl.zeros(
                (BLOCK_ORBITAL, BLOCK_ORBITAL),
                dtype=tl.float32,
            )

        for occ_start in tl.static_range(0, PADDED_OCC, BLOCK_OCC):
            offs_o = occ_start + tl.arange(0, BLOCK_OCC)
            left = tl.load(
                f_occ_ptr
                + graph * f_occ_stride_g
                + row_ids[:, None] * f_occ_stride_ao
                + offs_o[None, :],
                mask=row_valid[:, None]
                & (offs_b[:, None] < BLOCK_DIM)
                & (offs_o[None, :] < n_occ),
                other=0.0,
            )
            if USE_FP64:
                left = left.to(tl.float64)
            else:
                left = left.to(tl.float32)
            for virt_start in tl.static_range(0, PADDED_VIRT, BLOCK_VIRT):
                offs_v = virt_start + tl.arange(0, BLOCK_VIRT)
                right = tl.load(
                    f_virt_ptr
                    + graph * f_virt_stride_g
                    + col_ids[:, None] * f_virt_stride_ao
                    + offs_v[None, :],
                    mask=col_valid[:, None]
                    & (offs_b[:, None] < BLOCK_DIM)
                    & (offs_v[None, :] < n_virt),
                    other=0.0,
                )
                grad_q = tl.load(
                    grad_qov_ptr
                    + graph * grad_qov_stride_g
                    + offs_o[:, None] * grad_qov_stride_o
                    + offs_v[None, :],
                    mask=(offs_o[:, None] < n_occ)
                    & (offs_v[None, :] < n_virt),
                    other=0.0,
                )
                if USE_FP64:
                    right = right.to(tl.float64)
                    grad_q = grad_q.to(tl.float64)
                else:
                    right = right.to(tl.float32)
                    grad_q = grad_q.to(tl.float32)
                if USE_FP64:
                    left_grad_q = tl.sum(
                        left[:, :, None] * grad_q[None, :, :],
                        axis=1,
                    )
                    direct_grad = tl.sum(
                        left_grad_q[:, None, :] * right[None, :, :],
                        axis=2,
                    )
                else:
                    left_grad_q = tl.dot(
                        left,
                        grad_q,
                        input_precision="ieee",
                    )
                    direct_grad = tl.dot(
                        left_grad_q,
                        tl.trans(right),
                        input_precision="ieee",
                    )
                if SYMMETRIZE:
                    virt_row = tl.load(
                        f_virt_ptr
                        + graph * f_virt_stride_g
                        + row_ids[:, None] * f_virt_stride_ao
                        + offs_v[None, :],
                        mask=row_valid[:, None]
                        & (offs_b[:, None] < BLOCK_DIM)
                        & (offs_v[None, :] < n_virt),
                        other=0.0,
                    )
                    occ_col = tl.load(
                        f_occ_ptr
                        + graph * f_occ_stride_g
                        + col_ids[:, None] * f_occ_stride_ao
                        + offs_o[None, :],
                        mask=col_valid[:, None]
                        & (offs_b[:, None] < BLOCK_DIM)
                        & (offs_o[None, :] < n_occ),
                        other=0.0,
                    )
                    if USE_FP64:
                        virt_row = virt_row.to(tl.float64)
                        occ_col = occ_col.to(tl.float64)
                    else:
                        virt_row = virt_row.to(tl.float32)
                        occ_col = occ_col.to(tl.float32)
                    if USE_FP64:
                        virt_row_grad = tl.sum(
                            virt_row[:, None, :] * grad_q[None, :, :],
                            axis=2,
                        )
                        transpose_grad = tl.sum(
                            virt_row_grad[:, None, :] * occ_col[None, :, :],
                            axis=2,
                        )
                    else:
                        virt_row_grad = tl.dot(
                            virt_row,
                            tl.trans(grad_q),
                            input_precision="ieee",
                        )
                        transpose_grad = tl.dot(
                            virt_row_grad,
                            tl.trans(occ_col),
                            input_precision="ieee",
                        )
                    grad_block += 0.25 * (direct_grad + transpose_grad)
                else:
                    grad_block += 0.5 * direct_grad

        for row in tl.static_range(0, BLOCK_DIM):
            grad_row = tl.sum(
                tl.where(offs_b[:, None] == row, grad_block, 0.0),
                axis=0,
            )
            tl.store(
                grad_blocks_ptr
                + edge * grad_blocks_stride_e
                + row * BLOCK_DIM
                + offs_b,
                grad_row,
                mask=offs_b < BLOCK_DIM,
            )

    @triton.jit
    def _sparse_block_grad_to_labels_kernel(
        grad_blocks_ptr,
        transpose_values_ptr,
        transpose_indices_ptr,
        decoder_index_ptr,
        grad_labels_ptr,
        grad_blocks_stride_e,
        transpose_stride_t,
        transpose_stride_k,
        grad_labels_stride_e,
        NUM_COEFF: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ELL_TRANSPOSE_WIDTH: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        edge = tl.program_id(0)
        decoder_index = tl.load(decoder_index_ptr + edge)
        offs_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, ELL_TRANSPOSE_WIDTH)
        ell_ptrs = (
            decoder_index * transpose_stride_t
            + offs_k[:, None] * transpose_stride_k
            + offs_n[None, :]
        )
        output_index = tl.load(
            transpose_indices_ptr + ell_ptrs,
            mask=offs_k[:, None] < NUM_COEFF,
            other=-1,
        )
        value = tl.load(
            transpose_values_ptr + ell_ptrs,
            mask=(offs_k[:, None] < NUM_COEFF) & (output_index >= 0),
            other=0.0,
        )
        grad_output = tl.load(
            grad_blocks_ptr + edge * grad_blocks_stride_e + output_index,
            mask=(offs_k[:, None] < NUM_COEFF) & (output_index >= 0),
            other=0.0,
        )
        if USE_FP64:
            value = value.to(tl.float64)
            grad_output = grad_output.to(tl.float64)
        else:
            value = value.to(tl.float32)
            grad_output = grad_output.to(tl.float32)
        grad_coefficient = tl.sum(value * grad_output, axis=1)
        tl.store(
            grad_labels_ptr + edge * grad_labels_stride_e + offs_k,
            grad_coefficient,
            mask=offs_k < NUM_COEFF,
        )


def _require_triton_inputs(
    labels: torch.Tensor,
    decoder: torch.Tensor,
    f_occ: torch.Tensor,
    f_virt: torch.Tensor,
) -> None:
    if triton is None:
        raise RuntimeError("fused CG-to-QOV requires Triton")
    tensors = (labels, decoder, f_occ, f_virt)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused CG-to-QOV tensors must be CUDA tensors")
    if labels.dtype not in {torch.float32, torch.float64}:
        raise TypeError("fused CG-to-QOV supports float32 and float64")
    if any(tensor.dtype != labels.dtype for tensor in tensors[1:]):
        raise TypeError("labels, decoder, f_occ, and f_virt must share one dtype")
    if labels.shape[-1] != QH9_NUM_COEFFICIENTS:
        raise ValueError(
            f"expected {QH9_NUM_COEFFICIENTS} QH9 coefficients, "
            f"got {labels.shape[-1]}"
        )
    if decoder.ndim != 3 or tuple(decoder.shape[-2:]) != (
        QH9_NUM_COEFFICIENTS,
        QH9_NUM_COEFFICIENTS,
    ):
        raise ValueError("decoder bank must have shape (T, 196, 196)")


def _require_sparse_triton_inputs(
    labels: torch.Tensor,
    decoder: SparseQH9DecoderBank,
    f_occ: torch.Tensor,
    f_virt: torch.Tensor,
) -> None:
    if triton is None:
        raise RuntimeError("fused CG-to-QOV requires Triton")
    tensors = (
        labels,
        decoder.values,
        decoder.indices,
        decoder.transpose_values,
        decoder.transpose_indices,
        f_occ,
        f_virt,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused CG-to-QOV tensors must be CUDA tensors")
    if labels.dtype not in {torch.float32, torch.float64}:
        raise TypeError("fused CG-to-QOV supports float32 and float64")
    if labels.shape[-1] != QH9_NUM_COEFFICIENTS:
        raise ValueError(
            f"expected {QH9_NUM_COEFFICIENTS} QH9 coefficients, "
            f"got {labels.shape[-1]}"
        )
    if any(
        tensor.dtype != labels.dtype
        for tensor in (decoder.values, decoder.transpose_values, f_occ, f_virt)
    ):
        raise TypeError("labels, decoder values, f_occ, and f_virt must share dtype")
    if decoder.indices.dtype != torch.int32 or decoder.transpose_indices.dtype != torch.int32:
        raise TypeError("sparse decoder indices must be int32")
    expected_prefix = (decoder.num_decoders, QH9_NUM_COEFFICIENTS)
    if tuple(decoder.values.shape[:2]) != expected_prefix or tuple(
        decoder.indices.shape
    ) != tuple(decoder.values.shape):
        raise ValueError("sparse decoder forward arrays must have shape (T, 196, W)")
    if tuple(decoder.transpose_values.shape[:2]) != expected_prefix or tuple(
        decoder.transpose_indices.shape
    ) != tuple(decoder.transpose_values.shape):
        raise ValueError(
            "sparse decoder transpose arrays must have shape (T, 196, Wt)"
        )
    if decoder.width & (decoder.width - 1) or decoder.transpose_width & (
        decoder.transpose_width - 1
    ):
        raise ValueError("sparse decoder ELL widths must be powers of two")


@dataclass(frozen=True)
class _SparseLaunchConfig:
    block_occ: int
    block_virt: int
    num_warps: int
    num_stages: int


def _sparse_launch_config(
    dtype: torch.dtype,
    *,
    deterministic: bool = False,
    backward: bool = False,
) -> _SparseLaunchConfig:
    """Hopper-compatible launch table for sparse QH9 projection."""
    if backward:
        return _SparseLaunchConfig(16, 16, 4, 1)
    if deterministic:
        return _SparseLaunchConfig(
            16,
            0,
            4,
            1 if dtype == torch.float64 else 3,
        )
    if dtype == torch.float64:
        return _SparseLaunchConfig(16, 16, 4, 1)
    return _SparseLaunchConfig(16, 16, 8, 1)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _ensure_triton_c_compiler() -> None:
    """Point Triton's launcher build at this conda env's compiler if needed."""
    if os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang"):
        return
    candidate = os.path.join(sys.prefix, "bin", "x86_64-conda-linux-gnu-gcc")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        os.environ["CC"] = candidate


@torch.no_grad()
def build_maloq_qh9_decoder_bank(
    predictor: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize MALOQ CG decode, shell scatter, and fixed frame change.

    This is a one-time setup helper for ``DirectMALOQPredictor``-compatible
    objects.  The returned lookup has shape ``(100, 100)`` and maps
    ``lookup[row_atomic_number, col_atomic_number]`` to a decoder-bank row.
    Cache both returned tensors; do not rebuild them per training step.
    """
    if dtype not in {torch.float32, torch.float64}:
        raise TypeError("QH9 decoder bank supports float32 and float64")
    device = torch.device(device)
    predictor._move_basis_transform(device, dtype)
    eye = torch.eye(
        QH9_NUM_COEFFICIENTS,
        device=device,
        dtype=dtype,
    )
    uncoupled = predictor.basis_transform.get_H(eye)
    if tuple(uncoupled.shape) != (
        QH9_NUM_COEFFICIENTS,
        QH9_NUM_COEFFICIENTS,
    ):
        raise ValueError(
            "MALOQ basis transform is not the QH9 def2-SVP 196-to-196 map"
        )

    elements = [int(atomic_number) for atomic_number in predictor.elements]
    frame_change = None
    if predictor.internal_frame == "helm" and predictor.output_frame == "ml_dft":
        frame_change = predictor._frame_change_matrices(
            torch.tensor(elements, device=device, dtype=torch.long),
            dtype=dtype,
            device=device,
        )
    decoder_parts = []
    lookup = torch.full(
        (100, 100),
        -1,
        device=device,
        dtype=torch.int32,
    )
    for decoder_idx, (row_z, col_z) in enumerate(
        (row_z, col_z) for row_z in elements for col_z in elements
    ):
        row_atoms = torch.full(
            (QH9_NUM_COEFFICIENTS,),
            row_z,
            device=device,
            dtype=torch.long,
        )
        col_atoms = torch.full_like(row_atoms, col_z)
        padded = predictor._uncoupled_to_padded_blocks(
            uncoupled,
            row_atoms,
            col_atoms,
        )
        if frame_change is not None:
            padded = (
                frame_change[row_z]
                @ padded
                @ frame_change[col_z].transpose(-1, -2)
            )
        decoder_parts.append(padded.flatten(1))
        lookup[row_z, col_z] = decoder_idx
    return torch.stack(decoder_parts).contiguous(), lookup


@torch.no_grad()
def build_maloq_qh9_sparse_decoder_bank(
    predictor: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    zero_tolerance: float | None = None,
) -> tuple[SparseQH9DecoderBank, torch.Tensor]:
    """Build and compact the MALOQ QH9 decoder for repeated training use.

    FP32 construction removes entries at or below ``1e-7`` because the fixed
    frame/CG composition emits numerical noise below one FP32 ulp around unit
    scale; this reduces the real MALOQ ELL width from 32 to 8. FP64 defaults to
    exact nonzero preservation. Pass an explicit tolerance to override this.
    """
    dense, lookup = build_maloq_qh9_decoder_bank(
        predictor,
        device=device,
        dtype=dtype,
    )
    if zero_tolerance is None:
        zero_tolerance = 1.0e-7 if dtype == torch.float32 else 0.0
    return compress_qh9_decoder_bank(
        dense,
        zero_tolerance=zero_tolerance,
    ), lookup


class _FusedCGToQOV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        labels: torch.Tensor,
        decoder: torch.Tensor,
        decoder_index: torch.Tensor,
        f_occ: torch.Tensor,
        f_virt: torch.Tensor,
        edge_graph: torch.Tensor,
        row_ao: torch.Tensor,
        col_ao: torch.Tensor,
        row_valid: torch.Tensor,
        col_valid: torch.Tensor,
        n_occ: torch.Tensor,
        n_virt: torch.Tensor,
        qov_anchor: torch.Tensor,
        graph_edge_offsets: torch.Tensor,
        graph_edge_order: torch.Tensor,
        symmetrize: bool,
        deterministic: bool,
    ) -> torch.Tensor:
        _ensure_triton_c_compiler()
        _require_triton_inputs(labels, decoder, f_occ, f_virt)
        if (
            decoder.requires_grad
            or f_occ.requires_grad
            or f_virt.requires_grad
            or qov_anchor.requires_grad
        ):
            raise ValueError(
                "decoder, target-orbital factors, and QOV anchor must be detached"
            )
        labels = labels.contiguous()
        decoder = decoder.contiguous()
        decoder_index = decoder_index.to(
            device=labels.device,
            dtype=torch.int32,
        ).contiguous()
        f_occ = f_occ.contiguous()
        f_virt = f_virt.contiguous()
        edge_graph = edge_graph.to(device=labels.device, dtype=torch.int32).contiguous()
        row_ao = row_ao.to(device=labels.device, dtype=torch.int32).contiguous()
        col_ao = col_ao.to(device=labels.device, dtype=torch.int32).contiguous()
        row_valid = row_valid.to(device=labels.device, dtype=torch.uint8).contiguous()
        col_valid = col_valid.to(device=labels.device, dtype=torch.uint8).contiguous()
        n_occ = n_occ.to(device=labels.device, dtype=torch.int32).contiguous()
        n_virt = n_virt.to(device=labels.device, dtype=torch.int32).contiguous()
        qov_anchor = qov_anchor.to(
            device=labels.device,
            dtype=labels.dtype,
        ).contiguous()
        graph_edge_offsets = graph_edge_offsets.to(
            device=labels.device,
            dtype=torch.int32,
        ).contiguous()
        graph_edge_order = graph_edge_order.to(
            device=labels.device,
            dtype=torch.int32,
        ).contiguous()
        qov = torch.empty_like(qov_anchor) if deterministic else qov_anchor.clone()

        if labels.shape[0] > 0:
            if deterministic:
                deterministic_grid = (
                    f_occ.shape[0],
                    triton.cdiv(f_occ.shape[-1], 8),
                )
                _cg_labels_to_qov_deterministic_fwd_kernel[deterministic_grid](
                    labels,
                    decoder,
                    decoder_index,
                    f_occ,
                    f_virt,
                    graph_edge_offsets,
                    graph_edge_order,
                    row_ao,
                    col_ao,
                    row_valid,
                    col_valid,
                    n_occ,
                    n_virt,
                    qov_anchor,
                    qov,
                    labels.stride(0),
                    decoder.stride(0),
                    decoder.stride(1),
                    f_occ.stride(0),
                    f_occ.stride(1),
                    f_virt.stride(0),
                    f_virt.stride(1),
                    qov.stride(0),
                    qov.stride(1),
                    BLOCK_DIM=QH9_BLOCK_DIM,
                    NUM_COEFF=QH9_NUM_COEFFICIENTS,
                    BLOCK_K=256,
                    BLOCK_ORBITAL=16,
                    BLOCK_OCC=8,
                    BLOCK_VIRT=triton.next_power_of_2(f_virt.shape[-1]),
                    MAX_OCC=f_occ.shape[-1],
                    MAX_VIRT=f_virt.shape[-1],
                    SYMMETRIZE=bool(symmetrize),
                    USE_FP64=labels.dtype == torch.float64,
                    num_warps=8,
                )
            else:
                _cg_labels_to_qov_fwd_kernel[(labels.shape[0],)](
                    labels,
                    decoder,
                    decoder_index,
                    f_occ,
                    f_virt,
                    edge_graph,
                    row_ao,
                    col_ao,
                    row_valid,
                    col_valid,
                    n_occ,
                    n_virt,
                    qov,
                    labels.stride(0),
                    decoder.stride(0),
                    decoder.stride(1),
                    f_occ.stride(0),
                    f_occ.stride(1),
                    f_virt.stride(0),
                    f_virt.stride(1),
                    qov.stride(0),
                    qov.stride(1),
                    BLOCK_DIM=QH9_BLOCK_DIM,
                    NUM_COEFF=QH9_NUM_COEFFICIENTS,
                    BLOCK_K=256,
                    BLOCK_ORBITAL=16,
                    BLOCK_OCC=8,
                    BLOCK_VIRT=8,
                    MAX_OCC=f_occ.shape[-1],
                    MAX_VIRT=f_virt.shape[-1],
                    SYMMETRIZE=bool(symmetrize),
                    USE_FP64=labels.dtype == torch.float64,
                    num_warps=8,
                )
        elif deterministic:
            qov.copy_(qov_anchor)

        ctx.save_for_backward(
            decoder,
            decoder_index,
            f_occ,
            f_virt,
            edge_graph,
            row_ao,
            col_ao,
            row_valid,
            col_valid,
            n_occ,
            n_virt,
        )
        ctx.labels_shape = labels.shape
        ctx.symmetrize = bool(symmetrize)
        return qov

    @staticmethod
    def backward(ctx: Any, grad_qov: torch.Tensor):
        (
            decoder,
            decoder_index,
            f_occ,
            f_virt,
            edge_graph,
            row_ao,
            col_ao,
            row_valid,
            col_valid,
            n_occ,
            n_virt,
        ) = ctx.saved_tensors
        grad_qov = grad_qov.contiguous()
        grad_labels = torch.empty(
            ctx.labels_shape,
            device=grad_qov.device,
            dtype=grad_qov.dtype,
        )
        if grad_labels.shape[0] > 0:
            _cg_labels_to_qov_bwd_kernel[(grad_labels.shape[0],)](
                grad_qov,
                decoder,
                decoder_index,
                f_occ,
                f_virt,
                edge_graph,
                row_ao,
                col_ao,
                row_valid,
                col_valid,
                n_occ,
                n_virt,
                grad_labels,
                decoder.stride(0),
                decoder.stride(1),
                f_occ.stride(0),
                f_occ.stride(1),
                f_virt.stride(0),
                f_virt.stride(1),
                grad_qov.stride(0),
                grad_qov.stride(1),
                grad_labels.stride(0),
                BLOCK_DIM=QH9_BLOCK_DIM,
                NUM_COEFF=QH9_NUM_COEFFICIENTS,
                BLOCK_K=256,
                BLOCK_ORBITAL=16,
                BLOCK_OCC=8,
                BLOCK_VIRT=8,
                MAX_OCC=f_occ.shape[-1],
                MAX_VIRT=f_virt.shape[-1],
                SYMMETRIZE=ctx.symmetrize,
                USE_FP64=grad_qov.dtype == torch.float64,
                num_warps=8,
            )
        return (
            grad_labels,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SparseFusedCGToQOV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        labels: torch.Tensor,
        decoder_values: torch.Tensor,
        decoder_indices: torch.Tensor,
        transpose_values: torch.Tensor,
        transpose_indices: torch.Tensor,
        decoder_index: torch.Tensor,
        f_occ: torch.Tensor,
        f_virt: torch.Tensor,
        edge_graph: torch.Tensor,
        row_ao: torch.Tensor,
        col_ao: torch.Tensor,
        row_valid: torch.Tensor,
        col_valid: torch.Tensor,
        n_occ: torch.Tensor,
        n_virt: torch.Tensor,
        qov_anchor: torch.Tensor,
        graph_edge_offsets: torch.Tensor,
        graph_edge_order: torch.Tensor,
        symmetrize: bool,
        deterministic: bool,
    ) -> torch.Tensor:
        _ensure_triton_c_compiler()
        decoder = SparseQH9DecoderBank(
            values=decoder_values,
            indices=decoder_indices,
            transpose_values=transpose_values,
            transpose_indices=transpose_indices,
        )
        _require_sparse_triton_inputs(labels, decoder, f_occ, f_virt)
        if (
            decoder_values.requires_grad
            or transpose_values.requires_grad
            or f_occ.requires_grad
            or f_virt.requires_grad
            or qov_anchor.requires_grad
        ):
            raise ValueError(
                "decoder, target-orbital factors, and QOV anchor must be detached"
            )
        labels = labels.contiguous()
        decoder_values = decoder_values.contiguous()
        decoder_indices = decoder_indices.contiguous()
        transpose_values = transpose_values.contiguous()
        transpose_indices = transpose_indices.contiguous()
        decoder_index = decoder_index.to(
            device=labels.device,
            dtype=torch.int32,
        ).contiguous()
        f_occ = f_occ.contiguous()
        f_virt = f_virt.contiguous()
        edge_graph = edge_graph.to(device=labels.device, dtype=torch.int32).contiguous()
        row_ao = row_ao.to(device=labels.device, dtype=torch.int32).contiguous()
        col_ao = col_ao.to(device=labels.device, dtype=torch.int32).contiguous()
        row_valid = row_valid.to(device=labels.device, dtype=torch.uint8).contiguous()
        col_valid = col_valid.to(device=labels.device, dtype=torch.uint8).contiguous()
        n_occ = n_occ.to(device=labels.device, dtype=torch.int32).contiguous()
        n_virt = n_virt.to(device=labels.device, dtype=torch.int32).contiguous()
        qov_anchor = qov_anchor.to(
            device=labels.device,
            dtype=labels.dtype,
        ).contiguous()
        graph_edge_offsets = graph_edge_offsets.to(
            device=labels.device,
            dtype=torch.int32,
        ).contiguous()
        graph_edge_order = graph_edge_order.to(
            device=labels.device,
            dtype=torch.int32,
        ).contiguous()
        qov = torch.empty_like(qov_anchor) if deterministic else qov_anchor.clone()

        if labels.shape[0] > 0:
            config = _sparse_launch_config(
                labels.dtype,
                deterministic=bool(deterministic),
            )
            if deterministic:
                block_virt = max(16, triton.next_power_of_2(f_virt.shape[-1]))
                grid = (
                    f_occ.shape[0],
                    triton.cdiv(f_occ.shape[-1], config.block_occ),
                )
                _sparse_cg_labels_to_qov_deterministic_fwd_kernel[grid](
                    labels,
                    decoder_values,
                    decoder_indices,
                    decoder_index,
                    f_occ,
                    f_virt,
                    graph_edge_offsets,
                    graph_edge_order,
                    row_ao,
                    col_ao,
                    row_valid,
                    col_valid,
                    n_occ,
                    n_virt,
                    qov_anchor,
                    qov,
                    labels.stride(0),
                    decoder_values.stride(0),
                    decoder_values.stride(1),
                    f_occ.stride(0),
                    f_occ.stride(1),
                    f_virt.stride(0),
                    f_virt.stride(1),
                    qov.stride(0),
                    qov.stride(1),
                    f_occ.shape[-1],
                    f_virt.shape[-1],
                    BLOCK_DIM=QH9_BLOCK_DIM,
                    BLOCK_ORBITAL=16,
                    BLOCK_OCC=config.block_occ,
                    BLOCK_VIRT=block_virt,
                    ELL_WIDTH=decoder_values.shape[-1],
                    SYMMETRIZE=bool(symmetrize),
                    USE_FP64=labels.dtype == torch.float64,
                    num_warps=config.num_warps,
                    num_stages=config.num_stages,
                    default_dot_input_precision="ieee",
                )
            else:
                padded_occ = _round_up(f_occ.shape[-1], config.block_occ)
                padded_virt = _round_up(f_virt.shape[-1], config.block_virt)
                _sparse_cg_labels_to_qov_fwd_kernel[(labels.shape[0],)](
                    labels,
                    decoder_values,
                    decoder_indices,
                    decoder_index,
                    f_occ,
                    f_virt,
                    edge_graph,
                    row_ao,
                    col_ao,
                    row_valid,
                    col_valid,
                    n_occ,
                    n_virt,
                    qov,
                    labels.stride(0),
                    decoder_values.stride(0),
                    decoder_values.stride(1),
                    f_occ.stride(0),
                    f_occ.stride(1),
                    f_virt.stride(0),
                    f_virt.stride(1),
                    qov.stride(0),
                    qov.stride(1),
                    PADDED_OCC=padded_occ,
                    PADDED_VIRT=padded_virt,
                    BLOCK_DIM=QH9_BLOCK_DIM,
                    BLOCK_ORBITAL=16,
                    BLOCK_OCC=config.block_occ,
                    BLOCK_VIRT=config.block_virt,
                    ELL_WIDTH=decoder_values.shape[-1],
                    SYMMETRIZE=bool(symmetrize),
                    USE_FP64=labels.dtype == torch.float64,
                    num_warps=config.num_warps,
                    num_stages=config.num_stages,
                    default_dot_input_precision="ieee",
                )
        elif deterministic:
            qov.copy_(qov_anchor)

        ctx.save_for_backward(
            transpose_values,
            transpose_indices,
            decoder_index,
            f_occ,
            f_virt,
            edge_graph,
            row_ao,
            col_ao,
            row_valid,
            col_valid,
            n_occ,
            n_virt,
        )
        ctx.labels_shape = labels.shape
        ctx.symmetrize = bool(symmetrize)
        return qov

    @staticmethod
    def backward(ctx: Any, grad_qov: torch.Tensor):
        (
            transpose_values,
            transpose_indices,
            decoder_index,
            f_occ,
            f_virt,
            edge_graph,
            row_ao,
            col_ao,
            row_valid,
            col_valid,
            n_occ,
            n_virt,
        ) = ctx.saved_tensors
        grad_qov = grad_qov.contiguous()
        grad_labels = torch.empty(
            ctx.labels_shape,
            device=grad_qov.device,
            dtype=grad_qov.dtype,
        )
        if grad_labels.shape[0] > 0:
            grad_blocks = torch.empty(
                (grad_labels.shape[0], QH9_NUM_COEFFICIENTS),
                device=grad_qov.device,
                dtype=grad_qov.dtype,
            )
            config = _sparse_launch_config(grad_qov.dtype, backward=True)
            padded_occ = _round_up(f_occ.shape[-1], config.block_occ)
            padded_virt = _round_up(f_virt.shape[-1], config.block_virt)
            _sparse_qov_to_block_grad_kernel[(grad_labels.shape[0],)](
                grad_qov,
                f_occ,
                f_virt,
                edge_graph,
                row_ao,
                col_ao,
                row_valid,
                col_valid,
                n_occ,
                n_virt,
                grad_blocks,
                f_occ.stride(0),
                f_occ.stride(1),
                f_virt.stride(0),
                f_virt.stride(1),
                grad_qov.stride(0),
                grad_qov.stride(1),
                grad_blocks.stride(0),
                PADDED_OCC=padded_occ,
                PADDED_VIRT=padded_virt,
                BLOCK_DIM=QH9_BLOCK_DIM,
                BLOCK_ORBITAL=16,
                BLOCK_OCC=config.block_occ,
                BLOCK_VIRT=config.block_virt,
                SYMMETRIZE=ctx.symmetrize,
                USE_FP64=grad_qov.dtype == torch.float64,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                default_dot_input_precision="ieee",
            )
            adjoint_block_k = 32 if grad_qov.dtype == torch.float64 else 64
            adjoint_grid = (
                grad_labels.shape[0],
                triton.cdiv(QH9_NUM_COEFFICIENTS, adjoint_block_k),
            )
            _sparse_block_grad_to_labels_kernel[adjoint_grid](
                grad_blocks,
                transpose_values,
                transpose_indices,
                decoder_index,
                grad_labels,
                grad_blocks.stride(0),
                transpose_values.stride(0),
                transpose_values.stride(1),
                grad_labels.stride(0),
                NUM_COEFF=QH9_NUM_COEFFICIENTS,
                BLOCK_K=adjoint_block_k,
                ELL_TRANSPOSE_WIDTH=transpose_values.shape[-1],
                USE_FP64=grad_qov.dtype == torch.float64,
                num_warps=4,
                num_stages=1,
            )
        return (grad_labels,) + (None,) * 19


def fused_cg_labels_to_qov(
    labels: torch.Tensor,
    decoder: torch.Tensor | SparseQH9DecoderBank,
    f_occ: torch.Tensor,
    f_virt: torch.Tensor,
    edge_graph: torch.Tensor,
    row_ao: torch.Tensor,
    col_ao: torch.Tensor,
    row_valid: torch.Tensor,
    col_valid: torch.Tensor,
    n_occ: torch.Tensor,
    n_virt: torch.Tensor,
    qov_anchor: torch.Tensor | None = None,
    *,
    decoder_index: torch.Tensor | None = None,
    symmetrize: bool = False,
    deterministic: bool = False,
    graph_edge_offsets: torch.Tensor | None = None,
    graph_edge_order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project final QH9 CG labels directly into padded graph QOV matrices.

    ``decoder`` is either one ``(196, 196)`` map, a dense
    ``(num_element_pairs, 196, 196)`` bank, or a
    :class:`SparseQH9DecoderBank` made by
    :func:`build_maloq_qh9_sparse_decoder_bank`. ``f_occ`` and ``f_virt`` are
    detached, graph-padded slices of ``S @ C``
    with shapes ``(G, max_n_ao, max_occ)`` and
    ``(G, max_n_ao, max_virt)``.  AO maps are graph-local indices with shape
    ``(E, 14)``. Include node self-pairs among the ``E`` rows. With
    ``symmetrize=True``, the kernel applies the MALOQ global transpose average
    algebraically without materializing paired Hermitian blocks. Atomic mode
    is not bitwise reproducible. Deterministic mode performs one forward launch
    with one program per graph/occupied tile; pass a stable graph-grouped
    ``graph_edge_order`` and its CSR-style ``graph_edge_offsets``. Relative
    speed depends on decoder width and orbital shape; the compact QH9 sparse
    bank made deterministic mode faster in the production-shaped H200 audit.
    """
    if qov_anchor is None:
        qov_anchor = labels.new_zeros(
            f_occ.shape[0],
            f_occ.shape[-1],
            f_virt.shape[-1],
        )
    if not isinstance(decoder, SparseQH9DecoderBank) and decoder.ndim == 2:
        decoder = decoder.unsqueeze(0)
    num_decoders = (
        decoder.num_decoders
        if isinstance(decoder, SparseQH9DecoderBank)
        else decoder.shape[0]
    )
    if decoder_index is None:
        if num_decoders != 1:
            raise ValueError("decoder_index is required for a multi-decoder bank")
        decoder_index = torch.zeros(
            labels.shape[0],
            dtype=torch.int32,
            device=labels.device,
        )
    if tuple(decoder_index.shape) != (labels.shape[0],):
        raise ValueError(
            "decoder_index must have one entry per label row, got "
            f"{tuple(decoder_index.shape)} for {labels.shape[0]} rows"
        )
    if deterministic:
        if graph_edge_offsets is None or graph_edge_order is None:
            raise ValueError(
                "deterministic mode requires graph_edge_offsets and "
                "graph_edge_order"
            )
        if tuple(graph_edge_offsets.shape) != (f_occ.shape[0] + 1,):
            raise ValueError(
                "graph_edge_offsets must have shape "
                f"({f_occ.shape[0] + 1},)"
            )
        if tuple(graph_edge_order.shape) != (labels.shape[0],):
            raise ValueError(
                "graph_edge_order must have one entry per label row"
            )
    else:
        graph_edge_offsets = torch.empty(0, device=labels.device, dtype=torch.int32)
        graph_edge_order = torch.empty(0, device=labels.device, dtype=torch.int32)
    expected = (f_occ.shape[0], f_occ.shape[-1], f_virt.shape[-1])
    if tuple(qov_anchor.shape) != expected:
        raise ValueError(
            f"qov_anchor must have shape {expected}, got {tuple(qov_anchor.shape)}"
        )
    common_args = (
        decoder_index,
        f_occ,
        f_virt,
        edge_graph,
        row_ao,
        col_ao,
        row_valid,
        col_valid,
        n_occ,
        n_virt,
        qov_anchor,
        graph_edge_offsets,
        graph_edge_order,
        symmetrize,
        deterministic,
    )
    if isinstance(decoder, SparseQH9DecoderBank):
        return _SparseFusedCGToQOV.apply(
            labels,
            decoder.values,
            decoder.indices,
            decoder.transpose_values,
            decoder.transpose_indices,
            *common_args,
        )
    return _FusedCGToQOV.apply(labels, decoder, *common_args)
