from __future__ import annotations

import pytest
import torch

from maloq.helm.triton_kernels.fused_cg_qov_triton import (
    compress_qh9_decoder_bank,
    fused_cg_labels_to_qov,
)


def _dense_reference(
    labels: torch.Tensor,
    decoder: torch.Tensor,
    f_occ: torch.Tensor,
    f_virt: torch.Tensor,
    edge_graph: torch.Tensor,
    row_ao: torch.Tensor,
    col_ao: torch.Tensor,
    row_valid: torch.Tensor,
    col_valid: torch.Tensor,
    n_occ: torch.Tensor,
    n_virt: torch.Tensor,
    anchor: torch.Tensor,
    *,
    decoder_index: torch.Tensor | None = None,
    symmetrize: bool,
) -> torch.Tensor:
    if decoder.ndim == 3:
        assert decoder_index is not None
        selected_decoder = decoder.index_select(0, decoder_index.long())
        blocks = torch.einsum("ek,ekj->ej", labels, selected_decoder)
    else:
        blocks = labels @ decoder
    blocks = blocks.reshape(-1, 14, 14)
    out = anchor.clone()
    for edge in range(labels.shape[0]):
        graph = int(edge_graph[edge])
        occ = int(n_occ[graph])
        virt = int(n_virt[graph])
        left = f_occ[graph].index_select(0, row_ao[edge])[:, :occ]
        right = f_virt[graph].index_select(0, col_ao[edge])[:, :virt]
        mask = row_valid[edge, :, None] & col_valid[edge, None, :]
        block = blocks[edge] * mask.to(blocks.dtype)
        if symmetrize:
            left_t = f_occ[graph].index_select(0, col_ao[edge])[:, :occ]
            right_t = f_virt[graph].index_select(0, row_ao[edge])[:, :virt]
            out[graph, :occ, :virt] += 0.25 * (
                left.T @ block @ right + left_t.T @ block.T @ right_t
            )
        else:
            out[graph, :occ, :virt] += 0.5 * left.T @ block @ right
    return out


def _case(dtype: torch.dtype):
    torch.manual_seed(260721)
    device = torch.device("cuda")
    graphs, edges, max_ao = 2, 5, 19
    max_occ, max_virt = 8, 8
    labels = torch.randn(edges, 196, device=device, dtype=dtype, requires_grad=True)
    decoder = torch.randn(196, 196, device=device, dtype=dtype) / 14
    f_occ = torch.randn(graphs, max_ao, max_occ, device=device, dtype=dtype)
    f_virt = torch.randn(graphs, max_ao, max_virt, device=device, dtype=dtype)
    edge_graph = torch.tensor([0, 0, 1, 1, 1], device=device)
    row_ao = torch.randint(max_ao, (edges, 14), device=device)
    col_ao = torch.randint(max_ao, (edges, 14), device=device)
    row_valid = torch.rand(edges, 14, device=device) > 0.2
    col_valid = torch.rand(edges, 14, device=device) > 0.25
    n_occ = torch.tensor([3, 5], device=device)
    n_virt = torch.tensor([4, 7], device=device)
    anchor = torch.randn(graphs, max_occ, max_virt, device=device, dtype=dtype) / 20
    return (
        labels,
        decoder,
        f_occ,
        f_virt,
        edge_graph,
        row_ao,
        col_ao,
        row_valid,
        col_valid,
        n_occ,
        n_virt,
        anchor,
    )


def test_sparse_qh9_decoder_preserves_dense_map_and_exact_adjoint() -> None:
    torch.manual_seed(260722)
    dense = torch.randn(3, 196, 196, dtype=torch.float64)
    dense.masked_fill_(torch.rand_like(dense) > 0.04, 0.0)
    sparse = compress_qh9_decoder_bank(dense)

    output_major = torch.zeros_like(dense.transpose(1, 2))
    output_major.scatter_add_(
        2,
        sparse.indices.clamp_min(0).long(),
        sparse.values,
    )
    transpose = torch.zeros_like(dense)
    transpose.scatter_add_(
        2,
        sparse.transpose_indices.clamp_min(0).long(),
        sparse.transpose_values,
    )
    torch.testing.assert_close(output_major.transpose(1, 2), dense)
    torch.testing.assert_close(transpose, dense)
    assert sparse.width == 32
    assert sparse.transpose_width <= 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
def test_fused_cg_qov_selects_per_element_pair_decoder() -> None:
    case = list(_case(torch.float32))
    labels = case[0]
    reference_labels = labels.detach().clone().requires_grad_(True)
    base_decoder = case[1]
    decoder_bank = torch.stack(
        [base_decoder, 0.5 * base_decoder, -0.25 * base_decoder]
    )
    decoder_index = torch.tensor([0, 1, 2, 1, 0], device=labels.device)
    case[1] = decoder_bank

    fused = fused_cg_labels_to_qov(
        *case,
        decoder_index=decoder_index,
        symmetrize=True,
    )
    reference = _dense_reference(
        reference_labels,
        *case[1:],
        decoder_index=decoder_index,
        symmetrize=True,
    )
    torch.testing.assert_close(fused, reference, atol=2.0e-4, rtol=2.0e-4)

    probe = torch.randn_like(fused)
    fused_grad = torch.autograd.grad((fused * probe).sum(), labels)[0]
    reference_grad = torch.autograd.grad(
        (reference * probe).sum(),
        reference_labels,
    )[0]
    torch.testing.assert_close(
        fused_grad,
        reference_grad,
        atol=2.0e-4,
        rtol=2.0e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("symmetrize", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_fused_cg_qov_matches_dense_output_and_label_gradient(
    dtype: torch.dtype,
    symmetrize: bool,
    deterministic: bool,
) -> None:
    case = _case(dtype)
    labels = case[0]
    reference_labels = labels.detach().clone().requires_grad_(True)
    fused_kwargs = {"symmetrize": symmetrize, "deterministic": deterministic}
    if deterministic:
        edge_graph = case[4]
        graph_edge_order = torch.argsort(edge_graph, stable=True)
        counts = torch.bincount(edge_graph.long(), minlength=case[2].shape[0])
        graph_edge_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        fused_kwargs.update(
            graph_edge_offsets=graph_edge_offsets,
            graph_edge_order=graph_edge_order,
        )
    fused = fused_cg_labels_to_qov(*case, **fused_kwargs)
    reference = _dense_reference(
        reference_labels,
        *case[1:],
        symmetrize=symmetrize,
    )

    atol = 2.0e-4 if dtype == torch.float32 else 1.0e-10
    rtol = 2.0e-4 if dtype == torch.float32 else 1.0e-10
    torch.testing.assert_close(fused, reference, atol=atol, rtol=rtol)

    probe = torch.randn_like(fused)
    fused_grad = torch.autograd.grad((fused * probe).sum(), labels)[0]
    reference_grad = torch.autograd.grad(
        (reference * probe).sum(),
        reference_labels,
    )[0]
    torch.testing.assert_close(fused_grad, reference_grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("symmetrize", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_sparse_fused_cg_qov_matches_dense_output_and_label_gradient(
    dtype: torch.dtype,
    symmetrize: bool,
    deterministic: bool,
) -> None:
    case = list(_case(dtype))
    labels = case[0]
    row = torch.arange(196, device=labels.device)[:, None]
    col = torch.arange(196, device=labels.device)[None, :]
    structural_mask = ((row + 3 * col) % 31) < 2
    dense_decoder = case[1] * structural_mask
    sparse_decoder = compress_qh9_decoder_bank(dense_decoder)
    case[1] = sparse_decoder
    reference_labels = labels.detach().clone().requires_grad_(True)
    reference_case = [reference_labels, dense_decoder, *case[2:]]

    fused_kwargs = {"symmetrize": symmetrize, "deterministic": deterministic}
    if deterministic:
        edge_graph = case[4]
        graph_edge_order = torch.argsort(edge_graph, stable=True)
        counts = torch.bincount(edge_graph.long(), minlength=case[2].shape[0])
        graph_edge_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        fused_kwargs.update(
            graph_edge_offsets=graph_edge_offsets,
            graph_edge_order=graph_edge_order,
        )
    fused = fused_cg_labels_to_qov(*case, **fused_kwargs)
    reference = _dense_reference(
        *reference_case,
        symmetrize=symmetrize,
    )
    atol = 2.0e-4 if dtype == torch.float32 else 1.0e-10
    rtol = 2.0e-4 if dtype == torch.float32 else 1.0e-10
    torch.testing.assert_close(fused, reference, atol=atol, rtol=rtol)

    probe = torch.randn_like(fused)
    fused_grad = torch.autograd.grad((fused * probe).sum(), labels)[0]
    reference_grad = torch.autograd.grad(
        (reference * probe).sum(),
        reference_labels,
    )[0]
    torch.testing.assert_close(fused_grad, reference_grad, atol=atol, rtol=rtol)
