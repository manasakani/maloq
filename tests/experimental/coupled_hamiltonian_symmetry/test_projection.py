from __future__ import annotations

from types import SimpleNamespace

import torch
from e3nn import o3
from e3nn.o3 import Irreps

from maloq.experimental.coupled_hamiltonian_symmetry import (
    CoupledHamiltonianProjector,
    CoupledTranspose,
    SymmetryReducedMuonFockHead,
)
from maloq.experimental.flow_matching.conditioning import CoupledAOCodec
from maloq.fock_utils.utils_tensor_decomp import e3TensorDecomp


def _basis(dtype: torch.dtype = torch.float64) -> e3TensorDecomp:
    shells = (0, 1)
    return e3TensorDecomp(
        net_irreps_out=None,
        out_js_list=[(left, right) for left in shells for right in shells],
        default_dtype_torch=dtype,
    )


def _rotation(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return o3.angles_to_matrix(
        torch.tensor(0.31, dtype=dtype),
        torch.tensor(1.07, dtype=dtype),
        torch.tensor(-0.44, dtype=dtype),
    )


def _head(dtype: torch.dtype = torch.float32) -> SymmetryReducedMuonFockHead:
    basis = _basis(dtype)
    return SymmetryReducedMuonFockHead(
        irreps_in=Irreps("4x0e+4x1e+4x2e"),
        irreps_out=basis.required_irreps_out,
        lmax=2,
        sphere_channels=4,
        ls_list=(0, 1),
        open_shell=False,
        orbital_basis={1: [0, 1]},
    ).to(dtype=dtype)


def _inputs(dtype: torch.dtype = torch.float32):
    generator = torch.Generator().manual_seed(260728)
    embeddings = {
        "node_embeddings": torch.randn(
            2, 9, 4, dtype=dtype, generator=generator, requires_grad=True
        ),
        "edge_embeddings": torch.randn(
            2, 9, 4, dtype=dtype, generator=generator, requires_grad=True
        ),
    }
    batch = SimpleNamespace(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    return embeddings, batch


def test_coupled_transpose_matches_dense_ao_transpose() -> None:
    basis = _basis()
    codec = CoupledAOCodec(basis)
    transpose = CoupledTranspose(
        (0, 1),
        coupled_dim=basis.required_irreps_out.dim,
    )
    generator = torch.Generator().manual_seed(44)
    dense = torch.randn(3, 4, 4, dtype=torch.float64, generator=generator)
    coupled = codec.encode(dense)
    observed = codec.decode(transpose(coupled))
    torch.testing.assert_close(
        observed,
        dense.transpose(-1, -2),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        transpose(transpose(coupled)),
        coupled,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_node_and_reverse_edge_projection_is_exact_and_idempotent() -> None:
    basis = _basis(torch.float32)
    codec = CoupledAOCodec(basis)
    projector = CoupledHamiltonianProjector(
        (0, 1),
        coupled_dim=basis.required_irreps_out.dim,
    )
    generator = torch.Generator().manual_seed(1729)
    node_dense = torch.randn(1, 2, 4, 4, dtype=torch.float64, generator=generator)
    edge_dense = torch.randn(1, 2, 4, 4, dtype=torch.float64, generator=generator)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    node, edge = projector(
        codec.encode(node_dense.flatten(0, 1)).reshape(1, 2, -1),
        codec.encode(edge_dense.flatten(0, 1)).reshape(1, 2, -1),
        edge_index=edge_index,
    )
    decoded_node = codec.decode(node.flatten(0, 1)).reshape(1, 2, 4, 4)
    decoded_edge = codec.decode(edge.flatten(0, 1)).reshape(1, 2, 4, 4)
    torch.testing.assert_close(
        decoded_node,
        decoded_node.transpose(-1, -2),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        decoded_edge[:, 0],
        decoded_edge[:, 1].transpose(-1, -2),
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    node_twice, edge_twice = projector(node, edge, edge_index=edge_index)
    torch.testing.assert_close(node_twice, node, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(edge_twice, edge, atol=1.0e-12, rtol=1.0e-12)


def test_projection_commutes_with_rotation_and_backpropagates() -> None:
    basis = _basis(torch.float32)
    projector = CoupledHamiltonianProjector(
        (0, 1),
        coupled_dim=basis.required_irreps_out.dim,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    generator = torch.Generator().manual_seed(260728)
    node = torch.randn(
        1,
        2,
        basis.required_irreps_out.dim,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    edge = torch.randn(
        1,
        2,
        basis.required_irreps_out.dim,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    rotation = basis.required_irreps_out.D_from_matrix(_rotation())

    reference_node, reference_edge = projector(node, edge, edge_index=edge_index)
    rotated_node, rotated_edge = projector(
        node @ rotation.T,
        edge @ rotation.T,
        edge_index=edge_index,
    )
    torch.testing.assert_close(
        rotated_node,
        reference_node @ rotation.T,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        rotated_edge,
        reference_edge @ rotation.T,
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    (reference_node.square().mean() + reference_edge.square().mean()).backward()
    assert node.grad is not None and torch.isfinite(node.grad).all()
    assert edge.grad is not None and torch.isfinite(edge.grad).all()


def test_reduced_head_omits_diagonal_odd_l_and_reconstructs_symmetry() -> None:
    basis = _basis(torch.float32)
    codec = CoupledAOCodec(basis)
    head = _head()
    embeddings, batch = _inputs()

    assert head.reduce_node is True
    assert head.reduce_node_intra is True
    assert head.reduce_edge is True
    assert head.irreps_nodereduced == Irreps("0e+1e+0e+2e")
    assert head.irreps_nodereduced.dim == 10
    assert basis.required_irreps_out.dim == 16
    assert not hasattr(head, "node_lin_out_layers")
    assert not hasattr(head, "edge_lin_out_layers")

    node, edge = head(embeddings, batch)
    decoded_node = codec.decode(node.flatten(0, 1)).reshape(1, 2, 4, 4)
    decoded_edge = codec.decode(edge.flatten(0, 1)).reshape(1, 2, 4, 4)
    torch.testing.assert_close(
        decoded_node,
        decoded_node.transpose(-1, -2),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        decoded_edge[:, 0],
        decoded_edge[:, 1].transpose(-1, -2),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_reduced_head_is_equivariant_and_backpropagates() -> None:
    basis = _basis(torch.float32)
    head = _head()
    embeddings, batch = _inputs()
    node, edge = head(embeddings, batch)

    rotation = _rotation(torch.float32)
    input_rotation = Irreps("0e+1e+2e").D_from_matrix(rotation)
    output_rotation = basis.required_irreps_out.D_from_matrix(rotation)
    rotated_embeddings = {
        key: torch.einsum("ab,nbc->nac", input_rotation, value.detach())
        for key, value in embeddings.items()
    }
    rotated_node, rotated_edge = head(rotated_embeddings, batch)
    torch.testing.assert_close(
        rotated_node,
        node.detach() @ output_rotation.T,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        rotated_edge,
        edge.detach() @ output_rotation.T,
        atol=2.0e-5,
        rtol=2.0e-5,
    )

    (node.square().mean() + edge.square().mean()).backward()
    for value in embeddings.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
    semantic_parameters = list(head.semantic_matrix_parameters())
    assert len(semantic_parameters) == 3
    assert all(parameter.ndim == 2 for parameter in semantic_parameters)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in semantic_parameters
    )
    assert list(head.gate_matrix_parameters()) == []


def test_reduced_head_checkpoint_round_trip(tmp_path) -> None:
    head = _head()
    embeddings, batch = _inputs()
    reference = head(
        {key: value.detach() for key, value in embeddings.items()},
        batch,
    )

    checkpoint = tmp_path / "symmetry_reduced_head.pt"
    torch.save(head.state_dict(), checkpoint)
    restored = _head()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    observed = restored(
        {key: value.detach() for key, value in embeddings.items()},
        batch,
    )
    for actual, expected in zip(observed, reference, strict=True):
        torch.testing.assert_close(actual, expected)
