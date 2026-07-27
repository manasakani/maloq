from __future__ import annotations

import pytest
import torch
from e3nn import o3
from torch_geometric.data import Batch, Data

from maloq.experimental.flow_matching.backbone import (
    FlowConditionedBackbone,
    FlowConditionedQHFlow3Backbone,
)
from maloq.experimental.flow_matching.conditioning import (
    CoupledAOCodec,
    HamiltonianSymmetryProjector,
)
from maloq.fock_utils.utils_tensor_decomp import e3TensorDecomp
from maloq.helm.qhflow3 import QHFlow3Backbone


def _sp_basis() -> e3TensorDecomp:
    shells = (0, 1)
    return e3TensorDecomp(
        net_irreps_out=None,
        out_js_list=[(left, right) for left in shells for right in shells],
        default_dtype_torch=torch.float64,
    )


def _rotation() -> torch.Tensor:
    return o3.angles_to_matrix(
        torch.tensor(0.37, dtype=torch.float64),
        torch.tensor(1.11, dtype=torch.float64),
        torch.tensor(-0.62, dtype=torch.float64),
    )


def _orbital_rotation(rotation: torch.Tensor) -> torch.Tensor:
    return torch.block_diag(
        o3.Irrep(0, 1).D_from_matrix(rotation),
        o3.Irrep(1, 1).D_from_matrix(rotation),
    )


def _rotate_dense(dense: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    orbital_rotation = _orbital_rotation(rotation)
    return torch.einsum(
        "ij,njk,lk->nil",
        orbital_rotation,
        dense,
        orbital_rotation,
    )


def test_multishell_codec_roundtrip_and_rotation() -> None:
    basis = _sp_basis()
    codec = CoupledAOCodec(basis)
    dense = torch.tensor(
        [
            [
                [0.7, -0.2, 0.4, 0.1],
                [0.3, 1.1, -0.5, 0.8],
                [-0.6, 0.2, 0.9, -0.7],
                [0.5, -0.3, 0.6, 0.4],
            ]
        ],
        dtype=torch.float64,
    )

    coupled = codec.encode(dense)
    decoded = codec.decode(coupled)
    torch.testing.assert_close(decoded, dense, atol=1.0e-12, rtol=1.0e-12)

    # Shell-pair packed order differs from row-major dense order for [s, p].
    assert not torch.equal(basis.get_H(coupled).reshape_as(dense), dense)

    rotation = _rotation()
    coupled_rotation = basis.required_irreps_out.D_from_matrix(rotation)
    rotated_coupled = coupled @ coupled_rotation.T
    torch.testing.assert_close(
        codec.decode(rotated_coupled),
        _rotate_dense(dense, rotation),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_nabladft_br_max_shell_codec_roundtrip_and_rotation() -> None:
    shells = (0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2)
    basis = e3TensorDecomp(
        net_irreps_out=None,
        out_js_list=[(left, right) for left in shells for right in shells],
        default_dtype_torch=torch.float64,
    )
    codec = CoupledAOCodec(basis)
    assert codec.ao_dim == 32
    assert codec.packed_dim == 1024
    assert basis.required_irreps_out.dim == 1024

    generator = torch.Generator().manual_seed(3517)
    dense = torch.randn(1, 32, 32, dtype=torch.float64, generator=generator)
    coupled = codec.encode(dense)
    torch.testing.assert_close(codec.decode(coupled), dense, atol=1.0e-12, rtol=1.0e-12)
    assert not torch.equal(basis.get_H(coupled).reshape_as(dense), dense)

    rotation = _rotation()
    orbital_rotation = torch.block_diag(
        *(o3.Irrep(shell, 1).D_from_matrix(rotation) for shell in shells)
    )
    expected = torch.einsum(
        "ij,njk,lk->nil",
        orbital_rotation,
        dense,
        orbital_rotation,
    )
    coupled_rotation = basis.required_irreps_out.D_from_matrix(rotation)
    observed = codec.decode(coupled @ coupled_rotation.T)
    torch.testing.assert_close(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_projector_enforces_node_and_reverse_edge_symmetry_and_is_idempotent() -> None:
    basis = _sp_basis()
    codec = CoupledAOCodec(basis)
    projector = HamiltonianSymmetryProjector(basis)
    generator = torch.Generator().manual_seed(260728)
    node_dense = torch.randn(2, 4, 4, dtype=torch.float64, generator=generator)
    edge_dense = torch.randn(2, 4, 4, dtype=torch.float64, generator=generator)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    projected = projector(
        codec.encode(node_dense),
        codec.encode(edge_dense),
        edge_index=edge_index,
    )
    projected_node = codec.decode(projected.node)
    projected_edge = codec.decode(projected.edge)
    torch.testing.assert_close(
        projected_node,
        projected_node.transpose(-1, -2),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        projected_edge[0],
        projected_edge[1].T,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    projected_twice = projector(
        projected.node,
        projected.edge,
        edge_index=edge_index,
    )
    torch.testing.assert_close(
        projected_twice.node, projected.node, atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        projected_twice.edge, projected.edge, atol=1.0e-12, rtol=1.0e-12
    )


def test_projector_respects_shell_pair_masks_and_rejects_noninvolutive_pairs() -> None:
    basis = _sp_basis()
    projector = HamiltonianSymmetryProjector(basis)
    coupled_dim = basis.required_irreps_out.dim
    node = torch.randn(2, coupled_dim, dtype=torch.float64)
    edge = torch.randn(2, coupled_dim, dtype=torch.float64)
    mask = torch.zeros_like(node, dtype=torch.bool)
    mask[:, basis.in_slices[0] : basis.in_slices[1]] = True
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    projected = projector(
        node,
        edge,
        edge_index=edge_index,
        node_mask=mask,
        edge_mask=mask,
    )
    assert torch.equal(projected.node[~mask], torch.zeros_like(projected.node[~mask]))
    assert torch.equal(projected.edge[~mask], torch.zeros_like(projected.edge[~mask]))

    duplicate_pairs = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="involution"):
        projector(
            node,
            torch.randn(4, coupled_dim, dtype=torch.float64),
            edge_index=duplicate_pairs,
        )


def test_projection_commutes_with_general_rotation() -> None:
    basis = _sp_basis()
    codec = CoupledAOCodec(basis)
    projector = HamiltonianSymmetryProjector(basis)
    generator = torch.Generator().manual_seed(17)
    node = codec.encode(torch.randn(2, 4, 4, dtype=torch.float64, generator=generator))
    edge = codec.encode(torch.randn(2, 4, 4, dtype=torch.float64, generator=generator))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    rotation = _rotation()
    coupled_rotation = basis.required_irreps_out.D_from_matrix(rotation)

    reference = projector(node, edge, edge_index=edge_index)
    observed = projector(
        node @ coupled_rotation.T,
        edge @ coupled_rotation.T,
        edge_index=edge_index,
    )
    torch.testing.assert_close(
        observed.node,
        reference.node @ coupled_rotation.T,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        observed.edge,
        reference.edge @ coupled_rotation.T,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def _small_base() -> QHFlow3Backbone:
    return QHFlow3Backbone(
        sh_lmax=2,
        hidden_size=4,
        bottle_hidden_size=2,
        num_gnn_layers=1,
        num_ham_gnn_layers=1,
        radius_embed_dim=4,
        escn_edge_channels=4,
        escn_num_distance_basis=4,
        basis="def2-svp",
        grid_resolution=8,
        grid_ffn_chunk_size=None,
        module_dtype="float64",
    )


def _rotate_embeddings(
    embeddings: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    rotated = torch.empty_like(embeddings)
    lmax = int(embeddings.shape[1] ** 0.5) - 1
    for degree in range(lmax + 1):
        component_slice = slice(degree**2, (degree + 1) ** 2)
        degree_rotation = o3.Irrep(degree, 1).D_from_matrix(rotation)
        rotated[:, component_slice] = torch.einsum(
            "ij,njc->nic",
            degree_rotation,
            embeddings[:, component_slice],
        )
    return rotated


class _NativeEmbeddingBackbone(torch.nn.Module):
    architecture = "test-native-embedding"
    lmax = 2
    output_sphere_channels = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.node_output_projection = torch.nn.Identity()
        self.edge_output_projection = torch.nn.Identity()

    def forward(self, batch):
        node_count = int(batch.batch.numel())
        edge_count = int(batch.edge_index.shape[1])
        return {
            "node_embeddings": self.anchor.new_zeros(node_count, 9, 2),
            "edge_embeddings": self.anchor.new_zeros(edge_count, 9, 2),
            "sentinel": self.anchor.new_tensor(7.0),
        }


class _MutatingNativeEmbeddingBackbone(_NativeEmbeddingBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.observed_time: torch.Tensor | None = None

    def forward(self, batch):
        self.observed_time = batch.t.detach().clone()
        output = super().forward(batch)
        batch.edge_index = batch.edge_index.flip(1)
        return output


class _CastingNativeEmbeddingBackbone(_NativeEmbeddingBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.float()

    def forward(self, batch):
        batch.node_flow_t = batch.node_flow_t.float()
        batch.edge_flow_t = batch.edge_flow_t.float()
        batch.t = batch.t.float()
        return super().forward(batch)


def _generic_flow_batch(
    flow_dim: int,
    *,
    edge_index: torch.Tensor | None = None,
) -> Data:
    if edge_index is None:
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]],
            dtype=torch.long,
        )
    batch = Data(edge_index=edge_index)
    batch.batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    batch.ptr = torch.tensor([0, 2, 4], dtype=torch.long)
    batch.t = torch.tensor([0.2, 0.8], dtype=torch.float64)
    batch.node_flow_t = torch.zeros(4, flow_dim, dtype=torch.float64)
    batch.edge_flow_t = torch.zeros(
        edge_index.shape[1],
        flow_dim,
        dtype=torch.float64,
    )
    return batch


def test_forward_neutralizes_native_time_and_restores_canonical_edge_order() -> None:
    basis = _sp_basis()
    flow_dim = basis.required_irreps_out.dim
    torch.manual_seed(118)
    canonical_batch = _generic_flow_batch(flow_dim)
    canonical_batch.node_flow_t = torch.randn_like(canonical_batch.node_flow_t)
    canonical_batch.edge_flow_t = torch.randn_like(canonical_batch.edge_flow_t)
    mutating_batch = _generic_flow_batch(flow_dim)
    mutating_batch.node_flow_t = canonical_batch.node_flow_t.clone()
    mutating_batch.edge_flow_t = canonical_batch.edge_flow_t.clone()
    original_time = mutating_batch.t.clone()
    original_edge_index = mutating_batch.edge_index.clone()

    reference = FlowConditionedBackbone(
        _NativeEmbeddingBackbone(),
        flow_irreps=basis.required_irreps_out,
    ).eval()
    mutating_base = _MutatingNativeEmbeddingBackbone()
    observed = FlowConditionedBackbone(
        mutating_base,
        flow_irreps=basis.required_irreps_out,
    ).eval()
    observed.load_state_dict(reference.state_dict())

    with torch.no_grad():
        reference_output = reference(canonical_batch)
        observed_output = observed(mutating_batch)

    torch.testing.assert_close(
        mutating_base.observed_time,
        torch.ones_like(original_time),
    )
    torch.testing.assert_close(mutating_batch.t, original_time)
    torch.testing.assert_close(mutating_batch.edge_index, original_edge_index)
    torch.testing.assert_close(
        observed_output["node_embeddings"],
        reference_output["node_embeddings"],
    )
    torch.testing.assert_close(
        observed_output["edge_embeddings"],
        reference_output["edge_embeddings"],
    )


def test_forward_preserves_flow_time_value_across_native_dtype_cast() -> None:
    basis = _sp_basis()
    batch = _generic_flow_batch(basis.required_irreps_out.dim)
    original_time = batch.t
    wrapper = FlowConditionedBackbone(
        _CastingNativeEmbeddingBackbone(),
        flow_irreps=basis.required_irreps_out,
    ).eval()

    with torch.no_grad():
        output = wrapper(batch)

    assert batch.t is original_time
    assert batch.t.dtype == torch.float64
    assert output["node_embeddings"].dtype == torch.float32
    assert output["edge_embeddings"].dtype == torch.float32


def test_joint_flow_conditioning_is_so3_equivariant() -> None:
    basis = _sp_basis()
    torch.manual_seed(44)
    wrapper = FlowConditionedQHFlow3Backbone(
        _small_base(),
        flow_irreps=basis.required_irreps_out,
    ).eval()
    node_embeddings = torch.zeros(2, 9, 2, dtype=torch.float64)
    edge_embeddings = torch.zeros(2, 9, 2, dtype=torch.float64)
    node_state = torch.randn(2, basis.required_irreps_out.dim, dtype=torch.float64)
    edge_state = torch.randn(2, basis.required_irreps_out.dim, dtype=torch.float64)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    node_graph_index = torch.zeros(2, dtype=torch.long)
    graph_time = torch.tensor([0.37], dtype=torch.float64)
    rotation = _rotation()
    flow_rotation = wrapper.flow_so3_irreps.D_from_matrix(rotation)

    reference_node, reference_edge = wrapper.condition_embeddings(
        node_embeddings=node_embeddings,
        edge_embeddings=edge_embeddings,
        node_flow_state=node_state,
        edge_flow_state=edge_state,
        graph_time=graph_time,
        node_graph_index=node_graph_index,
        edge_index=edge_index,
    )
    observed_node, observed_edge = wrapper.condition_embeddings(
        node_embeddings=node_embeddings,
        edge_embeddings=edge_embeddings,
        node_flow_state=node_state @ flow_rotation.T,
        edge_flow_state=edge_state @ flow_rotation.T,
        graph_time=graph_time,
        node_graph_index=node_graph_index,
        edge_index=edge_index,
    )
    assert reference_node.abs().max() > 0
    assert reference_edge.abs().max() > 0
    torch.testing.assert_close(
        observed_node,
        _rotate_embeddings(reference_node, rotation),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        observed_edge,
        _rotate_embeddings(reference_edge, rotation),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_common_conditioner_separates_node_edge_incident_and_time_paths() -> None:
    basis = _sp_basis()
    torch.manual_seed(91)
    state_wrapper = FlowConditionedBackbone(
        _NativeEmbeddingBackbone(),
        flow_irreps=basis.required_irreps_out,
        time_scale=0.0,
    ).eval()
    node_embeddings = torch.zeros(4, 9, 2, dtype=torch.float64)
    edge_embeddings = torch.zeros(4, 9, 2, dtype=torch.float64)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]],
        dtype=torch.long,
    )
    node_graph_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    graph_time = torch.zeros(2, dtype=torch.float64)
    zero_node = torch.zeros(4, basis.required_irreps_out.dim, dtype=torch.float64)
    zero_edge = torch.zeros_like(zero_node)
    node_state = torch.randn_like(zero_node)
    edge_state = torch.randn_like(zero_edge)

    node_only, node_only_edge = state_wrapper.condition_embeddings(
        node_embeddings=node_embeddings,
        edge_embeddings=edge_embeddings,
        node_flow_state=node_state,
        edge_flow_state=zero_edge,
        graph_time=graph_time,
        node_graph_index=node_graph_index,
        edge_index=edge_index,
    )
    edge_incident_node, edge_only = state_wrapper.condition_embeddings(
        node_embeddings=node_embeddings,
        edge_embeddings=edge_embeddings,
        node_flow_state=zero_node,
        edge_flow_state=edge_state,
        graph_time=graph_time,
        node_graph_index=node_graph_index,
        edge_index=edge_index,
    )
    assert node_only.abs().max() > 0
    assert torch.count_nonzero(node_only_edge) == 0
    assert edge_incident_node.abs().max() > 0
    assert edge_only.abs().max() > 0

    time_wrapper = FlowConditionedBackbone(
        _NativeEmbeddingBackbone(),
        flow_irreps=basis.required_irreps_out,
        node_flow_scale=0.0,
        edge_flow_scale=0.0,
        incident_edge_scale=0.0,
    ).eval()
    with torch.no_grad():
        time_wrapper.node_time_projection.weight.fill_(1.0)
        time_wrapper.node_time_projection.bias.zero_()
        time_wrapper.edge_time_projection.weight.fill_(1.0)
        time_wrapper.edge_time_projection.bias.zero_()
    time_node, time_edge = time_wrapper.condition_embeddings(
        node_embeddings=node_embeddings,
        edge_embeddings=edge_embeddings,
        node_flow_state=zero_node,
        edge_flow_state=zero_edge,
        graph_time=torch.tensor([0.25, 0.75], dtype=torch.float64),
        node_graph_index=node_graph_index,
        edge_index=edge_index,
    )
    assert torch.count_nonzero(time_node[:, 1:]) == 0
    assert torch.count_nonzero(time_edge[:, 1:]) == 0
    torch.testing.assert_close(
        time_node[:2, 0], torch.full((2, 2), 0.25, dtype=torch.float64)
    )
    torch.testing.assert_close(
        time_node[2:, 0], torch.full((2, 2), 0.75, dtype=torch.float64)
    )
    torch.testing.assert_close(
        time_edge[:2, 0], torch.full((2, 2), 0.25, dtype=torch.float64)
    )
    torch.testing.assert_close(
        time_edge[2:, 0], torch.full((2, 2), 0.75, dtype=torch.float64)
    )


def test_common_conditioner_validates_state_dtype_time_and_edge_graph() -> None:
    basis = _sp_basis()
    wrapper = FlowConditionedBackbone(
        _NativeEmbeddingBackbone(),
        flow_irreps=basis.required_irreps_out,
    ).eval()
    valid = _generic_flow_batch(basis.required_irreps_out.dim)
    output = wrapper(valid)
    assert output["sentinel"].item() == 7.0

    bad_shape = _generic_flow_batch(basis.required_irreps_out.dim)
    bad_shape.node_flow_t = bad_shape.node_flow_t[:, :-1]
    with pytest.raises(ValueError, match="node_flow_t must have shape"):
        wrapper(bad_shape)

    bad_dtype = _generic_flow_batch(basis.required_irreps_out.dim)
    bad_dtype.edge_flow_t = bad_dtype.edge_flow_t.float()
    with pytest.raises(ValueError, match="share dtype/device"):
        wrapper(bad_dtype)

    bad_time = _generic_flow_batch(basis.required_irreps_out.dim)
    bad_time.t = torch.tensor([0.2], dtype=torch.float64)
    with pytest.raises(ValueError, match="incompatible with batch.t"):
        wrapper(bad_time)

    cross_graph = _generic_flow_batch(
        basis.required_irreps_out.dim,
        edge_index=torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="stay within one graph"):
        wrapper(cross_graph)


def _flow_forward_batch(
    node_flow_state: torch.Tensor,
    edge_flow_state: torch.Tensor,
) -> Batch:
    positions = torch.tensor(
        [[0.13, -0.21, 0.08], [0.74, 0.35, -0.42]],
        dtype=torch.float64,
    )
    atomic_numbers = torch.full((2,), 6, dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    batch = Batch.from_data_list(
        [
            Data(
                pos=positions,
                z=atomic_numbers,
                atomic_numbers=atomic_numbers,
                edge_index=edge_index,
                edge_attr=torch.cat(
                    [displacement.norm(dim=-1, keepdim=True), displacement],
                    dim=-1,
                ),
                num_atoms_in_molecule=torch.tensor([2]),
                charge=torch.zeros(1, dtype=torch.long),
                spin_multiplicity=torch.ones(1, dtype=torch.long),
            )
        ]
    )
    batch.overlap_matrix = [torch.eye(28, dtype=torch.float64).numpy()]
    batch.t = torch.tensor([0.37], dtype=torch.float64)
    batch.node_flow_t = node_flow_state
    batch.edge_flow_t = edge_flow_state
    return batch


def test_real_qhflow3_wrapper_forward_consumes_joint_flow_state() -> None:
    basis = _sp_basis()
    torch.manual_seed(1729)
    wrapper = FlowConditionedQHFlow3Backbone(
        _small_base(),
        flow_irreps=basis.required_irreps_out,
    ).eval()
    assert isinstance(wrapper, FlowConditionedBackbone)
    node_flow_state = torch.randn(
        2,
        basis.required_irreps_out.dim,
        dtype=torch.float64,
    )
    edge_flow_state = torch.randn_like(node_flow_state)
    zero_node = torch.zeros_like(node_flow_state)
    zero_edge = torch.zeros_like(edge_flow_state)

    with torch.no_grad():
        zero_output = wrapper(_flow_forward_batch(zero_node, zero_edge))
        node_output = wrapper(_flow_forward_batch(node_flow_state, zero_edge))
        edge_output = wrapper(_flow_forward_batch(zero_node, edge_flow_state))

    assert (
        node_output["node_embeddings"] - zero_output["node_embeddings"]
    ).abs().max() > 1.0e-8
    assert (
        edge_output["node_embeddings"] - zero_output["node_embeddings"]
    ).abs().max() > 1.0e-8
    assert (
        edge_output["edge_embeddings"] - zero_output["edge_embeddings"]
    ).abs().max() > 1.0e-8
