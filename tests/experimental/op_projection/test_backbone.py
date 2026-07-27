from __future__ import annotations

import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from maloq.experimental.op_projection import (
    OP_PROJECTION_ARCHITECTURE,
    OpProjectionBackbone,
    bind_operator_callback,
)
from maloq.helm.esen_osh_v2 import MaloqNTEV2Backbone


def _small_backbone() -> OpProjectionBackbone:
    return OpProjectionBackbone(
        Irreps("1x0e"),
        sphere_channels=4,
        hidden_channels=4,
        lmax=1,
        mmax=1,
        cutoff=8.0,
        edge_channels=4,
        num_layers=1,
        num_distance_basis=4,
        output_sphere_channels=2,
        conditioning_basis="def2-svp-nabla",
    )


def _small_v2_backbone() -> MaloqNTEV2Backbone:
    return MaloqNTEV2Backbone(
        Irreps("1x0e"),
        sphere_channels=4,
        hidden_channels=4,
        lmax=1,
        mmax=1,
        cutoff=8.0,
        edge_channels=4,
        num_layers=1,
        num_edge_layers=2,
        num_distance_basis=4,
        output_sphere_channels=2,
        conditioning_basis="def2-svp-nabla",
    )


def _conditioned_two_atom_batch() -> Batch:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.2, -0.1]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    data = Data(
        pos=positions,
        z=torch.tensor([6, 6], dtype=torch.long),
        atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
        edge_index=edge_index,
        edge_attr=torch.cat(
            (displacement.norm(dim=-1, keepdim=True), displacement),
            dim=-1,
        ),
        num_atoms_in_molecule=torch.tensor([2], dtype=torch.long),
        charge=torch.zeros(1, dtype=torch.long),
        spin_multiplicity=torch.ones(1, dtype=torch.long),
    )
    batch = Batch.from_data_list([data])
    batch.overlap_matrix = [torch.eye(28)]
    return batch


def test_backbone_returns_node_latents_and_geometry_without_edge_latents() -> None:
    model = _small_backbone().eval()

    output = model(_conditioned_two_atom_batch())

    assert model.architecture == OP_PROJECTION_ARCHITECTURE
    assert output["node_embeddings"].shape == (2, 4, 2)
    assert output["edge_index"].shape == (2, 2)
    assert output["edge_distance"].shape == (2,)
    assert output["edge_distance_vec"].shape == (2, 3)
    assert "edge_embeddings" not in output
    assert not hasattr(model, "edge_blocks")
    assert not hasattr(model, "edge_norm")
    assert not hasattr(model, "edge_output_projection")
    assert torch.isfinite(output["node_embeddings"]).all()


def test_shared_v2_node_checkpoint_reproduces_node_output() -> None:
    torch.manual_seed(7)
    baseline = _small_v2_backbone().eval()
    candidate = _small_backbone().eval()
    candidate_keys = candidate.state_dict()
    shared = {
        name: value
        for name, value in baseline.state_dict().items()
        if name in candidate_keys and value.shape == candidate_keys[name].shape
    }
    result = candidate.load_state_dict(shared, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys

    batch = _conditioned_two_atom_batch()
    with torch.no_grad():
        expected = baseline(batch)["node_embeddings"]
        actual = candidate(batch)["node_embeddings"]
    torch.testing.assert_close(actual, expected)


def test_bound_callback_is_matrix_free_and_differentiable() -> None:
    model = _small_backbone().train()
    batch = _conditioned_two_atom_batch()
    features = model(batch)

    def diagonal_projection(backbone_output, _batch, probe):
        diagonal = backbone_output["node_embeddings"][:, 0, 0].unsqueeze(-1)
        # A real projector also consumes edge_index and edge_distance here and
        # streams pair contributions without building edge feature tensors.
        assert backbone_output["edge_index"].shape[1] == 2
        return diagonal * probe

    operator = bind_operator_callback(features, batch, diagonal_projection)
    probe = torch.randn(2, 3)
    result = operator(probe)
    result.square().mean().backward()

    assert result.shape == probe.shape
    assert model.node_output_projection.weight.grad is not None
    assert torch.isfinite(model.node_output_projection.weight.grad).all()
