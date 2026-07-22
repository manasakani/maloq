from __future__ import annotations

from copy import deepcopy

import torch
from e3nn import o3
from e3nn.o3 import Irreps

from maloq.core.config import MaloqConfig
from maloq.fock_utils.basis_sets import orbital_basis_def2_svp_QM7
from maloq.fock_utils.utils_tensor_decomp import make_output_irreps
from maloq.helm.static_te_head import (
    SemanticPathContraction,
    StaticTensorExpansionHead,
)
from maloq.train_utils.training_workflow import TrainingWorkflow


def _explicit_semantic_contraction(
    contraction: SemanticPathContraction,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    output = embeddings.new_empty(embeddings.shape[0], contraction.irreps_out.dim)
    semantic_row = 0
    scalar_row = 0
    component_start = 0
    for multiplicity, irrep in contraction.irreps_out:
        width = irrep.dim
        features = embeddings[:, irrep.l**2 : (irrep.l + 1) ** 2, :]
        for copy_index in range(multiplicity):
            value = torch.einsum(
                "nmc,c->nm",
                features,
                contraction.weight[semantic_row],
            ) / contraction.input_channels
            if irrep.l == 0:
                value = value + contraction.bias[scalar_row] / contraction.input_channels
                scalar_row += 1
            start = component_start + copy_index * width
            output[:, start : start + width] = value
            semantic_row += 1
        component_start += multiplicity * width
    return output


def _qh9_irreps_and_shells() -> tuple[Irreps, list[int]]:
    _, irreps, _, shells, *_ = make_output_irreps(
        deepcopy(orbital_basis_def2_svp_QM7)
    )
    return irreps, shells


def test_static_te_scatter_preserves_semantic_path_and_channel_axes() -> None:
    irreps = Irreps("1x0e+1x1e+1x0e+1x2e+1x1e")
    contraction = SemanticPathContraction(irreps, 3)
    with torch.no_grad():
        contraction.weight.copy_(
            torch.arange(contraction.weight.numel()).reshape_as(contraction.weight)
        )
        contraction.bias.copy_(torch.tensor([0.25, -0.5]))
    embeddings = torch.arange(2 * 9 * 3, dtype=torch.float32).reshape(2, 9, 3)

    observed = contraction(embeddings)
    expected = _explicit_semantic_contraction(contraction, embeddings)

    assert contraction.PATH_LAYOUT == "path_offsets"
    assert tuple(contraction.weight.shape) == (5, 3)
    torch.testing.assert_close(observed, expected)


def test_static_te_contraction_is_equivariant() -> None:
    irreps = Irreps("1x0e+2x1e+1x2e+1x0e")
    contraction = SemanticPathContraction(
        irreps,
        4,
        init_mode="normal",
        init_std=0.2,
    ).double()
    embeddings = torch.randn(3, 9, 4, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    rotated = embeddings.clone()
    for degree in range(3):
        degree_rotation = o3.Irrep(degree, 1).D_from_matrix(rotation)
        block = embeddings[:, degree**2 : (degree + 1) ** 2, :]
        rotated[:, degree**2 : (degree + 1) ** 2, :] = torch.einsum(
            "ij,njc->nic", degree_rotation, block
        )

    expected = contraction(embeddings) @ irreps.D_from_matrix(rotation).T
    torch.testing.assert_close(
        contraction(rotated),
        expected,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_degreewise_l34_gate_is_identity_initialized_and_equivariant() -> None:
    irreps = Irreps("1x0e+1x1e+1x2e+1x3e+1x4e")
    contraction = SemanticPathContraction(
        irreps,
        4,
        init_mode="normal",
        init_std=0.2,
        gate_degrees=(3, 4),
        gate_activation="residual_tanh",
        gate_init=1.0,
    ).double()
    embeddings = torch.randn(3, 25, 4, dtype=torch.float64)
    ungated = SemanticPathContraction(
        irreps,
        4,
        init_mode="normal",
        init_std=0.2,
    ).double()
    with torch.no_grad():
        ungated.weight.copy_(contraction.weight)
        ungated.bias.copy_(contraction.bias)
    torch.testing.assert_close(contraction(embeddings), ungated(embeddings))

    rotation = o3.rand_matrix(dtype=torch.float64)
    rotated = embeddings.clone()
    for degree in range(5):
        degree_rotation = o3.Irrep(degree, 1).D_from_matrix(rotation)
        block = embeddings[:, degree**2 : (degree + 1) ** 2, :]
        rotated[:, degree**2 : (degree + 1) ** 2, :] = torch.einsum(
            "ij,njc->nic", degree_rotation, block
        )
    expected = contraction(embeddings) @ irreps.D_from_matrix(rotation).T
    torch.testing.assert_close(
        contraction(rotated), expected, rtol=1.0e-10, atol=1.0e-10
    )

    contraction(rotated).square().mean().backward()
    assert torch.isfinite(contraction.degree_gate.weight.grad).all()


def test_qh9_static_te_has_corrected_semantic_matrices_and_muon_routing() -> None:
    irreps, shells = _qh9_irreps_and_shells()
    head = StaticTensorExpansionHead(
        irreps_out=irreps,
        lmax=4,
        sphere_channels=4,
        ls_list=shells,
        reduce_node=True,
        reduce_node_intra=True,
        init_mode="normal",
        init_std=0.1,
    )
    embeddings = {
        "node_embeddings": torch.randn(2, 25, 4),
        "edge_embeddings": torch.randn(3, 25, 4),
    }

    node_output, edge_output = head(embeddings, batch=None)
    assert tuple(head.edge_contraction.weight.shape) == (56, 4)
    assert tuple(node_output.shape) == (1, 2, 196)
    assert tuple(edge_output.shape) == (1, 3, 196)
    assert int(head.edge_contraction._path_layout_version) == 1
    (node_output.square().mean() + edge_output.square().mean()).backward()
    assert torch.isfinite(head.node_contraction.weight.grad).all()
    assert torch.isfinite(head.edge_contraction.weight.grad).all()

    backbone = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.LayerNorm(4),
    )
    muon_parameters = TrainingWorkflow._collect_muon_parameters(backbone, head)
    muon_parameter_ids = {id(parameter) for parameter in muon_parameters}
    assert id(backbone[0].weight) in muon_parameter_ids
    assert id(backbone[1].weight) not in muon_parameter_ids
    assert id(head.node_contraction.weight) in muon_parameter_ids
    assert id(head.edge_contraction.weight) in muon_parameter_ids
    assert id(head.node_contraction.bias) not in muon_parameter_ids
    assert id(head.edge_contraction.bias) not in muon_parameter_ids


def test_static_te_config_round_trip() -> None:
    workflow = MaloqConfig(
        model={
            "head_type": "static_te",
            "static_te_init_mode": "normal",
            "static_te_init_std": 0.25,
            "static_te_gate_degrees": [3, 4],
            "static_te_gate_activation": "residual_tanh",
            "static_te_gate_init": 1.0,
        }
    ).to_workflow_config()

    assert workflow["head_type"] == "static_te"
    assert workflow["static_te_init_mode"] == "normal"
    assert workflow["static_te_init_std"] == 0.25
    assert workflow["static_te_gate_degrees"] == (3, 4)
    assert workflow["static_te_gate_activation"] == "residual_tanh"
    assert workflow["static_te_gate_init"] == 1.0
