from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
from e3nn.o3 import Irreps

from maloq.core.config import MaloqConfig
from maloq.fock_utils.basis_sets import orbital_basis_def2_svp_QM7
from maloq.fock_utils.utils_tensor_decomp import make_output_irreps
from maloq.helm.esen_osh import Fock_Irreps_Head
from maloq.helm.muon_fock_head import MuonFockIrrepsHead
from maloq.train_utils.optimizers import Muon
from maloq.train_utils.training_workflow import TrainingWorkflow


def _qh9_head_inputs(channels: int):
    _, irreps_out, _, shells, *_ = make_output_irreps(
        deepcopy(orbital_basis_def2_svp_QM7)
    )
    kwargs = {
        "irreps_in": Irreps(
            [(channels, (degree, 1)) for degree in range(irreps_out.lmax + 1)]
        ),
        "irreps_out": irreps_out,
        "lmax": irreps_out.lmax,
        "sphere_channels": channels,
        "reduce_edge": False,
        "open_shell": False,
        "ls_list": shells,
        "reduce_node": True,
        "reduce_node_intra": True,
        "orbital_basis": orbital_basis_def2_svp_QM7,
    }
    embeddings = {
        "node_embeddings": torch.randn(3, 25, channels),
        "edge_embeddings": torch.randn(4, 25, channels),
    }
    batch = SimpleNamespace(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    )
    return kwargs, embeddings, batch


def test_muon_maloq_head_preserves_native_forward_exactly() -> None:
    kwargs, embeddings, batch = _qh9_head_inputs(channels=4)
    torch.manual_seed(260722)
    native = Fock_Irreps_Head(**kwargs)
    torch.manual_seed(260722)
    semantic = MuonFockIrrepsHead(**kwargs)

    native_output = native(embeddings, batch)
    semantic_output = semantic(embeddings, batch)
    for native_part, semantic_part in zip(native_output, semantic_output, strict=True):
        torch.testing.assert_close(semantic_part, native_part)


def test_muon_maloq_head_routes_every_trainable_matrix() -> None:
    kwargs, embeddings, batch = _qh9_head_inputs(channels=4)
    head = MuonFockIrrepsHead(**kwargs)
    semantic_parameters = list(head.semantic_matrix_parameters())
    assert [tuple(parameter.shape) for parameter in semantic_parameters] == [
        (31, 4),
        (56, 4),
    ]

    backbone = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.LayerNorm(4),
    )
    muon_parameters = TrainingWorkflow._collect_muon_parameters(backbone, head)
    muon_ids = {id(parameter) for parameter in muon_parameters}
    assert id(backbone[0].weight) in muon_ids
    assert id(backbone[1].weight) not in muon_ids
    assert all(id(parameter) in muon_ids for parameter in semantic_parameters)
    assert all(
        (id(parameter) in muon_ids) == (parameter.ndim >= 2)
        for parameter in head.parameters()
    )

    outputs = head(embeddings, batch)
    loss = sum(output.square().mean() for output in outputs)
    loss.backward()
    before = [parameter.detach().clone() for parameter in semantic_parameters]
    auxiliary = [
        parameter
        for parameter in head.parameters()
        if id(parameter) not in {id(value) for value in semantic_parameters}
    ]
    optimizer = Muon(
        [
            {"params": semantic_parameters, "use_muon": True, "lr": 0.02},
            {"params": auxiliary, "use_muon": False, "lr": 5.0e-4},
        ],
        lr=0.02,
        ns_steps=2,
    )
    optimizer.step()
    assert all(
        not torch.equal(old, parameter)
        for old, parameter in zip(before, semantic_parameters, strict=True)
    )


def test_muon_maloq_head_config_round_trip() -> None:
    workflow = MaloqConfig(
        model={"head_type": "maloq_muon"},
        optimization={"muon_parameter_policy": "semantic"},
    ).to_workflow_config()
    assert workflow["head_type"] == "maloq_muon"
    assert "muon_parameter_policy" not in workflow


def test_fixed_muon_routing_includes_every_matrix() -> None:
    backbone = torch.nn.Sequential(
        torch.nn.Embedding(8, 4),
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 4),
    )
    head = torch.nn.Sequential(
        torch.nn.Linear(4, 2),
        torch.nn.LayerNorm(2),
    )
    muon_parameters = TrainingWorkflow._collect_muon_parameters(backbone, head)
    muon_ids = {id(parameter) for parameter in muon_parameters}

    assert all(
        (id(parameter) in muon_ids) == (parameter.ndim >= 2)
        for parameter in list(backbone.parameters()) + list(head.parameters())
    )
