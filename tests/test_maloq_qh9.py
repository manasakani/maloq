from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from e3nn.o3 import Irreps

from maloq.core.config import MaloqConfig
from maloq.helm.esen_block import DegreeLayerScale, GridAtomwise
from maloq.helm.esen_osh import eSEN_Backbone
from maloq.helm.nn.activation import GateActivation
from maloq.helm.qhflow3_clean import QHFlow3MaloqBackbone


def _load_comparison_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "_my_script/experiment/2026-07-21/compare_maloq_qh9.py"
    )
    spec = importlib.util.spec_from_file_location("compare_maloq_qh9", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_activation_supports_qhflow3_sigmoid_gate():
    sigmoid = GateActivation(1, 1, 2, gate_act_type="sigmoid")
    tanh = GateActivation(1, 1, 2, gate_act_type="tanh")
    gating = torch.zeros(1, 2)
    features = torch.ones(1, 4, 2)

    sigmoid_out = sigmoid(gating, features)
    tanh_out = tanh(gating, features)

    torch.testing.assert_close(sigmoid_out[:, 1:], torch.full((1, 3, 2), 0.5))
    torch.testing.assert_close(tanh_out[:, 1:], torch.zeros(1, 3, 2))


def test_bounded_degree_layerscale_has_reference_initialization_and_bounds():
    scale = DegreeLayerScale(
        lmax=4,
        mode="bounded_degree",
        init=1.0 / 64.0,
        log_range=math.log(64.0),
    )
    torch.testing.assert_close(
        scale.degree_scales(),
        torch.full((5,), 1.0 / 64.0),
    )

    with torch.no_grad():
        scale.raw.copy_(torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0]))
    values = scale.degree_scales()
    assert values.min().item() >= (1.0 / 4096.0) * (1.0 - 1.0e-6)
    assert values.max().item() <= 1.0 + 1.0e-6


def test_maloq_qh9_backbone_contract_matches_reference_axes():
    model = eSEN_Backbone(
        Irreps("1x0e"),
        sphere_channels=8,
        hidden_channels=8,
        lmax=4,
        mmax=4,
        cutoff=15.0,
        edge_channels=8,
        num_layers=3,
        num_edge_layers=2,
        num_distance_basis=16,
        gate_act_type="sigmoid",
        mlp_type="grid",
        message_passing_schedule="node_then_edge",
        output_sphere_channels=4,
        use_edge_envelope=True,
        use_edge_scalar_modulation=True,
        residual_update_scale_mode="bounded_degree",
        residual_update_scale_init=1.0 / 64.0,
        residual_update_scale_log_range=math.log(64.0),
    )

    assert len(model.node_blocks) == 3
    assert len(model.edge_blocks) == 2
    assert model.output_sphere_channels == 4
    assert model.message_passing_schedule == "node_then_edge"
    assert all(isinstance(block.atom_wise, GridAtomwise) for block in model.node_blocks)
    assert all(isinstance(block.atom_wise, GridAtomwise) for block in model.edge_blocks)
    assert all(block.edge_wise.use_edge_envelope for block in model.node_blocks)
    assert all(block.edge_wise.use_edge_scalar_modulation for block in model.edge_blocks)

    scale_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("_update_scale.raw")
    ]
    assert len(scale_parameters) == 10
    assert sum(parameter.numel() for parameter in scale_parameters) == 50


def test_maloq_qh9_config_round_trip_preserves_model_recipe():
    config = MaloqConfig(
        model={
            "model_variant": "maloq-qh9",
            "gate_act_type": "sigmoid",
            "mlp_type": "grid",
            "message_passing_schedule": "node_then_edge",
            "num_edge_layers": 2,
            "output_l_embedding_dim": 64,
            "use_edge_envelope": True,
            "use_edge_scalar_modulation": True,
            "residual_update_scale_mode": "bounded_degree",
            "residual_update_scale_init": 1.0 / 64.0,
            "residual_update_scale_log_range": math.log(64.0),
        },
        optimization={
            "scheduler_type": "warmup_polynomial",
            "warmup_steps": 1000,
            "gradient_clip_val": 1.0,
        },
        runtime={"seed": 44},
    )
    workflow = config.to_workflow_config()

    assert workflow["model_variant"] == "maloq-qh9"
    assert workflow["message_passing_schedule"] == "node_then_edge"
    assert workflow["output_l_embedding_dim"] == 64
    assert workflow["scheduler_type"] == "warmup_polynomial"
    assert workflow["gradient_clip_val"] == 1.0
    assert workflow["seed"] == 44


def test_comparison_loss_reader_uses_persisted_edge_node_order(tmp_path):
    module = _load_comparison_module()
    (tmp_path / "head_training_loss.txt").write_text("4.0\t2.0\n")
    (tmp_path / "head_validation_loss.txt").write_text("3.0\t1.0\n")

    assert module.last_losses(tmp_path) == {
        "train_node_loss": 2.0,
        "train_edge_loss": 4.0,
        "validation_node_loss": 1.0,
        "validation_edge_loss": 3.0,
    }


def test_qhflow3_native_overlap_bridge_uses_def2_svp_padding():
    batch = SimpleNamespace(
        overlap_matrix=[torch.eye(19).numpy()],
        ptr=torch.tensor([0, 2]),
        atomic_numbers=torch.tensor([1, 6]),
        pos=torch.zeros(2, 3),
    )

    blocks = QHFlow3MaloqBackbone._overlap_blocks(batch)

    assert blocks.shape == (2, 14, 14)
    hydrogen_mask = torch.tensor([0, 1, 3, 4, 5])
    torch.testing.assert_close(
        blocks[0][hydrogen_mask[:, None], hydrogen_mask[None, :]],
        torch.eye(5),
    )
    torch.testing.assert_close(blocks[1], torch.eye(14))


def test_qhflow3_config_selects_headless_native_bridge():
    workflow = MaloqConfig(
        model={
            "model_variant": "qhflow3-maloq-head",
            "backbone_type": "qhflow3_clean",
            "output_l_embedding_dim": 64,
            "num_edge_layers": 2,
        }
    ).to_workflow_config()

    assert workflow["backbone_type"] == "qhflow3_clean"
    assert workflow["output_l_embedding_dim"] == 64
