from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from e3nn.o3 import Irreps
from pydantic import ValidationError

from maloq.core.config import MaloqConfig
from maloq.helm.esen_osh import eSEN_Backbone
from maloq.train_utils.training_workflow import TrainingWorkflow
from maloq.train_utils.training_workflow_fixed import (
    _migrate_stored_signature_defaults,
)


def _small_backbone(**kwargs) -> eSEN_Backbone:
    options = {
        "sphere_channels": 4,
        "hidden_channels": 4,
        "lmax": 1,
        "mmax": 1,
        "cutoff": 8.0,
        "edge_channels": 4,
        "num_layers": 2,
        "num_edge_layers": 2,
        "num_distance_basis": 4,
        "message_passing_schedule": "node_then_edge",
    }
    options.update(kwargs)
    return eSEN_Backbone(Irreps("1x0e"), **options)


def _validate_workflow_model_options(**model_options) -> dict:
    model_options = dict(model_options)
    loss_target = model_options.pop("loss_target", None)
    config_options = {"model": model_options}
    if loss_target is not None:
        config_options["loss"] = {"loss_target": loss_target}
    workflow = object.__new__(TrainingWorkflow)
    workflow.config = MaloqConfig(**config_options).to_workflow_config()
    workflow.rank = 1
    workflow.device = torch.device("cpu")
    workflow.check_input_config()
    return workflow.config


def _load_nabladft_runner_module():
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_nabladft_qh9_density_initial_edge_zero_test",
        runner_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_edge_state_config_round_trip_and_validation() -> None:
    default = MaloqConfig().to_workflow_config()
    configured = MaloqConfig(
        model={
            "message_passing_schedule": "node_then_edge",
            "initial_edge_state_mode": "zero",
        }
    ).to_workflow_config()

    assert default["initial_edge_state_mode"] == "edge_degree"
    assert configured["initial_edge_state_mode"] == "zero"
    assert (
        _validate_workflow_model_options(
            message_passing_schedule="node_then_edge",
            initial_edge_state_mode="zero",
        )["initial_edge_state_mode"]
        == "zero"
    )
    with pytest.raises(ValidationError, match="initial_edge_state_mode"):
        MaloqConfig(model={"initial_edge_state_mode": "unknown"})


@pytest.mark.parametrize(
    ("model_options", "error_match"),
    (
        (
            {"initial_edge_state_mode": "zero"},
            "message_passing_schedule='node_then_edge'",
        ),
        (
            {
                "message_passing_schedule": "node_then_edge",
                "initial_edge_state_mode": "zero",
                "edge_stack_mode": "nte_parallel",
            },
            "edge_stack_mode='recurrent'",
        ),
        (
            {
                "message_passing_schedule": "node_then_edge",
                "initial_edge_state_mode": "zero",
                "message_type": "source-target-message",
            },
            "message_type='source-target'",
        ),
        (
            {
                "message_passing_schedule": "node_then_edge",
                "initial_edge_state_mode": "zero",
                "direct_edgewise_layers": (1,),
            },
            "redundant.*EdgeBlock 1",
        ),
        (
            {
                "backbone_type": "qhflow3_clean",
                "message_passing_schedule": "node_then_edge",
                "initial_edge_state_mode": "zero",
            },
            "backbone_type='esen'",
        ),
        (
            {
                "loss_target": "energies",
                "message_passing_schedule": "node_then_edge",
                "initial_edge_state_mode": "zero",
            },
            "matrix loss target",
        ),
    ),
)
def test_workflow_rejects_invalid_initial_edge_zero_combinations(
    model_options: dict,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _validate_workflow_model_options(**model_options)


@pytest.mark.parametrize(
    ("model_options", "error_match"),
    (
        (
            {
                "initial_edge_state_mode": "zero",
                "message_passing_schedule": "interleaved",
            },
            "message_passing_schedule='node_then_edge'",
        ),
        (
            {
                "initial_edge_state_mode": "zero",
                "edge_stack_mode": "nte_parallel",
            },
            "edge_stack_mode='recurrent'",
        ),
        (
            {
                "initial_edge_state_mode": "zero",
                "message_type": "source-target-message",
            },
            "message_type='source-target'",
        ),
        (
            {
                "initial_edge_state_mode": "zero",
                "direct_edgewise_layers": (1,),
            },
            "redundant.*EdgeBlock 1",
        ),
        (
            {
                "initial_edge_state_mode": "zero",
                "include_edges": False,
            },
            "include_edges=True",
        ),
    ),
)
def test_backbone_rejects_invalid_initial_edge_zero_combinations(
    model_options: dict,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _small_backbone(**model_options)


def test_default_initial_edge_mode_preserves_initialization_bitwise() -> None:
    torch.manual_seed(44)
    implicit = _small_backbone()
    implicit_rng = torch.get_rng_state().clone()

    torch.manual_seed(44)
    explicit = _small_backbone(initial_edge_state_mode="edge_degree")
    explicit_rng = torch.get_rng_state().clone()

    assert torch.equal(implicit_rng, explicit_rng)
    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    for name in implicit_state:
        assert torch.equal(implicit_state[name], explicit_state[name]), name


def test_zero_mode_preserves_parameters_and_rng_relative_to_default() -> None:
    torch.manual_seed(44)
    default = _small_backbone()
    default_rng = torch.get_rng_state().clone()

    torch.manual_seed(44)
    zero = _small_backbone(initial_edge_state_mode="zero")
    zero_rng = torch.get_rng_state().clone()

    assert torch.equal(default_rng, zero_rng)
    assert sum(parameter.numel() for parameter in default.parameters()) == sum(
        parameter.numel() for parameter in zero.parameters()
    )
    default_state = default.state_dict()
    zero_state = zero.state_dict()
    assert default_state.keys() == zero_state.keys()
    for name in default_state:
        assert torch.equal(default_state[name], zero_state[name]), name


class _NodeRecorder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_edges: list[torch.Tensor] = []

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        self.seen_edges.append(edge_state.detach().clone())
        return node_state + 1.0


class _EdgeRecorder(torch.nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment
        self.seen_edges: list[torch.Tensor] = []

    def forward(
        self,
        _node_state: torch.Tensor,
        edge_state: torch.Tensor,
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        self.seen_edges.append(edge_state.detach().clone())
        return edge_state + self.increment


def _message_passing_probe(initial_edge_state_mode: str):
    model = eSEN_Backbone.__new__(eSEN_Backbone)
    torch.nn.Module.__init__(model)
    model.include_edges = True
    model.message_passing_schedule = "node_then_edge"
    model.initial_edge_state_mode = initial_edge_state_mode
    model.node_stack_mode = "nte"
    model.edge_stack_mode = "recurrent"
    node_1 = _NodeRecorder()
    node_2 = _NodeRecorder()
    edge_1 = _EdgeRecorder(1.0)
    edge_2 = _EdgeRecorder(2.0)
    model.node_blocks = torch.nn.ModuleList((node_1, node_2))
    model.edge_blocks = torch.nn.ModuleList((edge_1, edge_2))

    initial_node = torch.zeros(2, 4, 3)
    initial_edge = torch.full((3, 4, 3), 10.0)
    final_node, final_edge = model._run_message_passing(
        initial_node,
        initial_edge,
        None,
        {
            "edge_distance": None,
            "edge_index": None,
            "partition": None,
        },
        None,
        None,
    )
    return (
        initial_node,
        initial_edge,
        final_node,
        final_edge,
        (node_1, node_2),
        (edge_1, edge_2),
    )


def test_initial_edge_zero_changes_only_first_recurrent_edge_input() -> None:
    (
        initial_node,
        initial_edge,
        zero_node,
        zero_edge,
        zero_nodes,
        zero_edges,
    ) = _message_passing_probe("zero")
    (
        _,
        _,
        default_node,
        default_edge,
        default_nodes,
        default_edges,
    ) = _message_passing_probe("edge_degree")

    for node_block in (*zero_nodes, *default_nodes):
        torch.testing.assert_close(node_block.seen_edges[0], initial_edge)
    torch.testing.assert_close(zero_node, initial_node + 2.0)
    torch.testing.assert_close(default_node, zero_node)

    torch.testing.assert_close(
        zero_edges[0].seen_edges[0],
        torch.zeros_like(initial_edge),
    )
    torch.testing.assert_close(
        zero_edges[1].seen_edges[0],
        torch.ones_like(initial_edge),
    )
    torch.testing.assert_close(zero_edge, torch.full_like(initial_edge, 3.0))

    torch.testing.assert_close(default_edges[0].seen_edges[0], initial_edge)
    torch.testing.assert_close(
        default_edges[1].seen_edges[0],
        initial_edge + 1.0,
    )
    torch.testing.assert_close(default_edge, initial_edge + 3.0)


def test_resume_migration_treats_missing_mode_as_edge_degree() -> None:
    migrated = _migrate_stored_signature_defaults({"world_size": 2})
    assert migrated["initial_edge_state_mode"] == "edge_degree"


def test_nabladft_runner_names_initial_edge_zero() -> None:
    runner = _load_nabladft_runner_module()
    identity = runner.nabladft_tracking_identity(
        "maloq-nte",
        {
            "head_type": "maloq_muon",
            "scale_and_shift": False,
            "l_embedding_dim": 128,
            "output_l_embedding_dim": 64,
            "num_mp_layers": 2,
            "num_edge_layers": 2,
            "seed": 44,
            "experiment_version": 1,
            "nte_input_conditioning": "qhflow3_exact",
            "edge_norm1_position": "pre_node",
            "initial_edge_state_mode": "zero",
        },
        smoke=False,
    )

    assert identity["experiment_id"].endswith(
        "-qcond-edgepre-edgezero-v1"
    )
    assert identity["display_name"].endswith(
        "QHFcond | EdgePre | InitialEdgeZero | V1"
    )
    assert "edge-norm1-position:pre-node" in identity["tags"]
    assert "ablation:edge-norm-ordering" in identity["tags"]
    assert "initial-edge-state:zero" in identity["tags"]
    assert "ablation:initial-edge-state" in identity["tags"]


def test_nabladft_runner_distinguishes_edgepre_initial_edge_zero() -> None:
    runner = _load_nabladft_runner_module()
    shared = {
        "head_type": "maloq_muon",
        "scale_and_shift": False,
        "l_embedding_dim": 128,
        "output_l_embedding_dim": 64,
        "num_mp_layers": 2,
        "num_edge_layers": 2,
        "seed": 44,
        "experiment_version": 1,
        "nte_input_conditioning": "qhflow3_exact",
        "initial_edge_state_mode": "zero",
    }
    post_edgewise = runner.nabladft_tracking_identity(
        "maloq-nte",
        shared,
        smoke=False,
    )
    edgepre = runner.nabladft_tracking_identity(
        "maloq-nte",
        {**shared, "edge_norm1_position": "pre_node"},
        smoke=False,
    )

    assert post_edgewise["experiment_id"] != edgepre["experiment_id"]
    assert post_edgewise["display_name"] != edgepre["display_name"]
    assert "-edgepre-" in edgepre["experiment_id"]
