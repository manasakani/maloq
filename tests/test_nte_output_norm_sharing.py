from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from e3nn.o3 import Irreps
from pydantic import ValidationError
from torch_geometric.data import Batch, Data

from maloq.core.config import MaloqConfig
from maloq.helm.esen_osh import eSEN_Backbone
from maloq.train_utils.training_workflow import TrainingWorkflow
from maloq.train_utils.training_workflow_fixed import (
    _migrate_stored_signature_defaults,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    PROJECT_ROOT
    / "_auto_script/layer_feature_analysis/analyze_nabladft_qhf_vs_nte.py"
)


def _small_backbone(**kwargs) -> eSEN_Backbone:
    options = {
        "sphere_channels": 4,
        "hidden_channels": 4,
        "lmax": 1,
        "mmax": 1,
        "cutoff": 8.0,
        "edge_channels": 4,
        "num_layers": 1,
        "num_edge_layers": 1,
        "num_distance_basis": 4,
        "message_passing_schedule": "node_then_edge",
    }
    options.update(kwargs)
    return eSEN_Backbone(Irreps("1x0e"), **options)


def _two_atom_batch() -> Batch:
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
    return Batch.from_data_list([data])


def _validate_workflow(**overrides) -> dict:
    workflow = object.__new__(TrainingWorkflow)
    workflow.config = MaloqConfig().to_workflow_config()
    workflow.config.update(overrides)
    workflow.rank = 1
    workflow.device = torch.device("cpu")
    workflow.check_input_config()
    return workflow.config


def _load_runner():
    path = (
        PROJECT_ROOT
        / "_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_nabladft_qh9_density_split_output_norm_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "sc26_split_output_norm_analyzer_test",
        ANALYZER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_output_norm_sharing_config_round_trip_and_literal_validation() -> None:
    assert MaloqConfig().to_workflow_config()["output_norm_sharing"] == "shared"
    configured = MaloqConfig(
        model={"output_norm_sharing": "separate"}
    ).to_workflow_config()
    assert configured["output_norm_sharing"] == "separate"

    with pytest.raises(ValidationError, match="output_norm_sharing"):
        MaloqConfig(model={"output_norm_sharing": "unknown"})


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    (
        (
            {
                "backbone_type": "qhflow3_clean",
                "output_norm_sharing": "separate",
            },
            "requires backbone_type='esen'",
        ),
        (
            {
                "loss_target": "energies",
                "output_norm_sharing": "separate",
            },
            "requires a matrix loss target",
        ),
        (
            {
                "edge_stack_mode": "qhflow3_exact_parallel",
                "message_passing_schedule": "node_then_edge",
                "output_norm_sharing": "separate",
            },
            "redundant.*separate QHFlow3 pair norm",
        ),
        (
            {"output_norm_sharing": "unknown"},
            "output_norm_sharing must be",
        ),
    ),
)
def test_workflow_rejects_output_norm_noop_or_invalid_scope(
    overrides: dict,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _validate_workflow(**overrides)


def test_backbone_rejects_output_norm_noop_or_invalid_scope() -> None:
    with pytest.raises(ValueError, match="output_norm_sharing must be"):
        _small_backbone(output_norm_sharing="unknown")
    with pytest.raises(ValueError, match="requires include_edges=True"):
        _small_backbone(
            output_norm_sharing="separate",
            include_edges=False,
        )
    with pytest.raises(
        ValueError,
        match="redundant.*separate QHFlow3 pair norm",
    ):
        _small_backbone(
            output_norm_sharing="separate",
            edge_stack_mode="qhflow3_exact_parallel",
        )


def test_shared_default_preserves_state_rng_parameters_and_forward_bitwise() -> None:
    torch.manual_seed(44)
    implicit = _small_backbone().eval()
    implicit_rng = torch.get_rng_state().clone()

    torch.manual_seed(44)
    explicit = _small_backbone(output_norm_sharing="shared").eval()
    explicit_rng = torch.get_rng_state().clone()

    assert torch.equal(implicit_rng, explicit_rng)
    assert implicit.edge_norm is None
    assert explicit.edge_norm is None
    assert sum(p.numel() for p in implicit.parameters()) == sum(
        p.numel() for p in explicit.parameters()
    )
    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    for name in implicit_state:
        assert torch.equal(implicit_state[name], explicit_state[name]), name

    batch = _two_atom_batch()
    with torch.no_grad():
        implicit_output = implicit(batch)
        explicit_output = explicit(batch)
    for key in ("node_embeddings", "edge_embeddings"):
        torch.testing.assert_close(
            explicit_output[key],
            implicit_output[key],
            rtol=0.0,
            atol=0.0,
        )


def test_separate_norm_matches_step_zero_and_has_independent_gradients() -> None:
    torch.manual_seed(44)
    shared = _small_backbone(output_norm_sharing="shared").eval()
    shared_rng = torch.get_rng_state().clone()

    torch.manual_seed(44)
    separate = _small_backbone(output_norm_sharing="separate").eval()
    separate_rng = torch.get_rng_state().clone()

    assert torch.equal(shared_rng, separate_rng)
    assert separate.edge_norm is not None
    assert type(separate.edge_norm) is type(separate.norm)
    for node_parameter, edge_parameter in zip(
        separate.norm.parameters(),
        separate.edge_norm.parameters(),
        strict=True,
    ):
        assert node_parameter.data_ptr() != edge_parameter.data_ptr()
        torch.testing.assert_close(
            edge_parameter,
            node_parameter,
            rtol=0.0,
            atol=0.0,
        )

    batch = _two_atom_batch()
    shared_output = shared(batch)
    separate_output = separate(batch)
    for key in ("node_embeddings", "edge_embeddings"):
        torch.testing.assert_close(
            separate_output[key],
            shared_output[key],
            rtol=0.0,
            atol=0.0,
        )

    node_probe = torch.randn_like(separate_output["node_embeddings"])
    edge_probe = torch.randn_like(separate_output["edge_embeddings"])
    (
        (separate_output["node_embeddings"] * node_probe).sum()
        + (separate_output["edge_embeddings"] * edge_probe).sum()
    ).backward()
    node_gradients = [
        parameter.grad for parameter in separate.norm.parameters()
    ]
    edge_gradients = [
        parameter.grad for parameter in separate.edge_norm.parameters()
    ]
    assert all(gradient is not None for gradient in node_gradients)
    assert all(gradient is not None for gradient in edge_gradients)
    assert any(
        not torch.equal(node_gradient, edge_gradient)
        for node_gradient, edge_gradient in zip(
            node_gradients,
            edge_gradients,
            strict=True,
        )
    )


def test_resume_migration_treats_missing_option_as_shared() -> None:
    migrated = _migrate_stored_signature_defaults({"world_size": 2})
    assert migrated["output_norm_sharing"] == "shared"


def test_nabladft_tracking_identity_is_unique_for_split_output_norm() -> None:
    runner = _load_runner()
    base_config = {
        "head_type": "maloq_muon",
        "scale_and_shift": False,
        "l_embedding_dim": 128,
        "output_l_embedding_dim": 64,
        "num_mp_layers": 2,
        "num_edge_layers": 2,
        "seed": 44,
        "experiment_version": 1,
        "nte_input_conditioning": "qhflow3_exact",
    }
    shared = runner.nabladft_tracking_identity(
        "maloq-nte",
        base_config,
        smoke=False,
    )
    separate = runner.nabladft_tracking_identity(
        "maloq-nte",
        {**base_config, "output_norm_sharing": "separate"},
        smoke=False,
    )

    assert shared["experiment_id"] != separate["experiment_id"]
    assert separate["experiment_id"].endswith(
        "-qcond-splitoutnorm-v1"
    )
    assert separate["display_name"].endswith(
        "QHFcond | SplitOutNorm | V1"
    )
    assert "matrixmuon-auxadamw" in separate["experiment_id"]
    assert "output-norm-sharing:separate" in separate["tags"]
    assert "ablation:output-norm-sharing" in separate["tags"]


def test_layer_feature_analyzer_propagates_output_norm_sharing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = _load_analyzer()

    class CapturedBackbone:
        kwargs: dict[str, object]

        def __init__(self, *_args, **kwargs) -> None:
            type(self).kwargs = kwargs

        def to(self, _device):
            return self

    from maloq.helm import esen_osh

    monkeypatch.setattr(esen_osh, "eSEN_Backbone", CapturedBackbone)
    config = analyzer.load_config(analyzer.DEFAULT_NTE_CONFIG)
    config["output_norm_sharing"] = "separate"
    analyzer.build_backbone(
        "nte",
        config,
        SimpleNamespace(lmax=4),
        torch.device("cpu"),
    )

    assert CapturedBackbone.kwargs["output_norm_sharing"] == "separate"
