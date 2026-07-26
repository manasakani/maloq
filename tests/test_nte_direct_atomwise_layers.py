from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from maloq.core.config import MaloqConfig
from maloq.helm.esen_osh import eSEN_Backbone
from maloq.train_utils.training_workflow import TrainingWorkflow
from maloq.train_utils.training_workflow_fixed import (
    _migrate_stored_signature_defaults,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_backbone(**kwargs) -> eSEN_Backbone:
    options = {
        "sphere_channels": 4,
        "hidden_channels": 4,
        "lmax": 1,
        "mmax": 1,
        "cutoff": 8.0,
        "edge_channels": 4,
        "num_layers": 1,
        "num_edge_layers": 2,
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_atomwise_config_round_trip() -> None:
    default = MaloqConfig().to_workflow_config()
    configured = MaloqConfig(
        model={"direct_atomwise_layers": [1]}
    ).to_workflow_config()

    assert default["direct_atomwise_layers"] == ()
    assert configured["direct_atomwise_layers"] == (1,)
    assert _validate_workflow(
        num_mp_layers=1,
        num_edge_layers=2,
        message_passing_schedule="node_then_edge",
        direct_atomwise_layers=(1,),
    )["direct_atomwise_layers"] == (1,)


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    (
        (
            {
                "num_edge_layers": 2,
                "direct_atomwise_layers": (1, 1),
            },
            "must not contain duplicates",
        ),
        (
            {
                "num_edge_layers": 2,
                "direct_atomwise_layers": (0,),
            },
            "1-based indices",
        ),
        (
            {
                "num_edge_layers": 2,
                "direct_atomwise_layers": (3,),
            },
            "1-based indices",
        ),
        (
            {
                "backbone_type": "qhflow3_clean",
                "direct_atomwise_layers": (1,),
            },
            "requires backbone_type='esen'",
        ),
        (
            {
                "loss_target": "energies",
                "direct_atomwise_layers": (1,),
            },
            "requires a matrix loss target",
        ),
        (
            {
                "edge_stack_mode": "qhflow3_parallel",
                "message_passing_schedule": "node_then_edge",
                "direct_atomwise_layers": (1,),
            },
            "not a QHFlow3 pair stack",
        ),
        (
            {
                "edge_stack_mode": "qhflow3_exact_parallel",
                "message_passing_schedule": "node_then_edge",
                "mlp_type": "grid",
                "direct_atomwise_layers": (1,),
            },
            "not a QHFlow3 pair stack",
        ),
    ),
)
def test_workflow_rejects_invalid_direct_atomwise_combinations(
    overrides: dict,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _validate_workflow(**overrides)


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    (
        (
            {"direct_atomwise_layers": (1, 1)},
            "must not contain duplicates",
        ),
        (
            {"direct_atomwise_layers": (0,)},
            "1-based indices",
        ),
        (
            {"direct_atomwise_layers": (3,)},
            "1-based indices",
        ),
        (
            {
                "direct_atomwise_layers": (1,),
                "include_edges": False,
            },
            "requires include_edges=True",
        ),
        (
            {
                "direct_atomwise_layers": (1,),
                "edge_stack_mode": "qhflow3_parallel",
            },
            "not a QHFlow3 pair stack",
        ),
        (
            {
                "direct_atomwise_layers": (1,),
                "edge_stack_mode": "qhflow3_exact_parallel",
                "mlp_type": "grid",
            },
            "not a QHFlow3 pair stack",
        ),
    ),
)
def test_backbone_rejects_invalid_direct_atomwise_combinations(
    overrides: dict,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _small_backbone(**overrides)


def test_default_direct_atomwise_layers_is_bitwise_backward_compatible() -> None:
    torch.manual_seed(44)
    implicit = _small_backbone(num_edge_layers=1).eval()
    implicit_rng = torch.get_rng_state().clone()

    torch.manual_seed(44)
    explicit = _small_backbone(
        num_edge_layers=1,
        direct_atomwise_layers=(),
    ).eval()
    explicit_rng = torch.get_rng_state().clone()

    assert torch.equal(implicit_rng, explicit_rng)
    assert [
        block.atomwise_output_mode for block in implicit.edge_blocks
    ] == ["residual_scaled"]
    assert [
        block.atomwise_output_mode for block in explicit.edge_blocks
    ] == ["residual_scaled"]
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


def test_selected_atomwise_layer_changes_only_execution_mode_not_state_or_rng() -> None:
    base_options = {
        "edge_norm1_position": "pre_node",
        "direct_edgewise_layers": (1,),
        "nte_output_projection_mode": "qhflow3_irrep_linear",
        "output_norm_sharing": "separate",
        "output_sphere_channels": 2,
    }
    torch.manual_seed(44)
    base = _small_backbone(**base_options)
    base_rng = torch.get_rng_state().clone()

    torch.manual_seed(44)
    selected = _small_backbone(
        **base_options,
        direct_atomwise_layers=(1,),
    )
    selected_rng = torch.get_rng_state().clone()

    assert torch.equal(base_rng, selected_rng)
    assert [
        block.edgewise_output_mode for block in selected.edge_blocks
    ] == ["direct", "residual_scaled"]
    assert [
        block.atomwise_output_mode for block in selected.edge_blocks
    ] == ["direct", "residual_scaled"]
    assert sum(parameter.numel() for parameter in base.parameters()) == sum(
        parameter.numel() for parameter in selected.parameters()
    )
    base_state = base.state_dict()
    selected_state = selected.state_dict()
    assert base_state.keys() == selected_state.keys()
    for name in base_state:
        assert torch.equal(base_state[name], selected_state[name]), name
    assert list(dict(base.named_parameters())) == list(
        dict(selected.named_parameters())
    )


def test_selected_layer_returns_exact_atomwise_transform_without_residual_scale() -> None:
    class FixedEdgewise(torch.nn.Module):
        def forward(self, _node_state, edge_state, *_args):
            return torch.full_like(edge_state, 3.0)

    class AddOne(torch.nn.Module):
        def forward(self, state):
            return state + 1.0

    class DoubleAtomwise(torch.nn.Module):
        def forward(self, state):
            return 2.0 * state

    class FailIfCalled(torch.nn.Module):
        def forward(self, _state):
            raise AssertionError(
                "direct atomwise output must not apply atom update scale"
            )

    model = _small_backbone(
        edge_norm1_position="pre_node",
        direct_edgewise_layers=(1,),
        direct_atomwise_layers=(1,),
    )
    block = model.edge_blocks[0]
    block.norm_1 = torch.nn.Identity()
    block.norm_2 = AddOne()
    block.post_residual_norm = torch.nn.Identity()
    block.edge_wise = FixedEdgewise()
    block.atom_wise = DoubleAtomwise()
    block.edge_update_scale = FailIfCalled()
    block.atom_update_scale = FailIfCalled()

    edge_state = torch.full((3, 4, 2), 7.0)
    output = block(
        torch.zeros(2, 4, 2),
        edge_state,
        None,
        None,
        None,
        None,
        None,
        "edge",
        None,
    )

    # Direct edgewise m1=3, then F1(norm2(m1))=2*(3+1)=8.
    torch.testing.assert_close(output, torch.full_like(edge_state, 8.0))
    assert model.edge_blocks[1].atomwise_output_mode == "residual_scaled"


def test_atomwise_selection_is_independent_from_edgewise_selection() -> None:
    class FixedEdgewise(torch.nn.Module):
        def forward(self, _node_state, edge_state, *_args):
            return torch.full_like(edge_state, 3.0)

    class DoubleAtomwise(torch.nn.Module):
        def forward(self, state):
            return 2.0 * state

    class FailIfCalled(torch.nn.Module):
        def forward(self, _state):
            raise AssertionError(
                "direct atomwise output must not apply atom update scale"
            )

    atomwise_only = _small_backbone(direct_atomwise_layers=(1,))
    both = _small_backbone(
        direct_edgewise_layers=(1,),
        direct_atomwise_layers=(1,),
    )

    assert [
        (
            block.edgewise_output_mode,
            block.atomwise_output_mode,
        )
        for block in atomwise_only.edge_blocks
    ] == [
        ("residual_scaled", "direct"),
        ("residual_scaled", "residual_scaled"),
    ]
    assert [
        (
            block.edgewise_output_mode,
            block.atomwise_output_mode,
        )
        for block in both.edge_blocks
    ] == [
        ("direct", "direct"),
        ("residual_scaled", "residual_scaled"),
    ]

    outputs = []
    incoming_edge = torch.full((3, 4, 2), 7.0)
    for model in (atomwise_only, both):
        block = model.edge_blocks[0]
        block.norm_1 = torch.nn.Identity()
        block.norm_2 = torch.nn.Identity()
        block.post_residual_norm = torch.nn.Identity()
        block.edge_wise = FixedEdgewise()
        block.atom_wise = DoubleAtomwise()
        block.edge_update_scale = torch.nn.Identity()
        block.atom_update_scale = FailIfCalled()
        outputs.append(
            block(
                torch.zeros(2, 4, 2),
                incoming_edge,
                None,
                None,
                None,
                None,
                None,
                "edge",
                None,
            )
        )

    # Atomwise-direct alone sees 3 + incoming 7; composing edgewise-direct
    # changes only that preceding boundary and supplies 3 to the same F1.
    torch.testing.assert_close(
        outputs[0],
        torch.full_like(incoming_edge, 20.0),
    )
    torch.testing.assert_close(
        outputs[1],
        torch.full_like(incoming_edge, 6.0),
    )


def test_resume_migration_treats_missing_layers_as_empty() -> None:
    migrated = _migrate_stored_signature_defaults({"world_size": 2})
    assert migrated["direct_atomwise_layers"] == []


def test_nabladft_tracking_names_direct_atomwise_layers() -> None:
    runner = _load_module(
        "run_nabladft_qh9_density_direct_atomwise_test",
        PROJECT_ROOT
        / "_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py",
    )
    direct_only = runner.nabladft_tracking_identity(
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
            "direct_atomwise_layers": (1,),
        },
        smoke=False,
    )
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
            "direct_edgewise_layers": (1,),
            "direct_atomwise_layers": (1,),
            "nte_output_projection_mode": "qhflow3_irrep_linear",
            "output_norm_sharing": "separate",
        },
        smoke=False,
    )

    assert direct_only["experiment_id"] == (
        "nabla-nte64e2-matrixmuon-auxadamw-raw-edge1atomdirect-v1"
    )
    assert direct_only["display_name"].endswith(
        "RAW | Edge1AtomDirect | V1"
    )
    assert identity["experiment_id"].endswith(
        "-qcond-edgepre-edge1direct-edge1atomdirect-qhfproj"
        "-splitoutnorm-v1"
    )
    assert identity["display_name"].endswith(
        "QHFcond | EdgePre | Edge1Direct | Edge1AtomDirect | "
        "QHFProj | SplitOutNorm | V1"
    )
    assert "atomwise-direct-layers:1" in identity["tags"]
    assert "ablation:edge-atomwise-residual" in identity["tags"]


def test_layer_feature_analyzer_propagates_direct_atomwise_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = _load_module(
        "sc26_direct_atomwise_analyzer_test",
        PROJECT_ROOT
        / "_auto_script/layer_feature_analysis/"
        "analyze_nabladft_qhf_vs_nte.py",
    )

    class CapturedBackbone:
        kwargs: dict[str, object]

        def __init__(self, *_args, **kwargs) -> None:
            type(self).kwargs = kwargs

        def to(self, _device):
            return self

    from maloq.helm import esen_osh

    monkeypatch.setattr(esen_osh, "eSEN_Backbone", CapturedBackbone)
    config = analyzer.load_config(analyzer.DEFAULT_NTE_CONFIG)
    config["direct_atomwise_layers"] = (1,)
    analyzer.build_backbone(
        "nte",
        config,
        SimpleNamespace(lmax=4),
        torch.device("cpu"),
    )

    assert CapturedBackbone.kwargs["direct_atomwise_layers"] == (1,)
