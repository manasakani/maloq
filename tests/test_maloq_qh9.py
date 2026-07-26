from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from maloq.core.config import MaloqConfig
from maloq.dataset_utils.get_loader import _qm7_matrix_target
from maloq.helm.esen_block import DegreeLayerScale, GridAtomwise, eSEN_Block
from maloq.helm.esen_osh import eSEN_Backbone
from maloq.helm.nn.activation import GateActivation
from maloq.helm.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
)
from maloq.helm.qhflow3_clean import (
    GridAtomwise as QHFlow3GridAtomwise,
    QHFlow3MaloqBackbone,
    _orbital_masks_for_basis,
)
from maloq.helm.nte_conditioning import NTEMatrixConditioning
from maloq.train_utils.splittrainer import SplitTrainer
from maloq.train_utils.training_workflow import TrainingWorkflow


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


def _load_nabladft_runner_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_nabladft_qh9_density",
        script,
    )
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
    assert all(
        block.edge_wise.use_edge_scalar_modulation for block in model.edge_blocks
    )

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


def test_nte_layer_structure_ablation_options_are_explicit():
    workflow = MaloqConfig(
        model={
            "message_passing_schedule": "node_then_edge",
            "num_mp_layers": 3,
            "unscaled_node_layers": [2],
            "repeat_system_embedding_each_node_block": True,
            "nte_input_conditioning": "qhflow3_exact",
            "edge_stack_mode": "qhflow3_parallel",
            "edge_atom_norm_type": "layer_norm_sh",
            "edge_post_residual_norm_type": "rms_norm_sh",
            "edge_atomwise_output_mode": "direct",
            "edge_norm1_position": "pre_node",
        },
        optimization={"muon_output_projection_policy": "adamw"},
    ).to_workflow_config()

    assert workflow["unscaled_node_layers"] == (2,)
    assert workflow["repeat_system_embedding_each_node_block"] is True
    assert workflow["edge_stack_mode"] == "qhflow3_parallel"
    assert workflow["edge_atom_norm_type"] == "layer_norm_sh"
    assert workflow["edge_post_residual_norm_type"] == "rms_norm_sh"
    assert workflow["edge_atomwise_output_mode"] == "direct"
    assert workflow["edge_norm1_position"] == "pre_node"
    assert workflow["muon_output_projection_policy"] == "adamw"


def test_nte_edge_only_norm_options_do_not_change_node_blocks():
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
        edge_atom_norm_type="layer_norm_sh",
        edge_post_residual_norm_type="rms_norm_sh",
    )

    assert all(
        isinstance(block.norm_2, EquivariantRMSNormArraySphericalHarmonicsV2)
        for block in model.node_blocks
    )
    assert all(
        isinstance(
            block.norm_2,
            EquivariantLayerNormArraySphericalHarmonics,
        )
        for block in model.edge_blocks
    )
    assert all(
        isinstance(
            block.post_residual_norm,
            EquivariantRMSNormArraySphericalHarmonicsV2,
        )
        for block in model.edge_blocks
    )


def test_nte_edge_degree_norm_is_per_degree():
    model = eSEN_Backbone(
        Irreps("1x0e"),
        sphere_channels=4,
        hidden_channels=4,
        lmax=2,
        mmax=2,
        cutoff=15.0,
        edge_channels=4,
        num_layers=1,
        num_edge_layers=1,
        num_distance_basis=8,
        message_passing_schedule="node_then_edge",
        edge_atom_norm_type="layer_norm",
    )

    assert isinstance(model.edge_blocks[0].norm_2, EquivariantLayerNormArray)
    assert isinstance(model.edge_blocks[0].post_residual_norm, torch.nn.Identity)


def test_nte_direct_edge_atomwise_output_skips_scale_and_residual():
    class FixedEdgewise(torch.nn.Module):
        def forward(self, _node_state, edge_state, *_args):
            return torch.full_like(edge_state, 3.0)

    class DoubleAtomwise(torch.nn.Module):
        def forward(self, state):
            return 2.0 * state

    class FailIfCalled(torch.nn.Module):
        def forward(self, _state):
            raise AssertionError("direct atomwise output must skip update scale")

    block = eSEN_Block.__new__(eSEN_Block)
    torch.nn.Module.__init__(block)
    block.norm_1 = torch.nn.Identity()
    block.norm_2 = torch.nn.Identity()
    block.post_residual_norm = torch.nn.Identity()
    block.edge_wise = FixedEdgewise()
    block.atom_wise = DoubleAtomwise()
    block.edge_update_scale = torch.nn.Identity()
    block.atom_update_scale = FailIfCalled()
    block.atomwise_output_mode = "direct"

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

    # edgewise output 3 + incoming residual 7 = 10, then direct atomwise = 20.
    torch.testing.assert_close(output, torch.full_like(edge_state, 20.0))


def test_nte_edge_norm1_can_move_before_edgewise_node_input():
    class AddFive(torch.nn.Module):
        def forward(self, state):
            return state + 5.0

    class CaptureEdgewise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_node = None

        def forward(self, node_state, edge_state, *_args):
            self.seen_node = node_state.detach().clone()
            return torch.ones_like(edge_state)

    class ZeroAtomwise(torch.nn.Module):
        def forward(self, state):
            return torch.zeros_like(state)

    block = eSEN_Block.__new__(eSEN_Block)
    torch.nn.Module.__init__(block)
    block.norm_1 = AddFive()
    block.norm_2 = torch.nn.Identity()
    block.post_residual_norm = torch.nn.Identity()
    block.edge_wise = CaptureEdgewise()
    block.atom_wise = ZeroAtomwise()
    block.edge_update_scale = torch.nn.Identity()
    block.atom_update_scale = torch.nn.Identity()
    block.atomwise_output_mode = "residual_scaled"
    block.edge_norm1_position = "pre_node"

    node_state = torch.full((2, 4, 2), 2.0)
    edge_state = torch.zeros(3, 4, 2)
    output = block(
        node_state,
        edge_state,
        None,
        None,
        None,
        None,
        None,
        "edge",
        None,
    )

    torch.testing.assert_close(
        block.edge_wise.seen_node,
        torch.full_like(node_state, 7.0),
    )
    # The same norm is not applied again to the edgewise result.
    torch.testing.assert_close(output, torch.ones_like(edge_state))


@pytest.mark.parametrize(
    "norm_class",
    (
        EquivariantLayerNormArray,
        EquivariantLayerNormArraySphericalHarmonics,
        EquivariantRMSNormArraySphericalHarmonicsV2,
    ),
)
def test_edge_norm_variants_are_rotation_equivariant(norm_class):
    torch.manual_seed(44)
    lmax = 3
    channels = 5
    irreps = o3.Irreps([(1, (degree, 1)) for degree in range(lmax + 1)])
    rotation = o3.rand_matrix()
    representation = irreps.D_from_matrix(rotation)
    features = torch.randn(7, (lmax + 1) ** 2, channels)
    rotated_features = torch.einsum(
        "ab,nbc->nac", representation, features
    )
    norm = norm_class(lmax=lmax, num_channels=channels)

    expected = torch.einsum(
        "ab,nbc->nac", representation, norm(features)
    )
    actual = norm(rotated_features)

    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_nte_can_unscale_only_node_block_two():
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
        residual_update_scale_mode="bounded_degree",
        residual_update_scale_init=1.0 / 64.0,
        residual_update_scale_log_range=math.log(64.0),
        unscaled_node_layers=(2,),
    )

    assert model.node_blocks[0].edge_update_scale.raw is not None
    assert model.node_blocks[1].edge_update_scale.raw is None
    assert model.node_blocks[1].atom_update_scale.raw is None
    assert model.node_blocks[2].edge_update_scale.raw is not None
    assert all(
        block.edge_update_scale.raw is not None for block in model.edge_blocks
    )


def test_nte_repeated_system_embedding_requires_qhflow3_conditioning():
    with pytest.raises(ValueError, match="qhflow3_exact"):
        eSEN_Backbone(
            Irreps("1x0e"),
            sphere_channels=4,
            hidden_channels=4,
            lmax=2,
            mmax=2,
            cutoff=15.0,
            edge_channels=4,
            num_layers=2,
            num_edge_layers=2,
            num_distance_basis=8,
            message_passing_schedule="node_then_edge",
            repeat_system_embedding_each_node_block=True,
        )

    conditioner = NTEMatrixConditioning(
        mode="qhflow3_exact",
        basis="def2-svp-nabla",
        hidden_size=8,
    )
    system_embedding = conditioner.system_embedding(
        3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert system_embedding.shape == (3, 8)
    assert torch.isfinite(system_embedding).all()
    torch.testing.assert_close(system_embedding[0], system_embedding[1])
    torch.testing.assert_close(system_embedding[1], system_embedding[2])


def test_nte_node_block_reinjects_system_embedding_after_first_norm():
    class CaptureEdgewise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, node_state, *_args):
            self.seen = node_state.detach().clone()
            return torch.zeros_like(node_state)

    class ZeroUpdate(torch.nn.Module):
        def forward(self, state):
            return torch.zeros_like(state)

    block = eSEN_Block.__new__(eSEN_Block)
    torch.nn.Module.__init__(block)
    block.norm_1 = torch.nn.Identity()
    block.norm_2 = torch.nn.Identity()
    block.edge_wise = CaptureEdgewise()
    block.atom_wise = ZeroUpdate()
    block.edge_update_scale = torch.nn.Identity()
    block.atom_update_scale = torch.nn.Identity()

    node_state = torch.zeros(2, 4, 3)
    system_embedding = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    output = block(
        node_state,
        None,
        None,
        None,
        None,
        None,
        None,
        "node",
        None,
        system_embedding,
    )

    assert block.edge_wise.seen is not None
    torch.testing.assert_close(
        block.edge_wise.seen[:, 0, :],
        system_embedding,
    )
    torch.testing.assert_close(
        block.edge_wise.seen[:, 1:, :],
        torch.zeros(2, 3, 3),
    )
    torch.testing.assert_close(output, node_state)


@pytest.mark.parametrize("edge_stack_mode", ("nte_parallel", "qhflow3_parallel"))
def test_parallel_edge_stacks_require_node_then_edge(edge_stack_mode):
    with pytest.raises(ValueError, match="node_then_edge"):
        eSEN_Backbone(
            Irreps("1x0e"),
            sphere_channels=4,
            hidden_channels=4,
            lmax=2,
            mmax=2,
            cutoff=15.0,
            edge_channels=4,
            num_layers=2,
            num_edge_layers=2,
            num_distance_basis=8,
            message_passing_schedule="interleaved",
            edge_stack_mode=edge_stack_mode,
        )


def test_nte_parallel_edge_stack_reuses_initial_state_and_sums_branches():
    class AddEdge(torch.nn.Module):
        def __init__(self, amount):
            super().__init__()
            self.amount = amount
            self.seen = []

        def forward(self, node_state, edge_state, *_args, **kwargs):
            self.seen.append(edge_state.detach().clone())
            return edge_state + self.amount

    model = eSEN_Backbone.__new__(eSEN_Backbone)
    torch.nn.Module.__init__(model)
    model.include_edges = True
    model.message_passing_schedule = "node_then_edge"
    model.edge_stack_mode = "nte_parallel"
    model.node_blocks = torch.nn.ModuleList()
    edge_1 = AddEdge(1.0)
    edge_2 = AddEdge(2.0)
    model.edge_blocks = torch.nn.ModuleList((edge_1, edge_2))

    node_state = torch.zeros(2, 4, 3)
    initial_edge_state = torch.full((3, 4, 3), 10.0)
    _, edge_state = model._run_message_passing(
        node_state,
        initial_edge_state,
        None,
        {
            "edge_distance": None,
            "edge_index": None,
            "partition": None,
        },
        None,
        None,
    )

    torch.testing.assert_close(edge_1.seen[0], initial_edge_state)
    torch.testing.assert_close(edge_2.seen[0], initial_edge_state)
    torch.testing.assert_close(
        edge_state,
        torch.full_like(initial_edge_state, 23.0),
    )


def test_delta_learning_config_round_trip_for_hamiltonian_and_density():
    for loss_target in ("fock_matrix", "density_matrix"):
        workflow = MaloqConfig(
            loss={"loss_target": loss_target, "delta_learning": True}
        ).to_workflow_config()

        assert workflow["loss_target"] == loss_target
        assert workflow["delta_learning"] is True


def test_wandb_tracking_defaults_to_ten_optimizer_steps():
    workflow = MaloqConfig().to_workflow_config()

    assert workflow["wandb_log_every_n_steps"] == 10
    assert workflow["wandb_run_name"] is None
    assert workflow["wandb_group"] is None
    assert workflow["wandb_job_type"] is None
    assert workflow["wandb_tags"] == ()
    assert workflow["experiment_version"] == 1
    assert SplitTrainer._should_log_wandb_step(10, 9, 11, 10)
    assert not SplitTrainer._should_log_wandb_step(9, 8, 11, 10)
    # The epoch summary owns the final step so W&B never receives that step twice.
    assert not SplitTrainer._should_log_wandb_step(10, 9, 10, 10)


def test_nabladft_tracking_identity_is_compact_and_grouped():
    module = _load_nabladft_runner_module()
    identity = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            "head_type": "maloq_muon",
            "scale_and_shift": True,
            "scale_shift_mode": "standardize",
            "output_l_embedding_dim": 128,
            "l_embedding_dim": 128,
            "num_edge_layers": 3,
            "num_mp_layers": 3,
            "seed": 44,
            "experiment_version": 1,
        },
        smoke=False,
    )

    assert identity["experiment_id"] == "nabla-nte128e3-muon-shift-std-v1"
    assert (
        identity["display_name"]
        == "NablaDFT | NTE-128/3 | Muon | SHIFT+STD | V1"
    )
    assert identity["group"] == "nabla-nte128e3-head-ss"
    assert identity["job_type"] == "full"
    assert "scale-shift:on" in identity["tags"]
    assert "normalization:l0-shift-std" in identity["tags"]
    assert "version:v1" in identity["tags"]


def test_nabladft_tracking_identity_distinguishes_shift_only():
    module = _load_nabladft_runner_module()
    identity = module.nabladft_tracking_identity(
        "maloq",
        {
            "head_type": "maloq_muon",
            "scale_and_shift": True,
            "scale_shift_mode": "shift_only",
            "seed": 44,
            "experiment_version": 1,
        },
        smoke=False,
    )

    assert identity["experiment_id"] == "nabla-maloq-muon-shift-v1"
    assert identity["display_name"] == "NablaDFT | MALOQ | Muon | SHIFT | V1"
    assert "normalization:l0-shift-only" in identity["tags"]


def test_nabladft_tracking_identity_names_semantic_global_muon_routing():
    module = _load_nabladft_runner_module()
    identity = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            "head_type": "maloq_semantic_global_muon",
            "scale_and_shift": False,
            "output_l_embedding_dim": 64,
            "l_embedding_dim": 128,
            "num_edge_layers": 2,
            "num_mp_layers": 3,
            "seed": 44,
            "experiment_version": 2,
        },
        smoke=False,
    )

    assert identity["experiment_id"] == (
        "nabla-nte64e2-matrixmuon-auxadamw-sghead-raw-v2"
    )
    assert identity["display_name"] == (
        "NablaDFT | NTE-64/2 | MatrixMuon+AuxAdamW+SGHead | RAW | V2"
    )
    assert "muon-routing:ndim-ge-2" in identity["tags"]
    assert "head-routing:semantic-global" in identity["tags"]


def test_nabladft_tracking_identity_names_semantic_gate_muon_routing():
    module = _load_nabladft_runner_module()
    identity = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            "head_type": "maloq_semantic_global_gate_muon",
            "scale_and_shift": False,
            "output_l_embedding_dim": 64,
            "l_embedding_dim": 128,
            "num_edge_layers": 2,
            "num_mp_layers": 3,
            "seed": 44,
            "experiment_version": 3,
        },
        smoke=False,
    )

    assert identity["experiment_id"] == (
        "nabla-nte64e2-matmuon-sghead-gatemuon-raw-v3"
    )
    assert identity["display_name"] == (
        "NablaDFT | NTE-64/2 | MatMuon+SGHead+GateMuon | RAW | V3"
    )
    assert "head-routing:semantic-global" in identity["tags"]
    assert "gate-optimizer:muon" in identity["tags"]
    assert "gate-routing:semantic-matrix" in identity["tags"]


def test_wandb_tracking_forwards_display_metadata(monkeypatch, tmp_path):
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name=kwargs["name"])

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    workflow = object.__new__(TrainingWorkflow)
    workflow.rank = 0
    workflow.config = {
        "use_wandb": True,
        "wandb_project": "maloq-nablaDFT",
        "wandb_entity": "kaist-korea",
        "wandb_mode": "online",
        "wandb_run_name": "NablaDFT | QHFlow3 | Muon | RAW | V2",
        "wandb_group": "nabla-qhflow3-ss",
        "wandb_job_type": "full",
        "wandb_tags": ("dataset:nabladft", "scale-shift:off"),
        "run_name": "nabla-qhf3-muon-raw-v2",
        "output_folder": str(tmp_path),
    }

    run = workflow.setup_tracking()

    assert run.name == "NablaDFT | QHFlow3 | Muon | RAW | V2"
    assert captured["group"] == "nabla-qhflow3-ss"
    assert captured["job_type"] == "full"
    assert captured["tags"] == ["dataset:nabladft", "scale-shift:off"]


def test_gradient_accumulation_config_and_optimizer_step_count():
    workflow = MaloqConfig(
        optimization={"gradient_accumulation_steps": 2}
    ).to_workflow_config()

    assert workflow["gradient_accumulation_steps"] == 2
    assert SplitTrainer.optimizer_steps_per_epoch(1208, 2) == 604
    assert SplitTrainer.optimizer_steps_per_epoch(3, 2) == 2

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        SplitTrainer.optimizer_steps_per_epoch(1, 0)


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


def test_qm7_loader_selects_explicit_density_target():
    hamiltonian = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    density = torch.tensor([[0.3, 0.1], [0.1, 0.7]])
    row = {"hamiltonian": hamiltonian, "density_matrix": density}

    torch.testing.assert_close(
        torch.from_numpy(_qm7_matrix_target(row, "fock_matrix")),
        hamiltonian,
    )
    torch.testing.assert_close(
        torch.from_numpy(_qm7_matrix_target(row, "density_matrix")),
        density,
    )


def test_qm7_loader_rejects_missing_density_target():
    row = {"hamiltonian": torch.eye(2)}
    try:
        _qm7_matrix_target(row, "density_matrix")
    except KeyError as error:
        assert "density_matrix" in str(error)
    else:
        raise AssertionError("Missing density target should fail explicitly")


def test_qhflow3_native_overlap_bridge_uses_def2_svp_padding():
    batch = SimpleNamespace(
        overlap_matrix=[torch.eye(19).numpy()],
        ptr=torch.tensor([0, 2]),
        atomic_numbers=torch.tensor([1, 6]),
        pos=torch.zeros(2, 3),
    )

    backbone = QHFlow3MaloqBackbone.__new__(QHFlow3MaloqBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.basis = "def2-svp"
    backbone._orbital_masks, backbone.output_matrix_dim = _orbital_masks_for_basis(
        backbone.basis
    )
    blocks = backbone._overlap_blocks(batch)

    assert blocks.shape == (2, 14, 14)
    hydrogen_mask = torch.tensor([0, 1, 3, 4, 5])
    torch.testing.assert_close(
        blocks[0][hydrogen_mask[:, None], hydrogen_mask[None, :]],
        torch.eye(5),
    )
    torch.testing.assert_close(blocks[1], torch.eye(14))


def test_qhflow3_native_matrix_bridge_handles_delta_inputs():
    initial_density = torch.diag(torch.arange(1, 20, dtype=torch.float32))
    batch = SimpleNamespace(
        initial_density_matrix=[initial_density.numpy()],
        ptr=torch.tensor([0, 2]),
        atomic_numbers=torch.tensor([1, 6]),
        pos=torch.zeros(2, 3),
    )

    backbone = QHFlow3MaloqBackbone.__new__(QHFlow3MaloqBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.basis = "def2-svp"
    backbone._orbital_masks, backbone.output_matrix_dim = _orbital_masks_for_basis(
        backbone.basis
    )
    blocks = backbone._matrix_blocks(
        batch,
        "initial_density_matrix",
        "initial density",
    )

    assert blocks.shape == (2, 14, 14)
    hydrogen_mask = torch.tensor([0, 1, 3, 4, 5])
    torch.testing.assert_close(
        blocks[0][hydrogen_mask[:, None], hydrogen_mask[None, :]],
        initial_density[:5, :5],
    )
    torch.testing.assert_close(blocks[1], initial_density[5:, 5:])


def test_qhflow3_nabladft_bridge_uses_32_ao_padding_for_heavy_elements():
    masks, matrix_dim = _orbital_masks_for_basis("def2-svp-nabla")

    assert matrix_dim == 32
    assert masks[1].tolist() == [0, 1, 5, 6, 7]
    assert masks[6].numel() == 14
    assert masks[16].numel() == 18
    assert masks[17].numel() == 18
    assert masks[35].tolist() == list(range(32))

    orbital_count = 5 + 14 + 18 + 32
    batch = SimpleNamespace(
        overlap_matrix=[torch.eye(orbital_count).numpy()],
        ptr=torch.tensor([0, 4]),
        atomic_numbers=torch.tensor([1, 6, 16, 35]),
        pos=torch.zeros(4, 3),
    )
    backbone = QHFlow3MaloqBackbone.__new__(QHFlow3MaloqBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.basis = "def2-svp-nabla"
    backbone._orbital_masks = masks
    backbone.output_matrix_dim = matrix_dim

    blocks = backbone._overlap_blocks(batch)

    assert blocks.shape == (4, 32, 32)
    for atom_index, atomic_number in enumerate((1, 6, 16, 35)):
        mask = masks[atomic_number]
        torch.testing.assert_close(
            blocks[atom_index][mask[:, None], mask[None, :]],
            torch.eye(mask.numel()),
        )


def test_splittrainer_adds_delta_baseline_to_closed_shell_outputs():
    node_delta = torch.tensor([[[0.1, -0.2], [0.3, 0.4]]])
    edge_delta = torch.tensor([[[0.5, -0.6]]])
    batch = SimpleNamespace(
        delta_node_base=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        delta_edge_base=torch.tensor([[5.0, 6.0]]),
    )

    node_final, edge_final = SplitTrainer.apply_delta_baseline(
        node_delta, edge_delta, batch
    )

    torch.testing.assert_close(
        node_final,
        torch.tensor([[[1.1, 1.8], [3.3, 4.4]]]),
    )
    torch.testing.assert_close(edge_final, torch.tensor([[[5.5, 5.4]]]))


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
    assert workflow["qhflow3_grid_resolution"] == 48
    assert workflow["qhflow3_grid_ffn_chunk_size"] == 512
    assert workflow["qhflow3_use_overlap"] is True
    assert workflow["esen_grid_resolution"] is None
    assert workflow["nte_input_conditioning"] == "none"


def test_structural_ablation_config_toggles_round_trip():
    qhflow3 = MaloqConfig(
        model={
            "backbone_type": "qhflow3_clean",
            "qhflow3_use_overlap": False,
        }
    ).to_workflow_config()
    nte = MaloqConfig(
        model={
            "backbone_type": "esen",
            "esen_grid_resolution": 48,
            "nte_input_conditioning": "overlap",
        }
    ).to_workflow_config()

    assert qhflow3["qhflow3_use_overlap"] is False
    assert nte["esen_grid_resolution"] == 48
    assert nte["nte_input_conditioning"] == "overlap"


def test_qhflow3_can_match_nte_default_grid():
    workflow = MaloqConfig(
        model={
            "backbone_type": "qhflow3_clean",
            "qhflow3_grid_resolution": None,
        }
    ).to_workflow_config()
    model = QHFlow3MaloqBackbone(
        sh_lmax=4,
        hidden_size=8,
        bottle_hidden_size=4,
        num_gnn_layers=1,
        num_ham_gnn_layers=1,
        radius_embed_dim=8,
        escn_edge_channels=8,
        escn_num_distance_basis=8,
        grid_resolution=workflow["qhflow3_grid_resolution"],
        basis="def2-svp-nabla",
        use_block_S=False,
        use_block_H=False,
        default_hamiltonian_input="zero",
    )

    grid = model.node_attr_backbone.SO3_grid["lmax_lmax"]
    assert workflow["qhflow3_grid_resolution"] is None
    assert model.grid_resolution is None
    assert grid.lat_resolution == 10
    assert grid.long_resolution == 11


def test_nabladft_tracking_names_structural_ablations():
    module = _load_nabladft_runner_module()
    common = {
        "head_type": "maloq_muon",
        "scale_and_shift": False,
        "l_embedding_dim": 128,
        "output_l_embedding_dim": 64,
        "num_mp_layers": 3,
        "num_edge_layers": 2,
        "seed": 44,
    }

    qhflow3 = module.nabladft_tracking_identity(
        "qhflow3",
        {
            **common,
            "experiment_version": 2,
            "qhflow3_use_overlap": False,
        },
        smoke=False,
    )
    nte = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            **common,
            "experiment_version": 1,
            "esen_grid_resolution": 48,
        },
        smoke=False,
    )

    assert qhflow3["experiment_id"] == "nabla-qhf3-muon-raw-ov0-v2"
    assert qhflow3["display_name"].endswith("RAW | OV0 | V2")
    assert "normalization:none" in qhflow3["tags"]
    assert "overlap:off" in qhflow3["tags"]
    assert nte["experiment_id"] == "nabla-nte64e2-muon-raw-g48-v1"
    assert nte["display_name"].endswith("RAW | Grid48 | V1")
    assert "grid:48x48" in nte["tags"]

    scond = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            **common,
            "experiment_version": 1,
            "nte_input_conditioning": "overlap",
        },
        smoke=False,
    )
    qcond = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            **common,
            "experiment_version": 1,
            "nte_input_conditioning": "qhflow3_exact",
        },
        smoke=False,
    )

    assert scond["experiment_id"] == "nabla-nte64e2-muon-raw-scond-v1"
    assert scond["display_name"].endswith("RAW | Scond | V1")
    assert "conditioning:overlap" in scond["tags"]
    assert qcond["experiment_id"] == "nabla-nte64e2-muon-raw-qcond-v1"
    assert qcond["display_name"].endswith("RAW | QHFcond | V1")
    assert "conditioning:qhflow3-exact" in qcond["tags"]

    node2 = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            **common,
            "experiment_version": 1,
            "nte_input_conditioning": "qhflow3_exact",
            "unscaled_node_layers": (2,),
        },
        smoke=False,
    )
    qhfpair = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            **common,
            "experiment_version": 1,
            "nte_input_conditioning": "qhflow3_exact",
            "edge_stack_mode": "qhflow3_parallel",
        },
        smoke=False,
    )
    nteparallel = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            **common,
            "experiment_version": 1,
            "nte_input_conditioning": "qhflow3_exact",
            "edge_stack_mode": "nte_parallel",
        },
        smoke=False,
    )
    projadamw = module.nabladft_tracking_identity(
        "maloq-nte",
        {
            **common,
            "experiment_version": 1,
            "nte_input_conditioning": "qhflow3_exact",
            "muon_output_projection_policy": "adamw",
        },
        smoke=False,
    )

    assert node2["experiment_id"].endswith("-qcond-n2nols-v1")
    assert node2["display_name"].endswith("QHFcond | N2NoLS | V1")
    assert "unscaled-node-layers:2" in node2["tags"]
    assert qhfpair["experiment_id"].endswith("-qcond-qhfpair-v1")
    assert qhfpair["display_name"].endswith("QHFcond | QHFPair | V1")
    assert "edge-stack:qhflow3-parallel" in qhfpair["tags"]
    assert nteparallel["experiment_id"].endswith("-qcond-ntepair-v1")
    assert nteparallel["display_name"].endswith(
        "QHFcond | NTEParallel | V1"
    )
    assert "edge-stack:nte-parallel-residual" in nteparallel["tags"]
    assert "pair-block-math:nte" in nteparallel["tags"]
    assert projadamw["experiment_id"].endswith("-qcond-projadamw-v1")
    assert projadamw["display_name"].endswith(
        "QHFcond | ProjAdamW | V1"
    )
    assert "output-projection-optimizer:adamw" in projadamw["tags"]


def _qhflow3_equivariance_batch(
    positions: torch.Tensor,
    overlap: torch.Tensor,
    device: torch.device,
    atomic_number: int = 6,
) -> Batch:
    atomic_numbers = torch.tensor(
        [atomic_number, atomic_number], dtype=torch.long, device=device
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    positions = positions.to(device)
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    edge_attr = torch.cat(
        [
            displacement.norm(dim=-1, keepdim=True),
            displacement,
        ],
        dim=-1,
    )
    data = Data(
        pos=positions,
        z=atomic_numbers,
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_atoms_in_molecule=torch.tensor([2], device=device),
        charge=torch.zeros(1, dtype=torch.long, device=device),
        spin_multiplicity=torch.ones(1, dtype=torch.long, device=device),
    )
    batch = Batch.from_data_list([data]).to(device)
    batch.overlap_matrix = [overlap.cpu().numpy()]
    return batch


def _qhflow3_sparse_chain_batch(device: torch.device) -> Batch:
    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.72, -0.18, 0.11],
            [1.51, 0.24, -0.09],
        ],
        dtype=torch.float32,
        device=device,
    )
    atomic_numbers = torch.full((3,), 6, dtype=torch.long, device=device)
    # Directed 0 <-> 1 <-> 2 chain. The deliberately shuffled order also
    # verifies that the QHFlow3 bridge restores the loader's edge order.
    edge_index = torch.tensor(
        [[1, 2, 1, 0], [2, 1, 0, 1]],
        dtype=torch.long,
        device=device,
    )
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    data = Data(
        pos=positions,
        z=atomic_numbers,
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        edge_attr=torch.cat(
            [
                displacement.norm(dim=-1, keepdim=True),
                displacement,
            ],
            dim=-1,
        ),
        num_atoms_in_molecule=torch.tensor([3], device=device),
        charge=torch.zeros(1, dtype=torch.long, device=device),
        spin_multiplicity=torch.ones(1, dtype=torch.long, device=device),
    )
    batch = Batch.from_data_list([data]).to(device)
    atom_block = torch.eye(14, dtype=torch.float32)
    batch.overlap_matrix = [torch.block_diag(*([atom_block] * 3)).numpy()]
    return batch


@pytest.mark.parametrize("mode", ("overlap", "qhflow3_exact"))
def test_nte_matrix_conditioning_is_atom_aligned_and_invariant(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(44)
    conditioner = (
        NTEMatrixConditioning(
            mode=mode,
            basis="def2-svp",
            hidden_size=8,
        )
        .to(device)
        .eval()
    )
    positions = torch.tensor(
        [[0.13, -0.21, 0.08], [0.74, 0.35, -0.42]],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(145)
    blocks = []
    for _ in range(2):
        raw = torch.randn(14, 14, generator=generator)
        blocks.append(raw @ raw.T / 14 + 0.25 * torch.eye(14))
    overlap = torch.block_diag(*blocks)
    atom_embedding = torch.randn(2, 8, generator=generator).to(device)
    base_scalar = torch.randn(2, 8, generator=generator).to(device)
    molecule_indices = torch.zeros(2, dtype=torch.long, device=device)

    def conditioned(matrix):
        batch = _qhflow3_equivariance_batch(
            positions,
            matrix,
            device,
            atomic_number=6,
        )
        return conditioner(
            batch,
            atom_embedding=atom_embedding,
            base_scalar=base_scalar,
            molecule_indices=molecule_indices,
        )

    with torch.no_grad():
        reference = conditioned(overlap)
        changed_blocks = [blocks[0], 1.5 * blocks[1]]
        changed = conditioned(torch.block_diag(*changed_blocks))

        rotation = o3.angles_to_matrix(
            torch.tensor(0.37),
            torch.tensor(1.11),
            torch.tensor(-0.62),
        )
        basis_rotation = Irreps("3x0e + 2x1e + 1x2e").D_from_matrix(rotation)
        rotated = conditioned(
            torch.block_diag(
                *(basis_rotation @ block @ basis_rotation.T for block in blocks)
            )
        )

    torch.testing.assert_close(changed[0], reference[0])
    assert torch.max(torch.abs(changed[1] - reference[1])).item() > 1.0e-5
    torch.testing.assert_close(rotated, reference, atol=2.0e-5, rtol=2.0e-5)
    assert torch.isfinite(reference).all()


def test_qhflow3_accepts_local_sparse_directed_pair_graph():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(44)
    model = (
        QHFlow3MaloqBackbone(
            sh_lmax=2,
            hidden_size=8,
            bottle_hidden_size=4,
            num_gnn_layers=1,
            num_ham_gnn_layers=1,
            max_radius=12.0,
            radius_embed_dim=8,
            escn_edge_channels=8,
            escn_num_distance_basis=8,
            esen_max_radius=15.0,
            basis="def2-svp",
            grid_resolution=12,
            grid_ffn_chunk_size=None,
        )
        .to(device)
        .train()
    )
    batch = _qhflow3_sparse_chain_batch(device)
    original_edge_index = batch.edge_index.clone()

    output = model(batch)

    assert output["node_embeddings"].shape[0] == 3
    assert output["edge_embeddings"].shape[0] == 4
    assert all(torch.isfinite(embedding).all() for embedding in output.values())
    torch.testing.assert_close(batch.edge_index, original_edge_index)

    sum(embedding.square().mean() for embedding in output.values()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_qhflow3_no_overlap_path_does_not_read_overlap_matrix():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(44)
    model = (
        QHFlow3MaloqBackbone(
            sh_lmax=2,
            hidden_size=8,
            bottle_hidden_size=4,
            num_gnn_layers=1,
            num_ham_gnn_layers=1,
            max_radius=12.0,
            radius_embed_dim=8,
            escn_edge_channels=8,
            escn_num_distance_basis=8,
            esen_max_radius=15.0,
            basis="def2-svp",
            grid_resolution=12,
            grid_ffn_chunk_size=None,
            use_block_S=False,
        )
        .to(device)
        .eval()
    )
    batch = _qhflow3_sparse_chain_batch(device)
    batch.overlap_matrix = None

    with torch.no_grad():
        output = model(batch)

    assert model.use_block_S is False
    assert all(torch.isfinite(embedding).all() for embedding in output.values())


def test_qhflow3_matrix_conditioning_is_atom_aligned():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(44)
    model = (
        QHFlow3MaloqBackbone(
            sh_lmax=2,
            hidden_size=8,
            bottle_hidden_size=4,
            num_gnn_layers=1,
            num_ham_gnn_layers=1,
            max_radius=12.0,
            radius_embed_dim=8,
            escn_edge_channels=8,
            escn_num_distance_basis=8,
            esen_max_radius=15.0,
            basis="def2-svp",
            grid_resolution=12,
            grid_ffn_chunk_size=None,
            delta_learning=True,
            delta_target="density_matrix",
        )
        .to(device)
        .eval()
    )
    positions = torch.tensor(
        [[0.13, -0.21, 0.08], [0.74, 0.35, -0.42]],
        dtype=torch.float32,
    )
    atom_block = torch.eye(14)
    matrices = {
        "overlap_matrix": torch.block_diag(atom_block, 2.0 * atom_block),
        "initial_density_matrix": torch.block_diag(
            3.0 * atom_block,
            4.0 * atom_block,
        ),
        "initial_hamiltonian": torch.block_diag(
            5.0 * atom_block,
            6.0 * atom_block,
        ),
    }
    captured_mix_inputs = []

    def capture_mix_input(_module, inputs):
        captured_mix_inputs.append(inputs[0].detach().clone())

    handle = model.node_attr_backbone.mix_matrix.register_forward_pre_hook(
        capture_mix_input
    )
    try:
        with torch.no_grad():
            for changed_attribute in (None, *matrices):
                batch = _qhflow3_equivariance_batch(
                    positions,
                    matrices["overlap_matrix"],
                    device,
                )
                for attribute, matrix in matrices.items():
                    value = matrix.clone()
                    if attribute == changed_attribute:
                        value[14:, 14:] *= 1.5
                    setattr(batch, attribute, [value.cpu().numpy()])
                model(batch)
    finally:
        handle.remove()

    reference = captured_mix_inputs[0]
    assert len(captured_mix_inputs) == 4
    for changed in captured_mix_inputs[1:]:
        torch.testing.assert_close(changed[0], reference[0])
        assert torch.max(torch.abs(changed[1] - reference[1])).item() > 1.0e-5


@pytest.mark.parametrize(
    ("basis", "atomic_number", "basis_irreps", "matrix_dim"),
    (
        ("def2-svp", 6, "3x0e + 2x1e + 1x2e", 14),
        ("def2-svp-nabla", 35, "5x0e + 4x1e + 3x2e", 32),
    ),
)
def test_qhflow3_grid48_backbone_is_equivariant_for_general_rotation(
    basis,
    atomic_number,
    basis_irreps,
    matrix_dim,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(44)
    model = (
        QHFlow3MaloqBackbone(
            sh_lmax=4,
            hidden_size=8,
            bottle_hidden_size=4,
            num_gnn_layers=1,
            num_ham_gnn_layers=1,
            max_radius=12.0,
            radius_embed_dim=8,
            escn_edge_channels=8,
            escn_num_distance_basis=8,
            esen_max_radius=15.0,
            basis=basis,
        )
        .to(device)
        .eval()
    )
    assert model.grid_resolution == 48
    assert model.grid_ffn_chunk_size == 512
    grid_modules = [
        module for module in model.modules() if isinstance(module, QHFlow3GridAtomwise)
    ]
    assert all(
        getattr(module, "grid_ffn_chunk_size", None) == 512 for module in grid_modules
    )
    # Force the checkpointed branch on this two-node test graph; production
    # keeps the same math but uses chunks of 512 nodes/pairs.
    for module in grid_modules:
        module.grid_ffn_chunk_size = 1

    positions = torch.tensor(
        [[0.13, -0.21, 0.08], [0.74, 0.35, -0.42]],
        dtype=torch.float32,
    )
    basis_irreps = Irreps(basis_irreps)
    generator = torch.Generator().manual_seed(145)
    overlap_blocks = []
    for _ in range(2):
        raw = torch.randn(matrix_dim, matrix_dim, generator=generator)
        overlap_blocks.append(raw @ raw.T / matrix_dim + 0.25 * torch.eye(matrix_dim))
    overlap = torch.block_diag(*overlap_blocks)

    cartesian_rotation = o3.angles_to_matrix(
        torch.tensor(0.37),
        torch.tensor(1.11),
        torch.tensor(-0.62),
    )
    # Positions are stored in physical xyz coordinates, while MALOQ matrix
    # irreps and eSEN spherical features use the internal (y, z, x) axes.
    xyz_to_yzx = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=cartesian_rotation.dtype,
    )
    internal_rotation = xyz_to_yzx @ cartesian_rotation @ xyz_to_yzx.T
    basis_rotation = basis_irreps.D_from_matrix(internal_rotation)
    rotated_overlap = torch.block_diag(
        *(basis_rotation @ block @ basis_rotation.T for block in overlap_blocks)
    )

    with torch.no_grad():
        reference = model(
            _qhflow3_equivariance_batch(positions, overlap, device, atomic_number)
        )
        observed = model(
            _qhflow3_equivariance_batch(
                positions @ cartesian_rotation.T,
                rotated_overlap,
                device,
                atomic_number,
            )
        )

    for embedding_name in ("node_embeddings", "edge_embeddings"):
        for degree in range(5):
            component_slice = slice(degree**2, (degree + 1) ** 2)
            degree_rotation = (
                o3.Irrep(degree, 1).D_from_matrix(internal_rotation).to(device)
            )
            expected = torch.einsum(
                "ij,njc->nic",
                degree_rotation,
                reference[embedding_name][:, component_slice, :],
            )
            torch.testing.assert_close(
                observed[embedding_name][:, component_slice, :],
                expected,
                atol=1.0e-4,
                rtol=1.0e-4,
            )

    model.train()
    training_output = model(
        _qhflow3_equivariance_batch(positions, overlap, device, atomic_number)
    )
    training_loss = sum(
        embedding.square().mean() for embedding in training_output.values()
    )
    training_loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
