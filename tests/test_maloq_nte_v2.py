from __future__ import annotations

import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from maloq.core.config import MaloqConfig
from maloq.helm.esen_block import eSEN_Block
from maloq.helm.esen_block_v2 import (
    EdgeRefinementBlock,
    InitialEdgeBlock,
    NodeBlock,
)
from maloq.experimental.nte_qhflow3_composition.backbone import (
    ConfigurableNTEBackbone,
)
from maloq.helm.esen_osh_v2 import (
    MALOQ_NTE_V2_ARCHITECTURE,
    MaloqNTEV2Backbone,
)
from maloq.train_utils.training_workflow import (
    TrainingWorkflow as CanonicalTrainingWorkflow,
)
from maloq.train_utils.training_workflow_v2 import TrainingWorkflowV2


def _small_v2_backbone(*, num_edge_layers: int = 2) -> MaloqNTEV2Backbone:
    return MaloqNTEV2Backbone(
        Irreps("1x0e"),
        sphere_channels=4,
        hidden_channels=4,
        lmax=1,
        mmax=1,
        cutoff=8.0,
        edge_channels=4,
        num_layers=1,
        num_edge_layers=num_edge_layers,
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


def _legacy_target_shape_backbone() -> ConfigurableNTEBackbone:
    return ConfigurableNTEBackbone(
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
        gate_act_type="sigmoid",
        mlp_type="grid",
        message_passing_schedule="node_then_edge",
        initial_edge_state_mode="edge_degree",
        output_sphere_channels=2,
        nte_output_projection_mode="qhflow3_irrep_linear",
        output_norm_sharing="separate",
        use_edge_envelope=True,
        use_edge_scalar_modulation=True,
        residual_update_scale_mode="bounded_degree",
        residual_update_scale_init=0.015625,
        residual_update_scale_log_range=4.1588830833596715,
        direct_edgewise_layers=(1,),
        direct_atomwise_layers=(2,),
        edge_norm1_position="pre_node",
        input_conditioning="qhflow3_exact",
        conditioning_basis="def2-svp-nabla",
    )


def _validated_workflow(
    backbone_type: str,
    *,
    head_type: str = "maloq",
    optimizer_type: str = "adam",
) -> dict:
    workflow = object.__new__(TrainingWorkflowV2)
    workflow.config = MaloqConfig(
        model={
            "backbone_type": backbone_type,
            "head_type": head_type,
            "num_edge_layers": 2 if backbone_type != "esen" else 3,
            "output_l_embedding_dim": (64 if backbone_type != "esen" else None),
        },
        optimization={"optimizer_type": optimizer_type},
    ).to_workflow_config()
    workflow.rank = 1
    workflow.world_size = 1
    workflow.device = torch.device("cpu")
    workflow.check_input_config()
    return workflow.config


def test_v2_does_not_force_head_optimizer_or_scale_settings() -> None:
    config = _validated_workflow(
        "maloq_nte_v2",
        head_type="maloq",
        optimizer_type="adamw",
    )

    assert config["head_type"] == "maloq"
    assert config["optimizer_type"] == "adamw"
    assert config["scale_and_shift"] is False
    assert config["output_l_embedding_dim"] == 64
    assert "message_passing_schedule" not in config
    assert "direct_edgewise_layers" not in config
    assert "direct_atomwise_layers" not in config
    assert "nte_input_conditioning" not in config


def test_v2_accepts_the_muon_visible_head_as_an_independent_axis() -> None:
    config = _validated_workflow(
        "maloq_nte_v2",
        head_type="maloq_muon",
        optimizer_type="muon",
    )

    assert config["head_type"] == "maloq_muon"
    assert config["optimizer_type"] == "muon"


def test_canonical_workflow_rejects_v2_backbones() -> None:
    workflow = object.__new__(CanonicalTrainingWorkflow)
    workflow.config = MaloqConfig(
        model={
            "backbone_type": "maloq_nte_v2",
            "output_l_embedding_dim": 64,
            "num_edge_layers": 2,
        }
    ).to_workflow_config()
    workflow.rank = 1
    workflow.world_size = 1
    workflow.device = torch.device("cpu")

    try:
        workflow.check_input_config()
    except ValueError as error:
        assert "backbone_type" in str(error)
    else:
        raise AssertionError("Canonical workflow accepted a V2 backbone.")


def test_v2_original_maloq_keeps_historical_defaults() -> None:
    config = _validated_workflow("esen")

    assert config["head_type"] == "maloq"
    assert config["optimizer_type"] == "adam"
    assert "message_passing_schedule" not in config


def test_v2_uses_fixed_block_types_and_initial_edge_envelope() -> None:
    model = _small_v2_backbone()

    assert not issubclass(MaloqNTEV2Backbone, ConfigurableNTEBackbone)
    assert all(
        not issubclass(block_type, eSEN_Block)
        for block_type in (NodeBlock, InitialEdgeBlock, EdgeRefinementBlock)
    )
    assert model.architecture == MALOQ_NTE_V2_ARCHITECTURE
    assert all(isinstance(block, NodeBlock) for block in model.node_blocks)
    assert isinstance(model.edge_blocks[0], InitialEdgeBlock)
    assert isinstance(model.edge_blocks[1], EdgeRefinementBlock)
    assert model.edge_degree_embedding.envelope is not None


def test_v2_supports_a_matched_three_layer_edge_stack() -> None:
    model = _small_v2_backbone(num_edge_layers=3)

    assert model.num_edge_layers == 3
    assert isinstance(model.edge_blocks[0], InitialEdgeBlock)
    assert all(
        isinstance(block, EdgeRefinementBlock) for block in model.edge_blocks[1:]
    )


def test_v2_backbone_runs_the_independent_fixed_path() -> None:
    model = _small_v2_backbone().eval()

    with torch.no_grad():
        output = model(_conditioned_two_atom_batch())

    assert output["node_embeddings"].shape == (2, 4, 2)
    assert output["edge_embeddings"].shape == (2, 4, 2)
    assert torch.isfinite(output["node_embeddings"]).all()
    assert torch.isfinite(output["edge_embeddings"]).all()


def test_v2_edge_blocks_have_the_selected_fixed_equations() -> None:
    class FixedEdgewise(torch.nn.Module):
        def edge_messages(
            self,
            _node_state,
            radial_features,
            *_args,
        ):
            return torch.full_like(radial_features, 3.0)

    class AddOne(torch.nn.Module):
        def forward(self, state):
            return state + 1.0

    class Double(torch.nn.Module):
        def forward(self, state):
            return 2.0 * state

    class Half(torch.nn.Module):
        def forward(self, state):
            return 0.5 * state

    class FailIfCalled(torch.nn.Module):
        def forward(self, _state):
            raise AssertionError("This residual scale is not part of V2.")

    model = _small_v2_backbone()
    first, second = model.edge_blocks
    for block in (first, second):
        block.norm_1 = torch.nn.Identity()
        block.post_residual_norm = torch.nn.Identity()
        block.edge_wise = FixedEdgewise()
        block.atom_wise = Double()

    first.norm_2 = torch.nn.Identity()
    first.edge_update_scale = FailIfCalled()
    first.atom_update_scale = Half()

    second.norm_2 = AddOne()
    second.edge_update_scale = Double()
    second.atom_update_scale = FailIfCalled()

    incoming = torch.full((3, 4, 2), 7.0)
    node_state = torch.zeros(2, 4, 2)
    radial_features = torch.empty_like(incoming)
    args = (radial_features, None, None, None, None)

    first_output = first(node_state, incoming, *args)
    second_output = second(node_state, incoming, *args)

    # EdgeBlock 1: F1 + Sa1 * A1(Norm(F1)) = 3 + 0.5 * 6.
    torch.testing.assert_close(first_output, torch.full_like(incoming, 6.0))
    # EdgeBlock 2: E1 + A2(Norm(Se2 * F2)) = 7 + 2 * (2 * 3 + 1).
    torch.testing.assert_close(second_output, torch.full_like(incoming, 21.0))


def test_v2_preserves_target_legacy_state_dict_keys_and_shapes() -> None:
    torch.manual_seed(17)
    v2 = _small_v2_backbone()
    v2_rng = torch.get_rng_state().clone()

    torch.manual_seed(17)
    legacy = _legacy_target_shape_backbone()
    legacy_rng = torch.get_rng_state().clone()

    assert torch.equal(v2_rng, legacy_rng)
    v2_state = v2.state_dict()
    legacy_state = legacy.state_dict()
    assert v2_state.keys() == legacy_state.keys()
    assert {name: tuple(value.shape) for name, value in v2_state.items()} == {
        name: tuple(value.shape) for name, value in legacy_state.items()
    }
    for name, value in v2_state.items():
        assert torch.equal(value, legacy_state[name]), name

    v2.load_state_dict(legacy_state, strict=True)
    legacy.load_state_dict(v2.state_dict(), strict=True)
