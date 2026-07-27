from __future__ import annotations

import pytest
import torch
from e3nn.o3 import Irreps

from maloq.core.config import MaloqConfig as CoreMaloqConfig
from maloq.experimental.nte_qhflow3_composition.config import (
    MaloqConfig as ExperimentalMaloqConfig,
)
from maloq.experimental.nte_qhflow3_composition.backbone import (
    ConfigurableNTEBackbone,
)
from maloq.experimental.nte_qhflow3_composition.workflow import (
    TrainingWorkflow as ExperimentalTrainingWorkflow,
)
from maloq.helm.esen_osh import eSEN_Backbone
from maloq.train_utils.training_workflow import (
    TrainingWorkflow as CanonicalTrainingWorkflow,
)


def _validate(
    workflow_type,
    *,
    feature_config: bool | None = None,
    **model_options,
):
    if feature_config is None:
        feature_config = issubclass(
            workflow_type,
            ExperimentalTrainingWorkflow,
        )
    config_type = ExperimentalMaloqConfig if feature_config else CoreMaloqConfig
    workflow = object.__new__(workflow_type)
    workflow.config = config_type(model=model_options).to_workflow_config()
    workflow.rank = 1
    workflow.world_size = 1
    workflow.device = torch.device("cpu")
    workflow.check_input_config()
    return workflow.config


def test_canonical_workflow_accepts_original_maloq_shape() -> None:
    config = _validate(
        CanonicalTrainingWorkflow,
        backbone_type="esen",
        num_mp_layers=3,
        num_edge_layers=3,
    )
    assert config["backbone_type"] == "esen"


def test_core_config_rejects_unmarked_nested_selector() -> None:
    with pytest.raises(ValueError, match="message_passing_schedule"):
        CoreMaloqConfig(
            model={"message_passing_schedule": "node_then_edge"}
        )


def test_core_config_rejects_unmarked_flat_selector() -> None:
    with pytest.raises(ValueError, match="direct_edgewise_layers"):
        CoreMaloqConfig(direct_edgewise_layers=(1,))


def test_feature_config_rejects_removed_unclaimed_selector() -> None:
    with pytest.raises(ValueError, match="initial_edge_degree_envelope"):
        ExperimentalMaloqConfig(
            model={"initial_edge_degree_envelope": True}
        )


def test_canonical_workflow_rejects_selector_based_esen() -> None:
    with pytest.raises(ValueError, match="explicit experimental workflow"):
        _validate(
            CanonicalTrainingWorkflow,
            feature_config=True,
            backbone_type="esen",
            num_mp_layers=2,
            num_edge_layers=2,
            message_passing_schedule="node_then_edge",
            direct_edgewise_layers=(1,),
        )


def test_feature_workflow_accepts_the_same_selector_configuration() -> None:
    config = _validate(
        ExperimentalTrainingWorkflow,
        backbone_type="esen",
        num_mp_layers=2,
        num_edge_layers=2,
        message_passing_schedule="node_then_edge",
        direct_edgewise_layers=(1,),
    )
    assert config["direct_edgewise_layers"] == (1,)



def test_canonical_default_matches_compatibility_state_and_rng() -> None:
    kwargs = {
        "sphere_channels": 4,
        "hidden_channels": 4,
        "lmax": 1,
        "mmax": 1,
        "cutoff": 8.0,
        "edge_channels": 4,
        "num_layers": 1,
        "num_distance_basis": 4,
    }
    torch.manual_seed(44)
    canonical = eSEN_Backbone(Irreps("1x0e"), **kwargs)
    canonical_rng = torch.get_rng_state().clone()

    torch.manual_seed(44)
    compatibility = ConfigurableNTEBackbone(Irreps("1x0e"), **kwargs)
    compatibility_rng = torch.get_rng_state().clone()

    assert torch.equal(canonical_rng, compatibility_rng)
    canonical_state = canonical.state_dict()
    compatibility_state = compatibility.state_dict()
    assert canonical_state.keys() == compatibility_state.keys()
    for name, value in canonical_state.items():
        assert torch.equal(value, compatibility_state[name]), name
