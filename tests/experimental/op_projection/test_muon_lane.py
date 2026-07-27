from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "_my_script/experiment/2026-07-28"
CONFIG_PATH = EXPERIMENT_ROOT / "nabladft_op_projection_muon.yaml"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from op_projection_muon_lane import (  # noqa: E402
    AUXILIARY_GROUP_NAME,
    EXPECTED_WANDB_RUN_NAME,
    MUON_GROUP_NAME,
    OpProjectionMuonOptimizationConfig,
    OpProjectionMuonTrainingConfig,
    build_muon_optimizer,
    optimizer_learning_rates,
    verify_muon_optimizer_state,
)


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(91)
        self.matrix = torch.nn.Parameter(torch.randn(5, 3, generator=generator))
        self.high_rank = torch.nn.Parameter(torch.randn(2, 3, 2, generator=generator))
        self.vector = torch.nn.Parameter(torch.randn(5, generator=generator))
        self.scalar = torch.nn.Parameter(torch.tensor(0.25))
        self.frozen = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)


def test_muon_config_is_strict_and_preserved_for_smoke() -> None:
    config = OpProjectionMuonTrainingConfig.from_yaml(CONFIG_PATH)
    smoke = config.for_scope("smoke")

    assert isinstance(config.optimization, OpProjectionMuonOptimizationConfig)
    assert isinstance(smoke, OpProjectionMuonTrainingConfig)
    assert isinstance(smoke.optimization, OpProjectionMuonOptimizationConfig)
    assert smoke.dataset.num_train == 20
    assert smoke.dataset.num_val == 20
    assert smoke.optimization.optimizer_type == "muon"

    payload = config.model_dump(mode="python")
    payload["optimization"]["optimizer_type"] = "adamw"
    with pytest.raises(ValidationError):
        OpProjectionMuonTrainingConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    payload["optimization"]["unknown_optimizer_knob"] = 1
    with pytest.raises(ValidationError):
        OpProjectionMuonTrainingConfig.model_validate(payload)


def test_muon_wandb_config_is_flat_and_names_both_learning_rates() -> None:
    config = OpProjectionMuonTrainingConfig.from_yaml(CONFIG_PATH)
    payload = config.wandb_config(output_folder="/tmp/op-projection-muon", scope="full")

    assert config.tracking.wandb_run_name == EXPECTED_WANDB_RUN_NAME
    assert payload["optimizer_type"] == "muon"
    assert payload["muon_lr"] == pytest.approx(0.02)
    assert payload["aux_adamw_lr"] == pytest.approx(0.0005)
    assert payload["validation_matrix_metrics_scope"] == "full"
    assert all(not isinstance(value, dict) for value in payload.values())


def test_shape_routing_is_exact_and_optimizer_state_round_trips() -> None:
    optimization = OpProjectionMuonOptimizationConfig(
        num_epochs=1,
        batch_size_per_rank=5,
        gradient_accumulation_steps=2,
        effective_batch_size=20,
        optimizer_type="muon",
        lr_init=5.0e-4,
        weight_decay=1.0e-4,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1.0e-10,
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
    )
    model = _ToyModel()
    optimizer, provenance = build_muon_optimizer(model, optimization)

    routed = {record["name"]: record["group"] for record in provenance["parameters"]}
    assert routed == {
        "matrix": MUON_GROUP_NAME,
        "high_rank": MUON_GROUP_NAME,
        "vector": AUXILIARY_GROUP_NAME,
        "scalar": AUXILIARY_GROUP_NAME,
    }
    assert provenance["groups"][MUON_GROUP_NAME]["tensor_count"] == 2
    assert provenance["groups"][AUXILIARY_GROUP_NAME]["tensor_count"] == 2
    assert "frozen" not in routed
    assert optimizer_learning_rates(optimizer) == {
        "optimizer/learning_rate": pytest.approx(0.02),
        "optimizer/muon_learning_rate": pytest.approx(0.02),
        "optimizer/aux_adamw_learning_rate": pytest.approx(0.0005),
    }

    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    verify_muon_optimizer_state(optimizer)

    restored_model = _ToyModel()
    restored_optimizer, restored_provenance = build_muon_optimizer(
        restored_model,
        optimization,
    )
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    verify_muon_optimizer_state(restored_optimizer)
    assert restored_provenance["routing_sha256"] == provenance["routing_sha256"]
