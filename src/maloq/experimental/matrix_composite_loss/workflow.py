"""Explicit adapter from a composite-loss profile to TrainingWorkflowV2."""

from __future__ import annotations

from typing import Any

from ...train_utils.training_workflow_v2 import TrainingWorkflowV2Fixed
from .loss import get_composite_loss_profile


def apply_matrix_composite_loss_profile(
    config: dict[str, Any],
    *,
    profile_id: str,
) -> dict[str, Any]:
    """Return a copy with one explicit experimental train-loss profile."""
    profile = get_composite_loss_profile(profile_id)
    workflow_config = dict(config)
    workflow_config.update(
        train_loss_fxn=profile.loss,
        matrix_composite_loss_profile=profile.id,
        matrix_composite_loss_formula=profile.formula,
        matrix_composite_loss_scale=profile.scale,
        matrix_composite_loss_callable=(
            f"{profile.loss.__module__}.{profile.loss.__name__}"
        ),
        matrix_composite_loss_space="masked_coupled_irrep_components",
        matrix_composite_loss_coordinate_invariance=(
            "componentwise_mae_coordinate_dependent"
        ),
    )
    return workflow_config


def build_matrix_composite_loss_workflow(
    config: dict[str, Any],
    *,
    profile_id: str,
) -> TrainingWorkflowV2Fixed:
    """Build the matched V2 workflow with only its train loss replaced."""
    return TrainingWorkflowV2Fixed(
        apply_matrix_composite_loss_profile(
            config,
            profile_id=profile_id,
        )
    )
