"""Explicit adapter from a composite-loss profile to TrainingWorkflowV2."""

from __future__ import annotations

from typing import Any

from ...train_utils.training_workflow_v2 import TrainingWorkflowV2Fixed
from .loss import get_composite_loss_profile


def build_matrix_composite_loss_workflow(
    config: dict[str, Any],
    *,
    profile_id: str,
) -> TrainingWorkflowV2Fixed:
    """Build the matched V2 workflow with only its train loss replaced."""
    profile = get_composite_loss_profile(profile_id)
    workflow_config = dict(config)
    workflow_config.update(
        train_loss_fxn=profile.loss,
        matrix_composite_loss_profile=profile.id,
        matrix_composite_loss_formula=profile.formula,
        matrix_composite_loss_scale=profile.scale,
    )
    return TrainingWorkflowV2Fixed(workflow_config)
