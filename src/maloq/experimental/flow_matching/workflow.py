"""Inherited NablaDFT workflow for full-matrix endpoint flow."""

from __future__ import annotations

from collections.abc import Mapping

from maloq.train_utils.training_workflow_v2 import TrainingWorkflowV2Fixed

from .backbone import FlowConditionedBackbone
from .config import (
    FEATURE_SLUG,
    PROFILE_ID,
    SUPPORTED_FLOW_BACKBONES,
    FlowMatchingConfig,
)
from .trainer import EndpointFlowTrainer


class FlowMatchingWorkflow(TrainingWorkflowV2Fixed):
    """Use canonical data/model/checkpoint paths with a flow trainer factory."""

    DEFAULTS = TrainingWorkflowV2Fixed.DEFAULTS | {
        "experimental_feature": FEATURE_SLUG,
        "experimental_profile": PROFILE_ID,
        "qhflow3_grid_resolution": None,
        FEATURE_SLUG: FlowMatchingConfig().model_dump(mode="python"),
    }

    def _validate_backbone_feature_config(self) -> None:
        selected = self.config.pop("experimental_feature", None)
        try:
            super()._validate_backbone_feature_config()
        finally:
            self.config["experimental_feature"] = selected
        if selected != FEATURE_SLUG:
            raise ValueError(
                f"This workflow requires experimental_feature={FEATURE_SLUG!r}."
            )
        if self.config.get("experimental_profile") != PROFILE_ID:
            raise ValueError(
                f"This workflow requires experimental_profile={PROFILE_ID!r}."
            )
        raw = self.config.get(FEATURE_SLUG)
        if not isinstance(raw, Mapping):
            raise TypeError("flow_matching must contain a mapping.")
        self.flow_matching_config = FlowMatchingConfig.model_validate(raw)

        c = self.config
        if c["backbone_type"] not in SUPPORTED_FLOW_BACKBONES:
            supported = ", ".join(sorted(SUPPORTED_FLOW_BACKBONES))
            raise ValueError(
                f"Endpoint flow backbone_type must be one of: {supported}."
            )
        if c.get("reduce_edge", False):
            raise ValueError("Full-matrix flow requires both directed edge blocks.")
        if c["open_shell"]:
            raise ValueError("Endpoint flow currently supports closed shell only.")
        if c["loss_target"] not in {"fock_matrix", "density_matrix"}:
            raise ValueError("Endpoint flow requires a matrix target.")
        if c.get("scale_and_shift", False):
            if c.get("scale_shift_mode") != "shift_only":
                raise ValueError("Endpoint flow supports SHIFT-only normalization.")
            if not c.get("scale_shift_path"):
                raise ValueError("SHIFT endpoint flow requires scale_shift_path.")
        if c.get("delta_learning", False):
            raise ValueError("Residual QHFlow2 parameterization is not yet ported.")
        if c.get("compute_uncoupled_loss", False):
            raise ValueError("Endpoint flow trains in coupled irrep coordinates.")
        if c.get("distribute_graphs", False):
            raise ValueError("Endpoint flow supports data parallelism only.")

    def _build_backbone(self, required_irreps):
        backbone = super()._build_backbone(required_irreps)
        return FlowConditionedBackbone(
            backbone,
            flow_irreps=required_irreps,
            embedding_lmax=required_irreps.lmax,
            embedding_channels=self._head_channels(backbone),
        ).to(self.device)

    def _build_trainer(self, *, backbone, head, head_irreps):
        return EndpointFlowTrainer(
            backbone=backbone,
            head=head,
            head_irreps=head_irreps,
            config=self.flow_matching_config,
            run_name=self.config.get("run_name", "flow_matching"),
            save_frequency=int(self.config.get("save_frequency", 10)),
            wandb_run=self.wandb_run,
            validation_inference_seed=int(self.config.get("seed", 42)),
        )

    def _backbone_summary(self, backbone):
        summary = dict(super()._backbone_summary(backbone))
        backbone_type = self.config.get("backbone_type")
        matched_grid = (
            backbone_type == "maloq_nte_v2"
            and self.config.get("esen_grid_resolution") is None
        ) or (
            backbone_type == "qhflow3"
            and self.config.get("qhflow3_grid_resolution") is None
        )
        summary.update(
            flow_state_scope=self.flow_matching_config.state_scope,
            flow_conditioning=(
                "equivariant_node_and_edge_projection_plus_incident_edge_and_time"
            ),
            flow_hamiltonian_symmetry="node_symmetric_reverse_edge_transpose",
            flow_equivariance_contract="SO(3)",
            flow_parameterization=self.flow_matching_config.parameterization,
            flow_prior=self.flow_matching_config.prior_type,
            flow_ode_steps=self.flow_matching_config.num_ode_steps,
            flow_validation_matrix_metrics="joint_euler_endpoint",
            flow_validation_inference_seed=int(self.config.get("seed", 42)),
            flow_effective_so3_grid_shape=(10, 11) if matched_grid else None,
        )
        return summary


class QHFlow2EndpointWorkflow(FlowMatchingWorkflow):
    """Compatibility name for the first audited QHFlow2-derived workflow."""
