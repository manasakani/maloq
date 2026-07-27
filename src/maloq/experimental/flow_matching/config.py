"""Feature-owned schema for full-matrix endpoint flow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maloq.core.config import (
    MaloqConfig as CoreMaloqConfig,
    ModelConfig as CoreModelConfig,
)

FEATURE_SLUG = "flow_matching"
CONFIG_NAMESPACE = f"experimental.{FEATURE_SLUG}"
PROFILE_ID = "full_matrix_endpoint_flow_v1"
UPSTREAM_COMMIT = "2b5193785c199dce57db43065142cc9a5759d556"
SUPPORTED_FLOW_BACKBONES = frozenset({"esen", "maloq_nte_v2", "qhflow3"})


class FlowMatchingConfig(BaseModel):
    """Strict endpoint-flow controls derived from active QHFlow2 QH9."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameterization: Literal["clean_endpoint"] = "clean_endpoint"
    prior_type: Literal["coupled_irrep_gaussian"] = "coupled_irrep_gaussian"
    sigma: float = Field(default=0.1, gt=0.0)
    time_distribution: Literal["uniform_per_graph"] = "uniform_per_graph"
    time_min: float = Field(default=0.01, ge=0.0, le=1.0)
    time_max: float = Field(default=0.99, ge=0.0, le=1.0)
    num_ode_steps: int = Field(default=3, ge=1)
    state_scope: Literal["node_and_edge"] = "node_and_edge"
    edge_parameterization: Literal["ode_endpoint"] = "ode_endpoint"
    enforce_hamiltonian_symmetry: Literal[True] = True
    endpoint_loss: Literal["masked_frobenius_mse"] = "masked_frobenius_mse"
    hamiltonian_weight: float = Field(default=10.0, gt=0.0)
    time_scaled_loss: Literal[False] = False
    architecture_version: Literal[2] = 2
    upstream_commit: Literal[UPSTREAM_COMMIT] = UPSTREAM_COMMIT

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> FlowMatchingConfig:
        if self.time_min >= self.time_max:
            raise ValueError("time_min must be strictly smaller than time_max.")
        if self.time_max >= 1.0:
            raise ValueError(
                "Training time_max must stay below one because the sampler "
                "derives (endpoint - state) / (1 - t)."
            )
        return self

    @classmethod
    def from_namespace(cls, value: Mapping[str, Any]) -> FlowMatchingConfig:
        """Validate either a direct payload or a ``flow_matching`` namespace."""
        if FEATURE_SLUG in value:
            namespaced = value[FEATURE_SLUG]
            if not isinstance(namespaced, Mapping):
                raise TypeError("flow_matching must contain a mapping.")
            return cls.model_validate(dict(namespaced))
        return cls.model_validate(dict(value))


class FlowMatchingModelConfig(CoreModelConfig):
    """Feature-scoped model defaults for matched endpoint-flow comparisons."""

    qhflow3_grid_resolution: int | None = None


class EndpointFlowMaloqConfig(CoreMaloqConfig):
    """Canonical typed config extended only by this experimental namespace."""

    model: FlowMatchingModelConfig = Field(default_factory=FlowMatchingModelConfig)
    experimental_feature: Literal[FEATURE_SLUG] = FEATURE_SLUG
    experimental_profile: Literal[PROFILE_ID] = PROFILE_ID
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)

    @model_validator(mode="after")
    def _validate_working_profile(self) -> EndpointFlowMaloqConfig:
        if self.model.backbone_type not in SUPPORTED_FLOW_BACKBONES:
            supported = ", ".join(sorted(SUPPORTED_FLOW_BACKBONES))
            raise ValueError(
                f"Endpoint flow backbone_type must be one of: {supported}."
            )
        if self.dataset.open_shell:
            raise ValueError("Endpoint flow currently supports closed shell only.")
        if self.splits.distribute_graphs:
            raise ValueError("Endpoint flow supports data parallelism only.")
        if self.model.reduce_edge:
            raise ValueError("Full-matrix flow requires both directed edge blocks.")
        if self.loss.loss_target not in {"fock_matrix", "density_matrix"}:
            raise ValueError("Endpoint flow requires a matrix-valued target.")
        if self.loss.scale_and_shift:
            if self.loss.scale_shift_mode != "shift_only":
                raise ValueError("Endpoint flow supports SHIFT-only normalization.")
            if self.loss.scale_shift_path is None:
                raise ValueError("SHIFT endpoint flow requires scale_shift_path.")
        if self.loss.delta_learning:
            raise ValueError(
                "The working endpoint-flow profile is direct; the upstream "
                "QHFlow2 residual parameterization has not been ported."
            )
        if self.loss.compute_uncoupled_loss:
            raise ValueError("Endpoint flow trains in the coupled irrep basis.")
        if self.tracking.validation_matrix_metrics_frequency < 1:
            raise ValueError(
                "validation_matrix_metrics_frequency must be at least one."
            )
        return self

    def to_workflow_config(self) -> dict[str, Any]:
        payload = super().to_workflow_config()
        payload.update(
            experimental_feature=self.experimental_feature,
            experimental_profile=self.experimental_profile,
            flow_matching=self.flow_matching.model_dump(mode="python"),
        )
        return payload
