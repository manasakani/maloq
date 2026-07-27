"""Feature-owned config schema for configurable NTE/QHFlow3 experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from maloq.core.config import MaloqConfig as CoreMaloqConfig
from maloq.core.config import ModelConfig as CoreModelConfig


FEATURE_SLUG = "nte_qhflow3_composition"
CONFIG_NAMESPACE = f"experimental.{FEATURE_SLUG}"
PROFILE_ID = "core_experiments_v1"

SELECTOR_FIELD_NAMES = (
    "gate_act_type",
    "message_passing_schedule",
    "initial_edge_state_mode",
    "nte_output_projection_mode",
    "output_norm_sharing",
    "use_edge_envelope",
    "use_edge_scalar_modulation",
    "residual_update_scale_mode",
    "residual_update_scale_init",
    "residual_update_scale_log_range",
    "unscaled_node_layers",
    "repeat_system_embedding_each_node_block",
    "node_stack_mode",
    "edge_stack_mode",
    "qhflow3_layer_gaussian_width",
    "qhflow3_layer_grid_ffn_chunk_size",
    "qhflow3_exact_pair_rng_aligned",
    "edge_atom_norm_type",
    "edge_post_residual_norm_type",
    "direct_edgewise_layers",
    "direct_atomwise_layers",
    "edge_atomwise_output_mode",
    "edge_norm1_position",
    "nte_input_conditioning",
)


class ModelConfig(CoreModelConfig):
    """Canonical model config plus this feature's deprecated selector schema."""

    head_type: Literal[
        "maloq",
        "maloq_muon",
        "maloq_semantic_global_gate_muon",
    ] = "maloq"
    gate_act_type: Literal["tanh", "sigmoid"] = "tanh"
    message_passing_schedule: Literal["interleaved", "node_then_edge"] = "interleaved"
    initial_edge_state_mode: Literal["edge_degree", "zero"] = "edge_degree"
    nte_output_projection_mode: Literal["so3_linear", "qhflow3_irrep_linear"] = (
        "so3_linear"
    )
    output_norm_sharing: Literal["shared", "separate"] = "shared"
    use_edge_envelope: bool = False
    use_edge_scalar_modulation: bool = False
    residual_update_scale_mode: Literal["none", "bounded_degree"] = "none"
    residual_update_scale_init: float = 1.0
    residual_update_scale_log_range: float = 0.0
    unscaled_node_layers: tuple[int, ...] = ()
    repeat_system_embedding_each_node_block: bool = False
    node_stack_mode: Literal["nte", "qhflow3_exact"] = "nte"
    edge_stack_mode: Literal[
        "recurrent",
        "nte_parallel",
        "qhflow3_parallel",
        "qhflow3_exact_parallel",
    ] = "recurrent"
    qhflow3_layer_gaussian_width: float = Field(default=2.0, gt=0.0)
    qhflow3_layer_grid_ffn_chunk_size: int | None = Field(
        default=512,
        gt=0,
    )
    qhflow3_exact_pair_rng_aligned: bool = False
    edge_atom_norm_type: (
        Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"] | None
    ) = None
    edge_post_residual_norm_type: (
        Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"] | None
    ) = None
    direct_edgewise_layers: tuple[int, ...] = ()
    direct_atomwise_layers: tuple[int, ...] = ()
    edge_atomwise_output_mode: Literal["residual_scaled", "direct"] = "residual_scaled"
    edge_norm1_position: Literal["post_edgewise", "pre_node"] = "post_edgewise"
    nte_input_conditioning: Literal["none", "overlap", "qhflow3_exact"] = "none"


SELECTOR_DEFAULTS = {
    name: ModelConfig.model_fields[name].default for name in SELECTOR_FIELD_NAMES
}


def with_selector_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a workflow dict with typed feature defaults and layer indices."""
    prepared = SELECTOR_DEFAULTS | dict(config)
    prepared["experimental_feature"] = FEATURE_SLUG
    prepared["experimental_profile"] = PROFILE_ID
    for key in (
        "unscaled_node_layers",
        "direct_edgewise_layers",
        "direct_atomwise_layers",
    ):
        prepared[key] = tuple(int(index) for index in prepared[key])
    return prepared


class MaloqConfig(CoreMaloqConfig):
    """Typed config entry point for this explicit experimental feature."""

    model: ModelConfig = Field(default_factory=ModelConfig)

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_selector_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        raw = dict(data)
        model_payload = raw.get("model", {})
        if not isinstance(model_payload, Mapping):
            return raw
        model = dict(model_payload)
        for key in SELECTOR_FIELD_NAMES:
            if key in raw and key not in model:
                model[key] = raw.pop(key)
        raw["model"] = model
        return raw

    def to_workflow_config(self) -> dict[str, Any]:
        return with_selector_defaults(super().to_workflow_config())


def _normalize_layer_indices(
    config: dict[str, Any],
    key: str,
    *,
    maximum: int,
) -> tuple[int, ...]:
    indices = tuple(int(index) for index in config[key])
    config[key] = indices
    if len(set(indices)) != len(indices):
        raise ValueError(f"{key} must not contain duplicates.")
    if any(index < 1 or index > maximum for index in indices):
        raise ValueError(
            f"{key} must contain 1-based indices within the configured stack."
        )
    return indices


def validate_selector_config(config: dict[str, Any]) -> None:
    """Validate cross-field constraints for the selected experiment profile."""
    if config["gate_act_type"] not in {"tanh", "sigmoid"}:
        raise ValueError("gate_act_type must be 'tanh' or 'sigmoid'.")
    if config["message_passing_schedule"] not in {"interleaved", "node_then_edge"}:
        raise ValueError(
            "message_passing_schedule must be 'interleaved' or 'node_then_edge'."
        )
    if config["initial_edge_state_mode"] not in {"edge_degree", "zero"}:
        raise ValueError("initial_edge_state_mode must be 'edge_degree' or 'zero'.")
    if config["residual_update_scale_mode"] not in {"none", "bounded_degree"}:
        raise ValueError(
            "residual_update_scale_mode must be 'none' or 'bounded_degree'."
        )

    num_node_layers = int(config["num_mp_layers"])
    _normalize_layer_indices(
        config,
        "unscaled_node_layers",
        maximum=num_node_layers,
    )
    if (
        config["repeat_system_embedding_each_node_block"]
        and config["nte_input_conditioning"] != "qhflow3_exact"
    ):
        raise ValueError(
            "repeat_system_embedding_each_node_block requires "
            "nte_input_conditioning='qhflow3_exact'."
        )
    if config["node_stack_mode"] not in {"nte", "qhflow3_exact"}:
        raise ValueError("node_stack_mode must be 'nte' or 'qhflow3_exact'.")
    if config["nte_output_projection_mode"] not in {
        "so3_linear",
        "qhflow3_irrep_linear",
    }:
        raise ValueError(
            "nte_output_projection_mode must be 'so3_linear' or 'qhflow3_irrep_linear'."
        )
    if (
        config["nte_output_projection_mode"] != "so3_linear"
        and config["backbone_type"] != "esen"
    ):
        raise ValueError(
            "nte_output_projection_mode='qhflow3_irrep_linear' requires "
            "backbone_type='esen'."
        )
    if config["output_norm_sharing"] not in {"shared", "separate"}:
        raise ValueError("output_norm_sharing must be 'shared' or 'separate'.")
    if (
        config["output_norm_sharing"] == "separate"
        and config["backbone_type"] != "esen"
    ):
        raise ValueError(
            "output_norm_sharing='separate' requires backbone_type='esen'."
        )
    if (
        config["output_norm_sharing"] == "separate"
        and "matrix" not in config["loss_target"]
    ):
        raise ValueError(
            "output_norm_sharing='separate' requires a matrix loss target "
            "with edge embeddings."
        )
    if config["edge_stack_mode"] not in {
        "recurrent",
        "nte_parallel",
        "qhflow3_parallel",
        "qhflow3_exact_parallel",
    }:
        raise ValueError(
            "edge_stack_mode must be 'recurrent', 'nte_parallel', "
            "'qhflow3_parallel', or 'qhflow3_exact_parallel'."
        )
    if (
        config["output_norm_sharing"] == "separate"
        and config["edge_stack_mode"] == "qhflow3_exact_parallel"
    ):
        raise ValueError(
            "output_norm_sharing='separate' is redundant with "
            "edge_stack_mode='qhflow3_exact_parallel', which already "
            "uses a separate QHFlow3 pair norm."
        )

    exact_qhflow3_layers = (
        config["node_stack_mode"] == "qhflow3_exact"
        or config["edge_stack_mode"] == "qhflow3_exact_parallel"
    )
    if (
        config["qhflow3_exact_pair_rng_aligned"]
        and config["edge_stack_mode"] != "qhflow3_exact_parallel"
    ):
        raise ValueError(
            "qhflow3_exact_pair_rng_aligned requires "
            "edge_stack_mode='qhflow3_exact_parallel'."
        )
    if exact_qhflow3_layers and config["backbone_type"] != "esen":
        raise ValueError(
            "Exact QHFlow3 layer transplants require backbone_type='esen'."
        )
    if exact_qhflow3_layers and config["mlp_type"] != "grid":
        raise ValueError("Exact QHFlow3 layer transplants require mlp_type='grid'.")
    if exact_qhflow3_layers and config["distribute_graphs"]:
        raise ValueError(
            "Exact QHFlow3 layer transplants do not support distributed graph training."
        )
    if float(config["qhflow3_layer_gaussian_width"]) <= 0.0:
        raise ValueError("qhflow3_layer_gaussian_width must be positive.")
    chunk_size = config["qhflow3_layer_grid_ffn_chunk_size"]
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError("qhflow3_layer_grid_ffn_chunk_size must be positive.")
    if (
        config["edge_stack_mode"]
        in {"nte_parallel", "qhflow3_parallel", "qhflow3_exact_parallel"}
        and config["message_passing_schedule"] != "node_then_edge"
    ):
        raise ValueError(
            "Parallel edge stacks require message_passing_schedule='node_then_edge'."
        )
    if (
        config["node_stack_mode"] == "qhflow3_exact"
        and config["message_passing_schedule"] != "node_then_edge"
    ):
        raise ValueError("The exact QHFlow3 node stack requires node_then_edge.")

    valid_norm_types = {None, "layer_norm", "layer_norm_sh", "rms_norm_sh"}
    for key in ("edge_atom_norm_type", "edge_post_residual_norm_type"):
        if config[key] not in valid_norm_types:
            raise ValueError(
                f"{key} must be None, 'layer_norm', 'layer_norm_sh', or 'rms_norm_sh'."
            )

    num_edge_layers = (
        num_node_layers
        if config["num_edge_layers"] is None
        else int(config["num_edge_layers"])
    )
    direct_edgewise = _normalize_layer_indices(
        config,
        "direct_edgewise_layers",
        maximum=num_edge_layers,
    )
    direct_atomwise = _normalize_layer_indices(
        config,
        "direct_atomwise_layers",
        maximum=num_edge_layers,
    )
    if direct_atomwise and config["backbone_type"] != "esen":
        raise ValueError("direct_atomwise_layers requires backbone_type='esen'.")
    if direct_atomwise and "matrix" not in config["loss_target"]:
        raise ValueError(
            "direct_atomwise_layers requires a matrix loss target with edge embeddings."
        )
    if direct_atomwise and config["edge_stack_mode"] in {
        "qhflow3_parallel",
        "qhflow3_exact_parallel",
    }:
        raise ValueError(
            "direct_atomwise_layers requires a native eSEN edge-block "
            "forward path, not a QHFlow3 pair stack."
        )

    if config["initial_edge_state_mode"] == "zero":
        if config["backbone_type"] != "esen":
            raise ValueError(
                "initial_edge_state_mode='zero' requires backbone_type='esen'."
            )
        if "matrix" not in config["loss_target"]:
            raise ValueError(
                "initial_edge_state_mode='zero' requires a matrix loss target."
            )
        if config["message_passing_schedule"] != "node_then_edge":
            raise ValueError(
                "initial_edge_state_mode='zero' requires "
                "message_passing_schedule='node_then_edge'."
            )
        if config["edge_stack_mode"] != "recurrent":
            raise ValueError(
                "initial_edge_state_mode='zero' requires edge_stack_mode='recurrent'."
            )
        if config.get("message_type", "source-target") != "source-target":
            raise ValueError(
                "initial_edge_state_mode='zero' requires message_type='source-target'."
            )
        if 1 in direct_edgewise:
            raise ValueError(
                "initial_edge_state_mode='zero' is redundant with "
                "direct_edgewise_layers containing EdgeBlock 1."
            )

    if config["edge_atomwise_output_mode"] not in {"residual_scaled", "direct"}:
        raise ValueError(
            "edge_atomwise_output_mode must be 'residual_scaled' or 'direct'."
        )
    if config["edge_norm1_position"] not in {"post_edgewise", "pre_node"}:
        raise ValueError("edge_norm1_position must be 'post_edgewise' or 'pre_node'.")
    if config["nte_input_conditioning"] not in {"none", "overlap", "qhflow3_exact"}:
        raise ValueError(
            "nte_input_conditioning must be 'none', 'overlap', or 'qhflow3_exact'."
        )
    if config["nte_input_conditioning"] != "none" and config["backbone_type"] != "esen":
        raise ValueError(
            "nte_input_conditioning is available only for the eSEN backbone."
        )
    if config["nte_input_conditioning"] != "none" and config["distribute_graphs"]:
        raise ValueError(
            "NTE matrix input conditioning requires distribute_graphs=False."
        )


__all__ = [
    "CONFIG_NAMESPACE",
    "FEATURE_SLUG",
    "MaloqConfig",
    "ModelConfig",
    "PROFILE_ID",
    "SELECTOR_DEFAULTS",
    "SELECTOR_FIELD_NAMES",
    "validate_selector_config",
    "with_selector_defaults",
]
