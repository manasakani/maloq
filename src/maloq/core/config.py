# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Pydantic configuration scaffold for MALOQ runs.

This module is intentionally minimal and designed to be extended.
It supports loading YAML/TOML/JSON config files and converting them to the
dictionary shape expected by ``TrainingWorkflow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


LossName = Literal[
    "mse_padded_loss",
    "l1_padded_loss",
    "rmse_padded_loss",
    "rmse_mse_padded_loss",
    "combined_padded_loss",
    "mse_unpadded_loss",
    "weighted_irrep_mse_loss",
    "l1_unpadded_loss",
    "geometric_mean_loss",
    "combined_unpadded_loss",
]


def _dataset_defaults() -> dict[str, Any]:
    return {
        "dataset_name": "run",
        "dbpath": "",
        "output_folder": "outputs",
        "run_name": "run",
        "open_shell": False,
    }


def _execution_defaults() -> dict[str, Any]:
    return {
        "train_or_eval": "train",
        "train_backbone": True,
        "train_head": True,
        "compute_total_energy": False,
    }


def _split_defaults() -> dict[str, Any]:
    return {
        "num_train": 0,
        "num_val": 0,
        "num_test": 0,
        "batch_size": 1,
        "shuffle": False,
        "distribute_graphs": False,
        "partition_type": None,
        "dist_backend": "nccl",
    }


def _model_defaults() -> dict[str, Any]:
    return {
        "model_variant": "maloq-baseline",
        "backbone_type": "esen",
        "head_type": "maloq",
        "wigner_backend": "torch",
        "l_embedding_dim": 128,
        "num_distance_basis": 128,
        "num_mp_layers": 3,
        "rcut_orbitals": 8.0,
        "rcut_gaussian": 16.0,
        "gaussian_width": 1.0,
        "reduce_edge": False,
        "reduce_node": True,
        "reduce_node_intra": True,
        "gate_act_type": "tanh",
        "mlp_type": "spectral",
        "message_passing_schedule": "interleaved",
        "num_edge_layers": None,
        "output_l_embedding_dim": None,
        "use_edge_envelope": False,
        "use_edge_scalar_modulation": False,
        "residual_update_scale_mode": "none",
        "residual_update_scale_init": 1.0,
        "residual_update_scale_log_range": 0.0,
        "unscaled_node_layers": (),
        "repeat_system_embedding_each_node_block": False,
        "node_stack_mode": "nte",
        "edge_stack_mode": "recurrent",
        "qhflow3_layer_gaussian_width": 2.0,
        "qhflow3_layer_grid_ffn_chunk_size": 512,
        "qhflow3_exact_pair_rng_aligned": False,
        "edge_atom_norm_type": None,
        "edge_post_residual_norm_type": None,
        "direct_edgewise_layers": (),
        "edge_atomwise_output_mode": "residual_scaled",
        "edge_norm1_position": "post_edgewise",
        "esen_grid_resolution": None,
        "nte_input_conditioning": "none",
        "qhflow3_max_radius": 12.0,
        "qhflow3_radius_embed_dim": 32,
        "qhflow3_grid_resolution": 48,
        "qhflow3_grid_ffn_chunk_size": 512,
        "qhflow3_use_overlap": True,
        "qhflow3_muonize_output_projection": False,
        "static_te_init_mode": "zero",
        "static_te_init_std": 1.0,
        "static_te_gate_degrees": (),
        "static_te_gate_activation": "none",
        "static_te_gate_init": 1.0,
    }


def _optimization_defaults() -> dict[str, Any]:
    return {
        "num_epochs": 1,
        "lr_init": 1e-4,
        "optimizer_type": "adam",
        "weight_decay": 0.0,
        "soap_lr": None,
        "soap_betas": (0.95, 0.95),
        "soap_shampoo_beta": -1.0,
        "soap_eps": 1e-8,
        "soap_precondition_frequency": 10,
        "soap_max_precondition_dim": 256,
        "soap_precondition_1d": False,
        "soap_normalize_grads": False,
        "muon_lr": 2e-2,
        "muon_momentum": 0.95,
        "muon_nesterov": True,
        "muon_ns_steps": 5,
        "muon_adamw_lr": None,
        "muon_adamw_betas": (0.9, 0.95),
        "muon_adamw_eps": 1e-10,
        "muon_output_projection_policy": "shape_muon",
        "gradient_clip_val": None,
        "gradient_accumulation_steps": 1,
        "scheduler_type": "cosine",
        "warmup_steps": 1000,
        "scheduler_power": 1.0,
        "min_lr_ratio": 0.0,
        "eta_min": 1e-8,
        "patience": 100,
        "threshold": 1e-8,
        "step_every_epoch": False,
    }


def _loss_defaults() -> dict[str, Any]:
    return {
        "loss_target": "fock_matrix",
        "train_loss": "rmse_mse_padded_loss",
        "test_loss": "l1_unpadded_loss",
        "scale_and_shift": False,
        "scale_shift_mode": "standardize",
        "scale_shift_path": None,
        "delta_learning": False,
    }


def _checkpoint_defaults() -> dict[str, Any]:
    return {
        "save_frequency": 10,
        "restart_backbone": False,
        "restart_head": False,
        "restart_optimizer": False,
        "backbone_checkpoint": "backbone.pt",
        "head_checkpoint": "head.pt",
    }


def _runtime_defaults() -> dict[str, Any]:
    return {"dtype": "float32", "seed": 42}


def _tracking_defaults() -> dict[str, Any]:
    return {
        "use_wandb": False,
        "wandb_project": "maloq",
        "wandb_entity": None,
        "wandb_mode": "online",
        "wandb_run_name": None,
        "wandb_group": None,
        "wandb_job_type": None,
        "wandb_tags": (),
        "experiment_version": 1,
        "wandb_log_every_n_steps": 10,
        "validation_matrix_metrics": False,
        "validation_matrix_metrics_frequency": 1,
    }


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_name: str = "run"
    dbpath: str = ""
    output_folder: str = "outputs"
    run_name: str = "run"
    open_shell: bool = False


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    train_or_eval: Literal["train", "eval"] = "train"
    train_backbone: bool = True
    train_head: bool = True
    compute_total_energy: bool = False


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    batch_size: int = 1
    shuffle: bool = False
    distribute_graphs: bool = False
    partition_type: str | None = None
    dist_backend: Literal["nccl", "gloo"] = "nccl"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_variant: str = "maloq-baseline"
    backbone_type: Literal["esen", "qhflow3_clean"] = "esen"
    head_type: Literal[
        "maloq",
        "maloq_muon",
        "maloq_semantic_global_muon",
        "maloq_semantic_global_gate_muon",
        "static_te",
    ] = "maloq"
    wigner_backend: Literal["torch", "triton"] = "torch"
    l_embedding_dim: int = 128
    num_distance_basis: int = 128
    num_mp_layers: int = 3
    rcut_orbitals: float = 8.0
    rcut_gaussian: float = 16.0
    gaussian_width: float = 1.0
    reduce_edge: bool = False
    reduce_node: bool = True
    reduce_node_intra: bool = True
    gate_act_type: Literal["tanh", "sigmoid"] = "tanh"
    mlp_type: Literal["spectral", "grid"] = "spectral"
    message_passing_schedule: Literal["interleaved", "node_then_edge"] = "interleaved"
    num_edge_layers: int | None = None
    output_l_embedding_dim: int | None = None
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
    qhflow3_layer_grid_ffn_chunk_size: int | None = Field(default=512, gt=0)
    qhflow3_exact_pair_rng_aligned: bool = False
    edge_atom_norm_type: Literal[
        "layer_norm",
        "layer_norm_sh",
        "rms_norm_sh",
    ] | None = None
    edge_post_residual_norm_type: Literal[
        "layer_norm",
        "layer_norm_sh",
        "rms_norm_sh",
    ] | None = None
    direct_edgewise_layers: tuple[int, ...] = ()
    edge_atomwise_output_mode: Literal[
        "residual_scaled",
        "direct",
    ] = "residual_scaled"
    edge_norm1_position: Literal[
        "post_edgewise",
        "pre_node",
    ] = "post_edgewise"
    esen_grid_resolution: int | None = Field(default=None, gt=0)
    nte_input_conditioning: Literal["none", "overlap", "qhflow3_exact"] = "none"
    qhflow3_max_radius: float = 12.0
    qhflow3_radius_embed_dim: int = 32
    qhflow3_grid_resolution: int | None = 48
    qhflow3_grid_ffn_chunk_size: int | None = 512
    qhflow3_use_overlap: bool = True
    qhflow3_muonize_output_projection: bool = False
    static_te_init_mode: Literal["zero", "normal"] = "zero"
    static_te_init_std: float = Field(default=1.0, gt=0.0)
    static_te_gate_degrees: tuple[int, ...] = ()
    static_te_gate_activation: Literal["none", "residual_tanh", "sigmoid"] = "none"
    static_te_gate_init: float = 1.0


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    num_epochs: int = 1
    lr_init: float = 1e-4
    optimizer_type: Literal["adam", "adamw", "soap", "muon"] = "adam"
    weight_decay: float = 0.0
    soap_lr: float | None = None
    soap_betas: tuple[float, float] = (0.95, 0.95)
    soap_shampoo_beta: float = -1.0
    soap_eps: float = 1e-8
    soap_precondition_frequency: int = 10
    soap_max_precondition_dim: int = 256
    soap_precondition_1d: bool = False
    soap_normalize_grads: bool = False
    muon_lr: float = 2e-2
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adamw_lr: float | None = None
    muon_adamw_betas: tuple[float, float] = (0.9, 0.95)
    muon_adamw_eps: float = 1e-10
    muon_output_projection_policy: Literal["shape_muon", "adamw"] = "shape_muon"
    gradient_clip_val: float | None = None
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    scheduler_type: Literal["plateau", "cosine", "warmup_polynomial"] = "cosine"
    warmup_steps: int = 1000
    scheduler_power: float = 1.0
    min_lr_ratio: float = 0.0
    eta_min: float = 1e-8
    patience: int = 100
    threshold: float = 1e-8
    step_every_epoch: bool = False


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    loss_target: str = "fock_matrix"
    train_loss: LossName = "rmse_mse_padded_loss"
    test_loss: LossName = "l1_unpadded_loss"
    scale_and_shift: bool = False
    scale_shift_mode: Literal["standardize", "shift_only"] = "standardize"
    scale_shift_path: str | None = None
    delta_learning: bool = False


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    save_frequency: int = 10
    restart_backbone: bool = False
    restart_head: bool = False
    restart_optimizer: bool = False
    backbone_checkpoint: str = "backbone.pt"
    head_checkpoint: str = "head.pt"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dtype: Literal["float32", "float64"] = "float32"
    seed: int = 42


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    use_wandb: bool = False
    wandb_project: str = "maloq"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline"] = "online"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None
    wandb_tags: tuple[str, ...] = ()
    experiment_version: int = Field(default=1, ge=1)
    wandb_log_every_n_steps: int = 10
    validation_matrix_metrics: bool = False
    validation_matrix_metrics_frequency: int = 1


class MaloqConfig(BaseModel):
    """Typed run configuration for MALOQ training/evaluation."""

    model_config = ConfigDict(extra="allow")

    dataset: DatasetConfig = Field(default_factory=lambda: DatasetConfig(**_dataset_defaults()))
    execution: ExecutionConfig = Field(default_factory=lambda: ExecutionConfig(**_execution_defaults()))
    splits: SplitConfig = Field(default_factory=lambda: SplitConfig(**_split_defaults()))
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(**_model_defaults()))
    optimization: OptimizationConfig = Field(default_factory=lambda: OptimizationConfig(**_optimization_defaults()))
    loss: LossConfig = Field(default_factory=lambda: LossConfig(**_loss_defaults()))
    checkpointing: CheckpointConfig = Field(default_factory=lambda: CheckpointConfig(**_checkpoint_defaults()))
    runtime: RuntimeConfig = Field(default_factory=lambda: RuntimeConfig(**_runtime_defaults()))
    tracking: TrackingConfig = Field(default_factory=lambda: TrackingConfig(**_tracking_defaults()))

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_flat_config(cls, data: Any) -> Any:
        """Allow legacy flat dictionaries as well as nested configs."""
        if not isinstance(data, Mapping):
            return data

        raw = dict(data)
        section_names = ("dataset", "execution", "splits", "model", "optimization", "loss", "checkpointing", "runtime", "tracking")
        nested = {name: dict(raw.pop(name, {}) or {}) for name in section_names}
        # Muon matrix routing is intentionally no longer configurable. Drop
        # the legacy option while loading old experiment files so every run
        # uses the one corrected routing rule.
        raw.pop("muon_parameter_policy", None)
        nested["optimization"].pop("muon_parameter_policy", None)

        flat_to_section = {
            # dataset
            "dataset_name": "dataset",
            "dbpath": "dataset",
            "output_folder": "dataset",
            "run_name": "dataset",
            "open_shell": "dataset",
            # execution
            "train_or_eval": "execution",
            "train_backbone": "execution",
            "train_head": "execution",
            "compute_total_energy": "execution",
            # splits
            "num_train": "splits",
            "num_val": "splits",
            "num_test": "splits",
            "batch_size": "splits",
            "shuffle": "splits",
            "distribute_graphs": "splits",
            "partition_type": "splits",
            "dist_backend": "splits",
            # model
            "model_variant": "model",
            "backbone_type": "model",
            "head_type": "model",
            "wigner_backend": "model",
            "l_embedding_dim": "model",
            "num_distance_basis": "model",
            "num_mp_layers": "model",
            "rcut_orbitals": "model",
            "rcut_gaussian": "model",
            "gaussian_width": "model",
            "reduce_edge": "model",
            "reduce_node": "model",
            "reduce_node_intra": "model",
            "gate_act_type": "model",
            "mlp_type": "model",
            "message_passing_schedule": "model",
            "num_edge_layers": "model",
            "output_l_embedding_dim": "model",
            "use_edge_envelope": "model",
            "use_edge_scalar_modulation": "model",
            "residual_update_scale_mode": "model",
            "residual_update_scale_init": "model",
            "residual_update_scale_log_range": "model",
            "unscaled_node_layers": "model",
            "repeat_system_embedding_each_node_block": "model",
            "node_stack_mode": "model",
            "edge_stack_mode": "model",
            "qhflow3_layer_gaussian_width": "model",
            "qhflow3_layer_grid_ffn_chunk_size": "model",
            "edge_atom_norm_type": "model",
            "edge_post_residual_norm_type": "model",
            "direct_edgewise_layers": "model",
            "edge_atomwise_output_mode": "model",
            "edge_norm1_position": "model",
            "esen_grid_resolution": "model",
            "nte_input_conditioning": "model",
            "qhflow3_max_radius": "model",
            "qhflow3_radius_embed_dim": "model",
            "qhflow3_grid_resolution": "model",
            "qhflow3_grid_ffn_chunk_size": "model",
            "qhflow3_use_overlap": "model",
            "qhflow3_muonize_output_projection": "model",
            "static_te_init_mode": "model",
            "static_te_init_std": "model",
            "static_te_gate_degrees": "model",
            "static_te_gate_activation": "model",
            "static_te_gate_init": "model",
            # optimization
            "num_epochs": "optimization",
            "lr_init": "optimization",
            "optimizer_type": "optimization",
            "weight_decay": "optimization",
            "soap_lr": "optimization",
            "soap_betas": "optimization",
            "soap_shampoo_beta": "optimization",
            "soap_eps": "optimization",
            "soap_precondition_frequency": "optimization",
            "soap_max_precondition_dim": "optimization",
            "soap_precondition_1d": "optimization",
            "soap_normalize_grads": "optimization",
            "muon_lr": "optimization",
            "muon_momentum": "optimization",
            "muon_nesterov": "optimization",
            "muon_ns_steps": "optimization",
            "muon_adamw_lr": "optimization",
            "muon_adamw_betas": "optimization",
            "muon_adamw_eps": "optimization",
            "muon_output_projection_policy": "optimization",
            "gradient_clip_val": "optimization",
            "gradient_accumulation_steps": "optimization",
            "scheduler_type": "optimization",
            "warmup_steps": "optimization",
            "scheduler_power": "optimization",
            "min_lr_ratio": "optimization",
            "eta_min": "optimization",
            "patience": "optimization",
            "threshold": "optimization",
            "step_every_epoch": "optimization",
            # loss
            "loss_target": "loss",
            "train_loss": "loss",
            "test_loss": "loss",
            "scale_and_shift": "loss",
            "scale_shift_mode": "loss",
            "scale_shift_path": "loss",
            "delta_learning": "loss",
            # checkpoint
            "save_frequency": "checkpointing",
            "restart_backbone": "checkpointing",
            "restart_head": "checkpointing",
            "restart_optimizer": "checkpointing",
            "backbone_checkpoint": "checkpointing",
            "head_checkpoint": "checkpointing",
            # runtime
            "dtype": "runtime",
            "seed": "runtime",
            # tracking and validation
            "use_wandb": "tracking",
            "wandb_project": "tracking",
            "wandb_entity": "tracking",
            "wandb_mode": "tracking",
            "wandb_run_name": "tracking",
            "wandb_group": "tracking",
            "wandb_job_type": "tracking",
            "wandb_tags": "tracking",
            "experiment_version": "tracking",
            "wandb_log_every_n_steps": "tracking",
            "validation_matrix_metrics": "tracking",
            "validation_matrix_metrics_frequency": "tracking",
        }

        for flat_key, section_name in flat_to_section.items():
            if flat_key in raw and flat_key not in nested[section_name]:
                nested[section_name][flat_key] = raw.pop(flat_key)

        raw.update(nested)
        return raw

    @classmethod
    def from_file(cls, path: str | Path) -> "MaloqConfig":
        """Load a config file from YAML/TOML/JSON."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".yaml", ".yml"}:
            import yaml  # type: ignore

            payload = yaml.safe_load(path.read_text())
        elif suffix == ".toml":
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
                import tomli as tomllib  # type: ignore

            payload = tomllib.loads(path.read_text())
        elif suffix == ".json":
            payload = json.loads(path.read_text())
        else:
            raise ValueError(f"Unsupported config extension: {suffix}")

        if not isinstance(payload, dict):
            raise ValueError("Configuration file must deserialize to a dictionary")

        return cls(**payload)

    def to_workflow_config(self) -> dict[str, Any]:
        """Convert to the dictionary expected by TrainingWorkflow."""
        import torch
        from ..train_utils import loss as loss_mod

        loss_map = {
            "mse_padded_loss": loss_mod.mse_padded_loss,
            "l1_padded_loss": loss_mod.l1_padded_loss,
            "rmse_padded_loss": loss_mod.rmse_padded_loss,
            "rmse_mse_padded_loss": loss_mod.rmse_mse_padded_loss,
            "combined_padded_loss": loss_mod.combined_padded_loss,
            "mse_unpadded_loss": loss_mod.mse_unpadded_loss,
            "weighted_irrep_mse_loss": loss_mod.weighted_irrep_mse_loss,
            "l1_unpadded_loss": loss_mod.l1_unpadded_loss,
            "geometric_mean_loss": loss_mod.geometric_mean_loss,
            "combined_unpadded_loss": loss_mod.combined_unpadded_loss,
        }

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
        }

        payload: dict[str, Any] = {}
        for section in (
            self.dataset,
            self.execution,
            self.splits,
            self.model,
            self.optimization,
            self.loss,
            self.checkpointing,
            self.runtime,
            self.tracking,
        ):
            payload.update(section.model_dump())

        payload["dtype"] = dtype_map[self.runtime.dtype]
        payload["train_loss_fxn"] = loss_map[self.loss.train_loss]
        payload["test_loss_fxn"] = loss_map[self.loss.test_loss]
        payload.pop("train_loss", None)
        payload.pop("test_loss", None)
        return payload
