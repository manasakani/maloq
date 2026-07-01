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
    }


def _optimization_defaults() -> dict[str, Any]:
    return {
        "num_epochs": 1,
        "lr_init": 1e-4,
        "optimizer_type": "adam",
        "weight_decay": 0.0,
        "scheduler_type": "cosine",
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
    return {"dtype": "float32"}


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


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    num_epochs: int = 1
    lr_init: float = 1e-4
    optimizer_type: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0
    scheduler_type: Literal["plateau", "cosine"] = "cosine"
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

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_flat_config(cls, data: Any) -> Any:
        """Allow legacy flat dictionaries as well as nested configs."""
        if not isinstance(data, Mapping):
            return data

        raw = dict(data)
        section_names = ("dataset", "execution", "splits", "model", "optimization", "loss", "checkpointing", "runtime")
        nested = {name: dict(raw.pop(name, {}) or {}) for name in section_names}

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
            # optimization
            "num_epochs": "optimization",
            "lr_init": "optimization",
            "optimizer_type": "optimization",
            "weight_decay": "optimization",
            "scheduler_type": "optimization",
            "eta_min": "optimization",
            "patience": "optimization",
            "threshold": "optimization",
            "step_every_epoch": "optimization",
            # loss
            "loss_target": "loss",
            "train_loss": "loss",
            "test_loss": "loss",
            "scale_and_shift": "loss",
            # checkpoint
            "save_frequency": "checkpointing",
            "restart_backbone": "checkpointing",
            "restart_head": "checkpointing",
            "restart_optimizer": "checkpointing",
            "backbone_checkpoint": "checkpointing",
            "head_checkpoint": "checkpointing",
            # runtime
            "dtype": "runtime",
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
        ):
            payload.update(section.model_dump())

        payload["dtype"] = dtype_map[self.runtime.dtype]
        payload["train_loss_fxn"] = loss_map[self.loss.train_loss]
        payload["test_loss_fxn"] = loss_map[self.loss.test_loss]
        payload.pop("train_loss", None)
        payload.pop("test_loss", None)
        return payload