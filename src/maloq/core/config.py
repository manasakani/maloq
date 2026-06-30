"""Pydantic configuration scaffold for MALOQ runs.

This module is intentionally minimal and designed to be extended.
It supports loading YAML/TOML/JSON config files and converting them to the
dictionary shape expected by ``TrainingWorkflow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import json

from pydantic import BaseModel, ConfigDict


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


class MaloqConfig(BaseModel):
    """Typed run configuration for MALOQ training/evaluation."""

    model_config = ConfigDict(extra="allow")

    # Dataset and run identity
    dataset_name: str
    dbpath: str
    output_folder: str = "outputs"
    run_name: str = "run"
    open_shell: bool = False

    # Execution flags
    train_or_eval: Literal["train", "eval"] = "train"
    train_backbone: bool = True
    train_head: bool = True

    # Splits / batching
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    batch_size: int = 1
    shuffle: bool = False
    distribute_graphs: bool = False
    partition_type: str | None = None
    dist_backend: Literal["nccl", "gloo"] = "nccl"

    # Model / geometry
    wigner_backend: Literal["torch", "triton"] = "torch"
    l_embedding_dim: int = 128
    num_distance_basis: int = 128
    num_mp_layers: int = 3
    rcut_orbitals: float = 8.0
    rcut_gaussian: float = 16.0
    gaussian_width: float = 1.0

    # Optimization
    num_epochs: int = 1
    dtype: Literal["float32", "float64"] = "float32"
    lr_init: float = 1e-4
    optimizer_type: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0
    scheduler_type: Literal["plateau", "cosine"] = "cosine"
    eta_min: float = 1e-8
    patience: int = 100
    threshold: float = 1e-8
    step_every_epoch: bool = False

    # Loss / target
    loss_target: str = "fock_matrix"
    train_loss: LossName = "rmse_mse_padded_loss"
    test_loss: LossName = "l1_unpadded_loss"

    # Checkpoint / eval toggles
    save_frequency: int = 10
    restart_backbone: bool = False
    restart_head: bool = False
    restart_optimizer: bool = False
    backbone_checkpoint: str = "backbone.pt"
    head_checkpoint: str = "head.pt"
    scale_and_shift: bool = False
    reduce_edge: bool = False
    reduce_node: bool = True
    reduce_node_intra: bool = True
    compute_total_energy: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> "MaloqConfig":
        """Load a config file from YAML/TOML/JSON."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".yaml", ".yml"}:
            import yaml  # type: ignore

            payload = yaml.safe_load(path.read_text())
        elif suffix == ".toml":
            import tomllib

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

        payload = self.model_dump()
        payload["dtype"] = dtype_map[self.dtype]
        payload["train_loss_fxn"] = loss_map[self.train_loss]
        payload["test_loss_fxn"] = loss_map[self.test_loss]
        payload.pop("train_loss", None)
        payload.pop("test_loss", None)
        return payload