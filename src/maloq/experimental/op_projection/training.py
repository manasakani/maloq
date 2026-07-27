# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Training contracts and matrix-free targets for operator projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .projection import OpProjectionHead


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpProjectionDatasetConfig(_StrictModel):
    dataset_name: Literal["nablaDFT"] = "nablaDFT"
    dbpath: Path
    num_train: int = Field(default=12081, gt=0)
    num_val: int = Field(default=64, gt=0)
    num_test: int = Field(default=0, ge=0)
    train_start: int = Field(default=0, ge=0)
    val_start: int = Field(default=12081, ge=0)


class OpProjectionGraphConfig(_StrictModel):
    local_graph_cutoff: float = Field(default=8.0, gt=0.0)
    operator_cutoff: float = Field(default=16.0, gt=0.0)
    include_self_edges: bool = False


class OpProjectionModelConfig(_StrictModel):
    model_variant: str = "nabladft-ntev2-op-projection"
    node_channels: int = Field(default=128, gt=0)
    hidden_channels: int = Field(default=128, gt=0)
    output_channels: int = Field(default=64, gt=0)
    pair_hidden_channels: int = Field(default=64, gt=0)
    pair_edge_channels: int = Field(default=64, gt=0)
    lmax: int | None = Field(default=None, ge=0)
    infer_lmax_from_basis: bool = True
    num_distance_basis: int = Field(default=512, gt=0)
    num_layers: int = Field(default=3, gt=0)
    pair_projection_chunk_size: int = Field(default=2048, gt=0)


class OpProjectionOperatorConfig(_StrictModel):
    probe_distribution: Literal["rademacher"] = "rademacher"
    train_num_probes: int = Field(default=2, gt=0)
    validation_num_probes: int = Field(default=8, gt=0)
    normalize_action_loss: bool = True


class OpProjectionMatrixMetricsConfig(_StrictModel):
    """Streamed exact-matrix evaluation without dense matrix materialization."""

    enabled: bool = True
    every_n_epochs: int = Field(default=1, gt=0)
    train_samples_per_rank: int = Field(default=1, gt=0)
    validation_scope: Literal["full"] = "full"
    identity_column_chunk_size: int = Field(default=64, gt=0)


class OpProjectionOptimizationConfig(_StrictModel):
    num_epochs: int = Field(default=20, gt=0)
    batch_size_per_rank: int = Field(default=1, gt=0)
    world_size: Literal[2] = 2
    gradient_accumulation_steps: int = Field(default=10, gt=0)
    effective_batch_size: int = Field(default=20, gt=0)
    optimizer_type: Literal["adamw"] = "adamw"
    lr_init: float = Field(default=5.0e-4, gt=0.0)
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = Field(default=1.0e-8, gt=0.0)
    gradient_clip_val: float = Field(default=1.0, gt=0.0)
    scheduler_type: Literal["warmup_polynomial"] = "warmup_polynomial"
    warmup_steps: int = Field(default=1000, ge=0)
    scheduler_power: float = Field(default=1.0, gt=0.0)
    min_lr_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class OpProjectionCheckpointConfig(_StrictModel):
    save_frequency: int = Field(default=1, gt=0)


class OpProjectionRuntimeConfig(_StrictModel):
    dtype: Literal["float32"] = "float32"
    seed: int = 44
    num_workers: Literal[0] = 0
    dist_backend: Literal["nccl"] = "nccl"
    log_every_n_steps: int = Field(default=10, gt=0)


class OpProjectionTrackingConfig(_StrictModel):
    use_wandb: bool = True
    wandb_project: str = "MALOQ-nablaDFT-v2"
    wandb_entity: str = "kaist-korea"
    wandb_mode: str = "online"
    wandb_run_name: str
    wandb_group: str
    wandb_job_type: str = "full"
    wandb_tags: list[str] = Field(default_factory=list)


class OpProjectionTrainingConfig(_StrictModel):
    """Strict feature-local configuration for the experimental trainer."""

    schema_version: Literal[1] = 1
    dataset: OpProjectionDatasetConfig
    graph: OpProjectionGraphConfig
    model: OpProjectionModelConfig
    operator: OpProjectionOperatorConfig
    matrix_metrics: OpProjectionMatrixMetricsConfig
    optimization: OpProjectionOptimizationConfig
    checkpointing: OpProjectionCheckpointConfig
    runtime: OpProjectionRuntimeConfig
    tracking: OpProjectionTrackingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpProjectionTrainingConfig":
        config_path = Path(path)
        with config_path.open() as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"configuration must be a mapping: {config_path}")
        config = cls.model_validate(payload)
        config.validate_contract()
        return config

    def for_scope(
        self,
        scope: Literal["validate", "smoke", "full"],
    ) -> "OpProjectionTrainingConfig":
        payload = self.model_dump(mode="python")
        if scope == "smoke":
            payload["dataset"].update(
                num_train=20,
                num_val=20,
                num_test=0,
                train_start=0,
                val_start=20,
            )
            payload["optimization"]["num_epochs"] = 1
            payload["operator"].update(
                train_num_probes=1,
                validation_num_probes=1,
            )
            payload["matrix_metrics"].update(
                train_samples_per_rank=1,
            )
            payload["tracking"].update(
                use_wandb=False,
                wandb_mode="disabled",
                wandb_job_type="smoke",
            )
        resolved = type(self).model_validate(payload)
        resolved.validate_contract()
        return resolved

    def validate_contract(self) -> None:
        if self.dataset.num_test != 0:
            raise ValueError("op_projection training currently expects num_test=0")
        if self.dataset.train_start != 0:
            raise ValueError("the canonical NablaDFT split must start at row zero")
        if self.dataset.val_start < (
            self.dataset.train_start + self.dataset.num_train
        ):
            raise ValueError("training and validation ranges overlap")
        if self.graph.include_self_edges:
            raise ValueError("onsite AO blocks are represented by node labels")
        if abs(
            self.graph.operator_cutoff - 2.0 * self.graph.local_graph_cutoff
        ) > 1.0e-8:
            raise ValueError(
                "operator cutoff must equal twice the per-atom graph cutoff"
            )
        if not self.model.infer_lmax_from_basis or self.model.lmax is not None:
            raise ValueError("lmax must be inferred from the orbital basis")
        if not self.operator.normalize_action_loss:
            raise ValueError("probe loss must be normalized molecule by molecule")
        opt = self.optimization
        local_train = self.dataset.num_train // opt.world_size
        if self.dataset.num_val % opt.world_size:
            raise ValueError(
                "validation rows must divide exactly across distributed ranks"
            )
        if (
            self.matrix_metrics.enabled
            and self.matrix_metrics.train_samples_per_rank > local_train
        ):
            raise ValueError(
                "exact train matrix metric subset must fit each rank-local dataset"
            )
        effective_batch = (
            opt.batch_size_per_rank
            * opt.gradient_accumulation_steps
            * opt.world_size
        )
        if effective_batch != opt.effective_batch_size or effective_batch != 20:
            raise ValueError(
                "the matched NablaDFT lane requires effective batch size 20; "
                f"got computed={effective_batch}, configured={opt.effective_batch_size}"
            )
        beta1, beta2 = opt.adamw_betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("AdamW betas must lie in [0, 1)")

    def wandb_config(
        self,
        *,
        output_folder: str | Path,
        scope: Literal["smoke", "full"],
    ) -> dict[str, Any]:
        """Return the flat config shape used by canonical NablaDFT runs."""
        return {
            "output_folder": str(output_folder),
            "run_name": self.tracking.wandb_run_name,
            "scope": scope,
            "dataset_name": self.dataset.dataset_name,
            "dbpath": str(self.dataset.dbpath),
            "num_train": self.dataset.num_train,
            "num_val": self.dataset.num_val,
            "num_test": self.dataset.num_test,
            "model_variant": self.model.model_variant,
            "node_channels": self.model.node_channels,
            "hidden_channels": self.model.hidden_channels,
            "output_channels": self.model.output_channels,
            "num_layers": self.model.num_layers,
            "train_num_probes": self.operator.train_num_probes,
            "validation_num_probes": self.operator.validation_num_probes,
            "batch_size": self.optimization.batch_size_per_rank,
            "world_size": self.optimization.world_size,
            "gradient_accumulation_steps": (
                self.optimization.gradient_accumulation_steps
            ),
            "effective_batch_size": self.optimization.effective_batch_size,
            "num_epochs": self.optimization.num_epochs,
            "lr_init": self.optimization.lr_init,
            "weight_decay": self.optimization.weight_decay,
            "seed": self.runtime.seed,
            "wandb_log_every_n_steps": self.runtime.log_every_n_steps,
            "validation_matrix_metrics": self.matrix_metrics.enabled,
            "validation_matrix_metrics_frequency": (
                self.matrix_metrics.every_n_epochs
            ),
            "validation_matrix_metrics_scope": (
                self.matrix_metrics.validation_scope
            ),
            "validation_matrix_metrics_aggregation": "raw-cutoff-ao-entry-micro",
            "validation_matrix_metrics_symmetrized": False,
            "identity_column_chunk_size": (
                self.matrix_metrics.identity_column_chunk_size
            ),
        }


def should_log_optimizer_step(
    *,
    optimizer_step: int,
    optimizer_step_in_epoch: int,
    optimizer_steps_per_epoch: int,
    every_n_steps: int,
) -> bool:
    """Match canonical cadence while reserving the final step for epoch metrics."""
    if min(
        optimizer_step,
        optimizer_step_in_epoch,
        optimizer_steps_per_epoch,
        every_n_steps,
    ) <= 0:
        raise ValueError("optimizer-step cadence values must be positive")
    if optimizer_step_in_epoch > optimizer_steps_per_epoch:
        raise ValueError("optimizer_step_in_epoch exceeds the epoch length")
    return (
        optimizer_step % every_n_steps == 0
        and optimizer_step_in_epoch < optimizer_steps_per_epoch
    )


def exact_matrix_sample_indices(
    dataset_size: int,
    *,
    split: Literal["train", "validation"],
    train_samples_per_rank: int | None = None,
) -> list[int]:
    """Resolve a bounded train subset or every validation sample."""
    if dataset_size <= 0:
        raise ValueError("exact matrix dataset must be non-empty")
    if split == "validation":
        if train_samples_per_rank is not None:
            raise ValueError("validation exact metrics do not accept a subset size")
        return list(range(dataset_size))
    if train_samples_per_rank is None:
        raise ValueError("train exact metrics require a subset size")
    if not 0 < train_samples_per_rank <= dataset_size:
        raise ValueError("train exact subset size must lie in [1, dataset_size]")
    return [
        ((2 * index + 1) * dataset_size) // (2 * train_samples_per_rank)
        for index in range(train_samples_per_rank)
    ]


def identity_column_ranges(
    matrix_size: int,
    configured_chunk_size: int,
) -> list[tuple[int, int]]:
    """Cover every column while forbidding an ``M x M`` chunk for ``M > 1``."""
    if matrix_size <= 0 or configured_chunk_size <= 0:
        raise ValueError("matrix and identity chunk sizes must be positive")
    effective_chunk_size = min(
        configured_chunk_size,
        max(matrix_size - 1, 1),
    )
    return [
        (start, min(start + effective_chunk_size, matrix_size))
        for start in range(0, matrix_size, effective_chunk_size)
    ]


def molecule_ao_bounds(
    head: OpProjectionHead,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return atom- and molecule-level AO prefix sums for a PyG batch."""
    atomic_numbers = batch.atomic_numbers.to(
        device=head.block_matvec.ao_counts.device,
        dtype=torch.long,
    )
    atom_ao_ptr = head.block_matvec.make_ao_ptr(atomic_numbers)
    atom_ptr = batch.ptr.to(device=atom_ao_ptr.device, dtype=torch.long)
    return atom_ao_ptr, atom_ao_ptr.index_select(0, atom_ptr)


@torch.no_grad()
def coupled_label_action(
    head: OpProjectionHead,
    batch,
    probe: torch.Tensor,
) -> torch.Tensor:
    """Apply loader-provided coupled AO labels without building a dense matrix."""
    atomic_numbers = batch.atomic_numbers.to(
        device=probe.device,
        dtype=torch.long,
    )
    atom_ao_ptr = head.block_matvec.make_ao_ptr(atomic_numbers)
    if probe.ndim != 2 or probe.shape[0] != int(atom_ao_ptr[-1].item()):
        raise ValueError("probe shape does not match the batch AO dimension")

    target = torch.zeros_like(probe)
    node_atoms = torch.arange(atomic_numbers.numel(), device=probe.device)
    head.block_matvec.add_coupled_blocks(
        target,
        batch.node_y.detach().to(device=probe.device, dtype=probe.dtype),
        node_atoms,
        node_atoms,
        atomic_numbers,
        atom_ao_ptr,
        probe,
    )
    edge_index = batch.edge_index.reshape(2, -1).to(
        device=probe.device,
        dtype=torch.long,
    )
    head.block_matvec.add_coupled_blocks(
        target,
        batch.y.detach().to(device=probe.device, dtype=probe.dtype),
        edge_index[0],
        edge_index[1],
        atomic_numbers,
        atom_ao_ptr,
        probe,
    )
    return target


def molecule_probe_statistics(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    molecule_ao_ptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute equal-molecule probe MSE and additive validation statistics."""
    if predicted_action.shape != target_action.shape or predicted_action.ndim != 2:
        raise ValueError("operator actions must share shape [total_ao, probes]")
    if molecule_ao_ptr.ndim != 1 or molecule_ao_ptr.numel() < 2:
        raise ValueError("molecule_ao_ptr must contain at least two boundaries")
    if int(molecule_ao_ptr[0].item()) != 0:
        raise ValueError("molecule AO boundaries must start at zero")
    if int(molecule_ao_ptr[-1].item()) != predicted_action.shape[0]:
        raise ValueError("molecule AO boundaries do not cover the action")

    difference = predicted_action - target_action
    num_probes = predicted_action.shape[1]
    per_molecule = []
    for start_value, stop_value in zip(
        molecule_ao_ptr[:-1],
        molecule_ao_ptr[1:],
        strict=True,
    ):
        start = int(start_value.item())
        stop = int(stop_value.item())
        num_ao = stop - start
        if num_ao <= 0:
            raise ValueError("every molecule must contain at least one AO")
        per_molecule.append(
            difference[start:stop].square().sum() / (num_probes * num_ao**2)
        )

    normalized_sum = torch.stack(per_molecule).sum()
    squared_error_sum = difference.square().sum()
    target_squared_sum = target_action.square().sum()
    return (
        normalized_sum / len(per_molecule),
        squared_error_sum,
        target_squared_sum,
        len(per_molecule),
    )


def matrix_column_error_sums(
    predicted_columns: torch.Tensor,
    target_columns: torch.Tensor,
    *,
    column_start: int,
    matrix_size: int,
) -> dict[str, torch.Tensor | int]:
    """Return additive exact statistics for contiguous identity columns.

    A callback evaluated on ``I[:, start:stop]`` returns exactly those matrix
    columns. The returned sums can be accumulated without storing either full
    matrix.
    """
    if (
        predicted_columns.shape != target_columns.shape
        or predicted_columns.ndim != 2
    ):
        raise ValueError("matrix column actions must share shape [rows, columns]")
    if matrix_size <= 0 or predicted_columns.shape[0] != matrix_size:
        raise ValueError("matrix column actions do not match matrix_size")
    num_columns = predicted_columns.shape[1]
    column_stop = int(column_start) + num_columns
    if column_start < 0 or column_stop > matrix_size:
        raise ValueError("matrix column range is outside the square matrix")

    difference = predicted_columns - target_columns
    absolute_difference = difference.abs()
    local_columns = torch.arange(num_columns, device=difference.device)
    global_columns = torch.arange(
        column_start,
        column_stop,
        device=difference.device,
    )
    diagonal_absolute_error = absolute_difference[
        global_columns,
        local_columns,
    ].sum()
    absolute_error = absolute_difference.sum()
    return {
        "squared_error_sum": difference.square().sum(),
        "absolute_error_sum": absolute_error,
        "target_squared_sum": target_columns.square().sum(),
        "diagonal_absolute_error_sum": diagonal_absolute_error,
        "off_diagonal_absolute_error_sum": (
            absolute_error - diagonal_absolute_error
        ),
        "entry_count": matrix_size * num_columns,
        "diagonal_entry_count": num_columns,
        "off_diagonal_entry_count": (matrix_size - 1) * num_columns,
    }


def deterministic_probe_seed(
    base_seed: int,
    *,
    epoch: int,
    batch_index: int,
    rank: int,
    validation: bool,
) -> int:
    """Map batch identity to a stable seed without checkpointing RNG state."""
    stream = 1 if validation else 0
    return (
        int(base_seed) * 1_000_003
        + int(epoch) * 100_003
        + int(batch_index) * 101
        + int(rank) * 10_007
        + stream * 1_000_000_007
    ) % (2**63 - 1)


__all__ = [
    "OpProjectionCheckpointConfig",
    "OpProjectionDatasetConfig",
    "OpProjectionGraphConfig",
    "OpProjectionMatrixMetricsConfig",
    "OpProjectionModelConfig",
    "OpProjectionOperatorConfig",
    "OpProjectionOptimizationConfig",
    "OpProjectionRuntimeConfig",
    "OpProjectionTrackingConfig",
    "OpProjectionTrainingConfig",
    "coupled_label_action",
    "deterministic_probe_seed",
    "exact_matrix_sample_indices",
    "identity_column_ranges",
    "matrix_column_error_sums",
    "molecule_ao_bounds",
    "molecule_probe_statistics",
    "should_log_optimizer_step",
]
