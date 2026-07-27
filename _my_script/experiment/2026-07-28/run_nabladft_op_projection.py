#!/usr/bin/env python3
"""Train the matrix-free NTE-V2 operator projection on NablaDFT-2k."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
SOURCE_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-28/nabladft_op_projection.yaml"
)
ORBITAL_CACHE = PROJECT_ROOT / "orbital_cache_nablaDFT.pkl"
EXPECTED_DATABASE = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/"
    "hamiltonian_databases/train_2k.db"
)
EXPECTED_DATABASE_BYTES = 15_118_426_112
EXPECTED_DATABASE_ROWS = 12_145
EXPECTED_ORBITAL_CACHE_SHA256 = (
    "4423cced9f770856f31bc99169f19dfca6b25dce54c417b634c2fbcb13d021a3"
)
FULL_COUNTS = {"train": 12081, "val": 64, "test": 0}
SMOKE_COUNTS = {"train": 20, "val": 20, "test": 0}
Scope = Literal["validate", "smoke", "full"]

for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    world_size = int(
        os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
    )
    local_rank = int(
        os.environ.get(
            "OMPI_COMM_WORLD_LOCAL_RANK",
            os.environ.get("LOCAL_RANK", "0"),
        )
    )
    return rank, world_size, local_rank


def _select_rank_local_cuda_before_maloq_import() -> None:
    _, world_size, local_rank = _distributed_context()
    if world_size <= 1:
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Two-rank op_projection training requires CUDA.")
    visible_devices = torch.cuda.device_count()
    device_index = 0 if visible_devices == 1 else local_rank
    if not 0 <= device_index < visible_devices:
        raise SystemExit(
            f"Local rank {local_rank} cannot select one of "
            f"{visible_devices} visible CUDA devices."
        )
    torch.cuda.set_device(device_index)


_select_rank_local_cuda_before_maloq_import()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from e3nn.o3 import Irreps  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from maloq.dataset_utils.get_loader import get_loader  # noqa: E402
from maloq.dataset_utils.nablaDFT_dataset_utils import (  # noqa: E402
    HamiltonianDatabase,
)
from maloq.experimental.op_projection import (  # noqa: E402
    OpProjectionBackbone,
    OpProjectionHead,
    OpProjectionModel,
    OpProjectionTrainingConfig,
    bind_operator_callback,
    coupled_label_action,
    deterministic_probe_seed,
    exact_matrix_sample_indices,
    identity_column_ranges,
    matrix_column_error_sums,
    molecule_ao_bounds,
    molecule_probe_statistics,
    rademacher_probes,
    should_log_optimizer_step,
)
from maloq.fock_utils import basis_sets  # noqa: E402
from maloq.fock_utils.utils_tensor_decomp import e3TensorDecomp  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--scope",
        choices=("validate", "smoke", "full"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_metadata(database_path: Path) -> dict[str, Any]:
    if database_path != EXPECTED_DATABASE:
        raise ValueError(
            f"database must be exactly {EXPECTED_DATABASE}, got {database_path}"
        )
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    database_bytes = database_path.stat().st_size
    if database_bytes != EXPECTED_DATABASE_BYTES:
        raise ValueError(
            f"database size drifted: {database_bytes} != {EXPECTED_DATABASE_BYTES}"
        )
    database = HamiltonianDatabase(str(database_path))
    rows = len(database)
    if rows != EXPECTED_DATABASE_ROWS:
        raise ValueError(f"database row count drifted: {rows} != {EXPECTED_DATABASE_ROWS}")
    atomic_numbers, positions, _, _, hamiltonian, overlap, *_ = database[0]
    if hamiltonian.ndim != 2 or hamiltonian.shape != overlap.shape:
        raise ValueError("row-zero Hamiltonian and overlap shapes disagree")
    return {
        "path": str(database_path),
        "bytes": database_bytes,
        "rows": rows,
        "row0_num_atoms": int(len(atomic_numbers)),
        "row0_positions_shape": list(positions.shape),
        "row0_matrix_shape": list(hamiltonian.shape),
    }


def _orbital_cache() -> dict[str, Any]:
    if not ORBITAL_CACHE.is_file():
        raise FileNotFoundError(ORBITAL_CACHE)
    actual_sha256 = _sha256(ORBITAL_CACHE)
    if actual_sha256 != EXPECTED_ORBITAL_CACHE_SHA256:
        raise ValueError(
            "orbital cache hash drifted: "
            f"{actual_sha256} != {EXPECTED_ORBITAL_CACHE_SHA256}"
        )
    with ORBITAL_CACHE.open("rb") as handle:
        return pickle.load(handle)


def _basis_transform_from_cache(
    cache: dict[str, Any],
    device: torch.device,
) -> tuple[Irreps, e3TensorDecomp]:
    required_irreps = Irreps(cache["req_output_irreps"])
    transform = e3TensorDecomp(
        required_irreps,
        cache["out_js_list"],
        default_dtype_torch=torch.float32,
        if_sort=False,
        device_torch=device,
    )
    return required_irreps, transform


def _normalize_orbital_basis(orbital_basis: dict) -> dict[int, list[int]]:
    normalized = {}
    for atomic_number, orbitals in orbital_basis.items():
        values = orbitals.tolist() if isinstance(orbitals, torch.Tensor) else orbitals
        normalized[int(atomic_number)] = [int(value) for value in values]
    return normalized


def _build_model(
    config: OpProjectionTrainingConfig,
    *,
    required_irreps: Irreps,
    basis_transform: e3TensorDecomp,
    orbital_basis: dict[int, list[int]],
    orbital_template: list,
    ls_list,
    device: torch.device,
) -> OpProjectionModel:
    model_config = config.model
    lmax = required_irreps.lmax
    backbone = OpProjectionBackbone(
        required_irreps,
        sphere_channels=model_config.node_channels,
        hidden_channels=model_config.hidden_channels,
        lmax=lmax,
        mmax=lmax,
        cutoff=config.graph.operator_cutoff,
        edge_channels=model_config.node_channels,
        num_distance_basis=model_config.num_distance_basis,
        num_layers=model_config.num_layers,
        output_sphere_channels=model_config.output_channels,
        conditioning_basis="def2-svp-nabla",
        wigner_backend="torch",
    )
    head = OpProjectionHead(
        required_irreps=required_irreps,
        ls_list=ls_list,
        orbital_basis=orbital_basis,
        orbital_template=orbital_template,
        basis_transformation=basis_transform,
        sphere_channels=model_config.output_channels,
        lmax=lmax,
        mmax=lmax,
        hidden_channels=model_config.pair_hidden_channels,
        edge_channels=model_config.pair_edge_channels,
        num_distance_basis=model_config.num_distance_basis,
        cutoff=config.graph.operator_cutoff,
        pair_chunk_size=model_config.pair_projection_chunk_size,
    )
    return OpProjectionModel(backbone, head).to(device)


def _validate_preview(
    config: OpProjectionTrainingConfig,
    config_path: Path,
) -> dict[str, Any]:
    database_metadata = _database_metadata(config.dataset.dbpath)
    cache = _orbital_cache()
    required_irreps, transform = _basis_transform_from_cache(
        cache,
        torch.device("cpu"),
    )
    model = _build_model(
        config,
        required_irreps=required_irreps,
        basis_transform=transform,
        orbital_basis=_normalize_orbital_basis(
            basis_sets.orbital_basis_def2_svp_nabla
        ),
        orbital_template=cache["orbital_template"],
        ls_list=cache["ls_list"],
        device=torch.device("cpu"),
    )
    result = {
        "lane": "nabladft-ntev2-op-projection",
        "scope": "validate",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "database": database_metadata,
        "orbital_cache": {
            "path": str(ORBITAL_CACHE),
            "sha256": EXPECTED_ORBITAL_CACHE_SHA256,
        },
        "required_irreps_lmax": required_irreps.lmax,
        "required_irreps_dim": required_irreps.dim,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "backbone_parameters": model.backbone.num_params,
        "head_parameters": sum(
            parameter.numel() for parameter in model.operator_head.parameters()
        ),
        "configured_split": {
            "train": config.dataset.num_train,
            "val": config.dataset.num_val,
            "test": config.dataset.num_test,
        },
        "two_rank_consumed_train_rows": (
            config.dataset.num_train // config.optimization.world_size
        )
        * config.optimization.world_size,
        "target_representation": "cutoff-local coupled AO node/pair labels",
        "prediction_constructs_dense_matrix": False,
        "target_action_constructs_dense_matrix": False,
        "exact_matrix_metrics_construct_dense_matrix": False,
        "loader_reads_dense_reference_hamiltonian": True,
        "backbone_consumes_dense_overlap_input": True,
        "resolved_config": config.model_dump(mode="json"),
    }
    del model
    return result


def _setup_distributed(config: OpProjectionTrainingConfig) -> tuple[int, int, torch.device]:
    rank, world_size, local_rank = _distributed_context()
    expected_world_size = config.optimization.world_size
    if world_size != expected_world_size:
        raise RuntimeError(
            f"expected {expected_world_size} MPI ranks, got {world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for smoke/full training")
    visible_devices = torch.cuda.device_count()
    device_index = 0 if visible_devices == 1 else local_rank
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    dist.init_process_group(
        backend=config.runtime.dist_backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return rank, world_size, device


def _rank_range(start: int, count: int, rank: int, world_size: int) -> tuple[int, int]:
    local_count = count // world_size
    local_start = start + rank * local_count
    return local_start, local_start + local_count


def _make_loaders(
    config: OpProjectionTrainingConfig,
    *,
    rank: int,
    world_size: int,
):
    train_start, train_stop = _rank_range(
        config.dataset.train_start,
        config.dataset.num_train,
        rank,
        world_size,
    )
    val_start, val_stop = _rank_range(
        config.dataset.val_start,
        config.dataset.num_val,
        rank,
        world_size,
    )
    database = HamiltonianDatabase(str(config.dataset.dbpath))
    loader_kwargs = {
        "database": database,
        "dataset_name": config.dataset.dataset_name,
        "rcut": config.graph.local_graph_cutoff,
        "batch_size": config.optimization.batch_size_per_rank,
        "dtype": torch.float32,
        "half_edges": False,
        "make_fock_targets": True,
        "scale_shift_data": None,
        "is_open_shell": False,
        "loss_target_string": "fock_matrix",
        "distribute_graphs": False,
        "train_or_eval": "train",
        "delta_learning": False,
        "shuffle": False,
    }
    train_loader, required_irreps, transform, orbital_basis, ls_list = get_loader(
        start_idx=train_start,
        end_idx=train_stop,
        **loader_kwargs,
    )
    val_loader, *_ = get_loader(
        start_idx=val_start,
        end_idx=val_stop,
        **loader_kwargs,
    )
    first_sample = train_loader.dataset[0]
    target_object = first_sample.fock_target_object
    orbital_template = target_object.orbital_template
    ranges = {
        "train": [train_start, train_stop],
        "val": [val_start, val_stop],
    }
    return (
        train_loader,
        val_loader,
        Irreps(required_irreps),
        transform,
        _normalize_orbital_basis(orbital_basis),
        ls_list,
        orbital_template,
        ranges,
    )


def _lr_lambda(config: OpProjectionTrainingConfig, total_steps: int):
    warmup_steps = config.optimization.warmup_steps
    power = config.optimization.scheduler_power
    minimum = config.optimization.min_lr_ratio

    def schedule(step: int) -> float:
        update = step + 1
        if warmup_steps > 0 and update <= warmup_steps:
            return max(minimum, update / warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((update - warmup_steps) / decay_steps, 0.0), 1.0)
        return max(minimum, (1.0 - progress) ** power)

    return schedule


def _gradient_norm(parameters) -> tuple[float, int]:
    squared_norm = 0.0
    tensors = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        tensors += 1
        squared_norm += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared_norm), tensors


def _missing_gradient_parameters(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


def _all_reduce_sum(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def _json_log(output_root: Path, payload: dict[str, Any], rank: int) -> None:
    if rank != 0:
        return
    print(json.dumps(payload, sort_keys=True), flush=True)
    with (output_root / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_checkpoint(
    output_root: Path,
    *,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: OpProjectionTrainingConfig,
    epoch: int,
    global_step: int,
    best_validation_error: float,
    metrics: dict[str, Any],
    is_best: bool,
    rank: int,
) -> Path:
    checkpoint_path = output_root / "checkpoint.pt"
    if rank == 0:
        temporary_path = output_root / f".checkpoint.pt.tmp-{os.getpid()}"
        torch.save(
            {
                "schema_version": 1,
                "epoch_next": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_relative_action_error": best_validation_error,
                "metrics": metrics,
                "config": config.model_dump(mode="json"),
            },
            temporary_path,
        )
        os.replace(temporary_path, checkpoint_path)
        if is_best:
            shutil.copy2(checkpoint_path, output_root / "best.pt")
    dist.barrier()
    return checkpoint_path


def _reload_checkpoint(
    checkpoint_path: Path,
    *,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    expected_epoch_next: int,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("checkpoint schema validation failed")
    if checkpoint.get("epoch_next") != expected_epoch_next:
        raise RuntimeError("checkpoint epoch validation failed")
    model.module.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    dist.barrier()


def _init_wandb(
    config: OpProjectionTrainingConfig,
    rank: int,
    *,
    output_root: Path,
    scope: Literal["smoke", "full"],
):
    run = None
    error = ""
    if rank == 0 and config.tracking.use_wandb:
        try:
            import wandb

            run = wandb.init(
                project=config.tracking.wandb_project,
                entity=config.tracking.wandb_entity,
                mode=config.tracking.wandb_mode,
                name=config.tracking.wandb_run_name,
                group=config.tracking.wandb_group,
                job_type=config.tracking.wandb_job_type,
                tags=list(config.tracking.wandb_tags),
                dir=str(output_root),
                config=config.wandb_config(
                    output_folder=output_root,
                    scope=scope,
                ),
            )
            print(
                "W&B logging enabled: "
                f"{config.tracking.wandb_entity}/"
                f"{config.tracking.wandb_project}/"
                f"{run.id}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - external service failure
            error = repr(exc)
    messages = [error]
    dist.broadcast_object_list(messages, src=0)
    if messages[0]:
        raise RuntimeError(f"W&B initialization failed on rank zero: {messages[0]}")
    return run


@torch.no_grad()
def _validate_epoch(
    model: DistributedDataParallel,
    val_loader,
    *,
    config: OpProjectionTrainingConfig,
    device: torch.device,
    rank: int,
) -> dict[str, float]:
    model.eval()
    head = model.module.operator_head
    local_normalized_sum = 0.0
    local_molecules = 0
    local_squared_error = 0.0
    local_target_squared = 0.0
    for batch_index, batch in enumerate(val_loader):
        batch = batch.to(device)
        _, molecule_ptr = molecule_ao_bounds(head, batch)
        generator = torch.Generator(device=device).manual_seed(
            deterministic_probe_seed(
                config.runtime.seed,
                epoch=0,
                batch_index=batch_index,
                rank=rank,
                validation=True,
            )
        )
        probes = rademacher_probes(
            int(molecule_ptr[-1].item()),
            config.operator.validation_num_probes,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        target_action = coupled_label_action(head, batch, probes)
        predicted_action = model(batch, probes)
        loss, squared_error, target_squared, molecule_count = (
            molecule_probe_statistics(
                predicted_action,
                target_action,
                molecule_ptr,
            )
        )
        local_normalized_sum += float(loss.item()) * molecule_count
        local_molecules += molecule_count
        local_squared_error += float(squared_error.item())
        local_target_squared += float(target_squared.item())

    normalized_sum, molecule_count, squared_error, target_squared = _all_reduce_sum(
        [
            local_normalized_sum,
            float(local_molecules),
            local_squared_error,
            local_target_squared,
        ],
        device,
    )
    probe_mse = normalized_sum / molecule_count
    relative_squared = squared_error / max(target_squared, 1.0e-30)
    return {
        "validation_probe_mse": probe_mse,
        "validation_relative_action_error": relative_squared,
        "validation/probe_matrix_mse_estimate": probe_mse,
        "validation/probe_matrix_rmse_estimate": math.sqrt(max(probe_mse, 0.0)),
        "validation/probe_relative_frobenius_squared_estimate": relative_squared,
        "validation/probe_relative_frobenius_estimate": math.sqrt(
            max(relative_squared, 0.0)
        ),
        "validation_molecules": molecule_count,
    }


@torch.no_grad()
def _exact_matrix_metrics(
    model: DistributedDataParallel,
    loader,
    *,
    split: Literal["train", "validation"],
    samples_per_rank: int | None,
    identity_column_chunk_size: int,
    expected_global_molecules: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure streamed exact cutoff-label matrix errors.

    Each identity chunk exposes exact matrix columns through the callback. Only
    scalar error sums are retained, so neither prediction nor target is ever
    assembled as an ``M x M`` tensor. Training uses a fixed diagnostic subset;
    validation always covers the entire rank-local validation split.
    """
    model.eval()
    head = model.module.operator_head
    local = {
        "squared_error_sum": 0.0,
        "absolute_error_sum": 0.0,
        "target_squared_sum": 0.0,
        "diagonal_absolute_error_sum": 0.0,
        "off_diagonal_absolute_error_sum": 0.0,
        "entry_count": 0.0,
        "diagonal_entry_count": 0.0,
        "off_diagonal_entry_count": 0.0,
        "macro_mse_sum": 0.0,
        "macro_mae_sum": 0.0,
        "macro_relative_frobenius_sum": 0.0,
        "molecule_count": 0.0,
        "ao_dimension_sum": 0.0,
    }
    dataset_size = len(loader.dataset)
    sample_indices = exact_matrix_sample_indices(
        dataset_size,
        split=split,
        train_samples_per_rank=samples_per_rank,
    )
    if split == "validation":
        prefix = "validation_exact"
    else:
        prefix = "train_subset_exact"

    for sample_index in sample_indices:
        batch = Batch.from_data_list([loader.dataset[sample_index]]).to(device)
        _, molecule_ptr = molecule_ao_bounds(head, batch)
        if molecule_ptr.numel() != 2:
            raise RuntimeError("exact matrix evaluator expects one molecule")
        matrix_size = int(molecule_ptr[-1].item())
        features = model.module.encode(batch)
        callback = bind_operator_callback(features, batch, head)
        molecule = {
            "squared_error_sum": 0.0,
            "absolute_error_sum": 0.0,
            "target_squared_sum": 0.0,
            "diagonal_absolute_error_sum": 0.0,
            "off_diagonal_absolute_error_sum": 0.0,
            "entry_count": 0,
            "diagonal_entry_count": 0,
            "off_diagonal_entry_count": 0,
        }
        for column_start, column_stop in identity_column_ranges(
            matrix_size,
            identity_column_chunk_size,
        ):
            num_columns = column_stop - column_start
            identity_columns = torch.zeros(
                matrix_size,
                num_columns,
                device=device,
                dtype=torch.float32,
            )
            local_columns = torch.arange(num_columns, device=device)
            identity_columns[column_start + local_columns, local_columns] = 1.0
            predicted_columns = callback(identity_columns)
            target_columns = coupled_label_action(head, batch, identity_columns)
            statistics = matrix_column_error_sums(
                predicted_columns,
                target_columns,
                column_start=column_start,
                matrix_size=matrix_size,
            )
            for name, value in statistics.items():
                molecule[name] += (
                    float(value.item())
                    if isinstance(value, torch.Tensor)
                    else int(value)
                )

        expected_entries = matrix_size**2
        if molecule["entry_count"] != expected_entries:
            raise RuntimeError("identity chunks did not cover the exact matrix")
        squared_error = float(molecule["squared_error_sum"])
        absolute_error = float(molecule["absolute_error_sum"])
        target_squared = float(molecule["target_squared_sum"])
        local["macro_mse_sum"] += squared_error / expected_entries
        local["macro_mae_sum"] += absolute_error / expected_entries
        local["macro_relative_frobenius_sum"] += math.sqrt(
            squared_error / max(target_squared, 1.0e-30)
        )
        for name in (
            "squared_error_sum",
            "absolute_error_sum",
            "target_squared_sum",
            "diagonal_absolute_error_sum",
            "off_diagonal_absolute_error_sum",
            "entry_count",
            "diagonal_entry_count",
            "off_diagonal_entry_count",
        ):
            local[name] += float(molecule[name])
        local["molecule_count"] += 1.0
        local["ao_dimension_sum"] += matrix_size

    names = list(local)
    reduced_values = _all_reduce_sum([local[name] for name in names], device)
    reduced = dict(zip(names, reduced_values, strict=True))
    molecule_count = reduced["molecule_count"]
    if int(molecule_count) != expected_global_molecules:
        raise RuntimeError("distributed exact matrix molecule count drifted")
    entry_count = reduced["entry_count"]
    squared_error = reduced["squared_error_sum"]
    macro_mse = reduced["macro_mse_sum"] / molecule_count
    micro_mse = squared_error / entry_count
    micro_mae = reduced["absolute_error_sum"] / entry_count
    metrics = {
        f"{prefix}/cutoff_matrix_mse_macro": macro_mse,
        f"{prefix}/cutoff_matrix_rmse_macro": math.sqrt(max(macro_mse, 0.0)),
        f"{prefix}/cutoff_matrix_mae_macro": (
            reduced["macro_mae_sum"] / molecule_count
        ),
        f"{prefix}/cutoff_relative_frobenius_macro": (
            reduced["macro_relative_frobenius_sum"] / molecule_count
        ),
        f"{prefix}/cutoff_matrix_mse_micro": micro_mse,
        f"{prefix}/cutoff_matrix_rmse_micro": math.sqrt(max(micro_mse, 0.0)),
        f"{prefix}/cutoff_matrix_mae_micro": micro_mae,
        f"{prefix}/cutoff_relative_frobenius_micro": math.sqrt(
            squared_error / max(reduced["target_squared_sum"], 1.0e-30)
        ),
        f"{prefix}/ao_diagonal_mae_micro": (
            reduced["diagonal_absolute_error_sum"]
            / reduced["diagonal_entry_count"]
        ),
        f"{prefix}/ao_off_diagonal_mae_micro": (
            reduced["off_diagonal_absolute_error_sum"]
            / max(reduced["off_diagonal_entry_count"], 1.0)
        ),
        f"{prefix}/evaluated_molecules": molecule_count,
        f"{prefix}/expected_molecules": float(expected_global_molecules),
        f"{prefix}/coverage_fraction": (
            molecule_count / expected_global_molecules
        ),
        f"{prefix}/evaluated_entries": entry_count,
        f"{prefix}/mean_ao_dimension": (
            reduced["ao_dimension_sum"] / molecule_count
        ),
    }
    if split == "validation":
        metrics.update(
            {
                "validation/matrix_mae": micro_mae,
                "validation/matrix_mse": micro_mse,
                "validation/matrix_rmse": math.sqrt(max(micro_mse, 0.0)),
                "validation/matrix_evaluated_molecules": molecule_count,
                "validation/matrix_expected_molecules": float(
                    expected_global_molecules
                ),
                "validation/matrix_evaluated_entries": entry_count,
                "validation/matrix_coverage": 1.0,
            }
        )
    return metrics


def _run_training(
    config: OpProjectionTrainingConfig,
    *,
    scope: Literal["smoke", "full"],
    config_path: Path,
    output_root: Path,
) -> None:
    rank, world_size, device = _setup_distributed(config)
    wandb_run = None
    try:
        torch.manual_seed(config.runtime.seed)
        torch.cuda.manual_seed_all(config.runtime.seed)
        np.random.seed(config.runtime.seed + rank)
        random.seed(config.runtime.seed + rank)
        torch.cuda.reset_peak_memory_stats(device)

        if rank == 0:
            output_root.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        database_metadata = _database_metadata(config.dataset.dbpath)
        _orbital_cache()
        (
            train_loader,
            val_loader,
            required_irreps,
            transform,
            orbital_basis,
            ls_list,
            orbital_template,
            local_ranges,
        ) = _make_loaders(config, rank=rank, world_size=world_size)
        accumulation = config.optimization.gradient_accumulation_steps
        if len(train_loader) % accumulation:
            raise RuntimeError(
                "train microbatches must divide exactly by gradient accumulation"
            )

        model = _build_model(
            config,
            required_irreps=required_irreps,
            basis_transform=transform,
            orbital_basis=orbital_basis,
            orbital_template=orbital_template,
            ls_list=ls_list,
            device=device,
        )
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=config.optimization.lr_init,
            weight_decay=config.optimization.weight_decay,
            betas=config.optimization.adamw_betas,
            eps=config.optimization.adamw_eps,
        )
        optimizer_steps_per_epoch = len(train_loader) // accumulation
        total_optimizer_steps = (
            optimizer_steps_per_epoch * config.optimization.num_epochs
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            _lr_lambda(config, total_optimizer_steps),
        )
        wandb_run = _init_wandb(
            config,
            rank,
            output_root=output_root,
            scope=scope,
        )

        gathered_ranges: list[Any] = [None] * world_size
        dist.all_gather_object(gathered_ranges, local_ranges)
        if rank == 0:
            resolved = {
                "scope": scope,
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "orbital_cache_sha256": EXPECTED_ORBITAL_CACHE_SHA256,
                "database": database_metadata,
                "local_rank_ranges": gathered_ranges,
                "configured_train_rows": config.dataset.num_train,
                "consumed_train_rows": (
                    config.dataset.num_train // world_size
                )
                * world_size,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "required_irreps_lmax": required_irreps.lmax,
                "required_irreps_dim": required_irreps.dim,
                "model_parameters": sum(
                    parameter.numel() for parameter in ddp_model.parameters()
                ),
                "prediction_constructs_dense_matrix": False,
                "target_action_constructs_dense_matrix": False,
                "exact_matrix_metrics_construct_dense_matrix": False,
                "loader_reads_dense_reference_hamiltonian": True,
                "backbone_consumes_dense_overlap_input": True,
                "target_representation": "cutoff-local coupled AO node/pair labels",
                "resolved_config": config.model_dump(mode="json"),
            }
            with (output_root / "resolved_config.json").open("w") as handle:
                json.dump(resolved, handle, indent=2, sort_keys=True)
                handle.write("\n")

        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        best_validation_error = math.inf
        max_backbone_grad_norm = 0.0
        max_head_grad_norm = 0.0
        last_epoch_payload: dict[str, Any] = {}
        started = time.perf_counter()

        for epoch in range(config.optimization.num_epochs):
            ddp_model.train()
            epoch_loss_sum = 0.0
            epoch_molecules = 0
            window_loss_sum = 0.0
            window_molecules = 0
            epoch_started = time.perf_counter()
            for batch_index, batch in enumerate(train_loader):
                batch = batch.to(device)
                head = ddp_model.module.operator_head
                _, molecule_ptr = molecule_ao_bounds(head, batch)
                generator = torch.Generator(device=device).manual_seed(
                    deterministic_probe_seed(
                        config.runtime.seed,
                        epoch=epoch,
                        batch_index=batch_index,
                        rank=rank,
                        validation=False,
                    )
                )
                probes = rademacher_probes(
                    int(molecule_ptr[-1].item()),
                    config.operator.train_num_probes,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                target_action = coupled_label_action(head, batch, probes)
                sync_gradients = (batch_index + 1) % accumulation == 0
                sync_context = (
                    contextlib.nullcontext()
                    if sync_gradients
                    else ddp_model.no_sync()
                )
                with sync_context:
                    predicted_action = ddp_model(batch, probes)
                    loss, _, _, molecule_count = molecule_probe_statistics(
                        predicted_action,
                        target_action,
                        molecule_ptr,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"non-finite loss at epoch={epoch}, batch={batch_index}"
                        )
                    (loss / accumulation).backward()

                loss_value = float(loss.detach().item())
                epoch_loss_sum += loss_value * molecule_count
                epoch_molecules += molecule_count
                window_loss_sum += loss_value * molecule_count
                window_molecules += molecule_count
                if not sync_gradients:
                    continue

                backbone_norm, backbone_grad_tensors = _gradient_norm(
                    ddp_model.module.backbone.parameters()
                )
                head_norm, head_grad_tensors = _gradient_norm(
                    ddp_model.module.operator_head.parameters()
                )
                if global_step == 0:
                    missing_gradients = _missing_gradient_parameters(
                        ddp_model.module
                    )
                    if missing_gradients:
                        raise RuntimeError(
                            "trainable parameters missing gradients: "
                            + ", ".join(missing_gradients)
                        )
                max_backbone_grad_norm = max(max_backbone_grad_norm, backbone_norm)
                max_head_grad_norm = max(max_head_grad_norm, head_norm)
                torch.nn.utils.clip_grad_norm_(
                    ddp_model.parameters(),
                    config.optimization.gradient_clip_val,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                optimizer_step_in_epoch = (batch_index + 1) // accumulation
                if should_log_optimizer_step(
                    optimizer_step=global_step,
                    optimizer_step_in_epoch=optimizer_step_in_epoch,
                    optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                    every_n_steps=config.runtime.log_every_n_steps,
                ):
                    global_loss_sum, global_molecules = _all_reduce_sum(
                        [window_loss_sum, float(window_molecules)],
                        device,
                    )
                    train_probe_mse = global_loss_sum / global_molecules
                    wandb_payload = {
                        "optimizer_step": global_step,
                        "epoch": epoch + 1,
                        "micro_batch_in_epoch": batch_index + 1,
                        "optimizer_step_in_epoch": optimizer_step_in_epoch,
                        "train_step/total_loss": train_probe_mse,
                        "train_step/probe_matrix_mse_estimate": train_probe_mse,
                        "train_step/probe_matrix_rmse_estimate": math.sqrt(
                            max(train_probe_mse, 0.0)
                        ),
                        "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                        "op_projection/backbone_gradient_norm": backbone_norm,
                        "op_projection/backbone_gradient_tensors": (
                            backbone_grad_tensors
                        ),
                        "op_projection/head_gradient_norm": head_norm,
                        "op_projection/head_gradient_tensors": head_grad_tensors,
                    }
                    payload = {
                        "event": "train_step",
                        "train_probe_mse": train_probe_mse,
                        "backbone_gradient_norm": backbone_norm,
                        "backbone_gradient_tensors": backbone_grad_tensors,
                        "head_gradient_norm": head_norm,
                        "head_gradient_tensors": head_grad_tensors,
                        **wandb_payload,
                    }
                    _json_log(output_root, payload, rank)
                    if wandb_run is not None:
                        wandb_run.log(wandb_payload, step=global_step)
                window_loss_sum = 0.0
                window_molecules = 0

            global_epoch_loss_sum, global_epoch_molecules = _all_reduce_sum(
                [epoch_loss_sum, float(epoch_molecules)],
                device,
            )
            train_epoch_seconds = time.perf_counter() - epoch_started
            validation = _validate_epoch(
                ddp_model,
                val_loader,
                config=config,
                device=device,
                rank=rank,
            )
            exact_matrix_metrics: dict[str, float] = {}
            if (
                config.matrix_metrics.enabled
                and (epoch + 1) % config.matrix_metrics.every_n_epochs == 0
            ):
                exact_matrix_metrics.update(
                    _exact_matrix_metrics(
                        ddp_model,
                        train_loader,
                        split="train",
                        samples_per_rank=(
                            config.matrix_metrics.train_samples_per_rank
                        ),
                        identity_column_chunk_size=(
                            config.matrix_metrics.identity_column_chunk_size
                        ),
                        expected_global_molecules=(
                            config.matrix_metrics.train_samples_per_rank
                            * world_size
                        ),
                        device=device,
                    )
                )
                exact_matrix_metrics.update(
                    _exact_matrix_metrics(
                        ddp_model,
                        val_loader,
                        split="validation",
                        samples_per_rank=None,
                        identity_column_chunk_size=(
                            config.matrix_metrics.identity_column_chunk_size
                        ),
                        expected_global_molecules=config.dataset.num_val,
                        device=device,
                    )
                )
            validation_error = validation["validation_relative_action_error"]
            is_best = validation_error < best_validation_error
            best_validation_error = min(best_validation_error, validation_error)
            train_probe_mse = global_epoch_loss_sum / global_epoch_molecules
            epoch_seconds = time.perf_counter() - epoch_started
            peak_cuda_memory_bytes = torch.cuda.max_memory_allocated(device)
            epoch_payload = {
                "event": "epoch",
                "epoch": epoch + 1,
                "optimizer_step": global_step,
                "train_probe_mse": train_probe_mse,
                "train/probe_matrix_mse_estimate": train_probe_mse,
                "train/probe_matrix_rmse_estimate": math.sqrt(
                    max(train_probe_mse, 0.0)
                ),
                **validation,
                **exact_matrix_metrics,
                "best_validation_relative_action_error": best_validation_error,
                "train_epoch_seconds": train_epoch_seconds,
                "epoch_seconds": epoch_seconds,
                "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            }
            _json_log(output_root, epoch_payload, rank)
            if wandb_run is not None:
                wandb_payload = {
                    "optimizer_step": global_step,
                    "epoch": epoch + 1,
                    "train/total_loss": train_probe_mse,
                    "train/probe_matrix_mse_estimate": train_probe_mse,
                    "train/probe_matrix_rmse_estimate": math.sqrt(
                        max(train_probe_mse, 0.0)
                    ),
                    "validation/total_loss": validation[
                        "validation_probe_mse"
                    ],
                    **{
                        key: value
                        for key, value in validation.items()
                        if "/" in key
                    },
                    **exact_matrix_metrics,
                    "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                    "time/train_epoch_seconds": train_epoch_seconds,
                    "time/epoch_seconds": epoch_seconds,
                    "system/gpu_peak_memory_mb": (
                        peak_cuda_memory_bytes / (1024.0**2)
                    ),
                }
                wandb_run.log(wandb_payload, step=global_step)
            last_epoch_payload = epoch_payload

            if (epoch + 1) % config.checkpointing.save_frequency == 0:
                checkpoint_path = _atomic_checkpoint(
                    output_root,
                    model=ddp_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    best_validation_error=best_validation_error,
                    metrics=epoch_payload,
                    is_best=is_best,
                    rank=rank,
                )

        if max_backbone_grad_norm <= 0.0 or max_head_grad_norm <= 0.0:
            raise RuntimeError(
                "training did not produce nonzero gradients in both backbone and head"
            )
        if scope == "smoke":
            _reload_checkpoint(
                checkpoint_path,
                model=ddp_model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                expected_epoch_next=config.optimization.num_epochs,
            )

        final = {
            "event": "complete",
            "scope": scope,
            "optimizer_steps": global_step,
            "epochs": config.optimization.num_epochs,
            "best_validation_relative_action_error": best_validation_error,
            "max_backbone_gradient_norm": max_backbone_grad_norm,
            "max_head_gradient_norm": max_head_grad_norm,
            "checkpoint_reload_verified": scope == "smoke",
            "prediction_constructs_dense_matrix": False,
            "target_action_constructs_dense_matrix": False,
            "exact_matrix_metrics_construct_dense_matrix": False,
            "loader_reads_dense_reference_hamiltonian": True,
            "backbone_consumes_dense_overlap_input": True,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        }
        final.update(
            {
                key: value
                for key, value in last_epoch_payload.items()
                if "/" in key
            }
        )
        _json_log(output_root, final, rank)
        if rank == 0:
            with (output_root / "final_metrics.json").open("w") as handle:
                json.dump(final, handle, indent=2, sort_keys=True)
                handle.write("\n")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")
    base_config = OpProjectionTrainingConfig.from_yaml(config_path)
    scope: Scope = args.scope
    config = base_config.for_scope(scope)

    if scope == "validate":
        if args.output_root is not None:
            raise SystemExit("validate does not accept --output-root")
        print(json.dumps(_validate_preview(config, config_path), indent=2, sort_keys=True))
        return

    if args.output_root is None:
        raise SystemExit("smoke/full requires --output-root")
    output_root = args.output_root.expanduser().resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_root == outputs_root or outputs_root not in output_root.parents:
        raise SystemExit(f"output must be a lane directory below {outputs_root}")
    _run_training(
        config,
        scope=scope,
        config_path=config_path,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
