#!/usr/bin/env python3
"""Recalculate final NablaDFT SHIFT+STD metrics in physical AO units.

The legacy SS1 (now SHIFT+STD) checkpoints were trained with element-wise l=0
node-label
standardization.  Their original per-epoch validation matrix metrics converted
normalized node errors directly to AO matrices, omitting the inverse standard
deviation.  This tool:

1. reconstructs each audited final checkpoint;
2. reproduces its normalized-space final validation metrics as a guard;
3. applies the missing per-element standard deviations to l=0 node errors;
4. writes a local provenance report; and
5. optionally backfills the corrected final summaries in W&B.

Historical W&B curve points are not rewritten.  The original final summary
values are preserved under ``validation_normalized/*`` before the canonical
``validation/*`` summary keys are corrected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
SOURCE_ROOT = PROJECT_ROOT / "src"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_REPORT = (
    OUTPUT_ROOT
    / "scale-shift-recalculation"
    / "ss1-final-physical-metrics-v1.json"
)
WANDB_PROJECT = "kaist-korea/maloq-nablaDFT"
STATS_PATH = (
    OUTPUT_ROOT
    / "scale-shift-statistics"
    / "nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt"
)
DATABASE_PATH = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/"
    "hamiltonian_databases/train_2k.db"
)
TRAIN_ROWS = 12081
VALIDATION_ROWS = 64
BACKFILL_VERSION = "ss1-final-physical-ao-v1"
DEFAULT_CHECKPOINT_SOURCE_COMMIT = (
    "8e49e1c46fbb47e74afd007d60c40b33e439881b"
)

METRIC_NAMES = (
    "matrix_mae",
    "matrix_mse",
    "node_matrix_mae",
    "node_matrix_mse",
    "edge_matrix_mae",
    "edge_matrix_mse",
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    display_name: str
    config_path: Path
    checkpoint_dir: Path
    source_commit: str


RUNS = (
    RunSpec(
        "qpa1dbz8",
        "NablaDFT | NTE-128/3 | Muon | SHIFT+STD | V1",
        PROJECT_ROOT
        / "_my_script/experiment/2026-07-23/"
        "maloq_nte_do128_le3_muon_head_scale_shift_nabladft.yaml",
        OUTPUT_ROOT
        / "nabladft-nte-do128-le3-muon-head-scale-shift-2gpu-eb20-mb5-"
        "ga2-full-e20-seed44-20260723-164854/run",
        "66911e0ffe06821c1f6aaf355d1a076a323c719c",
    ),
    RunSpec(
        "27dk4l35",
        "NablaDFT | NTE-128/3 | Native | SHIFT+STD | V1",
        PROJECT_ROOT
        / "_my_script/experiment/2026-07-23/"
        "maloq_nte_do128_le3_native_head_scale_shift_nabladft.yaml",
        OUTPUT_ROOT
        / "nabladft-nte-do128-le3-native-head-scale-shift-2gpu-eb20-mb5-"
        "ga2-full-e20-seed44-20260724-042801/run",
        DEFAULT_CHECKPOINT_SOURCE_COMMIT,
    ),
    RunSpec(
        "2ygp53bs",
        "NablaDFT | QHFlow3 | Muon | SHIFT+STD | V1",
        PROJECT_ROOT
        / "_my_script/experiment/2026-07-23/"
        "qhflow3_local_muon_head_scale_shift_nabladft.yaml",
        OUTPUT_ROOT
        / "nabladft-qhflow3-local-muon-head-scale-shift-2gpu-eb20-mb5-"
        "ga2-full-e20-seed44-20260724-043801/run",
        DEFAULT_CHECKPOINT_SOURCE_COMMIT,
    ),
    RunSpec(
        "jal9l7uk",
        "NablaDFT | MALOQ | Muon | SHIFT+STD | V1",
        PROJECT_ROOT
        / "_my_script/experiment/2026-07-24/"
        "maloq_muon_head_scale_shift_nabladft.yaml",
        OUTPUT_ROOT / "nabladft-maloq-muon-head-ss/run",
        DEFAULT_CHECKPOINT_SOURCE_COMMIT,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Backfill verified corrected final summaries in W&B.",
    )
    parser.add_argument(
        "--apply-existing-report",
        action="store_true",
        help=(
            "Apply the already verified --report payload without rerunning "
            "GPU inference. This is idempotent and intended for retrying a "
            "failed W&B write."
        ),
    )
    parser.add_argument(
        "--run-id",
        action="append",
        choices=[run.run_id for run in RUNS],
        help="Restrict recalculation to one or more audited run IDs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Validation inference batch size; 5 matches the original runs.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--normalized-rtol",
        type=float,
        default=5.0e-4,
        help="Relative tolerance for reproducing original W&B metrics.",
    )
    parser.add_argument(
        "--normalized-atol",
        type=float,
        default=5.0e-9,
        help="Absolute tolerance for reproducing original W&B metrics.",
    )
    return parser.parse_args()


def _parse_json_layers(value: Any) -> Any:
    value = getattr(value, "_json_dict", value)
    while isinstance(value, str):
        value = json.loads(value)
    return value


def _wandb_config(run) -> dict[str, Any]:
    raw = _parse_json_layers(run.config)
    return {
        key: (
            value.get("value")
            if isinstance(value, dict) and "value" in value
            else value
        )
        for key, value in raw.items()
    }


def _wandb_summary(run) -> dict[str, Any]:
    summary = _parse_json_layers(run.summary)
    if not isinstance(summary, dict):
        raise TypeError(f"Unexpected W&B summary type for {run.id}: {type(summary)}")
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()


def _load_model_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)


def _closed_shell_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected a closed-shell singleton spin dimension, got "
                f"{tuple(tensor.shape)}"
            )
        return tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected a rank-2 coupled tensor, got {tuple(tensor.shape)}")
    return tensor


def _physical_node_error(
    normalized_error: torch.Tensor,
    atomic_numbers,
    scale_shift_data: dict[str, Any],
) -> torch.Tensor:
    result = normalized_error.clone()
    scalar_indices = torch.as_tensor(
        scale_shift_data["scalar_irrep_indices"],
        dtype=torch.long,
        device=result.device,
    )
    atomic_numbers = torch.as_tensor(
        atomic_numbers,
        dtype=torch.long,
        device=result.device,
    )
    if result.shape[0] != atomic_numbers.numel():
        raise ValueError(
            f"Node/atomic-number mismatch: {result.shape[0]} vs "
            f"{atomic_numbers.numel()}"
        )

    stds = scale_shift_data["element_scalar_stds"]
    max_atomic_number = max(int(z) for z in stds)
    std_table = torch.ones(
        (max_atomic_number + 1, scalar_indices.numel()),
        dtype=result.dtype,
        device=result.device,
    )
    for atomic_number, values in stds.items():
        std_table[int(atomic_number)] = torch.as_tensor(
            values,
            dtype=result.dtype,
            device=result.device,
        )
    missing = sorted(
        set(int(z) for z in atomic_numbers.tolist()).difference(
            int(z) for z in stds
        )
    )
    if missing:
        raise KeyError(f"Scale/shift artifact lacks atomic numbers: {missing}")

    result[:, scalar_indices] *= std_table.index_select(0, atomic_numbers)
    return result


def _empty_sums() -> np.ndarray:
    return np.zeros(9, dtype=np.float64)


def _matrix_error_sums(
    node_error: torch.Tensor,
    edge_error: torch.Tensor,
    basis_transform,
    target_object,
    target_index: int,
) -> np.ndarray:
    from maloq.fock_utils import matrix2labels_kernels

    node_error = basis_transform.get_H(node_error)
    edge_error = basis_transform.get_H(edge_error)

    neighbour_list = target_object.neighbour_list_list[target_index]
    atomic_numbers = target_object.atomic_numbers_list[target_index]
    orbitals_per_atom = target_object.orbitals_per_atom_list[target_index]
    num_atoms = len(atomic_numbers)
    src_idxes = np.concatenate(
        [np.asarray(neighbour_list[0]), np.arange(num_atoms)]
    )
    target_idxes = np.concatenate(
        [np.asarray(neighbour_list[1]), np.arange(num_atoms)]
    )
    fock_block_offsets = np.concatenate(
        [np.array([0]), np.cumsum(orbitals_per_atom)]
    )
    matrix_size = int(fock_block_offsets[-1])
    error_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    error_targets = torch.cat(
        [edge_error, node_error],
        dim=0,
    ).detach().cpu().numpy()

    matrix2labels_kernels.numpy_single_matrix2label(
        target_object.orbital_template,
        fock_block_offsets,
        atomic_numbers,
        src_idxes,
        target_idxes,
        error_matrix,
        error_targets,
        forward=False,
    )
    if not np.allclose(error_matrix, error_matrix.T, atol=1.0e-4):
        error_matrix = (error_matrix + error_matrix.T) / 2
    error_matrix = error_matrix.astype(np.float64, copy=False)

    sums = _empty_sums()
    sums[0] = np.abs(error_matrix).sum()
    sums[1] = np.square(error_matrix).sum()
    sums[2] = error_matrix.size

    for atom_index in range(num_atoms):
        block_start = fock_block_offsets[atom_index]
        block_end = fock_block_offsets[atom_index + 1]
        node_block = error_matrix[
            block_start:block_end,
            block_start:block_end,
        ]
        sums[3] += np.abs(node_block).sum()
        sums[4] += np.square(node_block).sum()
        sums[5] += node_block.size

    for source_index, destination_index in zip(
        neighbour_list[0],
        neighbour_list[1],
        strict=True,
    ):
        row_start = fock_block_offsets[source_index]
        row_end = fock_block_offsets[source_index + 1]
        column_start = fock_block_offsets[destination_index]
        column_end = fock_block_offsets[destination_index + 1]
        edge_block = error_matrix[
            row_start:row_end,
            column_start:column_end,
        ]
        sums[6] += np.abs(edge_block).sum()
        sums[7] += np.square(edge_block).sum()
        sums[8] += edge_block.size
    return sums


def _finalize_sums(sums: np.ndarray) -> dict[str, float]:
    if np.any(sums[[2, 5, 8]] <= 0):
        raise ValueError(f"Metric denominator is non-positive: {sums.tolist()}")
    return {
        "matrix_mae": float(sums[0] / sums[2]),
        "matrix_mse": float(sums[1] / sums[2]),
        "node_matrix_mae": float(sums[3] / sums[5]),
        "node_matrix_mse": float(sums[4] / sums[5]),
        "edge_matrix_mae": float(sums[6] / sums[8]),
        "edge_matrix_mse": float(sums[7] / sums[8]),
    }


def _build_model(
    spec: RunSpec,
    device: torch.device,
    required_irreps,
    orbital_basis,
    ls_list,
):
    from maloq.core.config import MaloqConfig
    from maloq.train_utils.training_workflow import TrainingWorkflow

    config = MaloqConfig.from_file(spec.config_path).to_workflow_config()
    build_output = (
        OUTPUT_ROOT
        / "scale-shift-recalculation"
        / "model-build"
        / spec.run_id
    )
    build_output.mkdir(parents=True, exist_ok=True)
    config.update(
        train_or_eval="eval",
        output_folder=str(build_output),
        run_name=f"ss1-metric-backfill-{spec.run_id}",
        use_wandb=False,
        restart_backbone=False,
        restart_head=False,
        restart_optimizer=False,
    )

    workflow = object.__new__(TrainingWorkflow)
    workflow.config = TrainingWorkflow.DEFAULTS | config
    workflow.device = device
    workflow.rank = 0
    workflow.world_size = 1
    workflow.local_rank = 0
    workflow.check_input_config()
    backbone, head, optimizer = workflow.build_model(
        required_irreps,
        orbital_basis,
        ls_list,
    )
    del optimizer
    _load_model_state(backbone, spec.checkpoint_dir / "backbone.pt")
    _load_model_state(head, spec.checkpoint_dir / "head.pt")
    backbone.eval()
    head.eval()
    return backbone, head, config


def _recalculate_run(
    spec: RunSpec,
    device: torch.device,
    validation_loader,
    required_irreps,
    basis_transform,
    orbital_basis,
    ls_list,
    scale_shift_data: dict[str, Any],
) -> dict[str, Any]:
    backbone, head, config = _build_model(
        spec,
        device,
        required_irreps,
        orbital_basis,
        ls_list,
    )
    normalized_sums = _empty_sums()
    physical_sums = _empty_sums()
    molecule_count = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_index, batch in enumerate(validation_loader):
            batch = batch.to(device)
            backbone_output = backbone(batch)
            node_output, edge_output = head(backbone_output, batch)
            node_output = _closed_shell_tensor(node_output)
            edge_output = _closed_shell_tensor(edge_output)
            node_target = _closed_shell_tensor(batch.node_y)
            edge_target = _closed_shell_tensor(batch.y)

            node_slices = batch.ptr
            edge_graph_indices = batch.batch[batch.edge_index[0]]
            target_indices = batch.fock_target_id.reshape(-1)

            for graph_index in range(batch.num_graphs):
                node_start = int(node_slices[graph_index].item())
                node_end = int(node_slices[graph_index + 1].item())
                edge_mask = edge_graph_indices == graph_index
                target_object = batch.fock_target_object[graph_index]
                target_index = int(target_indices[graph_index].item())
                atomic_numbers = target_object.atomic_numbers_list[target_index]

                normalized_node_error = (
                    node_output[node_start:node_end]
                    - node_target[node_start:node_end]
                )
                edge_error = edge_output[edge_mask] - edge_target[edge_mask]
                physical_node_error = _physical_node_error(
                    normalized_node_error,
                    atomic_numbers,
                    scale_shift_data,
                )

                normalized_sums += _matrix_error_sums(
                    normalized_node_error,
                    edge_error,
                    basis_transform,
                    target_object,
                    target_index,
                )
                physical_sums += _matrix_error_sums(
                    physical_node_error,
                    edge_error,
                    basis_transform,
                    target_object,
                    target_index,
                )
                molecule_count += 1

            print(
                f"{spec.run_id}: batch {batch_index + 1}/"
                f"{len(validation_loader)}, molecules={molecule_count}",
                flush=True,
            )
            del batch, backbone_output, node_output, edge_output

    result = {
        "run_id": spec.run_id,
        "display_name": spec.display_name,
        "config_path": str(spec.config_path),
        "checkpoint_dir": str(spec.checkpoint_dir),
        "model_variant": config["model_variant"],
        "molecules": molecule_count,
        "normalized": _finalize_sums(normalized_sums),
        "physical": _finalize_sums(physical_sums),
        "element_count_totals": {
            "matrix": int(normalized_sums[2]),
            "node_matrix": int(normalized_sums[5]),
            "edge_matrix": int(normalized_sums[8]),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_provenance": {
            "source_commit": spec.source_commit,
            "backbone_sha256": _sha256(spec.checkpoint_dir / "backbone.pt"),
            "head_sha256": _sha256(spec.checkpoint_dir / "head.pt"),
        },
    }
    if molecule_count != VALIDATION_ROWS:
        raise RuntimeError(
            f"{spec.run_id} evaluated {molecule_count} molecules; "
            f"expected {VALIDATION_ROWS}"
        )
    del backbone, head
    torch.cuda.empty_cache()
    return result


def _validate_inventory(specs: tuple[RunSpec, ...]):
    import wandb

    api = wandb.Api(timeout=90)
    remote_runs = {
        spec.run_id: api.run(f"{WANDB_PROJECT}/{spec.run_id}")
        for spec in specs
    }
    selected: dict[str, Any] = {}
    for spec in specs:
        if spec.run_id not in remote_runs:
            raise RuntimeError(f"W&B run not found: {spec.run_id}")
        run = remote_runs[spec.run_id]
        if run.state == "running":
            raise RuntimeError(f"Refusing to modify running W&B run: {spec.run_id}")
        if run.name != spec.display_name:
            raise RuntimeError(
                f"{spec.run_id} name mismatch: {run.name!r} != "
                f"{spec.display_name!r}"
            )
        config = _wandb_config(run)
        if config.get("scale_and_shift") is not True:
            raise RuntimeError(f"{spec.run_id} is not a scale_and_shift=True run")
        if Path(str(config.get("output_folder"))).resolve() != (
            spec.checkpoint_dir.resolve()
        ):
            raise RuntimeError(
                f"{spec.run_id} checkpoint path mismatch: "
                f"{config.get('output_folder')!r}"
            )
        for path in (
            spec.config_path,
            spec.checkpoint_dir / "backbone.pt",
            spec.checkpoint_dir / "head.pt",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_revision = spec.checkpoint_dir.parent / "source_revision.tsv"
        if not source_revision.is_file():
            raise FileNotFoundError(source_revision)
        revision_text = source_revision.read_text()
        if spec.source_commit not in revision_text:
            raise RuntimeError(
                f"{spec.run_id} source revision is not "
                f"{spec.source_commit}"
            )
        selected[spec.run_id] = run
    return selected


def _validate_normalized_reproduction(
    result: dict[str, Any],
    run,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    summary = _wandb_summary(run)
    comparisons = {}
    for metric in METRIC_NAMES:
        key = f"validation/{metric}"
        remote_value = float(summary[key])
        recalculated = float(result["normalized"][metric])
        close = bool(
            np.isclose(
                recalculated,
                remote_value,
                rtol=rtol,
                atol=atol,
            )
        )
        comparisons[metric] = {
            "wandb": remote_value,
            "recalculated": recalculated,
            "absolute_difference": abs(recalculated - remote_value),
            "close": close,
        }
    if not all(item["close"] for item in comparisons.values()):
        raise RuntimeError(
            f"{result['run_id']} did not reproduce original normalized "
            f"metrics: {json.dumps(comparisons, sort_keys=True)}"
        )
    return comparisons


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    output_root = OUTPUT_ROOT.resolve()
    if path != output_root and output_root not in path.parents:
        raise ValueError(f"Report must be below {output_root}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _apply_wandb_backfill(
    report: dict[str, Any],
    remote_runs: dict[str, Any],
) -> None:
    note_line = (
        "SHIFT+STD final validation summary corrected with inverse l=0 standard "
        "deviation before AO reconstruction "
        f"({BACKFILL_VERSION}); historical epoch curves remain in normalized "
        "space."
    )
    for result in report["runs"]:
        run = remote_runs[result["run_id"]]
        existing_summary = _wandb_summary(run)
        updates: dict[str, Any] = {
            "validation/metric_space": "physical_ao_hartree",
            "validation/metric_backfill_version": BACKFILL_VERSION,
            "validation/metric_backfill_validation_rows": VALIDATION_ROWS,
            "validation/metric_backfill_checkpoint_source_commit": (
                result["checkpoint_provenance"]["source_commit"]
            ),
            "validation/metric_backfill_recalculation_git_head": report["git_head"],
            "validation/metric_backfill_recalculated_at": report["created_at"],
        }
        for metric in METRIC_NAMES:
            canonical_key = f"validation/{metric}"
            updates[f"validation_normalized/{metric}"] = float(
                result["normalized_reproduction"][metric]["wandb"]
            )
            updates[f"validation_physical/{metric}"] = float(
                result["physical"][metric]
            )
            updates[canonical_key] = float(result["physical"][metric])

        # W&B 0.22 can expose summaryMetrics as a JSON string.  Normalize the
        # public API object's backing dictionary before committing one mutation.
        run.summary._json_dict = existing_summary
        run.summary._dict = {}
        run.summary.update(updates)

        desired_tags = sorted(
            set(run.tags or ())
            | {
                "normalization:l0-zscore",
                "metrics:physical-backfill-v1",
            }
        )
        existing_notes = (run.notes or "").rstrip()
        run.tags = desired_tags
        if note_line not in existing_notes:
            run.notes = (
                f"{existing_notes}\n\n{note_line}".strip()
                if existing_notes
                else note_line
            )
        run.update()
        print(f"Updated W&B summary: {run.id} {run.name}", flush=True)

    # Re-fetch and verify every canonical and preserved value.
    import wandb

    api = wandb.Api(timeout=90)
    for result in report["runs"]:
        run = api.run(f"{WANDB_PROJECT}/{result['run_id']}")
        summary = _wandb_summary(run)
        for metric in METRIC_NAMES:
            expected_physical = float(result["physical"][metric])
            for key in (
                f"validation/{metric}",
                f"validation_physical/{metric}",
            ):
                actual = float(summary[key])
                if not np.isclose(actual, expected_physical, rtol=1e-12, atol=0):
                    raise RuntimeError(
                        f"W&B verification failed for {run.id} {key}: "
                        f"{actual} != {expected_physical}"
                    )
            expected_normalized = float(result["normalized_reproduction"][metric]["wandb"])
            actual_normalized = float(summary[f"validation_normalized/{metric}"])
            if not np.isclose(
                actual_normalized,
                expected_normalized,
                rtol=1e-12,
                atol=0,
            ):
                raise RuntimeError(
                    f"W&B normalized preservation failed for {run.id} {metric}"
                )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.normalized_rtol < 0 or args.normalized_atol < 0:
        raise SystemExit("Normalized metric tolerances must be non-negative")

    selected_ids = set(args.run_id or [run.run_id for run in RUNS])
    specs = tuple(run for run in RUNS if run.run_id in selected_ids)
    if not specs:
        raise SystemExit("No SHIFT+STD runs selected")

    if args.apply_existing_report:
        if args.run_id:
            raise SystemExit(
                "--apply-existing-report applies the complete report; "
                "do not combine it with --run-id"
            )
        report_path = args.report.expanduser().resolve()
        report = json.loads(report_path.read_text())
        if report.get("backfill_version") != BACKFILL_VERSION:
            raise RuntimeError(
                f"Unexpected report backfill version: "
                f"{report.get('backfill_version')!r}"
            )
        report_ids = [result["run_id"] for result in report.get("runs", ())]
        expected_ids = [spec.run_id for spec in specs]
        if report_ids != expected_ids:
            raise RuntimeError(
                f"Report run inventory mismatch: {report_ids} != {expected_ids}"
            )
        remote_runs = _validate_inventory(specs)
        _apply_wandb_backfill(report, remote_runs)
        report["wandb_applied"] = True
        report["wandb_applied_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(report_path, report)
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "run_ids": report_ids,
                    "wandb_applied": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    if not STATS_PATH.is_file():
        raise FileNotFoundError(STATS_PATH)
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(DATABASE_PATH)

    sys.path.insert(0, str(SOURCE_ROOT))
    remote_runs = _validate_inventory(specs)
    stats_payload = torch.load(
        STATS_PATH,
        map_location="cpu",
        weights_only=False,
    )
    provenance = stats_payload["provenance"]
    expected_provenance = {
        "dataset_name": "nablaDFT",
        "database_path": str(DATABASE_PATH),
        "database_rows": TRAIN_ROWS + VALIDATION_ROWS,
        "training_index_start": 0,
        "training_index_end_exclusive": TRAIN_ROWS,
        "num_train": TRAIN_ROWS,
        "validation_rows_in_statistics": 0,
        "loss_target": "fock_matrix",
        "rcut_orbitals": 8.0,
        "dtype": "float32",
        "normalization": "elementwise_standardize_l0_node_labels",
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Scale/shift provenance mismatch: {mismatches}")
    scale_shift_data = {
        "element_scalar_means": stats_payload["element_scalar_means"],
        "element_scalar_stds": stats_payload["element_scalar_stds"],
        "scalar_irrep_indices": stats_payload["scalar_irrep_indices"],
    }

    from maloq.dataset_utils import get_loader
    from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
    from maloq.train_utils import utils_compute

    device = utils_compute.setup_env(0, 1, backend="nccl", local_rank=0)
    try:
        database = HamiltonianDatabase(str(DATABASE_PATH))
        (
            validation_loader,
            required_irreps,
            basis_transform,
            orbital_basis,
            ls_list,
        ) = get_loader.get_loader(
            database=database,
            start_idx=TRAIN_ROWS,
            end_idx=TRAIN_ROWS + VALIDATION_ROWS,
            dataset_name="nablaDFT",
            rcut=8.0,
            batch_size=args.batch_size,
            dtype=torch.float32,
            half_edges=False,
            loss_target_string="fock_matrix",
            is_open_shell=False,
            scale_shift_data=scale_shift_data,
            distribute_graphs=False,
            partition_type=None,
            train_or_eval="eval",
            delta_learning=False,
            load_delta_auxiliary_matrix=False,
        )

        report: dict[str, Any] = {
            "schema_version": 1,
            "backfill_version": BACKFILL_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_head": _git_head(),
            "git_worktree_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=PROJECT_ROOT,
                    text=True,
                    check=True,
                    capture_output=True,
                ).stdout
            ),
            "wandb_project": WANDB_PROJECT,
            "database_path": str(DATABASE_PATH),
            "validation_index_start": TRAIN_ROWS,
            "validation_index_end_exclusive": TRAIN_ROWS + VALIDATION_ROWS,
            "validation_rows": VALIDATION_ROWS,
            "batch_size": args.batch_size,
            "scale_shift_artifact": str(STATS_PATH),
            "scale_shift_sha256": _sha256(STATS_PATH),
            "scale_shift_provenance": provenance,
            "metric_definition": {
                "normalized": (
                    "Original buggy validation path: AO reconstruction of "
                    "standardized l=0 node error."
                ),
                "physical": (
                    "Element std multiplied into standardized l=0 node error "
                    "before AO reconstruction; mean cancels in prediction-target."
                ),
                "matrix_scope": (
                    "Full dense AO error matrix reconstructed on the original "
                    "rcut=8 directed graph, with the same symmetrization and "
                    "denominators as SplitTrainer."
                ),
            },
            "runs": [],
            "wandb_applied": False,
        }

        for spec in specs:
            result = _recalculate_run(
                spec,
                device,
                validation_loader,
                required_irreps,
                basis_transform,
                orbital_basis,
                ls_list,
                scale_shift_data,
            )
            result["normalized_reproduction"] = (
                _validate_normalized_reproduction(
                    result,
                    remote_runs[spec.run_id],
                    args.normalized_rtol,
                    args.normalized_atol,
                )
            )
            report["runs"].append(result)
            _write_report(args.report, report)

        if args.apply:
            _apply_wandb_backfill(report, remote_runs)
            report["wandb_applied"] = True
            report["wandb_applied_at"] = datetime.now(timezone.utc).isoformat()
            _write_report(args.report, report)
    finally:
        utils_compute.cleanup_process_group(sync_barrier=True)

    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "run_ids": [result["run_id"] for result in report["runs"]],
                "wandb_applied": report["wandb_applied"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
