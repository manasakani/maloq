#!/usr/bin/env python3
"""Run one isolated NablaDFT V2 native-head RAW comparison lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
SOURCE_ROOT = PROJECT_ROOT / "src"
EXPERIMENT_ROOT = PROJECT_ROOT / "_my_script/experiment/2026-07-28/native_raw"
DEFAULT_BASE_CONFIG = EXPERIMENT_ROOT / "nabladft_native_raw_common.yaml"
EXPECTED_DATABASE = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db"
)
WANDB_PROJECT = "MALOQ-nablaDFT-v2"
FULL_COUNTS = {"train": 12081, "val": 64, "test": 0}
SMOKE_COUNTS = {"train": 20, "val": 20, "test": 0}

for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _select_rank_local_cuda_before_workflow_import() -> None:
    """Select the MPI-local CUDA device before eSEN/CuPy is imported."""
    world_size = int(
        os.environ.get(
            "OMPI_COMM_WORLD_SIZE",
            os.environ.get("WORLD_SIZE", "1"),
        )
    )
    if world_size <= 1:
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Two-rank training requires visible CUDA devices.")
    local_rank = int(
        os.environ.get(
            "OMPI_COMM_WORLD_LOCAL_RANK",
            os.environ.get("LOCAL_RANK", "0"),
        )
    )
    visible_devices = torch.cuda.device_count()
    device_index = 0 if visible_devices == 1 else local_rank
    if not 0 <= device_index < visible_devices:
        raise SystemExit(
            f"Local rank {local_rank} cannot select one of "
            f"{visible_devices} visible CUDA devices."
        )
    torch.cuda.set_device(device_index)


_select_rank_local_cuda_before_workflow_import()

from maloq.core.config import MaloqConfig  # noqa: E402
from maloq.train_utils.training_workflow_v2 import (  # noqa: E402
    TrainingWorkflowV2Fixed,
)


Architecture = Literal["maloq", "qhflow3", "ntev2"]
Scope = Literal["validate", "smoke", "full"]


@dataclass(frozen=True)
class Lane:
    id: str
    architecture: Architecture
    display_name: str
    expected_parameters: int


LANES = (
    Lane(
        id="maloq-e3",
        architecture="maloq",
        display_name="NablaDFT | MALOQ-E3 | Native | RAW | V2",
        expected_parameters=34_489_297,
    ),
    Lane(
        id="qhflow3-e3",
        architecture="qhflow3",
        display_name="NablaDFT | QHFlow3-E3 | Native | RAW | V2",
        expected_parameters=34_382_227,
    ),
    Lane(
        id="ntev2-e3",
        architecture="ntev2",
        display_name="NablaDFT | NTEV2-E3 | Native | RAW | V2",
        expected_parameters=33_891_021,
    ),
)
LANE_BY_ID = {lane.id: lane for lane in LANES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}.")


def _validate_base_contract(base: MaloqConfig) -> None:
    """Protect every matched full-run setting from silent config drift."""
    checks = (
        ("dataset.dataset_name", base.dataset.dataset_name, "nablaDFT"),
        ("dataset.dbpath", Path(base.dataset.dbpath), EXPECTED_DATABASE),
        ("dataset.open_shell", base.dataset.open_shell, False),
        ("splits.num_train", base.splits.num_train, 12081),
        ("splits.num_val", base.splits.num_val, 64),
        ("splits.num_test", base.splits.num_test, 0),
        ("splits.batch_size", base.splits.batch_size, 5),
        ("splits.shuffle", base.splits.shuffle, False),
        ("splits.distribute_graphs", base.splits.distribute_graphs, False),
        ("splits.dist_backend", base.splits.dist_backend, "nccl"),
        ("model.head_type", base.model.head_type, "maloq"),
        ("model.l_embedding_dim", base.model.l_embedding_dim, 128),
        ("model.hidden_dim", base.model.hidden_dim, 128),
        ("model.num_distance_basis", base.model.num_distance_basis, 512),
        ("model.num_mp_layers", base.model.num_mp_layers, 3),
        ("model.num_edge_layers", base.model.num_edge_layers, 3),
        ("model.message_type", base.model.message_type, "source-target"),
        ("model.rcut_orbitals", base.model.rcut_orbitals, 8.0),
        ("model.rcut_gaussian", base.model.rcut_gaussian, 16.0),
        ("model.gaussian_width", base.model.gaussian_width, 1.0),
        ("model.reduce_edge", base.model.reduce_edge, False),
        ("model.reduce_node", base.model.reduce_node, True),
        ("model.reduce_node_intra", base.model.reduce_node_intra, True),
        ("model.qhflow3_max_radius", base.model.qhflow3_max_radius, 12.0),
        ("model.qhflow3_radius_embed_dim", base.model.qhflow3_radius_embed_dim, 32),
        ("model.qhflow3_grid_resolution", base.model.qhflow3_grid_resolution, 48),
        (
            "model.qhflow3_grid_ffn_chunk_size",
            base.model.qhflow3_grid_ffn_chunk_size,
            512,
        ),
        ("model.qhflow3_use_overlap", base.model.qhflow3_use_overlap, True),
        (
            "model.qhflow3_muonize_output_projection",
            base.model.qhflow3_muonize_output_projection,
            False,
        ),
        ("optimization.num_epochs", base.optimization.num_epochs, 20),
        ("optimization.lr_init", base.optimization.lr_init, 0.0005),
        ("optimization.optimizer_type", base.optimization.optimizer_type, "muon"),
        ("optimization.weight_decay", base.optimization.weight_decay, 0.0001),
        ("optimization.muon_lr", base.optimization.muon_lr, 0.02),
        ("optimization.muon_momentum", base.optimization.muon_momentum, 0.95),
        ("optimization.muon_nesterov", base.optimization.muon_nesterov, True),
        ("optimization.muon_ns_steps", base.optimization.muon_ns_steps, 5),
        ("optimization.muon_adamw_lr", base.optimization.muon_adamw_lr, 0.0005),
        (
            "optimization.muon_adamw_betas",
            base.optimization.muon_adamw_betas,
            (0.9, 0.95),
        ),
        ("optimization.muon_adamw_eps", base.optimization.muon_adamw_eps, 1e-10),
        (
            "optimization.muon_output_projection_policy",
            base.optimization.muon_output_projection_policy,
            "shape_muon",
        ),
        (
            "optimization.gradient_clip_val",
            base.optimization.gradient_clip_val,
            1.0,
        ),
        (
            "optimization.gradient_accumulation_steps",
            base.optimization.gradient_accumulation_steps,
            2,
        ),
        (
            "optimization.scheduler_type",
            base.optimization.scheduler_type,
            "warmup_polynomial",
        ),
        ("optimization.warmup_steps", base.optimization.warmup_steps, 1000),
        ("optimization.scheduler_power", base.optimization.scheduler_power, 1.0),
        ("optimization.min_lr_ratio", base.optimization.min_lr_ratio, 0.0),
        ("optimization.step_every_epoch", base.optimization.step_every_epoch, False),
        ("loss.loss_target", base.loss.loss_target, "fock_matrix"),
        ("loss.train_loss", base.loss.train_loss, "rmse_mse_padded_loss"),
        ("loss.test_loss", base.loss.test_loss, "l1_unpadded_loss"),
        ("loss.scale_and_shift", base.loss.scale_and_shift, False),
        ("loss.scale_shift_mode", base.loss.scale_shift_mode, "shift_only"),
        ("loss.scale_shift_path", base.loss.scale_shift_path, None),
        ("loss.delta_learning", base.loss.delta_learning, False),
        ("runtime.dtype", base.runtime.dtype, "float32"),
        ("runtime.seed", base.runtime.seed, 44),
        ("tracking.wandb_project", base.tracking.wandb_project, WANDB_PROJECT),
        (
            "tracking.wandb_group",
            base.tracking.wandb_group,
            "nabladft-v2-ofat-seed44",
        ),
        (
            "tracking.validation_matrix_metrics",
            base.tracking.validation_matrix_metrics,
            True,
        ),
        (
            "tracking.validation_matrix_metrics_frequency",
            base.tracking.validation_matrix_metrics_frequency,
            1,
        ),
    )
    for label, actual, expected in checks:
        _assert_equal(label, actual, expected)


def _validate_suite_inputs(base: MaloqConfig) -> dict[str, object]:
    _validate_base_contract(base)
    if not EXPECTED_DATABASE.is_file():
        raise FileNotFoundError(EXPECTED_DATABASE)

    from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

    database = HamiltonianDatabase(str(EXPECTED_DATABASE))
    rows = len(database)
    if rows != sum(FULL_COUNTS.values()):
        raise ValueError(
            f"Expected {sum(FULL_COUNTS.values())} fixed rows, got {rows}."
        )
    atomic_numbers, positions, _, _, hamiltonian, overlap, *_ = database[0]
    if hamiltonian.ndim != 2 or hamiltonian.shape != overlap.shape:
        raise ValueError(
            "NablaDFT row 0 has incompatible Hamiltonian/overlap shapes: "
            f"{hamiltonian.shape}, {overlap.shape}."
        )
    return {
        "database": str(EXPECTED_DATABASE),
        "rows": rows,
        "row0_atoms": len(atomic_numbers),
        "row0_positions_shape": list(positions.shape),
        "row0_matrix_shape": list(hamiltonian.shape),
        "target_treatment": "raw",
        "scale_and_shift": False,
        "scale_shift_path": None,
    }


def _architecture_overrides(lane: Lane) -> dict[str, object]:
    if lane.architecture == "maloq":
        return {
            "backbone_type": "esen",
            "output_l_embedding_dim": None,
            "mlp_type": "spectral",
        }
    if lane.architecture == "ntev2":
        return {
            "backbone_type": "maloq_nte_v2",
            "output_l_embedding_dim": 64,
            "mlp_type": "grid",
        }
    return {
        "backbone_type": "qhflow3",
        "output_l_embedding_dim": 64,
        "mlp_type": "grid",
    }


def _build_lane_config(
    base: MaloqConfig,
    lane: Lane,
    scope: Scope,
    output_root: Path,
) -> MaloqConfig:
    payload = base.model_dump(mode="python")
    run_name = f"nabladft-v2-ofat-{lane.id}-native-raw"
    payload["dataset"].update(
        run_name=run_name,
        output_folder=str(output_root),
    )
    counts = SMOKE_COUNTS if scope == "smoke" else FULL_COUNTS
    payload["splits"].update(
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
    )
    payload["model"].update(
        model_variant=run_name,
        head_type="maloq",
        num_mp_layers=3,
        num_edge_layers=3,
        **_architecture_overrides(lane),
    )
    payload["optimization"].update(
        num_epochs=1 if scope == "smoke" else 20,
        optimizer_type="muon",
    )
    payload["loss"].update(
        scale_and_shift=False,
        scale_shift_mode="shift_only",
        scale_shift_path=None,
    )
    lane_tags = [
        f"architecture:{lane.architecture}",
        "edge-layers:3",
        "head:native",
        "head-implementation:maloq",
        "normalization:raw",
        "axis:head-normalization-cross",
    ]
    payload["tracking"].update(
        use_wandb=scope == "full",
        wandb_project=WANDB_PROJECT,
        wandb_run_name=lane.display_name,
        wandb_job_type=scope,
        wandb_tags=tuple(
            dict.fromkeys([*payload["tracking"]["wandb_tags"], *lane_tags])
        ),
    )
    config = MaloqConfig.model_validate(payload)
    _validate_lane_contract(config, lane, scope)
    return config


def _validate_lane_contract(
    config: MaloqConfig,
    lane: Lane,
    scope: Scope,
) -> None:
    expected_backbones = {
        "maloq": "esen",
        "ntev2": "maloq_nte_v2",
        "qhflow3": "qhflow3",
    }
    checks = (
        (
            "backbone_type",
            config.model.backbone_type,
            expected_backbones[lane.architecture],
        ),
        ("head_type", config.model.head_type, "maloq"),
        ("num_mp_layers", config.model.num_mp_layers, 3),
        ("num_edge_layers", config.model.num_edge_layers, 3),
        ("optimizer_type", config.optimization.optimizer_type, "muon"),
        (
            "num_epochs",
            config.optimization.num_epochs,
            1 if scope == "smoke" else 20,
        ),
        ("num_train", config.splits.num_train, 20 if scope == "smoke" else 12081),
        ("num_val", config.splits.num_val, 20 if scope == "smoke" else 64),
        ("num_test", config.splits.num_test, 0),
        ("batch_size", config.splits.batch_size, 5),
        (
            "gradient_accumulation_steps",
            config.optimization.gradient_accumulation_steps,
            2,
        ),
        ("shuffle", config.splits.shuffle, False),
        ("distribute_graphs", config.splits.distribute_graphs, False),
        ("scale_and_shift", config.loss.scale_and_shift, False),
        ("scale_shift_mode", config.loss.scale_shift_mode, "shift_only"),
        ("scale_shift_path", config.loss.scale_shift_path, None),
        ("wandb_project", config.tracking.wandb_project, WANDB_PROJECT),
        (
            "wandb_group",
            config.tracking.wandb_group,
            "nabladft-v2-ofat-seed44",
        ),
        ("wandb_run_name", config.tracking.wandb_run_name, lane.display_name),
    )
    for label, actual, expected in checks:
        _assert_equal(f"{lane.id}.{label}", actual, expected)


def _config_preview(config: MaloqConfig, lane: Lane) -> dict[str, object]:
    return {
        **asdict(lane),
        "workflow": (
            f"{TrainingWorkflowV2Fixed.__module__}.{TrainingWorkflowV2Fixed.__name__}"
        ),
        "backbone_type": config.model.backbone_type,
        "head_type": config.model.head_type,
        "optimizer_type": config.optimization.optimizer_type,
        "muon_lr": config.optimization.muon_lr,
        "aux_adamw_lr": config.optimization.muon_adamw_lr,
        "output_l_embedding_dim": config.model.output_l_embedding_dim,
        "num_mp_layers": config.model.num_mp_layers,
        "num_edge_layers": config.model.num_edge_layers,
        "num_epochs": config.optimization.num_epochs,
        "num_train": config.splits.num_train,
        "num_val": config.splits.num_val,
        "num_test": config.splits.num_test,
        "batch_size_per_rank": config.splits.batch_size,
        "gradient_accumulation_steps": (
            config.optimization.gradient_accumulation_steps
        ),
        "effective_batch_size_at_world_size_2": (
            config.splits.batch_size
            * 2
            * config.optimization.gradient_accumulation_steps
        ),
        "shuffle": config.splits.shuffle,
        "scale_and_shift": config.loss.scale_and_shift,
        "scale_shift_mode": config.loss.scale_shift_mode,
        "scale_shift_path": config.loss.scale_shift_path,
        "wandb_project": config.tracking.wandb_project,
        "wandb_run_name": config.tracking.wandb_run_name,
        "wandb_tags": list(config.tracking.wandb_tags),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
    )
    parser.add_argument(
        "--lane",
        choices=("all", *LANE_BY_ID),
        required=True,
    )
    parser.add_argument(
        "--scope",
        choices=("validate", "smoke", "full"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def _rank() -> int:
    return int(
        os.environ.get(
            "OMPI_COMM_WORLD_RANK",
            os.environ.get("RANK", "0"),
        )
    )


def main() -> None:
    args = _parse_args()
    base_config_path = args.base_config.expanduser().resolve()
    if not base_config_path.is_file():
        raise SystemExit(f"Base config not found: {base_config_path}")
    base = MaloqConfig.from_file(base_config_path)
    input_metadata = _validate_suite_inputs(base)

    scope: Scope = args.scope
    if args.lane == "all":
        if scope != "validate":
            raise SystemExit("--lane all is allowed only with --scope validate.")
        selected_lanes = LANES
    else:
        selected_lanes = (LANE_BY_ID[args.lane],)

    if scope == "validate":
        preview_base = (
            PROJECT_ROOT / "outputs/_config-preview/nabladft-v2-ofat-native-raw"
        )
        previews = [
            _config_preview(
                _build_lane_config(
                    base,
                    lane,
                    scope,
                    preview_base / lane.id,
                ),
                lane,
            )
            for lane in selected_lanes
        ]
        print(
            json.dumps(
                {
                    "suite": "nabladft-v2-ofat-native-raw",
                    "workflow": (
                        f"{TrainingWorkflowV2Fixed.__module__}."
                        f"{TrainingWorkflowV2Fixed.__name__}"
                    ),
                    "base_config": str(base_config_path),
                    "base_config_sha256": _sha256(base_config_path),
                    "inputs": input_metadata,
                    "lanes": previews,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if len(selected_lanes) != 1:
        raise SystemExit("Training requires exactly one lane.")
    if args.output_root is None:
        raise SystemExit("--output-root is required for smoke/full.")
    output_root = args.output_root.expanduser().resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_root == outputs_root or outputs_root not in output_root.parents:
        raise SystemExit(
            f"Output must be a lane directory below {outputs_root}: {output_root}"
        )

    lane = selected_lanes[0]
    typed_config = _build_lane_config(base, lane, scope, output_root)
    if _rank() == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        resolved_payload = {
            "suite": "nabladft-v2-ofat-native-raw",
            "scope": scope,
            "lane": asdict(lane),
            "base_config": str(base_config_path),
            "base_config_sha256": _sha256(base_config_path),
            "inputs": input_metadata,
            "config": typed_config.model_dump(mode="json"),
        }
        (output_root / "resolved_native_raw_config.json").write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n"
        )

    workflow_config = typed_config.to_workflow_config()
    TrainingWorkflowV2Fixed(workflow_config).run()


if __name__ == "__main__":
    main()
