#!/usr/bin/env python3
"""Run one typed lane from the NablaDFT MALOQ/NTEV2/QHFlow3 OFAT suite."""

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
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "_my_script/experiment/2026-07-27/nabladft_v2_ofat_common.yaml"
)
EXPECTED_DATABASE = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db"
)
EXPECTED_SCALE_SHIFT = (
    PROJECT_ROOT / "outputs/scale-shift-statistics/"
    "nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt"
)
EXPECTED_SCALE_SHIFT_SHA256 = (
    "375167ad551fb0b60dbe9cd049a4995276b54ce075e09906639ef3daa4f79475"
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


Architecture = Literal["maloq", "ntev2", "qhflow3"]
Head = Literal["native", "muon"]
Normalization = Literal["raw", "shift"]
Scope = Literal["validate", "smoke", "full"]


@dataclass(frozen=True)
class Lane:
    id: str
    architecture: Architecture
    edge_layers: Literal[2, 3]
    head: Head
    normalization: Normalization
    axes: tuple[str, ...]


LANES = (
    Lane(
        "maloq-e3-muon-shift",
        "maloq",
        3,
        "muon",
        "shift",
        ("structure",),
    ),
    Lane(
        "ntev2-e3-muon-shift",
        "ntev2",
        3,
        "muon",
        "shift",
        ("structure", "head", "normalization"),
    ),
    Lane(
        "qhflow3-e3-muon-shift",
        "qhflow3",
        3,
        "muon",
        "shift",
        ("structure", "head", "normalization"),
    ),
    Lane(
        "ntev2-e2-muon-shift",
        "ntev2",
        2,
        "muon",
        "shift",
        ("structure",),
    ),
    Lane(
        "qhflow3-e2-muon-shift",
        "qhflow3",
        2,
        "muon",
        "shift",
        ("structure",),
    ),
    Lane(
        "ntev2-e3-native-shift",
        "ntev2",
        3,
        "native",
        "shift",
        ("head",),
    ),
    Lane(
        "qhflow3-e3-native-shift",
        "qhflow3",
        3,
        "native",
        "shift",
        ("head",),
    ),
    Lane(
        "ntev2-e3-muon-raw",
        "ntev2",
        3,
        "muon",
        "raw",
        ("normalization",),
    ),
    Lane(
        "qhflow3-e3-muon-raw",
        "qhflow3",
        3,
        "muon",
        "raw",
        ("normalization",),
    ),
)
LANE_BY_ID = {lane.id: lane for lane in LANES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_suite_inputs(base: MaloqConfig) -> dict[str, object]:
    if Path(base.dataset.dbpath) != EXPECTED_DATABASE:
        raise ValueError(
            f"Base config must use {EXPECTED_DATABASE}, got {base.dataset.dbpath!r}."
        )
    if not EXPECTED_DATABASE.is_file():
        raise FileNotFoundError(EXPECTED_DATABASE)
    if Path(base.loss.scale_shift_path or "") != EXPECTED_SCALE_SHIFT:
        raise ValueError(
            "Base config does not reference the fixed SHIFT statistics "
            f"artifact: {base.loss.scale_shift_path!r}."
        )
    if not EXPECTED_SCALE_SHIFT.is_file():
        raise FileNotFoundError(EXPECTED_SCALE_SHIFT)
    scale_shift_sha256 = _sha256(EXPECTED_SCALE_SHIFT)
    if scale_shift_sha256 != EXPECTED_SCALE_SHIFT_SHA256:
        raise ValueError(
            "SHIFT artifact SHA-256 mismatch: "
            f"{scale_shift_sha256} != {EXPECTED_SCALE_SHIFT_SHA256}."
        )
    if base.tracking.wandb_project != WANDB_PROJECT:
        raise ValueError(
            f"W&B project must be exactly {WANDB_PROJECT!r}, got "
            f"{base.tracking.wandb_project!r}."
        )

    import torch
    from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

    scale_shift = torch.load(
        EXPECTED_SCALE_SHIFT,
        map_location="cpu",
        weights_only=False,
    )
    provenance = scale_shift.get("provenance", {})
    expected_provenance = {
        "dataset_name": "nablaDFT",
        "database_path": str(EXPECTED_DATABASE),
        "num_train": FULL_COUNTS["train"],
        "validation_rows_in_statistics": 0,
        "test_rows_in_statistics": 0,
        "loss_target": "fock_matrix",
        "rcut_orbitals": 8.0,
        "normalization": "elementwise_standardize_l0_node_labels",
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"SHIFT artifact provenance mismatch: {mismatches}")

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
        "scale_shift": str(EXPECTED_SCALE_SHIFT),
        "scale_shift_sha256": scale_shift_sha256,
        "scale_shift_num_train": provenance["num_train"],
        "scale_shift_validation_rows": provenance["validation_rows_in_statistics"],
    }


def _architecture_overrides(lane: Lane) -> dict[str, object]:
    if lane.architecture == "maloq":
        if lane.edge_layers != 3:
            raise ValueError("Canonical MALOQ is fixed to three interleaved edges.")
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


def _display_architecture(lane: Lane) -> str:
    labels = {
        "maloq": "MALOQ",
        "ntev2": "NTEV2",
        "qhflow3": "QHFlow3",
    }
    return f"{labels[lane.architecture]}-E{lane.edge_layers}"


def _build_lane_config(
    base: MaloqConfig,
    lane: Lane,
    scope: Scope,
    output_root: Path,
) -> MaloqConfig:
    payload = base.model_dump(mode="python")
    run_name = f"nabladft-v2-ofat-{lane.id}"
    display_name = (
        f"NablaDFT | {_display_architecture(lane)} | "
        f"{'Muon' if lane.head == 'muon' else 'Native'} | "
        f"{'SHIFT' if lane.normalization == 'shift' else 'RAW'} | V2"
    )

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
        head_type="maloq_muon" if lane.head == "muon" else "maloq",
        num_edge_layers=lane.edge_layers,
        **_architecture_overrides(lane),
    )
    payload["optimization"]["num_epochs"] = 1 if scope == "smoke" else 20
    shift_enabled = lane.normalization == "shift"
    payload["loss"].update(
        scale_and_shift=shift_enabled,
        scale_shift_mode="shift_only",
        scale_shift_path=(str(EXPECTED_SCALE_SHIFT) if shift_enabled else None),
    )
    lane_tags = [
        f"architecture:{lane.architecture}",
        f"edge-layers:{lane.edge_layers}",
        f"head:{lane.head}",
        ("normalization:l0-shift-only" if shift_enabled else "normalization:raw"),
        *[f"axis:{axis}" for axis in lane.axes],
    ]
    payload["tracking"].update(
        use_wandb=scope == "full",
        wandb_project=WANDB_PROJECT,
        wandb_run_name=display_name,
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
    if config.model.backbone_type != expected_backbones[lane.architecture]:
        raise ValueError("Lane architecture/backbone mismatch.")
    if config.model.num_edge_layers != lane.edge_layers:
        raise ValueError("Lane edge-depth mismatch.")
    expected_head = "maloq_muon" if lane.head == "muon" else "maloq"
    if config.model.head_type != expected_head:
        raise ValueError("Lane head mismatch.")
    if config.optimization.optimizer_type != "muon":
        raise ValueError(
            "Both head lanes must keep optimizer_type='muon'; only the head "
            "parameterization and routing may change."
        )
    shift_enabled = lane.normalization == "shift"
    if config.loss.scale_and_shift is not shift_enabled:
        raise ValueError("Lane normalization mismatch.")
    if shift_enabled and config.loss.scale_shift_mode != "shift_only":
        raise ValueError("SHIFT lanes must use mean subtraction only.")
    if config.tracking.wandb_project != WANDB_PROJECT:
        raise ValueError("W&B project drifted from the V2 project.")
    if config.optimization.num_epochs != (1 if scope == "smoke" else 20):
        raise ValueError("Scope/epoch mismatch.")
    if config.splits.batch_size != 5:
        raise ValueError("Per-rank micro-batch must remain 5.")
    if config.optimization.gradient_accumulation_steps != 2:
        raise ValueError("Gradient accumulation must remain 2.")
    if config.splits.shuffle or config.splits.distribute_graphs:
        raise ValueError("The OFAT suite requires ordered data parallelism.")


def _config_preview(config: MaloqConfig, lane: Lane) -> dict[str, object]:
    return {
        **asdict(lane),
        "backbone_type": config.model.backbone_type,
        "head_type": config.model.head_type,
        "output_l_embedding_dim": config.model.output_l_embedding_dim,
        "num_mp_layers": config.model.num_mp_layers,
        "num_edge_layers": config.model.num_edge_layers,
        "num_epochs": config.optimization.num_epochs,
        "num_train": config.splits.num_train,
        "num_val": config.splits.num_val,
        "batch_size_per_rank": config.splits.batch_size,
        "gradient_accumulation_steps": (
            config.optimization.gradient_accumulation_steps
        ),
        "effective_batch_size_at_world_size_2": (
            config.splits.batch_size
            * 2
            * config.optimization.gradient_accumulation_steps
        ),
        "scale_and_shift": config.loss.scale_and_shift,
        "scale_shift_mode": (
            config.loss.scale_shift_mode if config.loss.scale_and_shift else None
        ),
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
        preview_base = PROJECT_ROOT / "outputs/_config-preview/nabladft-v2-ofat"
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
                    "suite": "nabladft-v2-ofat",
                    "workflow": (
                        f"{TrainingWorkflowV2Fixed.__module__}."
                        f"{TrainingWorkflowV2Fixed.__name__}"
                    ),
                    "base_config": str(base_config_path),
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
            "suite": "nabladft-v2-ofat",
            "scope": scope,
            "lane": asdict(lane),
            "base_config": str(base_config_path),
            "base_config_sha256": _sha256(base_config_path),
            "inputs": input_metadata,
            "config": typed_config.model_dump(mode="json"),
        }
        (output_root / "resolved_ofat_config.json").write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n"
        )

    workflow_config = typed_config.to_workflow_config()
    TrainingWorkflowV2Fixed(workflow_config).run()


if __name__ == "__main__":
    main()
