#!/usr/bin/env python3
"""Run one matched E3/Muon/SHIFT endpoint-flow lane on NablaDFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
SOURCE_ROOT = PROJECT_ROOT / "src"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "_my_script/experiment/2026-07-28/"
    "flow_matching_e3_muon_shift_nabladft.yaml"
)
EXPECTED_DATABASE = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db"
)
EXPECTED_SCALE_SHIFT = (
    OUTPUTS_ROOT / "scale-shift-statistics/"
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
    """Select the torchrun-local CUDA device before eSEN/CuPy is imported."""
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("OMPI_COMM_WORLD_SIZE", "1"),
        )
    )
    if world_size <= 1:
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Two-rank training requires visible CUDA devices.")
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"),
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

from maloq.experimental.flow_matching import (  # noqa: E402
    FEATURE_SLUG,
    PROFILE_ID,
    EndpointFlowMaloqConfig,
    FlowMatchingWorkflow,
)

Architecture = Literal["maloq", "ntev2", "qhflow3"]
Scope = Literal["validate", "smoke", "full"]


@dataclass(frozen=True)
class Lane:
    id: str
    architecture: Architecture
    display_architecture: str
    backbone_type: Literal["esen", "maloq_nte_v2", "qhflow3"]
    output_l_embedding_dim: int | None
    mlp_type: Literal["spectral", "grid"]


LANES = (
    Lane(
        id="maloq-e3-muon-shift",
        architecture="maloq",
        display_architecture="MALOQ-E3",
        backbone_type="esen",
        output_l_embedding_dim=None,
        mlp_type="spectral",
    ),
    Lane(
        id="ntev2-e3-muon-shift",
        architecture="ntev2",
        display_architecture="NTEV2-E3",
        backbone_type="maloq_nte_v2",
        output_l_embedding_dim=64,
        mlp_type="grid",
    ),
    Lane(
        id="qhflow3-e3-muon-shift",
        architecture="qhflow3",
        display_architecture="QHFlow3-E3",
        backbone_type="qhflow3",
        output_l_embedding_dim=64,
        mlp_type="grid",
    ),
)
LANE_BY_ID = {lane.id: lane for lane in LANES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_suite_inputs(base: EndpointFlowMaloqConfig) -> dict[str, object]:
    if base.experimental_feature != FEATURE_SLUG:
        raise ValueError("Experimental feature slug mismatch.")
    if base.experimental_profile != PROFILE_ID:
        raise ValueError("Experimental profile mismatch.")
    if Path(base.dataset.dbpath) != EXPECTED_DATABASE:
        raise ValueError(
            f"Base config must use {EXPECTED_DATABASE}, got {base.dataset.dbpath!r}."
        )
    if not EXPECTED_DATABASE.is_file():
        raise FileNotFoundError(EXPECTED_DATABASE)
    if Path(base.loss.scale_shift_path or "") != EXPECTED_SCALE_SHIFT:
        raise ValueError(
            "Base config does not reference the fixed SHIFT artifact: "
            f"{base.loss.scale_shift_path!r}."
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
            f"W&B project must be {WANDB_PROJECT!r}, got "
            f"{base.tracking.wandb_project!r}."
        )

    from maloq.helm.qhf_layer.so3 import SO3_Grid

    grid = SO3_Grid(
        4,
        4,
        resolution=base.model.qhflow3_grid_resolution,
        rescale=True,
    )
    grid_shape = (int(grid.lat_resolution), int(grid.long_resolution))
    if grid_shape != (10, 11):
        raise ValueError(
            f"FlowMatching QHFlow3 grid must resolve to 10x11; got {grid_shape}."
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
        key: (provenance.get(key), expected)
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
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
        "effective_qhflow3_grid_shape": list(grid_shape),
    }


def _build_lane_config(
    base: EndpointFlowMaloqConfig,
    lane: Lane,
    scope: Scope,
    output_root: Path,
) -> EndpointFlowMaloqConfig:
    payload = base.model_dump(mode="python")
    run_name = f"nabladft-flow-matching-{lane.id}-v2"
    display_name = (
        f"NablaDFT | {lane.display_architecture} | Muon | SHIFT | FlowMatching | V2"
    )
    counts = SMOKE_COUNTS if scope == "smoke" else FULL_COUNTS

    payload["dataset"].update(
        run_name=run_name,
        output_folder=str(output_root),
    )
    payload["splits"].update(
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
    )
    payload["model"].update(
        model_variant=run_name,
        backbone_type=lane.backbone_type,
        head_type="maloq_muon",
        num_mp_layers=3,
        num_edge_layers=3,
        output_l_embedding_dim=lane.output_l_embedding_dim,
        mlp_type=lane.mlp_type,
    )
    payload["optimization"]["num_epochs"] = 1 if scope == "smoke" else 20
    payload["loss"].update(
        scale_and_shift=True,
        scale_shift_mode="shift_only",
        scale_shift_path=str(EXPECTED_SCALE_SHIFT),
    )
    lane_tags = [
        f"architecture:{lane.architecture}",
        "edge-layers:3",
        "head:muon",
        "normalization:l0-shift-only",
        "axis:structure",
    ]
    if lane.architecture in {"ntev2", "qhflow3"}:
        lane_tags.append("grid:10x11")

    payload["tracking"].update(
        use_wandb=scope == "full",
        wandb_project=WANDB_PROJECT,
        wandb_run_name=display_name,
        wandb_job_type=scope,
        wandb_tags=tuple(
            dict.fromkeys([*payload["tracking"]["wandb_tags"], *lane_tags])
        ),
    )

    config = EndpointFlowMaloqConfig.model_validate(payload)
    _validate_lane_contract(config, lane, scope)
    return config


def _validate_lane_contract(
    config: EndpointFlowMaloqConfig,
    lane: Lane,
    scope: Scope,
) -> None:
    if config.model.backbone_type != lane.backbone_type:
        raise ValueError("Lane architecture/backbone mismatch.")
    if config.model.num_mp_layers != 3 or config.model.num_edge_layers != 3:
        raise ValueError("Every lane must retain matched MP3/E3 depth.")
    if config.model.head_type != "maloq_muon":
        raise ValueError("Every lane must use the Muon-visible matrix head.")
    if config.model.output_l_embedding_dim != lane.output_l_embedding_dim:
        raise ValueError("Lane output embedding width mismatch.")
    if config.model.mlp_type != lane.mlp_type:
        raise ValueError("Lane MLP type mismatch.")
    expected_model = {
        "atom_scalar_embedding_mode": "element_charge_spin",
        "wigner_backend": "torch",
        "l_embedding_dim": 128,
        "hidden_dim": 128,
        "num_distance_basis": 512,
        "message_type": "source-target",
        "rcut_orbitals": 8.0,
        "rcut_gaussian": 16.0,
        "gaussian_width": 1.0,
        "reduce_edge": False,
        "reduce_node": True,
        "reduce_node_intra": True,
        "esen_grid_resolution": None,
        "qhflow3_max_radius": 12.0,
        "qhflow3_radius_embed_dim": 32,
        "qhflow3_grid_resolution": None,
        "qhflow3_grid_ffn_chunk_size": 512,
        "qhflow3_use_overlap": True,
        "qhflow3_muonize_output_projection": False,
    }
    model_mismatches = {
        key: (getattr(config.model, key), expected)
        for key, expected in expected_model.items()
        if getattr(config.model, key) != expected
    }
    if model_mismatches:
        raise ValueError(f"Matched model settings drifted: {model_mismatches}")

    optimizer = config.optimization
    expected_optimizer = {
        "optimizer_type": "muon",
        "lr_init": 0.0005,
        "weight_decay": 0.0001,
        "muon_lr": 0.02,
        "muon_momentum": 0.95,
        "muon_nesterov": True,
        "muon_ns_steps": 5,
        "muon_adamw_lr": 0.0005,
        "muon_adamw_eps": 1.0e-10,
        "muon_output_projection_policy": "shape_muon",
        "gradient_clip_val": 1.0,
        "gradient_accumulation_steps": 2,
        "scheduler_type": "warmup_polynomial",
        "warmup_steps": 1000,
        "scheduler_power": 1.0,
        "min_lr_ratio": 0.0,
        "step_every_epoch": False,
    }
    mismatches = {
        key: (getattr(optimizer, key), expected)
        for key, expected in expected_optimizer.items()
        if getattr(optimizer, key) != expected
    }
    if tuple(optimizer.muon_adamw_betas) != (0.9, 0.95):
        mismatches["muon_adamw_betas"] = (
            tuple(optimizer.muon_adamw_betas),
            (0.9, 0.95),
        )
    if mismatches:
        raise ValueError(f"Matched Muon/AuxAdamW settings drifted: {mismatches}")

    expected_epochs = 1 if scope == "smoke" else 20
    if optimizer.num_epochs != expected_epochs:
        raise ValueError("Scope/epoch mismatch.")
    expected_counts = SMOKE_COUNTS if scope == "smoke" else FULL_COUNTS
    if (
        config.splits.num_train != expected_counts["train"]
        or config.splits.num_val != expected_counts["val"]
        or config.splits.num_test != expected_counts["test"]
    ):
        raise ValueError("Scope/split mismatch.")
    if config.splits.batch_size != 5:
        raise ValueError("Per-rank micro-batch must remain 5.")
    if config.splits.shuffle or config.splits.distribute_graphs:
        raise ValueError("The suite requires ordered data parallelism.")
    if (
        not config.loss.scale_and_shift
        or config.loss.scale_shift_mode != "shift_only"
        or Path(config.loss.scale_shift_path or "") != EXPECTED_SCALE_SHIFT
    ):
        raise ValueError("Every lane must use the frozen SHIFT-only artifact.")
    if config.loss.compute_uncoupled_loss or config.loss.delta_learning:
        raise ValueError("Endpoint flow must stay coupled and direct.")
    if (
        config.loss.loss_target != "fock_matrix"
        or config.loss.train_loss != "mse_padded_loss"
        or config.loss.test_loss != "mse_padded_loss"
    ):
        raise ValueError("Endpoint-flow matrix loss declarations drifted.")
    if (
        not config.tracking.validation_matrix_metrics
        or config.tracking.validation_matrix_metrics_frequency != 1
    ):
        raise ValueError("Euler-endpoint matrix metrics must run every epoch.")
    if config.tracking.wandb_project != WANDB_PROJECT:
        raise ValueError("W&B project drifted from MALOQ-nablaDFT-v2.")
    if "FlowMatching" not in (config.tracking.wandb_run_name or ""):
        raise ValueError("W&B identity must distinguish FlowMatching from baselines.")
    if config.runtime.seed != 44 or config.runtime.dtype != "float32":
        raise ValueError("Runtime seed/dtype drifted from the matched baselines.")
    execution = config.execution
    if (
        execution.train_or_eval != "train"
        or not execution.train_backbone
        or not execution.train_head
        or execution.compute_total_energy
        or not execution.compute_eigenvalues
    ):
        raise ValueError("Matched training execution settings drifted.")


def _config_preview(
    config: EndpointFlowMaloqConfig,
    lane: Lane,
) -> dict[str, object]:
    return {
        **asdict(lane),
        "run_name": config.dataset.run_name,
        "backbone_type": config.model.backbone_type,
        "head_type": config.model.head_type,
        "num_mp_layers": config.model.num_mp_layers,
        "num_edge_layers": config.model.num_edge_layers,
        "output_l_embedding_dim": config.model.output_l_embedding_dim,
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
        "scale_shift_mode": config.loss.scale_shift_mode,
        "qhflow3_grid_resolution": config.model.qhflow3_grid_resolution,
        "validation_matrix_metrics": config.tracking.validation_matrix_metrics,
        "flow_matching": config.flow_matching.model_dump(mode="json"),
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
            "RANK",
            os.environ.get("OMPI_COMM_WORLD_RANK", "0"),
        )
    )


def _world_size() -> int:
    return int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("OMPI_COMM_WORLD_SIZE", "1"),
        )
    )


def main() -> None:
    args = _parse_args()
    base_config_path = args.base_config.expanduser().resolve()
    if not base_config_path.is_file():
        raise SystemExit(f"Base config not found: {base_config_path}")
    base = EndpointFlowMaloqConfig.from_file(base_config_path)
    input_metadata = _validate_suite_inputs(base)

    scope: Scope = args.scope
    if args.lane == "all":
        if scope != "validate":
            raise SystemExit("--lane all is allowed only with --scope validate.")
        selected_lanes = LANES
    else:
        selected_lanes = (LANE_BY_ID[args.lane],)

    if scope == "validate":
        preview_base = OUTPUTS_ROOT / "_config-preview/nabladft-flow-matching-e3"
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
                    "suite": "nabladft-flow-matching-e3-muon-shift-v2",
                    "workflow": (
                        f"{FlowMatchingWorkflow.__module__}."
                        f"{FlowMatchingWorkflow.__name__}"
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
    if _world_size() != 2:
        raise SystemExit(f"Smoke/full requires exactly two ranks; got {_world_size()}.")
    if args.output_root is None:
        raise SystemExit("--output-root is required for smoke/full.")
    output_root = args.output_root.expanduser().resolve()
    canonical_outputs = OUTPUTS_ROOT.resolve()
    if output_root == canonical_outputs or canonical_outputs not in output_root.parents:
        raise SystemExit(
            f"Output must be a lane directory below {canonical_outputs}: {output_root}"
        )

    lane = selected_lanes[0]
    typed_config = _build_lane_config(base, lane, scope, output_root)
    launch_token = os.environ.get("MALOQ_FLOW_LAUNCH_TOKEN")
    if (
        not launch_token
        or len(launch_token) > 128
        or any(
            not (character.isalnum() or character in "-._")
            for character in launch_token
        )
    ):
        raise SystemExit("Training requires a valid MALOQ_FLOW_LAUNCH_TOKEN.")
    ready_marker = output_root / ".flow_matching_ready"

    rank = _rank()
    if rank == 0:
        if output_root.exists():
            raise SystemExit(f"Output already exists: {output_root}")
        output_root.mkdir(parents=True, exist_ok=False)
        resolved_payload = {
            "suite": "nabladft-flow-matching-e3-muon-shift-v2",
            "scope": scope,
            "lane": asdict(lane),
            "base_config": str(base_config_path),
            "base_config_sha256": _sha256(base_config_path),
            "inputs": input_metadata,
            "config": typed_config.model_dump(mode="json"),
        }
        (output_root / "resolved_flow_matching_config.json").write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n"
        )
        temporary_ready = output_root / f".flow_matching_ready.rank-{rank}.tmp"
        temporary_ready.write_text(launch_token + "\n")
        os.replace(temporary_ready, ready_marker)

    else:
        deadline = time.monotonic() + 30.0
        while True:
            try:
                observed_token = ready_marker.read_text().strip()
            except FileNotFoundError:
                observed_token = None
            if observed_token == launch_token:
                break
            if observed_token is not None:
                raise RuntimeError("Rank 0 ready marker has a different launch token.")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Rank {rank} timed out waiting for rank 0 readiness at {output_root}."
                )
            time.sleep(0.1)

    FlowMatchingWorkflow(typed_config.to_workflow_config()).run()


if __name__ == "__main__":
    main()
