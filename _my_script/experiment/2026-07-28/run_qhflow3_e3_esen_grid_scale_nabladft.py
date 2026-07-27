#!/usr/bin/env python3
"""Run the matched QHFlow3-E3 eSEN-grid SCALE lane on NablaDFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
SOURCE_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-28/"
    "qhflow3_e3_esen_grid_scale_nabladft.yaml"
)
EXPECTED_DATABASE = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/"
    "hamiltonian_databases/train_2k.db"
)
EXPECTED_SCALE_SHIFT = (
    PROJECT_ROOT
    / "outputs/scale-shift-statistics/"
    "nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt"
)
EXPECTED_SCALE_SHIFT_SHA256 = (
    "375167ad551fb0b60dbe9cd049a4995276b54ce075e09906639ef3daa4f79475"
)
EXPECTED_RUN_NAME = "nabladft-v2-qhflow3-e3-10x11-muon-scale"
EXPECTED_DISPLAY_NAME = "NablaDFT | QHFlow3-E3-10x11 | Muon | SCALE | V2"
WANDB_PROJECT = "MALOQ-nablaDFT-v2"
FULL_COUNTS = {"train": 12081, "val": 64, "test": 0}
SMOKE_COUNTS = {"train": 20, "val": 20, "test": 0}
Scope = Literal["validate", "smoke", "full"]

for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _select_rank_local_cuda_before_workflow_import() -> None:
    """Select the MPI-local device before eSEN/CuPy is imported."""
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}.")


def _validate_base_contract(config: MaloqConfig) -> None:
    """Reject any drift outside the requested grid and SCALE comparison."""
    _require_equal("dataset_name", config.dataset.dataset_name, "nablaDFT")
    _require_equal("database", Path(config.dataset.dbpath), EXPECTED_DATABASE)
    _require_equal("dataset_format", config.dataset.dataset_format, "auto")
    _require_equal("open_shell", config.dataset.open_shell, False)

    _require_equal("train_or_eval", config.execution.train_or_eval, "train")
    _require_equal("train_backbone", config.execution.train_backbone, True)
    _require_equal("train_head", config.execution.train_head, True)
    _require_equal("compute_total_energy", config.execution.compute_total_energy, False)
    _require_equal("compute_eigenvalues", config.execution.compute_eigenvalues, True)

    _require_equal("num_train", config.splits.num_train, FULL_COUNTS["train"])
    _require_equal("num_val", config.splits.num_val, FULL_COUNTS["val"])
    _require_equal("num_test", config.splits.num_test, FULL_COUNTS["test"])
    _require_equal("batch_size", config.splits.batch_size, 5)
    _require_equal("shuffle", config.splits.shuffle, False)
    _require_equal("distribute_graphs", config.splits.distribute_graphs, False)
    _require_equal("dist_backend", config.splits.dist_backend, "nccl")

    model = config.model
    _require_equal("model_variant", model.model_variant, EXPECTED_RUN_NAME)
    _require_equal("backbone_type", model.backbone_type, "qhflow3")
    _require_equal("head_type", model.head_type, "maloq_muon")
    _require_equal("wigner_backend", model.wigner_backend, "torch")
    _require_equal("l_embedding_dim", model.l_embedding_dim, 128)
    _require_equal("hidden_dim", model.hidden_dim, 128)
    _require_equal("output_l_embedding_dim", model.output_l_embedding_dim, 64)
    _require_equal("num_distance_basis", model.num_distance_basis, 512)
    _require_equal("num_mp_layers", model.num_mp_layers, 3)
    _require_equal("num_edge_layers", model.num_edge_layers, 3)
    _require_equal("message_type", model.message_type, "source-target")
    _require_equal("rcut_orbitals", model.rcut_orbitals, 8.0)
    _require_equal("rcut_gaussian", model.rcut_gaussian, 16.0)
    _require_equal("gaussian_width", model.gaussian_width, 1.0)
    _require_equal("reduce_edge", model.reduce_edge, False)
    _require_equal("reduce_node", model.reduce_node, True)
    _require_equal("reduce_node_intra", model.reduce_node_intra, True)
    _require_equal("mlp_type", model.mlp_type, "grid")
    _require_equal("esen_grid_resolution", model.esen_grid_resolution, None)
    _require_equal("qhflow3_max_radius", model.qhflow3_max_radius, 12.0)
    _require_equal("qhflow3_radius_embed_dim", model.qhflow3_radius_embed_dim, 32)
    _require_equal(
        "qhflow3_grid_resolution",
        model.qhflow3_grid_resolution,
        None,
    )
    _require_equal(
        "qhflow3_grid_ffn_chunk_size",
        model.qhflow3_grid_ffn_chunk_size,
        512,
    )
    _require_equal("qhflow3_use_overlap", model.qhflow3_use_overlap, True)
    _require_equal(
        "qhflow3_muonize_output_projection",
        model.qhflow3_muonize_output_projection,
        False,
    )

    optimization = config.optimization
    _require_equal("num_epochs", optimization.num_epochs, 20)
    _require_equal("lr_init", optimization.lr_init, 0.0005)
    _require_equal("optimizer_type", optimization.optimizer_type, "muon")
    _require_equal("weight_decay", optimization.weight_decay, 0.0001)
    _require_equal("muon_lr", optimization.muon_lr, 0.02)
    _require_equal("muon_momentum", optimization.muon_momentum, 0.95)
    _require_equal("muon_nesterov", optimization.muon_nesterov, True)
    _require_equal("muon_ns_steps", optimization.muon_ns_steps, 5)
    _require_equal("muon_adamw_lr", optimization.muon_adamw_lr, 0.0005)
    _require_equal("muon_adamw_betas", optimization.muon_adamw_betas, (0.9, 0.95))
    _require_equal("muon_adamw_eps", optimization.muon_adamw_eps, 1.0e-10)
    _require_equal(
        "muon_output_projection_policy",
        optimization.muon_output_projection_policy,
        "shape_muon",
    )
    _require_equal("gradient_clip_val", optimization.gradient_clip_val, 1.0)
    _require_equal(
        "gradient_accumulation_steps",
        optimization.gradient_accumulation_steps,
        2,
    )
    _require_equal("scheduler_type", optimization.scheduler_type, "warmup_polynomial")
    _require_equal("warmup_steps", optimization.warmup_steps, 1000)
    _require_equal("scheduler_power", optimization.scheduler_power, 1.0)
    _require_equal("min_lr_ratio", optimization.min_lr_ratio, 0.0)
    _require_equal("step_every_epoch", optimization.step_every_epoch, False)

    loss = config.loss
    _require_equal("loss_target", loss.loss_target, "fock_matrix")
    _require_equal("train_loss", loss.train_loss, "rmse_mse_padded_loss")
    _require_equal("test_loss", loss.test_loss, "l1_unpadded_loss")
    _require_equal("scale_and_shift", loss.scale_and_shift, True)
    _require_equal("scale_shift_mode", loss.scale_shift_mode, "standardize")
    _require_equal("scale_shift_path", Path(loss.scale_shift_path or ""), EXPECTED_SCALE_SHIFT)
    _require_equal("compute_uncoupled_loss", loss.compute_uncoupled_loss, False)
    _require_equal("delta_learning", loss.delta_learning, False)

    _require_equal("dtype", config.runtime.dtype, "float32")
    _require_equal("seed", config.runtime.seed, 44)
    _require_equal("wandb_project", config.tracking.wandb_project, WANDB_PROJECT)
    _require_equal("wandb_entity", config.tracking.wandb_entity, "kaist-korea")
    _require_equal("wandb_run_name", config.tracking.wandb_run_name, EXPECTED_DISPLAY_NAME)
    required_tags = {
        "architecture:qhflow3",
        "edge-layers:3",
        "head:muon",
        "grid:esen-default-10x11",
        "normalization:l0-standardize",
    }
    missing_tags = required_tags.difference(config.tracking.wandb_tags)
    if missing_tags:
        raise ValueError(f"Required W&B tags are missing: {sorted(missing_tags)}")


def _validate_inputs(config: MaloqConfig) -> dict[str, object]:
    if not EXPECTED_DATABASE.is_file():
        raise FileNotFoundError(EXPECTED_DATABASE)
    if not EXPECTED_SCALE_SHIFT.is_file():
        raise FileNotFoundError(EXPECTED_SCALE_SHIFT)

    scale_shift_sha256 = _sha256(EXPECTED_SCALE_SHIFT)
    if scale_shift_sha256 != EXPECTED_SCALE_SHIFT_SHA256:
        raise ValueError(
            "SCALE artifact SHA-256 mismatch: "
            f"{scale_shift_sha256} != {EXPECTED_SCALE_SHIFT_SHA256}."
        )

    import torch
    from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
    from maloq.helm.qhf_layer.so3 import SO3_Grid

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
        raise ValueError(f"SCALE artifact provenance mismatch: {mismatches}")

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

    grid = SO3_Grid(
        lmax=4,
        mmax=4,
        resolution=config.model.qhflow3_grid_resolution,
        rescale=True,
    )
    grid_shape = (grid.lat_resolution, grid.long_resolution)
    if grid_shape != (10, 11):
        raise ValueError(f"Expected eSEN default grid 10x11, got {grid_shape}.")

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
        "configured_grid_resolution": config.model.qhflow3_grid_resolution,
        "effective_esen_grid_shape": list(grid_shape),
    }


def _build_scope_config(
    base: MaloqConfig,
    scope: Scope,
    output_root: Path | None,
) -> MaloqConfig:
    payload = base.model_dump(mode="python")
    counts = SMOKE_COUNTS if scope == "smoke" else FULL_COUNTS
    payload["splits"].update(
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
    )
    payload["optimization"]["num_epochs"] = 1 if scope == "smoke" else 20
    payload["tracking"].update(
        use_wandb=scope == "full",
        wandb_job_type=scope,
    )
    if output_root is not None:
        payload["dataset"].update(
            output_folder=str(output_root),
            run_name=EXPECTED_RUN_NAME,
        )

    config = MaloqConfig.model_validate(payload)
    _require_equal(
        "scope num_epochs",
        config.optimization.num_epochs,
        1 if scope == "smoke" else 20,
    )
    _require_equal("scope num_train", config.splits.num_train, counts["train"])
    _require_equal("scope num_val", config.splits.num_val, counts["val"])
    _require_equal("scope use_wandb", config.tracking.use_wandb, scope == "full")
    _require_equal("scope wandb_job_type", config.tracking.wandb_job_type, scope)
    _require_equal("scope grid", config.model.qhflow3_grid_resolution, None)
    _require_equal("scope normalization", config.loss.scale_shift_mode, "standardize")
    return config


def _preview(config: MaloqConfig, metadata: dict[str, object]) -> dict[str, object]:
    return {
        "experiment": EXPECTED_RUN_NAME,
        "display_name": config.tracking.wandb_run_name,
        "backbone_type": config.model.backbone_type,
        "head_type": config.model.head_type,
        "num_mp_layers": config.model.num_mp_layers,
        "num_edge_layers": config.model.num_edge_layers,
        "configured_grid_resolution": config.model.qhflow3_grid_resolution,
        "effective_esen_grid_shape": metadata["effective_esen_grid_shape"],
        "qhflow3_use_overlap": config.model.qhflow3_use_overlap,
        "scale_and_shift": config.loss.scale_and_shift,
        "scale_shift_mode": config.loss.scale_shift_mode,
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
        "seed": config.runtime.seed,
        "wandb_project": config.tracking.wandb_project,
        "inputs": metadata,
    }


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


def _rank() -> int:
    return int(
        os.environ.get(
            "OMPI_COMM_WORLD_RANK",
            os.environ.get("RANK", "0"),
        )
    )


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    base = MaloqConfig.from_file(config_path)
    _validate_base_contract(base)
    metadata = _validate_inputs(base)
    scope: Scope = args.scope

    if scope == "validate":
        if args.output_root is not None:
            raise SystemExit("--output-root is not used with --scope validate.")
        print(json.dumps(_preview(base, metadata), indent=2, sort_keys=True))
        return

    if args.output_root is None:
        raise SystemExit("--output-root is required for smoke/full.")
    output_root = args.output_root.expanduser().resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_root == outputs_root or outputs_root not in output_root.parents:
        raise SystemExit(
            f"Output must be a run directory below {outputs_root}: {output_root}"
        )

    typed_config = _build_scope_config(base, scope, output_root)
    if _rank() == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        resolved_payload = {
            "experiment": EXPECTED_RUN_NAME,
            "scope": scope,
            "config_source": str(config_path),
            "config_source_sha256": _sha256(config_path),
            "inputs": metadata,
            "config": typed_config.model_dump(mode="json"),
        }
        (output_root / "resolved_qhflow3_esen_grid_scale_config.json").write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n"
        )

    TrainingWorkflowV2Fixed(typed_config.to_workflow_config()).run()


if __name__ == "__main__":
    main()
