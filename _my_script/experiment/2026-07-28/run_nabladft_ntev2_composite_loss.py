#!/usr/bin/env python3
"""Run matched NTEV2-E3 Muon SHIFT composite-loss profiles on NablaDFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project/MALOQ")
SOURCE_ROOT = PROJECT_ROOT / "src"
EXPERIMENT_ROOT = PROJECT_ROOT / "_my_script/experiment/2026-07-28"
DEFAULT_BASE_CONFIG = EXPERIMENT_ROOT / "nabladft_ntev2_composite_loss_common.yaml"
EXPECTED_DATABASE = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db"
)
EXPECTED_SCALE_SHIFT = PROJECT_ROOT / (
    "outputs/scale-shift-statistics/"
    "nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt"
)
EXPECTED_SCALE_SHIFT_SHA256 = (
    "375167ad551fb0b60dbe9cd049a4995276b54ce075e09906639ef3daa4f79475"
)
WANDB_PROJECT = "MALOQ-nablaDFT-v2"
FULL_COUNTS = {"train": 12081, "val": 64, "test": 0}
SMOKE_COUNTS = {"train": 20, "val": 20, "test": 0}
EXPECTED_PARAMETERS = 33_891_021
BASELINE_OUTPUT = PROJECT_ROOT / (
    "outputs/nabladft-v2-ofat-ntev2-e3-muon-shift-2gpu-eb20-mb5-ga2-"
    "full-e20-20260727-181411-3321867"
)

for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _select_rank_local_cuda_before_workflow_import() -> None:
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    if world_size <= 1:
        return
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Two-rank training requires visible CUDA devices.")
    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))
    )
    visible_devices = torch.cuda.device_count()
    device_index = 0 if visible_devices == 1 else local_rank
    if not 0 <= device_index < visible_devices:
        raise SystemExit(
            f"Local rank {local_rank} cannot select one of {visible_devices} visible GPUs."
        )
    torch.cuda.set_device(device_index)


_select_rank_local_cuda_before_workflow_import()

from maloq.core.config import MaloqConfig  # noqa: E402
from maloq.experimental.matrix_composite_loss import (  # noqa: E402
    build_matrix_composite_loss_workflow,
    get_composite_loss_profile,
)


Scope = Literal["validate", "smoke", "full"]


@dataclass(frozen=True)
class Variant:
    id: str
    profile_id: str
    display_loss: str


VARIANTS = (
    Variant("rmse-mse-mae", "rmse_mse_mae", "RMSE+MSE+MAE"),
    Variant("10x-rmse-mse-mae", "10x_rmse_mse_mae", "10x(RMSE+MSE+MAE)"),
)
VARIANT_BY_ID = {variant.id: variant for variant in VARIANTS}


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
    checks = (
        ("dataset", base.dataset.dataset_name, "nablaDFT"),
        ("database", Path(base.dataset.dbpath), EXPECTED_DATABASE),
        ("open_shell", base.dataset.open_shell, False),
        ("num_train", base.splits.num_train, 12081),
        ("num_val", base.splits.num_val, 64),
        ("num_test", base.splits.num_test, 0),
        ("batch_size", base.splits.batch_size, 5),
        ("shuffle", base.splits.shuffle, False),
        ("distribute_graphs", base.splits.distribute_graphs, False),
        ("backbone_type", base.model.backbone_type, "maloq_nte_v2"),
        ("head_type", base.model.head_type, "maloq_muon"),
        ("l_embedding_dim", base.model.l_embedding_dim, 128),
        ("hidden_dim", base.model.hidden_dim, 128),
        ("output_l_embedding_dim", base.model.output_l_embedding_dim, 64),
        ("num_distance_basis", base.model.num_distance_basis, 512),
        ("num_mp_layers", base.model.num_mp_layers, 3),
        ("num_edge_layers", base.model.num_edge_layers, 3),
        ("reduce_edge", base.model.reduce_edge, False),
        ("reduce_node", base.model.reduce_node, True),
        ("reduce_node_intra", base.model.reduce_node_intra, True),
        ("num_epochs", base.optimization.num_epochs, 20),
        ("lr_init", base.optimization.lr_init, 0.0005),
        ("optimizer_type", base.optimization.optimizer_type, "muon"),
        ("weight_decay", base.optimization.weight_decay, 0.0001),
        ("muon_lr", base.optimization.muon_lr, 0.02),
        ("muon_adamw_lr", base.optimization.muon_adamw_lr, 0.0005),
        ("gradient_clip_val", base.optimization.gradient_clip_val, 1.0),
        ("gradient_accumulation_steps", base.optimization.gradient_accumulation_steps, 2),
        ("scheduler_type", base.optimization.scheduler_type, "warmup_polynomial"),
        ("warmup_steps", base.optimization.warmup_steps, 1000),
        ("loss_target", base.loss.loss_target, "fock_matrix"),
        ("typed_loss_placeholder", base.loss.train_loss, "rmse_mse_padded_loss"),
        ("scale_and_shift", base.loss.scale_and_shift, True),
        ("scale_shift_mode", base.loss.scale_shift_mode, "shift_only"),
        ("scale_shift_path", Path(base.loss.scale_shift_path or ""), EXPECTED_SCALE_SHIFT),
        ("dtype", base.runtime.dtype, "float32"),
        ("seed", base.runtime.seed, 44),
        ("wandb_project", base.tracking.wandb_project, WANDB_PROJECT),
        ("validation_matrix_metrics", base.tracking.validation_matrix_metrics, True),
    )
    for label, actual, expected in checks:
        _assert_equal(label, actual, expected)


def _validate_inputs(base: MaloqConfig) -> dict[str, object]:
    _validate_base_contract(base)
    if not EXPECTED_DATABASE.is_file():
        raise FileNotFoundError(EXPECTED_DATABASE)
    if not EXPECTED_SCALE_SHIFT.is_file():
        raise FileNotFoundError(EXPECTED_SCALE_SHIFT)
    scale_sha = _sha256(EXPECTED_SCALE_SHIFT)
    _assert_equal("SHIFT SHA-256", scale_sha, EXPECTED_SCALE_SHIFT_SHA256)

    import torch
    from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

    artifact = torch.load(EXPECTED_SCALE_SHIFT, map_location="cpu", weights_only=False)
    provenance = artifact.get("provenance", {})
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
    _assert_equal("database rows", rows, sum(FULL_COUNTS.values()))
    atomic_numbers, positions, _, _, hamiltonian, overlap, *_ = database[0]
    if hamiltonian.ndim != 2 or hamiltonian.shape != overlap.shape:
        raise ValueError("NablaDFT row 0 has incompatible Hamiltonian/overlap shapes.")
    return {
        "database": str(EXPECTED_DATABASE),
        "rows": rows,
        "row0_atoms": len(atomic_numbers),
        "row0_positions_shape": list(positions.shape),
        "row0_matrix_shape": list(hamiltonian.shape),
        "scale_shift": str(EXPECTED_SCALE_SHIFT),
        "scale_shift_sha256": scale_sha,
        "scale_shift_num_train": provenance["num_train"],
        "scale_shift_validation_rows": provenance["validation_rows_in_statistics"],
    }


def _build_config(
    base: MaloqConfig,
    variant: Variant,
    scope: Scope,
    output_root: Path,
) -> MaloqConfig:
    profile = get_composite_loss_profile(variant.profile_id)
    payload = base.model_dump(mode="python")
    run_name = f"nabladft-matrix-composite-loss-ntev2-e3-muon-shift-{variant.id}"
    payload["dataset"].update(run_name=run_name, output_folder=str(output_root))
    counts = SMOKE_COUNTS if scope == "smoke" else FULL_COUNTS
    payload["splits"].update(
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
    )
    payload["model"]["model_variant"] = run_name
    payload["optimization"]["num_epochs"] = 1 if scope == "smoke" else 20
    payload["tracking"].update(
        use_wandb=scope == "full",
        wandb_run_name=(
            f"NablaDFT | NTEV2-E3 | Muon | SHIFT | {variant.display_loss} | V2"
        ),
        wandb_job_type=scope,
        wandb_tags=tuple(
            dict.fromkeys(
                [
                    *payload["tracking"]["wandb_tags"],
                    f"loss-profile:{profile.id}",
                    f"loss-scale:{profile.scale:g}",
                ]
            )
        ),
    )
    config = MaloqConfig.model_validate(payload)
    _assert_equal("scope epochs", config.optimization.num_epochs, 1 if scope == "smoke" else 20)
    _assert_equal("profile formula", profile.formula, "rmse+mse+mae" if profile.scale == 1 else "10*(rmse+mse+mae)")
    _validate_base_contract(
        MaloqConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "optimization": {
                    **config.optimization.model_dump(mode="python"),
                    "num_epochs": 20,
                },
                "splits": base.splits.model_dump(mode="python"),
            }
        )
    )
    return config


def _preview(config: MaloqConfig, variant: Variant) -> dict[str, object]:
    profile = get_composite_loss_profile(variant.profile_id)
    return {
        **asdict(variant),
        "formula": profile.formula,
        "loss_scale": profile.scale,
        "backbone_type": config.model.backbone_type,
        "head_type": config.model.head_type,
        "num_mp_layers": config.model.num_mp_layers,
        "num_edge_layers": config.model.num_edge_layers,
        "num_epochs": config.optimization.num_epochs,
        "num_train": config.splits.num_train,
        "num_val": config.splits.num_val,
        "effective_batch_size_at_world_size_2": (
            config.splits.batch_size * 2 * config.optimization.gradient_accumulation_steps
        ),
        "optimizer_type": config.optimization.optimizer_type,
        "gradient_clip_val": config.optimization.gradient_clip_val,
        "scale_and_shift": config.loss.scale_and_shift,
        "scale_shift_mode": config.loss.scale_shift_mode,
        "expected_parameters": EXPECTED_PARAMETERS,
        "wandb_run_name": config.tracking.wandb_run_name,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--variant", choices=("all", *VARIANT_BY_ID), required=True)
    parser.add_argument("--scope", choices=("validate", "smoke", "full"), required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def _rank() -> int:
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))


def main() -> None:
    args = _parse_args()
    base_path = args.base_config.expanduser().resolve()
    if not base_path.is_file():
        raise SystemExit(f"Base config not found: {base_path}")
    base = MaloqConfig.from_file(base_path)
    inputs = _validate_inputs(base)
    scope: Scope = args.scope
    variants = VARIANTS if args.variant == "all" else (VARIANT_BY_ID[args.variant],)

    if scope == "validate":
        preview_root = PROJECT_ROOT / "outputs/_config-preview/nabladft-matrix-composite-loss"
        print(
            json.dumps(
                {
                    "suite": "nabladft-matrix-composite-loss",
                    "baseline_output": str(BASELINE_OUTPUT),
                    "base_config": str(base_path),
                    "base_config_sha256": _sha256(base_path),
                    "inputs": inputs,
                    "variants": [
                        _preview(_build_config(base, variant, scope, preview_root / variant.id), variant)
                        for variant in variants
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if len(variants) != 1 or args.output_root is None:
        raise SystemExit("smoke/full requires one variant and --output-root.")
    output_root = args.output_root.expanduser().resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_root == outputs_root or outputs_root not in output_root.parents:
        raise SystemExit(f"Output must be a run directory below {outputs_root}.")

    variant = variants[0]
    profile = get_composite_loss_profile(variant.profile_id)
    typed_config = _build_config(base, variant, scope, output_root)
    if _rank() == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "resolved_matrix_composite_loss_config.json").write_text(
            json.dumps(
                {
                    "suite": "nabladft-matrix-composite-loss",
                    "scope": scope,
                    "variant": asdict(variant),
                    "experimental_loss": {
                        "profile_id": profile.id,
                        "formula": profile.formula,
                        "scale": profile.scale,
                    },
                    "baseline_output": str(BASELINE_OUTPUT),
                    "base_config": str(base_path),
                    "base_config_sha256": _sha256(base_path),
                    "inputs": inputs,
                    "expected_parameters": EXPECTED_PARAMETERS,
                    "config": typed_config.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )

    workflow_config = typed_config.to_workflow_config()
    workflow = build_matrix_composite_loss_workflow(
        workflow_config,
        profile_id=variant.profile_id,
    )
    workflow.run()


if __name__ == "__main__":
    main()
