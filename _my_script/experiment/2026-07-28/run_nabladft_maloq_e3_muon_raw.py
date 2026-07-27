#!/usr/bin/env python3
"""Run the matched NablaDFT MALOQ-E3 + Muon + RAW control lane."""

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
    PROJECT_ROOT / "_my_script/experiment/2026-07-28/nabladft_maloq_e3_muon_raw.yaml"
)
EXPECTED_DATABASE = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db"
)
EXPECTED_RUN_NAME = "nabladft-v2-ofat-maloq-e3-muon-raw"
EXPECTED_DISPLAY_NAME = "NablaDFT | MALOQ-E3 | Muon | RAW | V2"
EXPECTED_WANDB_PROJECT = "MALOQ-nablaDFT-v2"
FULL_COUNTS = {"train": 12081, "val": 64, "test": 0}
SMOKE_COUNTS = {"train": 20, "val": 20, "test": 0}
Scope = Literal["validate", "smoke", "full"]

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank() -> int:
    return int(
        os.environ.get(
            "OMPI_COMM_WORLD_RANK",
            os.environ.get("RANK", "0"),
        )
    )


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


def _validate_database() -> dict[str, object]:
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
    }


def _build_config(
    base: MaloqConfig,
    scope: Scope,
    output_root: Path,
) -> MaloqConfig:
    payload = base.model_dump(mode="python")
    counts = SMOKE_COUNTS if scope == "smoke" else FULL_COUNTS
    payload["dataset"].update(
        run_name=EXPECTED_RUN_NAME,
        output_folder=str(output_root),
    )
    payload["splits"].update(
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
    )
    payload["optimization"]["num_epochs"] = 1 if scope == "smoke" else 20
    payload["tracking"].update(
        use_wandb=scope == "full",
        wandb_run_name=EXPECTED_DISPLAY_NAME,
        wandb_job_type=scope,
    )
    config = MaloqConfig.model_validate(payload)
    _validate_contract(config, scope)
    return config


def _validate_contract(config: MaloqConfig, scope: Scope) -> None:
    if Path(config.dataset.dbpath) != EXPECTED_DATABASE:
        raise ValueError(
            f"Database must be exactly {EXPECTED_DATABASE}, got "
            f"{config.dataset.dbpath!r}."
        )
    if config.dataset.run_name != EXPECTED_RUN_NAME:
        raise ValueError("Run identity drifted from the dedicated RAW control.")
    if config.model.backbone_type != "esen":
        raise ValueError("The MALOQ RAW control requires backbone_type='esen'.")
    if config.model.num_mp_layers != 3 or config.model.num_edge_layers != 3:
        raise ValueError("The MALOQ RAW control requires exactly three layers.")
    if config.model.head_type != "maloq_muon":
        raise ValueError("The MALOQ RAW control requires head_type='maloq_muon'.")
    if config.model.output_l_embedding_dim is not None:
        raise ValueError("Canonical MALOQ must retain its native 128-channel output.")
    if config.model.mlp_type != "spectral":
        raise ValueError("Canonical MALOQ must retain the spectral eSEN stack.")
    if config.optimization.optimizer_type != "muon":
        raise ValueError("The MALOQ RAW control requires optimizer_type='muon'.")
    if config.loss.scale_and_shift:
        raise ValueError("RAW targets require scale_and_shift=false.")
    if config.loss.scale_shift_path is not None:
        raise ValueError("RAW targets require scale_shift_path=null.")
    if config.loss.delta_learning:
        raise ValueError("The matched control uses absolute Fock targets.")
    if config.runtime.seed != 44:
        raise ValueError("The matched control requires seed 44.")
    if config.splits.batch_size != 5:
        raise ValueError("Per-rank micro-batch must remain 5.")
    if config.optimization.gradient_accumulation_steps != 2:
        raise ValueError("Gradient accumulation must remain 2.")
    if config.splits.shuffle or config.splits.distribute_graphs:
        raise ValueError("The matched suite requires ordered data parallelism.")
    expected_counts = SMOKE_COUNTS if scope == "smoke" else FULL_COUNTS
    actual_counts = {
        "train": config.splits.num_train,
        "val": config.splits.num_val,
        "test": config.splits.num_test,
    }
    if actual_counts != expected_counts:
        raise ValueError(f"Scope/split mismatch: {actual_counts} != {expected_counts}.")
    expected_epochs = 1 if scope == "smoke" else 20
    if config.optimization.num_epochs != expected_epochs:
        raise ValueError("Scope/epoch mismatch.")
    if config.tracking.wandb_project != EXPECTED_WANDB_PROJECT:
        raise ValueError("W&B project drifted from MALOQ-nablaDFT-v2.")
    if config.tracking.wandb_run_name != EXPECTED_DISPLAY_NAME:
        raise ValueError("W&B display name drifted from the requested identity.")


def _preview(
    config: MaloqConfig,
    config_path: Path,
    database_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "lane": "maloq-e3-muon-raw",
        "workflow": (
            f"{TrainingWorkflowV2Fixed.__module__}.{TrainingWorkflowV2Fixed.__name__}"
        ),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "inputs": database_metadata,
        "contract": {
            "backbone_type": config.model.backbone_type,
            "num_mp_layers": config.model.num_mp_layers,
            "num_edge_layers": config.model.num_edge_layers,
            "head_type": config.model.head_type,
            "optimizer_type": config.optimization.optimizer_type,
            "scale_and_shift": config.loss.scale_and_shift,
            "scale_shift_path": config.loss.scale_shift_path,
            "seed": config.runtime.seed,
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
            "wandb_project": config.tracking.wandb_project,
            "wandb_run_name": config.tracking.wandb_run_name,
        },
    }


def main() -> None:
    args = _parse_args()
    scope: Scope = args.scope
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    base = MaloqConfig.from_file(config_path)
    database_metadata = _validate_database()

    if scope == "validate":
        preview_root = (
            PROJECT_ROOT / "outputs/_config-preview/nabladft-v2-ofat-maloq-e3-muon-raw"
        )
        typed_config = _build_config(base, scope, preview_root)
        print(
            json.dumps(
                _preview(typed_config, config_path, database_metadata),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.output_root is None:
        raise SystemExit("--output-root is required for smoke/full.")
    output_root = args.output_root.expanduser().resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_root == outputs_root or outputs_root not in output_root.parents:
        raise SystemExit(
            f"Output must be a lane directory below {outputs_root}: {output_root}"
        )

    typed_config = _build_config(base, scope, output_root)
    if _rank() == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        resolved_payload = {
            "scope": scope,
            "lane": "maloq-e3-muon-raw",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "inputs": database_metadata,
            "resolved_config": typed_config.model_dump(mode="json"),
        }
        (output_root / "resolved_maloq_e3_muon_raw_config.json").write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n"
        )

    TrainingWorkflowV2Fixed(typed_config.to_workflow_config()).run()


if __name__ == "__main__":
    main()
