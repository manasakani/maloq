#!/usr/bin/env python3
"""Run MALOQ comparisons on NablaDFT or converted QH9 matrix data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
REFERENCE_CONFIG_ROOT = PROJECT_ROOT / "_my_script/experiment/2026-07-21"
for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

NABLA_DB = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db"
)
QH9_MATRICES_FULL_DB = Path(
    "/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9StableMatrices_random.db"
)
QH9_MATRICES_SMOKE_DB = Path(
    "/dataset_tmp/qh9_matrix_maloq_ase/QH9StableMatrices_random_2_1_1.db"
)
QH9_HAMILTONIAN_FULL_DB = Path(
    "/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db"
)
QH9_HAMILTONIAN_SMOKE_DB = Path(
    "/dataset_tmp/qh9_maloq_ase_verification/"
    "QH9StableHamiltonianDelta_random_2_1_1.db"
)
NABLA_COUNTS = {"train": 12081, "val": 64, "test": 0}
NABLA_SMOKE_COUNTS = {"train": 2, "val": 1, "test": 0}
QH9_COUNTS = {"train": 104664, "val": 13083, "test": 13084}
QH9_SMOKE_COUNTS = {"train": 2, "val": 1, "test": 1}
CONFIGS = {
    "maloq": REFERENCE_CONFIG_ROOT / "maloq_baseline_qh9stable.yaml",
    "maloq-nte": REFERENCE_CONFIG_ROOT / "maloq_qh9stable.yaml",
    "qhflow3": REFERENCE_CONFIG_ROOT / "qhflow3_maloq_head_qh9stable.yaml",
}
MODEL_NAMES = {
    "maloq": "MALOQ",
    "maloq-nte": "MALOQ-NTE",
    "qhflow3": "QHFlow3",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("nabladft", "qh9-hamiltonian", "qh9-density"),
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=("maloq", "maloq-nte", "qhflow3", "all"),
        default="all",
        help="Train one named model, or all three sequentially for comparison.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help=(
            "Override the reference YAML for one explicit model variant. "
            "This is intentionally unavailable with --variant all."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-size-smoke", action="store_true")
    parser.add_argument(
        "--keep-smoke-output",
        action="store_true",
        help="Keep successful smoke artifacts instead of deleting them.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dbpath", type=Path, default=None)
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA_VISIBLE_DEVICES value, for example 0 or 6,7.",
    )
    parser.add_argument("--master-port", type=int, default=29545)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Molecules contributed per rank and micro-batch. The effective "
            "global batch also includes WORLD_SIZE and gradient accumulation."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Micro-batches accumulated before each optimizer update.",
    )
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=None)
    parser.add_argument(
        "--distribute-graphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Partition each global molecular supergraph across all ranks.",
    )
    parser.add_argument(
        "--partition-type",
        choices=("linear-atomwise", "linear-edgewise"),
        default="linear-edgewise",
    )
    parser.add_argument(
        "--optimizer-type",
        choices=("adam", "adamw", "soap", "muon"),
        default=None,
        help="Override the optimizer in the reference model config.",
    )
    parser.add_argument(
        "--head-type",
        choices=("maloq", "maloq_muon", "static_te"),
        default=None,
        help="Override the matrix head while keeping the selected backbone.",
    )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable W&B tracking explicitly.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-log-every-n-steps",
        type=int,
        default=None,
        help="Record rank-averaged training losses every N optimizer steps.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default=None,
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="For a single variant, write it directly into --output-root.",
    )
    parser.add_argument(
        "--delta-learning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict QH9 H or D residuals from the matching initial matrix.",
    )
    return parser


def validate_nabladft(path: Path, counts: dict[str, int]) -> dict[str, object]:
    from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

    database = HamiltonianDatabase(str(path))
    if len(database) != 12145:
        raise ValueError(f"{path} has {len(database)} rows; expected 12145")
    if sum(counts.values()) > len(database):
        raise ValueError(f"Requested split {counts} exceeds {len(database)} rows")
    z, positions, energy, forces, hamiltonian, overlap, *_ = database[0]
    if hamiltonian.shape != overlap.shape or hamiltonian.ndim != 2:
        raise ValueError(
            f"NablaDFT row 0 has incompatible H/S shapes: "
            f"{hamiltonian.shape}, {overlap.shape}"
        )
    return {
        "dataset_name": "nablaDFT",
        "native_maloq_schema": True,
        "rows": len(database),
        "row0_atoms": len(z),
        "row0_matrix_shape": list(hamiltonian.shape),
        "position_unit": "Bohr",
        "matrix_convention": "native_nabladft_psi4",
    }


def validate_qh9_target_database(
    path: Path,
    counts: dict[str, int],
    target: str,
) -> dict[str, object]:
    from maloq.dataset_utils.ASEDataset import ASEAtomsData

    database = ASEAtomsData(str(path))
    metadata = database.metadata
    stored_counts = metadata.get("selected_subset_counts")
    if not isinstance(stored_counts, dict):
        raise ValueError(f"{path} has no selected_subset_counts metadata")
    if len(database) != sum(int(value) for value in stored_counts.values()):
        raise ValueError(
            f"{path} has {len(database)} rows but metadata declares {stored_counts}"
        )
    for subset, requested_count in counts.items():
        if requested_count > int(stored_counts.get(subset, 0)):
            raise ValueError(
                f"Requested {subset}={requested_count} exceeds stored "
                f"{stored_counts.get(subset, 0)}"
            )
    if target == "hamiltonian":
        expected = {
            "dataset_name": "QH9Stable",
            "complete": True,
            "basis": "def2-svp",
            "xc": "b3lyp5",
            "maloq_loader_dataset_name": "QM7",
            "hamiltonian_storage_convention": "maloq_qm7_pre_orca_to_e3nn",
            "initial_hamiltonian_storage_convention": (
                "maloq_qm7_pre_orca_to_e3nn"
            ),
            "initial_density_storage_convention": (
                "maloq_qm7_pre_orca_to_e3nn"
            ),
            "overlap_storage_convention": "maloq_e3nn_def2svp",
            "target_properties": ["hamiltonian"],
            "loss_targets_supported": ["fock_matrix"],
            "delta_learning_supported": True,
            "delta_baseline_properties": {
                "fock_matrix": "initial_hamiltonian",
            },
            "delta_learning_scope": "hamiltonian_only",
        }
        required = {
            "hamiltonian",
            "initial_hamiltonian",
            "initial_density_matrix",
            "overlap",
            "energy",
            "forces",
        }
    elif target == "density":
        expected = {
            "dataset_name": "QH9StableMatrices",
            "complete": True,
            "basis": "def2-svp",
            "xc": "b3lyp5",
            "maloq_loader_dataset_name": "QM7",
            "density_storage_convention": "maloq_qm7_pre_orca_to_e3nn",
            "initial_density_storage_convention": (
                "maloq_qm7_pre_orca_to_e3nn"
            ),
            "initial_hamiltonian_storage_convention": (
                "maloq_qm7_pre_orca_to_e3nn"
            ),
            "overlap_storage_convention": "maloq_e3nn_def2svp",
            "target_properties": ["density_matrix"],
            "loss_targets_supported": ["density_matrix"],
            "delta_learning_supported": True,
            "delta_baseline_properties": {
                "density_matrix": "initial_density_matrix",
            },
            "delta_learning_scope": "density_only",
        }
        required = {
            "density_matrix",
            "initial_density_matrix",
            "initial_hamiltonian",
            "overlap",
            "energy",
            "forces",
        }
    else:
        raise ValueError(f"Unsupported QH9 target profile: {target}")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Converted QH9 matrix metadata {key}={metadata.get(key)!r}; "
                f"expected {value!r}"
            )
    if not required.issubset(database.available_properties):
        raise ValueError(
            f"Converted {target} properties are {database.available_properties}; "
            f"expected {sorted(required)}"
        )
    return metadata


def last_losses(output_dir: Path) -> dict[str, float]:
    def read_last(path: Path) -> tuple[float, float]:
        edge, node = (
            float(value) for value in path.read_text().strip().splitlines()[-1].split()
        )
        return node, edge

    train_node, train_edge = read_last(output_dir / "head_training_loss.txt")
    val_node, val_edge = read_last(output_dir / "head_validation_loss.txt")
    return {
        "train_node_loss": train_node,
        "train_edge_loss": train_edge,
        "validation_node_loss": val_node,
        "validation_edge_loss": val_edge,
    }


def prepare_config(
    dataset_name: str,
    variant: str,
    dbpath: Path,
    counts: dict[str, int],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    from maloq.core.config import MaloqConfig
    from maloq.train_utils.utils_compute import distributed_context

    model_config = args.model_config or CONFIGS[variant]
    config = MaloqConfig.from_file(model_config).to_workflow_config()
    _, world_size, _ = distributed_context()
    per_device_batch_size = args.batch_size or (
        1 if args.smoke else (10 if dataset_name == "nabladft" else 32)
    )
    config.update(
        dbpath=str(dbpath),
        output_folder=str(output_dir),
        run_name=f"{output_dir.parent.name}-{output_dir.name}",
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
        num_epochs=args.num_epochs or (1 if args.smoke else config["num_epochs"]),
        save_frequency=1 if args.smoke else config["save_frequency"],
        use_wandb=False if args.smoke else config["use_wandb"],
        batch_size=per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=(
            per_device_batch_size * world_size * args.gradient_accumulation_steps
        ),
        distribute_graphs=args.distribute_graphs,
        partition_type=(args.partition_type if args.distribute_graphs else None),
        validation_matrix_metrics=(
            False if args.distribute_graphs else config["validation_matrix_metrics"]
        ),
    )
    if args.optimizer_type is not None:
        config["optimizer_type"] = args.optimizer_type
    if args.head_type is not None:
        config["head_type"] = args.head_type
    if args.use_wandb is not None:
        config["use_wandb"] = args.use_wandb
    if args.wandb_project is not None:
        config["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        config["wandb_entity"] = args.wandb_entity
    if args.wandb_mode is not None:
        config["wandb_mode"] = args.wandb_mode
    if args.wandb_log_every_n_steps is not None:
        config["wandb_log_every_n_steps"] = args.wandb_log_every_n_steps
    if dataset_name == "nabladft":
        config.update(
            dataset_name="nablaDFT",
            loss_target="fock_matrix",
            # QHFlow3's pair trunk requires the complete directed graph.
            rcut_orbitals=100.0 if variant == "qhflow3" else 8.0,
            rcut_gaussian=16.0,
        )
    elif dataset_name == "qh9-density":
        config.update(
            dataset_name="QM7",
            loss_target="density_matrix",
            delta_learning=args.delta_learning,
        )
    else:
        config.update(
            dataset_name="QM7",
            loss_target="fock_matrix",
            delta_learning=args.delta_learning,
        )

    if args.smoke and not args.full_size_smoke:
        config.update(
            l_embedding_dim=16,
            hidden_dim=16,
            num_distance_basis=16,
            output_l_embedding_dim=8 if variant != "maloq" else None,
        )
    return config


def run_variant(
    dataset_name: str,
    variant: str,
    dbpath: Path,
    counts: dict[str, int],
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    from maloq.train_utils.training_workflow import TrainingWorkflow

    output_dir = output_root if args.flat_output else output_root / variant
    config = prepare_config(dataset_name, variant, dbpath, counts, output_dir, args)
    started = time.perf_counter()
    workflow = TrainingWorkflow(config)
    workflow.run()
    elapsed = time.perf_counter() - started
    if workflow.rank != 0:
        return None
    model_summary = json.loads((output_dir / "model_summary.json").read_text())
    losses = last_losses(output_dir)
    removed_checkpoint_bytes = 0
    if args.smoke:
        for checkpoint in output_dir.glob("*.pt"):
            removed_checkpoint_bytes += checkpoint.stat().st_size
            checkpoint.unlink()
    return {
        "dataset": dataset_name,
        "loss_target": config["loss_target"],
        "delta_learning": bool(config.get("delta_learning", False)),
        "model_name": MODEL_NAMES[variant],
        "variant": variant,
        "num_epochs": int(config["num_epochs"]),
        "batch_size": int(config["batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "world_size": int(workflow.world_size),
        "effective_batch_size": int(config["effective_batch_size"]),
        "distribute_graphs": bool(config["distribute_graphs"]),
        "partition_type": config["partition_type"],
        "optimizer_type": config["optimizer_type"],
        "head_type": config["head_type"],
        "wandb": {
            "enabled": bool(config["use_wandb"]),
            "project": config["wandb_project"],
            "entity": config.get("wandb_entity"),
            "mode": config["wandb_mode"],
            "log_every_n_steps": config["wandb_log_every_n_steps"],
            "run_name": config["run_name"],
        },
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed,
        "losses": losses,
        "model": model_summary,
        "removed_smoke_checkpoint_bytes": removed_checkpoint_bytes,
    }


def selected_variants(choice: str) -> tuple[str, ...]:
    if choice == "all":
        return ("maloq", "maloq-nte", "qhflow3")
    return (choice,)


def write_comparison_csv(path: Path, results: list[dict[str, object]]) -> None:
    columns = (
        "dataset",
        "loss_target",
        "delta_learning",
        "model_name",
        "variant",
        "head_type",
        "trainable_parameters",
        "elapsed_seconds",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "train_node_loss",
        "train_edge_loss",
        "validation_node_loss",
        "validation_edge_loss",
        "output_dir",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            losses = result["losses"]
            model = result["model"]
            writer.writerow(
                {
                    "dataset": result["dataset"],
                    "loss_target": result["loss_target"],
                    "delta_learning": result["delta_learning"],
                    "model_name": result["model_name"],
                    "variant": result["variant"],
                    "head_type": result["head_type"],
                    "trainable_parameters": model["trainable_parameters"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "micro_batch_size": result["batch_size"],
                    "gradient_accumulation_steps": result[
                        "gradient_accumulation_steps"
                    ],
                    "effective_batch_size": result["effective_batch_size"],
                    **losses,
                    "output_dir": result["output_dir"],
                }
            )


def main() -> None:
    args = build_parser().parse_args()
    if args.num_epochs is not None and args.num_epochs <= 0:
        raise SystemExit("--num-epochs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    if args.wandb_log_every_n_steps is not None and args.wandb_log_every_n_steps <= 0:
        raise SystemExit("--wandb-log-every-n-steps must be positive")
    for name in ("num_train", "num_val", "num_test"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} cannot be negative")
    if args.num_train == 0:
        raise SystemExit("--num-train must be positive")
    if args.full_size_smoke and not args.smoke:
        raise SystemExit("--full-size-smoke requires --smoke")
    if args.delta_learning and not args.dataset.startswith("qh9-"):
        raise SystemExit("--delta-learning is only available with a QH9 dataset")
    if args.flat_output and args.variant == "all":
        raise SystemExit("--flat-output requires one explicit model variant")
    if args.model_config is not None:
        if args.variant == "all":
            raise SystemExit("--model-config requires one explicit model variant")
        args.model_config = args.model_config.resolve()
        if not args.model_config.is_file():
            raise SystemExit(f"Model config not found: {args.model_config}")
    if not 1 <= args.master_port <= 65535:
        raise SystemExit("--master-port must be between 1 and 65535")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    from maloq.train_utils.utils_compute import distributed_context

    rank, world_size, local_rank = distributed_context()
    if world_size > 1:
        import torch

        visible_device_count = torch.cuda.device_count()
        if visible_device_count == 0:
            raise SystemExit("Multi-GPU training requires visible CUDA devices")
        device_index = 0 if visible_device_count == 1 else local_rank
        if not 0 <= device_index < visible_device_count:
            raise SystemExit(
                f"Local rank {local_rank} cannot select one of "
                f"{visible_device_count} visible CUDA devices"
            )
        # eSEN imports a CuPy NCCL communicator before TrainingWorkflow is
        # constructed, so select the rank-local CUDA device before that import.
        torch.cuda.set_device(device_index)
    if args.distribute_graphs and args.dataset != "nabladft":
        raise SystemExit(
            "Distributed-graph testing is currently restricted to NablaDFT; "
            "the QH9 streaming path has not been validated."
        )

    if args.dataset == "nabladft":
        counts = dict(NABLA_SMOKE_COUNTS if args.smoke else NABLA_COUNTS)
        dbpath = (args.dbpath or NABLA_DB).resolve()
    else:
        counts = dict(QH9_SMOKE_COUNTS if args.smoke else QH9_COUNTS)
        use_smoke_db = args.smoke and not args.full_size_smoke
        if args.dataset == "qh9-hamiltonian":
            default_db = (
                QH9_HAMILTONIAN_SMOKE_DB
                if use_smoke_db
                else QH9_HAMILTONIAN_FULL_DB
            )
        else:
            default_db = QH9_MATRICES_SMOKE_DB if use_smoke_db else QH9_MATRICES_FULL_DB
        dbpath = (args.dbpath or default_db).resolve()

    for split in ("train", "val", "test"):
        override = getattr(args, f"num_{split}")
        if override is not None:
            counts[split] = override

    if args.dataset == "nabladft":
        metadata = validate_nabladft(dbpath, counts)
    else:
        if not dbpath.is_file():
            converter = (
                "_auto_script/qh9_raw_to_maloq/"
                "process_qh9_raw_to_maloq_ase.py"
                if args.dataset == "qh9-hamiltonian"
                else "_auto_script/qh9_matrix_lmdb_to_maloq/"
                "process_qh9_matrix_lmdb_to_maloq_ase.py"
            )
            raise SystemExit(
                f"Converted QH9 database not found: {dbpath}. Run {converter} first."
            )
        target = "hamiltonian" if args.dataset == "qh9-hamiltonian" else "density"
        metadata = validate_qh9_target_database(dbpath, counts, target)

    variants = selected_variants(args.variant)
    if args.distribute_graphs and "qhflow3" in variants:
        raise SystemExit(
            "QHFlow3 supports multi-GPU data parallelism, but its separate "
            "pair graph is not implemented for distributed-graph training."
        )
    validation = {
        "dataset": args.dataset,
        "dbpath": str(dbpath),
        "model_config_override": (
            str(args.model_config) if args.model_config is not None else None
        ),
        "counts": counts,
        "variants": variants,
        "model_names": [MODEL_NAMES[variant] for variant in variants],
        "loss_target": (
            "density_matrix" if args.dataset == "qh9-density" else "fock_matrix"
        ),
        "metadata": metadata,
        "distributed": {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "per_device_batch_size": args.batch_size,
            "effective_batch_size": (
                args.batch_size * world_size * args.gradient_accumulation_steps
                if args.batch_size is not None
                else None
            ),
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "distribute_graphs": args.distribute_graphs,
            "partition_type": (args.partition_type if args.distribute_graphs else None),
        },
    }
    if args.validate_only:
        preview_root = PROJECT_ROOT / "outputs" / "_config-preview"
        run_config = {}
        for variant in variants:
            config = prepare_config(
                args.dataset,
                variant,
                dbpath,
                counts,
                preview_root / variant,
                args,
            )
            run_config[variant] = {
                key: config[key]
                for key in (
                    "model_variant",
                    "backbone_type",
                    "head_type",
                    "optimizer_type",
                    "num_epochs",
                    "batch_size",
                    "gradient_accumulation_steps",
                    "effective_batch_size",
                    "distribute_graphs",
                    "partition_type",
                    "seed",
                    "loss_target",
                    "dataset_name",
                    "use_wandb",
                    "wandb_project",
                    "wandb_entity",
                    "wandb_mode",
                    "wandb_log_every_n_steps",
                    "run_name",
                )
            }
            if config["optimizer_type"] == "muon":
                run_config[variant]["muon_routing"] = (
                    "all_trainable_ndim_ge_2"
                )
        validation["run_config"] = run_config
        if rank == 0:
            print(json.dumps(validation, indent=2, default=str))
        return

    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
    scope = "smoke" if args.smoke else "full"
    if args.full_size_smoke:
        scope = "full-size-smoke"
    learning = "delta" if args.delta_learning else "absolute"
    selection = "three-model-comparison" if args.variant == "all" else args.variant
    default_output = (
        PROJECT_ROOT
        / "outputs"
        / f"{args.dataset}-{learning}-{selection}-{scope}-seed44-{timestamp}"
    )
    output_root = (args.output_root or default_output).resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_root != outputs_root and outputs_root not in output_root.parents:
        raise SystemExit(f"--output-root must be below {outputs_root}")
    if rank == 0:
        if output_root.exists():
            raise SystemExit(f"Output directory already exists: {output_root}")
        output_root.mkdir(parents=True)

    rank_results = [
        run_variant(args.dataset, variant, dbpath, counts, output_root, args)
        for variant in variants
    ]
    if rank != 0:
        return
    results = [result for result in rank_results if result is not None]
    summary = {
        **validation,
        "smoke": args.smoke,
        "delta_learning": args.delta_learning,
        "results": results,
    }
    by_variant = {result["variant"]: result for result in results}
    if "maloq" in by_variant:
        maloq_losses = by_variant["maloq"]["losses"]
        summary["loss_ratio_to_maloq"] = {
            variant: {
                key: result["losses"][key] / maloq_losses[key]
                for key in maloq_losses
                if maloq_losses[key] != 0.0
            }
            for variant, result in by_variant.items()
            if variant != "maloq"
        }
    summary_path = output_root / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    comparison_csv_path = output_root / "comparison.csv"
    write_comparison_csv(comparison_csv_path, results)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"Comparison summary: {summary_path}", flush=True)
    print(f"Comparison table: {comparison_csv_path}", flush=True)
    if args.smoke and not args.keep_smoke_output:
        shutil.rmtree(output_root)
        print(f"Successful smoke output discarded: {output_root}", flush=True)


if __name__ == "__main__":
    main()
