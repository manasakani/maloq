#!/usr/bin/env python3
"""Run a matched QH9Stable comparison of MALOQ baseline and maloq-qh9."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

DEFAULT_FULL_DB = Path(
    "/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db"
)
DEFAULT_SMOKE_DB = Path(
    "/dataset_tmp/qh9_maloq_ase_verification/QH9Stable_random_2_1_1.db"
)
OFFICIAL_COUNTS = {"train": 104664, "val": 13083, "test": 13084}
SMOKE_COUNTS = {"train": 2, "val": 1, "test": 1}
EXPECTED_METADATA = {
    "basis": "def2-svp",
    "maloq_loader_dataset_name": "QM7",
    "hamiltonian_storage_convention": "maloq_qm7_pre_orca_to_e3nn",
    "overlap_storage_convention": "maloq_e3nn_def2svp",
}
CONFIGS = {
    "baseline": EXPERIMENT_ROOT / "maloq_baseline_qh9stable.yaml",
    "maloq-qh9": EXPERIMENT_ROOT / "maloq_qh9stable.yaml",
}
REFERENCE_RUN = (
    "qh9_b3lyp5_maloq0713_nte_qhflow3_parity_"
    "bounded_degree_layerscale_s64_full_seed44"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("baseline", "maloq-qh9", "both"),
        default="both",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the ordered 2/1/1 QH9Stable database for one epoch.",
    )
    parser.add_argument(
        "--full-size-smoke",
        action="store_true",
        help="With --smoke, retain the production channel dimensions.",
    )
    parser.add_argument("--dbpath", type=Path, default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--master-port", type=int, default=29534)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Comparison directory below the repository outputs tree.",
    )
    return parser


def validate_database(dbpath: Path, expected_counts: dict[str, int]) -> None:
    from maloq.dataset_utils.ASEDataset import ASEAtomsData

    database = ASEAtomsData(str(dbpath))
    if len(database) != sum(expected_counts.values()):
        raise ValueError(
            f"{dbpath} contains {len(database)} rows; expected "
            f"{sum(expected_counts.values())}."
        )
    metadata = database.metadata
    if metadata.get("dataset_name") != "QH9Stable":
        raise ValueError(f"{dbpath} is not marked as QH9Stable.")
    if metadata.get("complete") is not True:
        raise ValueError(f"{dbpath} is not marked as a complete conversion.")
    for key, expected in EXPECTED_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Converted database metadata {key!r} is "
                f"{metadata.get(key)!r}; expected {expected!r}."
            )
    if metadata.get("selected_subset_counts") != expected_counts:
        raise ValueError(
            "Converted split metadata does not match the ordered split: "
            f"{metadata.get('selected_subset_counts')} != {expected_counts}."
        )


def last_losses(output_dir: Path) -> dict[str, float]:
    def read_last(path: Path) -> tuple[float, float]:
        line = path.read_text().strip().splitlines()[-1]
        # SplitTrainer's persisted matrix-loss format is edge first, node second.
        edge, node = (float(value) for value in line.split())
        return node, edge

    train_node, train_edge = read_last(output_dir / "head_training_loss.txt")
    val_node, val_edge = read_last(output_dir / "head_validation_loss.txt")
    return {
        "train_node_loss": train_node,
        "train_edge_loss": train_edge,
        "validation_node_loss": val_node,
        "validation_edge_loss": val_edge,
    }


def run_variant(
    variant: str,
    *,
    dbpath: Path,
    counts: dict[str, int],
    output_root: Path,
    smoke: bool,
    full_size_smoke: bool,
    num_epochs: int | None,
    batch_size: int | None,
) -> dict[str, object]:
    from maloq.core.config import MaloqConfig
    from maloq.train_utils.training_workflow import TrainingWorkflow

    config = MaloqConfig.from_file(CONFIGS[variant]).to_workflow_config()
    output_dir = output_root / variant
    config.update(
        dbpath=str(dbpath),
        output_folder=str(output_dir),
        run_name=output_dir.name,
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
        num_epochs=num_epochs or (1 if smoke else config["num_epochs"]),
        batch_size=batch_size or (1 if smoke else config["batch_size"]),
        save_frequency=1 if smoke else config["save_frequency"],
        use_wandb=False if smoke else config["use_wandb"],
    )
    if smoke and not full_size_smoke:
        config.update(
            l_embedding_dim=16,
            hidden_dim=16,
            num_distance_basis=16,
            output_l_embedding_dim=8 if variant == "maloq-qh9" else None,
        )

    started = time.perf_counter()
    workflow = TrainingWorkflow(config)
    workflow.run()
    elapsed = time.perf_counter() - started

    model_summary = json.loads((output_dir / "model_summary.json").read_text())
    losses = last_losses(output_dir)
    removed_smoke_checkpoint_bytes = 0
    if smoke:
        for checkpoint in output_dir.glob("*.pt"):
            removed_smoke_checkpoint_bytes += checkpoint.stat().st_size
            checkpoint.unlink()
    return {
        "variant": variant,
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed,
        "losses": losses,
        "model": model_summary,
        "removed_smoke_checkpoint_bytes": removed_smoke_checkpoint_bytes,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.num_epochs is not None and args.num_epochs <= 0:
        raise SystemExit("--num-epochs must be positive.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.full_size_smoke and not args.smoke:
        raise SystemExit("--full-size-smoke requires --smoke.")
    if not 1 <= args.master_port <= 65535:
        raise SystemExit("--master-port must be between 1 and 65535.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)

    counts = SMOKE_COUNTS if args.smoke else OFFICIAL_COUNTS
    dbpath = (args.dbpath or (DEFAULT_SMOKE_DB if args.smoke else DEFAULT_FULL_DB)).resolve()
    if not dbpath.is_file():
        raise SystemExit(
            f"Converted QH9Stable database not found: {dbpath}. Run "
            "_auto_script/qh9_raw_to_maloq/process_qh9_raw_to_maloq_ase.py first."
        )
    validate_database(dbpath, counts)

    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
    if args.full_size_smoke:
        run_scope = "full-size-smoke"
    elif args.smoke:
        run_scope = "smoke"
    else:
        run_scope = "full"
    default_name = f"maloq-qh9-{run_scope}-{timestamp}"
    output_root = (args.output_root or (PROJECT_ROOT / "outputs" / default_name)).resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_root != outputs_root and outputs_root not in output_root.parents:
        raise SystemExit(f"--output-root must be below {outputs_root}.")
    if output_root.exists():
        raise SystemExit(f"Output directory already exists: {output_root}")
    output_root.mkdir(parents=True)

    variants = (
        ("baseline", "maloq-qh9")
        if args.variant == "both"
        else (args.variant,)
    )
    results = [
        run_variant(
            variant,
            dbpath=dbpath,
            counts=counts,
            output_root=output_root,
            smoke=args.smoke,
            full_size_smoke=args.full_size_smoke,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )
        for variant in variants
    ]
    summary: dict[str, object] = {
        "reference_ml_dft_run": REFERENCE_RUN,
        "comparison_contract": (
            "Both lanes use the same official QH9Stable order, native MALOQ "
            "coupled-irrep head, target labels, loss, optimizer, and seed."
        ),
        "dbpath": str(dbpath),
        "split_counts": counts,
        "smoke": args.smoke,
        "full_size_smoke": args.full_size_smoke,
        "results": results,
    }
    if len(results) == 2:
        baseline = results[0]["losses"]
        candidate = results[1]["losses"]
        summary["candidate_over_baseline"] = {
            key: candidate[key] / baseline[key]
            for key in baseline
            if baseline[key] != 0.0
        }
    summary_path = output_root / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Comparison summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
