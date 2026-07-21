#!/usr/bin/env python3
"""Train MALOQ with an officially ordered QH9Stable ASE database."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FULL_DB = Path(
    "/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db"
)
DEFAULT_SMOKE_DB = Path(
    "/dataset_tmp/qh9_maloq_ase_verification/QH9Stable_random_2_1_1.db"
)
OFFICIAL_COUNTS = {"train": 104664, "val": 13083, "test": 13084}
SMOKE_COUNTS = {"train": 2, "val": 1, "test": 1}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch on the ordered 2/1/1 Stable smoke database.",
    )
    parser.add_argument(
        "--dbpath",
        type=Path,
        default=None,
        help="Converted QH9Stable ASE database; defaults depend on --smoke.",
    )
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-frequency", type=int, default=None)
    parser.add_argument("--wigner-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--optimizer-type", choices=("adam", "adamw", "soap", "muon"), default="adamw")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_database(dbpath: Path, expected_counts: dict[str, int]) -> None:
    from dataset_utils.ASEDataset import ASEAtomsData

    database = ASEAtomsData(str(dbpath))
    expected_total = sum(expected_counts.values())
    if len(database) != expected_total:
        raise ValueError(
            f"{dbpath} contains {len(database)} rows; expected {expected_total} "
            f"for {expected_counts}."
        )
    metadata = database.metadata
    if metadata.get("dataset_name") != "QH9Stable":
        raise ValueError(f"{dbpath} is not marked as QH9Stable.")
    if metadata.get("complete") is not True:
        raise ValueError(f"{dbpath} is not marked as a complete conversion.")
    if metadata.get("selected_subset_counts") != expected_counts:
        raise ValueError(
            "Converted split metadata does not match the requested ordered split: "
            f"{metadata.get('selected_subset_counts')} != {expected_counts}."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    requested_counts = (args.num_train, args.num_val, args.num_test)
    if args.smoke and any(value is not None for value in requested_counts):
        parser.error("Do not combine --smoke with explicit split counts.")
    if not args.smoke and any(value is not None for value in requested_counts):
        if not all(value is not None for value in requested_counts):
            parser.error("Provide --num-train, --num-val, and --num-test together.")
        if any(value <= 0 for value in requested_counts):
            parser.error("Explicit split counts must all be positive.")
        counts = dict(zip(("train", "val", "test"), requested_counts))
    else:
        counts = SMOKE_COUNTS if args.smoke else OFFICIAL_COUNTS
    dbpath = (args.dbpath or (DEFAULT_SMOKE_DB if args.smoke else DEFAULT_FULL_DB)).resolve()
    if not dbpath.is_file():
        parser.error(
            f"Converted QH9Stable database not found: {dbpath}. "
            "Run _auto_script/qh9_raw_to_maloq/process_qh9_raw_to_maloq_ase.py first."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")

    import torch.distributed as dist

    from run_QM7 import config as qm7_config
    from train_utils.training_workflow import TrainingWorkflow

    validate_database(dbpath, counts)
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
    run_kind = (
        "smoke"
        if args.smoke
        else "full"
        if counts == OFFICIAL_COUNTS
        else "subset"
    )
    output_folder = args.output_folder or f"outputs/qh9Stable_random_{run_kind}_{timestamp}"
    run_name = Path(output_folder).name
    resolved_output = Path(
        TrainingWorkflow.resolve_output_folder(output_folder, run_name)
    )
    if args.resume and args.output_folder is None:
        parser.error("--resume requires --output-folder for the existing run.")
    if args.resume and not resolved_output.is_dir():
        parser.error(f"Resume output directory not found: {resolved_output}")
    if not args.resume and resolved_output.exists():
        parser.error(
            f"Output directory already exists: {resolved_output}. "
            "Choose a new --output-folder or use --resume."
        )

    run_config = dict(qm7_config)
    run_config.update(
        dbpath=str(dbpath),
        output_folder=output_folder,
        run_name=run_name,
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
        batch_size=args.batch_size or (1 if args.smoke else 5),
        num_epochs=args.num_epochs or (1 if args.smoke else 200),
        save_frequency=args.save_frequency or (1 if args.smoke else 10),
        shuffle=False,
        distribute_graphs=False,
        wigner_backend=args.wigner_backend,
        optimizer_type=args.optimizer_type,
        restart_backbone=args.resume,
        restart_head=args.resume,
        restart_optimizer=args.resume,
        scale_and_shift=False,
    )
    if args.smoke:
        run_config.update(
            l_embedding_dim=16,
            hidden_dim=16,
            num_distance_basis=16,
            num_mp_layers=1,
        )

    workflow = None
    try:
        workflow = TrainingWorkflow(run_config)
        print(f"QH9Stable database: {dbpath}", flush=True)
        print(f"Ordered split counts: {counts}", flush=True)
        print(f"Model output: {workflow.config['output_folder']}", flush=True)
        workflow.run()
    finally:
        if workflow is not None:
            workflow.finish_tracking()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
