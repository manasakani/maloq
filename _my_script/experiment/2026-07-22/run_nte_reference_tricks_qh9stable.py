#!/usr/bin/env python3
"""Validate or train the two audited NTE reference-trick adaptations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
FULL_DB = Path(
    "/dataset/seongsu/shared-home/data/QH9_maloq_ase/"
    "QH9StableMatrices_random.db"
)
SMOKE_DB = Path(
    "/dataset_tmp/qh9_matrix_maloq_ase/"
    "QH9StableMatrices_random_2_1_1.db"
)
PRESETS = {
    "zero-channel-mean-layerscale-s64": (
        SCRIPT_ROOT
        / "nte_corrected_static_zero_channel_mean_layerscale_s64_qh9stable.yaml"
    ),
    "degreewise-l34-gate": (
        SCRIPT_ROOT / "nte_corrected_static_degreewise_l34_gate_qh9stable.yaml"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--scope", choices=("validate", "smoke", "full"), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--master-port", type=int, default=29621)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def validate_database(path: Path, *, smoke: bool) -> dict[str, object]:
    from maloq.dataset_utils.ASEDataset import ASEAtomsData

    if not path.is_file():
        raise SystemExit(f"QH9Stable density database not found: {path}")
    database = ASEAtomsData(str(path))
    metadata = database.metadata
    expected_rows = 4 if smoke else 130831
    if len(database) != expected_rows:
        raise SystemExit(f"{path} has {len(database)} rows; expected {expected_rows}.")
    required = {
        "density_matrix",
        "initial_density_matrix",
        "initial_hamiltonian",
        "overlap",
    }
    missing = required.difference(database.available_properties)
    if missing:
        raise SystemExit(f"{path} is missing properties: {sorted(missing)}")
    if metadata.get("delta_baseline_properties", {}).get("density_matrix") != (
        "initial_density_matrix"
    ):
        raise SystemExit("Database does not declare density delta-learning support.")
    return {
        "path": str(path),
        "rows": len(database),
        "target": "density_matrix",
        "delta_baseline": "initial_density_matrix",
        "initial_hamiltonian_available": True,
        "overlap_available": True,
    }


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.master_port <= 65535:
        raise SystemExit("--master-port must be between 1 and 65535.")
    for source_path in (PROJECT_ROOT, SOURCE_ROOT):
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))

    from maloq.core.config import MaloqConfig

    config = MaloqConfig.from_file(PRESETS[args.preset]).to_workflow_config()
    database = SMOKE_DB if args.scope == "smoke" else FULL_DB
    database_summary = validate_database(database, smoke=args.scope == "smoke")
    if args.scope == "validate":
        print(
            json.dumps(
                {
                    "preset": args.preset,
                    "run_name": config["run_name"],
                    "database": database_summary,
                    "head_type": config["head_type"],
                    "static_te_init_mode": config["static_te_init_mode"],
                    "static_te_gate_degrees": config["static_te_gate_degrees"],
                    "static_te_gate_activation": config["static_te_gate_activation"],
                    "residual_update_scale_mode": config[
                        "residual_update_scale_mode"
                    ],
                    "muon_routing": "all_trainable_ndim_ge_2",
                },
                indent=2,
            )
        )
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    from maloq.train_utils.training_workflow import TrainingWorkflow

    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "outputs" / f"{config['run_name']}-{args.scope}-{timestamp}"
    ).resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_dir == outputs_root or outputs_root not in output_dir.parents:
        raise SystemExit(f"--output-dir must be below {outputs_root}.")
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")

    config.update(
        dbpath=str(database),
        output_folder=str(output_dir),
        use_wandb=args.scope == "full",
    )
    if args.scope == "smoke":
        config.update(
            num_train=2,
            num_val=1,
            num_test=1,
            batch_size=1,
            num_epochs=1,
            save_frequency=1,
            l_embedding_dim=16,
            hidden_dim=16,
            output_l_embedding_dim=8,
            num_distance_basis=16,
            validation_matrix_metrics=False,
        )

    succeeded = False
    try:
        TrainingWorkflow(config).run()
        succeeded = True
    finally:
        if args.scope == "smoke" and succeeded and output_dir.is_dir():
            shutil.rmtree(output_dir)
            print(f"Smoke passed; temporary output removed: {output_dir}")


if __name__ == "__main__":
    main()
