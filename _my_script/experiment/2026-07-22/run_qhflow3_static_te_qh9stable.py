#!/usr/bin/env python3
"""Run the corrected static-TE QHFlow3 lane on QH9Stable."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).with_name("qhflow3_static_te_qh9stable.yaml")
FULL_DB = Path("/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db")
SMOKE_DB = Path("/dataset_tmp/qh9_maloq_ase_verification/QH9Stable_random_2_1_1.db")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--master-port", type=int, default=29541)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.master_port <= 65535:
        raise SystemExit("--master-port must be between 1 and 65535.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    source_root = PROJECT_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from maloq.core.config import MaloqConfig
    from maloq.train_utils.training_workflow import TrainingWorkflow

    config = MaloqConfig.from_file(CONFIG_PATH).to_workflow_config()
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
    scope = "smoke" if args.smoke else "full"
    output_dir = (
        args.output_dir
        or PROJECT_ROOT
        / "outputs"
        / f"qh9stable-qhflow3-static-te-{scope}-seed44-{timestamp}"
    ).resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_dir == outputs_root or outputs_root not in output_dir.parents:
        raise SystemExit(f"--output-dir must be below {outputs_root}.")
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")

    database = SMOKE_DB if args.smoke else FULL_DB
    if not database.is_file():
        raise SystemExit(f"QH9Stable database not found: {database}")
    config.update(
        dbpath=str(database),
        output_folder=str(output_dir),
        run_name=output_dir.name,
        use_wandb=False if args.smoke else config["use_wandb"],
    )
    if args.smoke:
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
            qhflow3_radius_embed_dim=8,
            qhflow3_grid_ffn_chunk_size=1,
        )
    TrainingWorkflow(config).run()


if __name__ == "__main__":
    main()
