"""Data-independent CLI runner for MALOQ."""

from __future__ import annotations

import argparse
import json

from ..core.config import MaloqConfig
from ..train_utils.training_workflow import TrainingWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MALOQ from a config file")
    parser.add_argument("--config", required=True, help="Path to YAML/TOML/JSON config file")
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print resolved config and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = MaloqConfig.from_file(args.config)
    wf_config = cfg.to_workflow_config()

    if args.print_effective_config:
        printable = {k: (v.__name__ if hasattr(v, "__name__") else str(v)) for k, v in wf_config.items()}
        print(json.dumps(printable, indent=2))
        return

    workflow = TrainingWorkflow(wf_config)
    workflow.run()


if __name__ == "__main__":
    main()