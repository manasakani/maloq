#!/usr/bin/env python3
"""Run the existing NablaDFT/QH9 experiment CLI with TrainingWorkflowFixed."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
LEGACY_RUNNER = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py"
)
for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def main() -> None:
    fixed_parser = argparse.ArgumentParser(add_help=False)
    fixed_parser.add_argument("--resume-from", type=Path)
    fixed_parser.add_argument(
        "--allow-resume-config-mismatch",
        action="store_true",
        help="Diagnostic escape hatch; unsafe for scientific continuation.",
    )
    fixed_parser.add_argument("--fixed-stop-after-epoch", type=int)
    fixed_args, legacy_args = fixed_parser.parse_known_args()

    if fixed_args.resume_from is not None:
        resume_from = fixed_args.resume_from.expanduser().resolve()
        if not resume_from.exists():
            raise SystemExit(f"Resume checkpoint path does not exist: {resume_from}")
        os.environ["MALOQ_FIXED_RESUME_FROM"] = str(resume_from)
    else:
        os.environ.pop("MALOQ_FIXED_RESUME_FROM", None)
    if fixed_args.allow_resume_config_mismatch:
        os.environ["MALOQ_FIXED_ALLOW_CONFIG_MISMATCH"] = "1"
    else:
        os.environ.pop("MALOQ_FIXED_ALLOW_CONFIG_MISMATCH", None)
    if fixed_args.fixed_stop_after_epoch is not None:
        if fixed_args.fixed_stop_after_epoch <= 0:
            raise SystemExit("--fixed-stop-after-epoch must be positive")
        os.environ["MALOQ_FIXED_STOP_AFTER_EPOCH"] = str(
            fixed_args.fixed_stop_after_epoch
        )
    else:
        os.environ.pop("MALOQ_FIXED_STOP_AFTER_EPOCH", None)

    # eSEN imports a CuPy NCCL communicator while training_workflow is being
    # imported. Select the rank-local CUDA device before that import, matching
    # the ordering in the underlying experiment runner.
    import torch

    if torch.cuda.is_available():
        visible_device_count = torch.cuda.device_count()
        local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"),
            )
        )
        device_index = 0 if visible_device_count == 1 else local_rank
        if not 0 <= device_index < visible_device_count:
            raise SystemExit(
                f"Local rank {local_rank} cannot select one of "
                f"{visible_device_count} visible CUDA devices"
            )
        torch.cuda.set_device(device_index)

    from maloq.experimental.nte_qhflow3_composition import (
        workflow as experimental_workflow,
    )
    from maloq.experimental.nte_qhflow3_composition.workflow import (
        TrainingWorkflowFixed as ExperimentalTrainingWorkflowFixed,
    )
    from maloq.train_utils import training_workflow
    from maloq.train_utils.training_workflow_fixed import (
        TrainingWorkflowFixed as CanonicalTrainingWorkflowFixed,
    )

    training_workflow.TrainingWorkflow = CanonicalTrainingWorkflowFixed
    experimental_workflow.TrainingWorkflow = ExperimentalTrainingWorkflowFixed
    sys.argv = [sys.argv[0], *legacy_args]
    runpy.run_path(str(LEGACY_RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
