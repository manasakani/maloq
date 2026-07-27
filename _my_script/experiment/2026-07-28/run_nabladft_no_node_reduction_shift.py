#!/usr/bin/env python3
"""Run the E3 Muon+SHIFT node-reduction ablation on NablaDFT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Literal


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
SOURCE_ROOT = PROJECT_ROOT / "src"
BASE_RUNNER = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-27/run_nabladft_v2_ofat.py"
)
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-27/nabladft_v2_ofat_common.yaml"
)
ALLOWED_LANES = (
    "maloq-e3-muon-shift",
    "ntev2-e3-muon-shift",
    "qhflow3-e3-muon-shift",
)
Scope = Literal["validate", "smoke", "full"]

for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _load_base_suite() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_nabladft_v2_ofat_base",
        BASE_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base OFAT runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE_SUITE = _load_base_suite()
MaloqConfig = BASE_SUITE.MaloqConfig
TrainingWorkflowV2Fixed = BASE_SUITE.TrainingWorkflowV2Fixed


def _ablation_slug(reduce_node: bool, reduce_node_intra: bool) -> str:
    if not reduce_node and not reduce_node_intra:
        return "no-node-reduction"
    return (
        f"node-reduction-r{int(reduce_node)}-"
        f"intra{int(reduce_node_intra)}"
    )


def _display_suffix(reduce_node: bool, reduce_node_intra: bool) -> str:
    if not reduce_node and not reduce_node_intra:
        return "NoNodeReduce"
    return (
        f"NodeReduce-r{int(reduce_node)}-"
        f"intra{int(reduce_node_intra)}"
    )


def _build_ablation_config(
    base,
    lane,
    scope: Scope,
    output_root: Path,
    *,
    reduce_node: bool,
    reduce_node_intra: bool,
):
    config = BASE_SUITE._build_lane_config(base, lane, scope, output_root)
    payload = config.model_dump(mode="python")
    slug = _ablation_slug(reduce_node, reduce_node_intra)
    run_name = f"{config.dataset.run_name}-{slug}"

    display_name = config.tracking.wandb_run_name
    if display_name.endswith(" | V2"):
        display_name = display_name[: -len(" | V2")]
    display_name = (
        f"{display_name} | "
        f"{_display_suffix(reduce_node, reduce_node_intra)} | V2"
    )

    payload["dataset"].update(
        run_name=run_name,
        output_folder=str(output_root),
    )
    payload["model"].update(
        model_variant=run_name,
        reduce_node=reduce_node,
        reduce_node_intra=reduce_node_intra,
    )
    payload["tracking"].update(
        wandb_run_name=display_name,
        wandb_tags=tuple(
            dict.fromkeys(
                [
                    *payload["tracking"]["wandb_tags"],
                    f"reduce-node:{str(reduce_node).lower()}",
                    (
                        "reduce-node-intra:"
                        f"{str(reduce_node_intra).lower()}"
                    ),
                    "axis:node-reduction",
                    f"ablation:{slug}",
                ]
            )
        ),
    )

    typed_config = MaloqConfig.model_validate(payload)
    BASE_SUITE._validate_lane_contract(typed_config, lane, scope)
    if typed_config.model.reduce_node is not reduce_node:
        raise ValueError("reduce_node option was not preserved.")
    if typed_config.model.reduce_node_intra is not reduce_node_intra:
        raise ValueError("reduce_node_intra option was not preserved.")
    if typed_config.model.reduce_edge:
        raise ValueError("This ablation must keep reduce_edge=false.")
    return typed_config


def _config_preview(config, lane) -> dict[str, object]:
    preview = BASE_SUITE._config_preview(config, lane)
    preview.update(
        run_name=config.dataset.run_name,
        model_variant=config.model.model_variant,
        reduce_edge=config.model.reduce_edge,
        reduce_node=config.model.reduce_node,
        reduce_node_intra=config.model.reduce_node_intra,
    )
    return preview


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
    )
    parser.add_argument(
        "--lane",
        choices=("all", *ALLOWED_LANES),
        required=True,
    )
    parser.add_argument(
        "--scope",
        choices=("validate", "smoke", "full"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--reduce-node",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reduce-node-intra",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
    base_config_path = args.base_config.expanduser().resolve()
    if not base_config_path.is_file():
        raise SystemExit(f"Base config not found: {base_config_path}")

    base = MaloqConfig.from_file(base_config_path)
    input_metadata = BASE_SUITE._validate_suite_inputs(base)
    scope: Scope = args.scope

    if args.lane == "all":
        if scope != "validate":
            raise SystemExit("--lane all is allowed only with --scope validate.")
        selected_lanes = tuple(
            BASE_SUITE.LANE_BY_ID[lane_id] for lane_id in ALLOWED_LANES
        )
    else:
        selected_lanes = (BASE_SUITE.LANE_BY_ID[args.lane],)

    if scope == "validate":
        slug = _ablation_slug(
            args.reduce_node,
            args.reduce_node_intra,
        )
        preview_base = (
            PROJECT_ROOT
            / "outputs/_config-preview"
            / "nabladft-v2-node-reduction-ablation"
        )
        previews = [
            _config_preview(
                _build_ablation_config(
                    base,
                    lane,
                    scope,
                    preview_base / f"{lane.id}-{slug}",
                    reduce_node=args.reduce_node,
                    reduce_node_intra=args.reduce_node_intra,
                ),
                lane,
            )
            for lane in selected_lanes
        ]
        print(
            json.dumps(
                {
                    "suite": "nabladft-v2-node-reduction-ablation",
                    "workflow": (
                        f"{TrainingWorkflowV2Fixed.__module__}."
                        f"{TrainingWorkflowV2Fixed.__name__}"
                    ),
                    "base_runner": str(BASE_RUNNER),
                    "base_config": str(base_config_path),
                    "inputs": input_metadata,
                    "options": {
                        "reduce_node": args.reduce_node,
                        "reduce_node_intra": args.reduce_node_intra,
                    },
                    "lanes": previews,
                },
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
            f"Output must be a lane directory below {outputs_root}: "
            f"{output_root}"
        )

    lane = selected_lanes[0]
    typed_config = _build_ablation_config(
        base,
        lane,
        scope,
        output_root,
        reduce_node=args.reduce_node,
        reduce_node_intra=args.reduce_node_intra,
    )
    if _rank() == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        resolved_payload = {
            "suite": "nabladft-v2-node-reduction-ablation",
            "scope": scope,
            "lane": asdict(lane),
            "base_runner": str(BASE_RUNNER),
            "base_config": str(base_config_path),
            "base_config_sha256": BASE_SUITE._sha256(base_config_path),
            "inputs": input_metadata,
            "options": {
                "reduce_node": args.reduce_node,
                "reduce_node_intra": args.reduce_node_intra,
            },
            "config": typed_config.model_dump(mode="json"),
        }
        (output_root / "resolved_node_reduction_config.json").write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n"
        )

    TrainingWorkflowV2Fixed(typed_config.to_workflow_config()).run()


if __name__ == "__main__":
    main()
