#!/usr/bin/env python3
"""Run matched E3 Muon+SHIFT models with symmetry-reduced node/edge matrix outputs."""

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
BASE_RUNNER = PROJECT_ROOT / "_my_script/experiment/2026-07-27/run_nabladft_v2_ofat.py"
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "_my_script/experiment/2026-07-27/nabladft_v2_ofat_common.yaml"
)
ALLOWED_LANES = (
    "maloq-e3-muon-shift",
    "ntev2-e3-muon-shift",
    "qhflow3-e3-muon-shift",
)
FEATURE_SLUG = "coupled-hamiltonian-symmetry"
QHFLOW3_GRID_SHAPE = (10, 11)
Scope = Literal["validate", "smoke", "full"]

for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _load_base_suite() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_nabladft_v2_ofat_symmetry_base",
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

from maloq.experimental.coupled_hamiltonian_symmetry import (  # noqa: E402
    CoupledHamiltonianSymmetryWorkflow,
)


def _feature_slug(lane) -> str:
    if lane.architecture == "qhflow3":
        return f"{FEATURE_SLUG}-grid10x11"
    return FEATURE_SLUG


def _validate_qhflow3_grid(config, lane) -> tuple[int, int] | None:
    if lane.architecture != "qhflow3":
        return None
    if config.model.qhflow3_grid_resolution is not None:
        raise ValueError(
            "The matched QHFlow3 10x11 lane must use the eSEN default grid "
            "policy (qhflow3_grid_resolution=None)."
        )

    from maloq.helm.qhf_layer.so3 import SO3_Grid

    grid = SO3_Grid(
        lmax=4,
        mmax=4,
        resolution=config.model.qhflow3_grid_resolution,
        rescale=True,
    )
    shape = (grid.lat_resolution, grid.long_resolution)
    if shape != QHFLOW3_GRID_SHAPE:
        raise ValueError(
            f"Expected the lmax=4 eSEN default grid {QHFLOW3_GRID_SHAPE}, "
            f"got {shape}."
        )
    return shape


def _build_feature_config(base, lane, scope: Scope, output_root: Path):
    config = BASE_SUITE._build_lane_config(base, lane, scope, output_root)
    payload = config.model_dump(mode="python")
    feature_slug = _feature_slug(lane)
    run_name = f"{config.dataset.run_name}-{feature_slug}"

    display_name = config.tracking.wandb_run_name
    if display_name.endswith(" | V2"):
        display_name = display_name[: -len(" | V2")]
    if lane.architecture == "qhflow3":
        display_name = display_name.replace("QHFlow3-E3", "QHFlow3-E3-10x11")
    display_name = f"{display_name} | Node+EdgeIrrepSym | V2"

    payload["dataset"].update(
        run_name=run_name,
        output_folder=str(output_root),
    )
    payload["model"].update(
        model_variant=run_name,
        reduce_node=False,
        reduce_node_intra=False,
        reduce_edge=False,
    )
    if lane.architecture == "qhflow3":
        payload["model"]["qhflow3_grid_resolution"] = None

    grid_tags = (
        ("grid:esen-default-10x11", "grid-policy:esen-default")
        if lane.architecture == "qhflow3"
        else ()
    )
    payload["tracking"].update(
        wandb_run_name=display_name,
        wandb_tags=tuple(
            dict.fromkeys(
                [
                    *payload["tracking"]["wandb_tags"],
                    "feature:coupled-hamiltonian-symmetry",
                    "symmetry:node-intra-edge-pair-irreps",
                    "symmetry-space:reduced-coupled-irreps",
                    "legacy-reduction-flags:false",
                    "head-reduce-node:true",
                    "head-reduce-node-intra:true",
                    "head-reduce-edge:true",
                    "axis:output-symmetry",
                    *grid_tags,
                ]
            )
        ),
    )

    typed_config = MaloqConfig.model_validate(payload)
    BASE_SUITE._validate_lane_contract(typed_config, lane, scope)
    if any(
        (
            typed_config.model.reduce_node,
            typed_config.model.reduce_node_intra,
            typed_config.model.reduce_edge,
        )
    ):
        raise ValueError(
            "Legacy reduction flags must remain off; the experimental head owns reduction."
        )
    if typed_config.model.head_type != "maloq_muon":
        raise ValueError("The matched symmetry experiment requires the Muon head.")
    _validate_qhflow3_grid(typed_config, lane)
    return typed_config


def _config_preview(config, lane) -> dict[str, object]:
    preview = BASE_SUITE._config_preview(config, lane)
    grid_shape = _validate_qhflow3_grid(config, lane)
    preview.update(
        run_name=config.dataset.run_name,
        model_variant=config.model.model_variant,
        workflow=(
            f"{CoupledHamiltonianSymmetryWorkflow.__module__}."
            f"{CoupledHamiltonianSymmetryWorkflow.__name__}"
        ),
        symmetry_profile=CoupledHamiltonianSymmetryWorkflow.feature_profile,
        node_symmetry="upper_triangle_plus_even_diagonal_irreps",
        edge_symmetry="reverse_pair_alpha_beta_irreps",
        symmetry_reduction=True,
        configured_qhflow3_grid_resolution=(
            config.model.qhflow3_grid_resolution
            if lane.architecture == "qhflow3"
            else None
        ),
        effective_qhflow3_grid_shape=(list(grid_shape) if grid_shape else None),
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
        preview_base = (
            PROJECT_ROOT
            / "outputs/_config-preview/nabladft-coupled-hamiltonian-symmetry"
        )
        previews = [
            _config_preview(
                _build_feature_config(
                    base,
                    lane,
                    scope,
                    preview_base / lane.id,
                ),
                lane,
            )
            for lane in selected_lanes
        ]
        print(
            json.dumps(
                {
                    "suite": "nabladft-coupled-hamiltonian-symmetry",
                    "feature": FEATURE_SLUG,
                    "workflow": (
                        f"{CoupledHamiltonianSymmetryWorkflow.__module__}."
                        f"{CoupledHamiltonianSymmetryWorkflow.__name__}"
                    ),
                    "base_runner": str(BASE_RUNNER),
                    "base_config": str(base_config_path),
                    "inputs": input_metadata,
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
            f"Output must be a lane directory below {outputs_root}: {output_root}"
        )

    lane = selected_lanes[0]
    typed_config = _build_feature_config(base, lane, scope, output_root)
    if _rank() == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        resolved_payload = {
            "suite": "nabladft-coupled-hamiltonian-symmetry",
            "scope": scope,
            "feature": _feature_slug(lane),
            "symmetry_profile": (CoupledHamiltonianSymmetryWorkflow.feature_profile),
            "lane": asdict(lane),
            "base_runner": str(BASE_RUNNER),
            "base_config": str(base_config_path),
            "base_config_sha256": BASE_SUITE._sha256(base_config_path),
            "inputs": input_metadata,
            "config": typed_config.model_dump(mode="json"),
        }
        (output_root / "resolved_coupled_hamiltonian_symmetry_config.json").write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n"
        )

    CoupledHamiltonianSymmetryWorkflow(typed_config.to_workflow_config()).run()


if __name__ == "__main__":
    main()
