#!/usr/bin/env python3
"""Validate or run the bounded inherited full-matrix endpoint-flow smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project/MALOQ")
SOURCE_ROOT = PROJECT_ROOT / "src"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-28/qhflow2_endpoint_flow_nabladft.yaml"
)

for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from maloq.experimental.flow_matching import (  # noqa: E402
    FEATURE_SLUG,
    PROFILE_ID,
    EndpointFlowMaloqConfig,
    EndpointFlowTrainer,
    QHFlow2EndpointWorkflow,
)
from maloq.train_utils.splittrainer import SplitTrainer  # noqa: E402
from maloq.train_utils.training_workflow_v2 import (  # noqa: E402
    TrainingWorkflowV2Fixed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scope", choices=("validate", "smoke"), required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> EndpointFlowMaloqConfig:
    config = EndpointFlowMaloqConfig.from_file(path)
    if config.experimental_feature != FEATURE_SLUG:
        raise ValueError("Experimental feature slug mismatch.")
    if config.experimental_profile != PROFILE_ID:
        raise ValueError("Experimental profile mismatch.")
    if config.optimization.num_epochs != 20:
        raise ValueError("The matched candidate config must retain 20 epochs.")
    if config.model.num_edge_layers != 2:
        raise ValueError("The full-matrix endpoint candidate requires two edge layers.")
    flow = config.flow_matching
    if flow.state_scope != "node_and_edge":
        raise ValueError("The full-matrix endpoint profile requires node+edge state.")
    if flow.edge_parameterization != "ode_endpoint":
        raise ValueError("The edge state must use endpoint-parameterized ODE flow.")
    if not flow.enforce_hamiltonian_symmetry:
        raise ValueError(
            "The full-matrix endpoint profile requires symmetry projection."
        )
    if flow.architecture_version != 2:
        raise ValueError("The full-matrix endpoint profile requires architecture v2.")
    return config


def validation_payload(config: EndpointFlowMaloqConfig) -> dict[str, object]:
    flow = config.flow_matching
    return {
        "feature_slug": FEATURE_SLUG,
        "profile_id": PROFILE_ID,
        "status": "draft",
        "workflow": (
            f"{QHFlow2EndpointWorkflow.__module__}.{QHFlow2EndpointWorkflow.__name__}"
        ),
        "workflow_subclasses_fixed_v2": issubclass(
            QHFlow2EndpointWorkflow,
            TrainingWorkflowV2Fixed,
        ),
        "workflow_inherits_run": "run" not in QHFlow2EndpointWorkflow.__dict__,
        "trainer_subclasses_splittrainer": issubclass(
            EndpointFlowTrainer,
            SplitTrainer,
        ),
        "objective": {
            "state_scope": "node and directed-edge Hamiltonian blocks",
            "shared_graph_time": "one t per graph, shared by its nodes and edges",
            "node_path": "H_node,t=(1-t)H_node,0+tH_node,1",
            "edge_path": "H_edge,t=(1-t)H_edge,0+tH_edge,1",
            "prediction": "clean node and edge endpoints",
            "sampling_velocity": "(Hhat_1-H_t)/(1-t) for node and edge",
            "sampling": "three fixed joint Euler steps",
            "loss": config.loss.train_loss,
        },
        "matrix_contract": {
            "ao_codec": "coupled <-> shell-pair-packed <-> padded dense AO",
            "symmetry": "node transpose and reverse-edge transpose projection",
            "edge_conditioning": (
                "SO(3)-equivariant edge projection injected into edge and "
                "incident-node QHFlow3 embeddings"
            ),
            "parity": (
                "proper-rotation SO(3) contract; parity labels are relabeled "
                "by degree only for the equivariant edge projection"
            ),
        },
        "hyperparameters": flow.model_dump(mode="json"),
        "matched_run": {
            "epochs": config.optimization.num_epochs,
            "train": config.splits.num_train,
            "validation": config.splits.num_val,
            "backbone": config.model.backbone_type,
            "edge_layers": config.model.num_edge_layers,
            "head": config.model.head_type,
            "optimizer": config.optimization.optimizer_type,
            "raw": not config.loss.scale_and_shift,
            "seed": config.runtime.seed,
        },
        "launcher_modes": ["validate", "smoke"],
        "limitations": [
            "direct full-H endpoint rather than upstream residual H-Hinit",
            "SO(3) proper rotations only; no O(3)/reflection claim",
            "no CUDA/DDP/checkpoint evidence until smoke is explicitly run",
            "no full or queue mode",
        ],
    }


def smoke_config(
    config: EndpointFlowMaloqConfig,
    output_root: Path,
) -> EndpointFlowMaloqConfig:
    resolved = output_root.expanduser().resolve()
    canonical_outputs = OUTPUTS_ROOT.resolve()
    if resolved == canonical_outputs or canonical_outputs not in resolved.parents:
        raise ValueError(f"Smoke output must be a child of {canonical_outputs}.")
    if resolved.exists():
        raise FileExistsError(f"Smoke output already exists: {resolved}")

    payload = config.model_dump(mode="python")
    payload["dataset"].update(
        output_folder=str(resolved),
        run_name="nabladft-full-matrix-endpoint-flow-qhflow3-e2-muon-raw-v2-smoke",
    )
    payload["splits"].update(num_train=20, num_val=20, num_test=0, batch_size=2)
    payload["optimization"].update(num_epochs=1, warmup_steps=1)
    payload["checkpointing"].update(save_frequency=1)
    payload["tracking"].update(
        use_wandb=False,
        wandb_job_type="smoke",
        validation_matrix_metrics=False,
    )
    return EndpointFlowMaloqConfig.model_validate(payload)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    config = load_config(config_path)

    if args.scope == "validate":
        print(json.dumps(validation_payload(config), indent=2, sort_keys=True))
        return

    if args.output_root is None:
        raise SystemExit("--output-root is required for smoke.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise SystemExit(
            f"The bounded smoke requires exactly 2 ranks; got {world_size}."
        )
    candidate = smoke_config(config, args.output_root)
    workflow = QHFlow2EndpointWorkflow(candidate.to_workflow_config())
    workflow.run()
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "status": "smoke_complete",
                    "output_root": str(args.output_root.expanduser().resolve()),
                    "num_train": candidate.splits.num_train,
                    "num_val": candidate.splits.num_val,
                    "epochs": candidate.optimization.num_epochs,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
