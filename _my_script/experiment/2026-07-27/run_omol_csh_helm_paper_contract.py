#!/usr/bin/env python3
"""Validate or run the OMol_CSH paper-contract HELM experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from maloq.core.config import MaloqConfig
from maloq.dataset_utils.omol_csh_58k_dataset_utils import (
    OMolCSH58kDatabase,
    def2_tzvpd_basis_by_atomic_number,
    load_key_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SCALE_SHIFT = (
    PROJECT_ROOT / "src/maloq/fock_utils/element_scale_shifts_omol.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scope",
        choices=("validate", "smoke", "full"),
        required=True,
    )
    return parser.parse_args()


def validate_contract(config: dict) -> None:
    expected = {
        "dataset_name": "omol",
        "dataset_format": "omol_csh_h5",
        "omol_csh_metadata_policy": "paper_contract",
        "open_shell": False,
        "backbone_type": "esen",
        "atom_scalar_embedding_mode": "element_only",
        "loss_target": "fock_matrix",
        "nte_input_conditioning": "none",
        "distribute_graphs": False,
        "scale_and_shift": True,
        "compute_uncoupled_loss": True,
        "compute_eigenvalues": False,
        "compute_total_energy": False,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r}, expected {expected_value!r}"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError("OMol_CSH paper contract mismatch: " + details)

    keys = load_key_manifest(config["dbpath"])
    requested = (
        int(config["num_train"])
        + int(config["num_val"])
        + int(config["num_test"])
    )
    if requested > len(keys):
        raise ValueError(
            f"Requested {requested} structures from {len(keys)} H5 entries."
        )


def validate_scale_shift_artifact() -> None:
    if not LEGACY_SCALE_SHIFT.is_file():
        raise FileNotFoundError(
            f"Required OMol scale-shift artifact is missing: "
            f"{LEGACY_SCALE_SHIFT}"
        )
    payload = torch.load(
        LEGACY_SCALE_SHIFT,
        map_location="cpu",
        weights_only=False,
    )
    required = {
        "element_scalar_means",
        "element_scalar_stds",
        "scalar_irrep_indices",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"OMol scale-shift artifact is missing fields {missing}."
        )
    expected_elements = (
        set(def2_tzvpd_basis_by_atomic_number())
        - set(range(21, 31))
        - {84, 85}
    )
    mean_elements = set(map(int, payload["element_scalar_means"]))
    std_elements = set(map(int, payload["element_scalar_stds"]))
    if mean_elements != expected_elements or std_elements != expected_elements:
        raise ValueError(
            "OMol scale-shift artifact element keys do not match the "
            "58-element OMol_CSH basis contract."
        )


def validate_sample(config: dict) -> None:
    database = OMolCSH58kDatabase(
        config["dbpath"],
        indices=[0],
        metadata_policy=config["omol_csh_metadata_policy"],
    )
    sample = database[0]
    matrix = sample["fock_matrix"]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Corrected Fock matrix is not square: {matrix.shape}")
    print(
        "OMol_CSH sample validated: "
        f"name={sample['name']} atoms={len(sample['atomic_numbers'])} "
        f"fock={matrix.shape} source_charge={sample['source_charge']} "
        f"source_spin={sample['source_spin']} "
        f"model_charge={sample['charge']} "
        f"model_spin={sample['spin_multiplicity']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = MaloqConfig.from_file(args.config).to_workflow_config()
    validate_contract(config)
    validate_scale_shift_artifact()

    output_override = os.environ.get("OMOL_CSH_OUTPUT_FOLDER")
    if output_override:
        config["output_folder"] = output_override

    if args.scope == "validate":
        validate_sample(config)
        return

    if args.scope == "smoke":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        config.update(
            num_train=world_size,
            num_val=world_size,
            num_test=0,
            num_epochs=1,
            batch_size=1,
            shuffle=False,
            save_frequency=1,
            use_wandb=False,
            validation_matrix_metrics=True,
            validation_matrix_metrics_frequency=1,
            run_name="omol-csh-helm-paper-contract-smoke",
        )

    from maloq.train_utils.training_workflow import TrainingWorkflow

    TrainingWorkflow(config).run()


if __name__ == "__main__":
    main()
