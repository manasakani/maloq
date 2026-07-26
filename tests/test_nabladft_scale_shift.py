from pathlib import Path

import pytest
import torch

from maloq.core.config import MaloqConfig
from maloq.fock_utils.fock_targets_batched import Fock_Targets
from maloq.train_utils.training_workflow import TrainingWorkflow


def test_scale_shift_standardizes_and_restores_scalar_components():
    targets = object.__new__(Fock_Targets)
    targets.scale_shift_data = {
        "element_scalar_means": {
            1: [2.0, -1.0],
            6: [4.0, 3.0],
        },
        "element_scalar_stds": {
            1: [2.0, 4.0],
            6: [1.0, 2.0],
        },
        "scalar_irrep_indices": [0, 2],
    }
    labels = torch.tensor(
        [
            [6.0, 11.0, 7.0],
            [5.0, 13.0, 9.0],
        ]
    )
    atomic_numbers = torch.tensor([1, 6])

    scaled = targets.scale_shift_node_blocks(
        labels.clone(),
        atomic_numbers,
    )

    torch.testing.assert_close(
        scaled,
        torch.tensor(
            [
                [2.0, 11.0, 2.0],
                [1.0, 13.0, 3.0],
            ]
        ),
    )
    restored = targets.unscale_shift_node_blocks(scaled, atomic_numbers)
    torch.testing.assert_close(restored, labels)


def test_scale_shift_can_shift_without_standardizing():
    targets = object.__new__(Fock_Targets)
    targets.scale_shift_data = {
        "element_scalar_means": {
            1: [2.0, -1.0],
            6: [4.0, 3.0],
        },
        "element_scalar_stds": {
            1: [2.0, 4.0],
            6: [1.0, 2.0],
        },
        "scalar_irrep_indices": [0, 2],
        "normalization_mode": "shift_only",
    }
    labels = torch.tensor(
        [
            [6.0, 11.0, 7.0],
            [5.0, 13.0, 9.0],
        ]
    )
    atomic_numbers = torch.tensor([1, 6])

    shifted = targets.scale_shift_node_blocks(
        labels.clone(),
        atomic_numbers,
    )

    torch.testing.assert_close(
        shifted,
        torch.tensor(
            [
                [4.0, 11.0, 8.0],
                [1.0, 13.0, 6.0],
            ]
        ),
    )
    restored = targets.unscale_shift_node_blocks(shifted, atomic_numbers)
    torch.testing.assert_close(restored, labels)


def test_config_preserves_explicit_scale_shift_path(tmp_path: Path):
    stats_path = tmp_path / "stats.pt"
    config = MaloqConfig.model_validate(
        {
            "loss": {
                "scale_and_shift": True,
                "scale_shift_mode": "shift_only",
                "scale_shift_path": str(stats_path),
            }
        }
    ).to_workflow_config()

    assert config["scale_and_shift"] is True
    assert config["scale_shift_mode"] == "shift_only"
    assert config["scale_shift_path"] == str(stats_path)


def test_workflow_loads_matching_scale_shift_provenance(tmp_path: Path):
    stats_path = tmp_path / "stats.pt"
    payload = {
        "element_scalar_means": {1: [0.5]},
        "element_scalar_stds": {1: [2.0]},
        "scalar_irrep_indices": [0],
        "provenance": {
            "dataset_name": "nablaDFT",
            "loss_target": "fock_matrix",
            "rcut_orbitals": 8.0,
        },
    }
    torch.save(payload, stats_path)
    workflow = object.__new__(TrainingWorkflow)
    workflow.config = {
        "scale_and_shift": True,
        "scale_shift_path": str(stats_path),
        "dataset_name": "nablaDFT",
        "loss_target": "fock_matrix",
        "rcut_orbitals": 8.0,
        "open_shell": False,
        "scale_shift_mode": "shift_only",
    }

    loaded = workflow._handle_scale_shift()

    assert loaded == {
        "element_scalar_means": {1: [0.5]},
        "element_scalar_stds": {1: [2.0]},
        "scalar_irrep_indices": [0],
        "normalization_mode": "shift_only",
    }


def test_workflow_rejects_scale_shift_provenance_mismatch(tmp_path: Path):
    stats_path = tmp_path / "stats.pt"
    torch.save(
        {
            "element_scalar_means": {1: [0.5]},
            "element_scalar_stds": {1: [2.0]},
            "scalar_irrep_indices": [0],
            "provenance": {
                "dataset_name": "nablaDFT",
                "loss_target": "fock_matrix",
                "rcut_orbitals": 6.0,
            },
        },
        stats_path,
    )
    workflow = object.__new__(TrainingWorkflow)
    workflow.config = {
        "scale_and_shift": True,
        "scale_shift_path": str(stats_path),
        "dataset_name": "nablaDFT",
        "loss_target": "fock_matrix",
        "rcut_orbitals": 8.0,
        "open_shell": False,
    }

    with pytest.raises(ValueError, match="provenance mismatch"):
        workflow._handle_scale_shift()
