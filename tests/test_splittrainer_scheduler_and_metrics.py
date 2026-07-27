from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maloq.fock_utils.fock_targets_batched import Fock_Targets
from maloq.train_utils import splittrainer as splittrainer_module
from maloq.train_utils.splittrainer import SplitTrainer


class _PlateauScheduler:
    patience = 2

    def __init__(self) -> None:
        self.metrics = []

    def step(self, metric) -> None:
        self.metrics.append(float(metric))


class _EpochScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_scheduler_steps_once_after_reduced_validation() -> None:
    plateau = _PlateauScheduler()
    assert SplitTrainer.step_scheduler_after_validation(
        plateau,
        reduced_validation_loss=1.25,
        step_every_epoch=True,
    )
    assert plateau.metrics == [1.25]

    per_step_scheduler = _EpochScheduler()
    assert not SplitTrainer.step_scheduler_after_validation(
        per_step_scheduler,
        reduced_validation_loss=1.25,
        step_every_epoch=False,
    )
    assert per_step_scheduler.steps == 0

    epoch_scheduler = _EpochScheduler()
    assert SplitTrainer.step_scheduler_after_validation(
        epoch_scheduler,
        reduced_validation_loss=1.25,
        step_every_epoch=True,
    )
    assert epoch_scheduler.steps == 1


def test_batch_target_reference_uses_sample_id_not_loader_position() -> None:
    first = object()
    second = object()
    batch = SimpleNamespace(
        fock_target_object=[first, second],
        fock_target_id=torch.tensor([0, 7]),
    )
    target, target_index = SplitTrainer._batch_target_reference(
        batch,
        graph_index=1,
    )
    assert target is second
    assert target_index == 7


def test_validation_matrix_metrics_are_in_physical_units(
    monkeypatch,
) -> None:
    target = object.__new__(Fock_Targets)
    target.scale_shift_data = {
        "element_scalar_means": {1: [3.0]},
        "element_scalar_stds": {1: [4.0]},
        "scalar_irrep_indices": [0],
        "normalization_mode": "standardize",
    }
    target.neighbour_list_list = [
        np.empty((2, 0), dtype=np.int64),
    ]
    target.atomic_numbers_list = [np.array([1], dtype=np.int64)]
    target.orbitals_per_atom_list = [[1]]
    target.orbital_template = object()

    batch = SimpleNamespace(
        ptr=torch.tensor([0, 1]),
        batch=torch.tensor([0], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        fock_target_id=torch.tensor([0]),
        fock_target_object=[target],
        num_graphs=1,
    )

    def fake_label_to_matrix(
        orbital_template,
        fock_block_offsets,
        atomic_numbers,
        source_indices,
        target_indices,
        matrix,
        labels,
        *,
        forward,
    ):
        assert not forward
        matrix[0, 0] = labels[-1, 0]

    monkeypatch.setattr(
        splittrainer_module.matrix2labels_kernels,
        "numpy_single_matrix2label",
        fake_label_to_matrix,
    )

    class IdentityBasis:
        @staticmethod
        def get_H(values):
            return values

    trainer = object.__new__(SplitTrainer)
    trainer.open_shell = False
    stats = trainer.compute_validation_matrix_error_sums(
        batch,
        node_output=torch.tensor([[2.0]]),
        edge_output=torch.empty((0, 1)),
        node_target=torch.tensor([[0.5]]),
        edge_target=torch.empty((0, 1)),
        basis_transform=IdentityBasis(),
    )

    # Standardized difference 1.5 becomes 1.5 * std(4) = 6 Hartree.
    assert stats[0] == pytest.approx(6.0)
    assert stats[1] == pytest.approx(36.0)
    assert stats[2] == 1
    assert stats[3] == pytest.approx(6.0)
    assert stats[4] == pytest.approx(36.0)
    assert stats[5] == 1
