from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maloq.fock_utils.fock_targets_batched import (
    Fock_Targets,
    compute_outside_cutoff_reference_error_sums,
)
from maloq.train_utils import splittrainer as splittrainer_module
from maloq.train_utils.splittrainer import (
    SplitTrainer,
    MATRIX_EVAL_COLUMNS,
    finalize_validation_matrix_error_sums,
)


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


def test_matrix_eval_columns_preserve_legacy_prefix() -> None:
    assert MATRIX_EVAL_COLUMNS[:4] == (
        "Total_MAE",
        "Eigenvalue_MAE",
        "Total_Energy_Error",
        "Num_Atoms",
    )
    assert MATRIX_EVAL_COLUMNS[4:] == ("Full_Dense_MAE",)




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
    target.outside_cutoff_reference_error_sums_list = [((0.0,), (0.0,), 0)]

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
    assert stats[9:] == pytest.approx((0.0, 0.0, 0))


def test_outside_cutoff_reference_sums_use_neighbour_complement() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    block_starts = np.array([0, 1, 2, 3])
    neighbour_list = np.array([[0, 1], [1, 0]], dtype=np.int64)

    absolute_sums, squared_sums, entry_count = (
        compute_outside_cutoff_reference_error_sums(
            matrix,
            block_starts,
            neighbour_list,
        )
    )

    # The represented entries are the diagonal and directed 0<->1 blocks.
    # The four outside-cutoff values are [3, 3, 5, 5].
    assert absolute_sums == pytest.approx((16.0,))
    assert squared_sums == pytest.approx((68.0,))
    assert entry_count == 4


def test_validation_full_matrix_stats_include_outside_cutoff_reference(
    monkeypatch,
) -> None:
    target = object.__new__(Fock_Targets)
    target.scale_shift_data = None
    target.neighbour_list_list = [np.empty((2, 0), dtype=np.int64)]
    target.atomic_numbers_list = [np.array([1, 1], dtype=np.int64)]
    target.orbitals_per_atom_list = [[1, 1]]
    target.orbital_template = object()
    target.outside_cutoff_reference_error_sums_list = [((4.0,), (8.0,), 2)]

    batch = SimpleNamespace(
        ptr=torch.tensor([0, 2]),
        batch=torch.tensor([0, 0], dtype=torch.long),
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
        matrix[0, 0] = labels[-2, 0]
        matrix[1, 1] = labels[-1, 0]

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
        node_output=torch.zeros((2, 1)),
        edge_output=torch.empty((0, 1)),
        node_target=torch.zeros((2, 1)),
        edge_target=torch.empty((0, 1)),
        basis_transform=IdentityBasis(),
    )

    assert stats[:3] == pytest.approx((0.0, 0.0, 4))
    assert stats[3:6] == pytest.approx((0.0, 0.0, 2))
    assert stats[6:9] == pytest.approx((0.0, 0.0, 0))
    assert stats[9:] == pytest.approx((4.0, 8.0, 2))
    assert (stats[0] + stats[9]) / stats[2] == pytest.approx(1.0)
    assert (stats[1] + stats[10]) / stats[2] == pytest.approx(2.0)


def test_finalize_validation_matrix_error_sums_preserves_legacy_keys(
) -> None:
    matrix_stats = torch.tensor(
        [
            6.0,
            14.0,
            9.0,
            2.0,
            4.0,
            3.0,
            4.0,
            10.0,
            2.0,
            3.0,
            5.0,
            4.0,
        ],
        dtype=torch.float64,
    )
    metrics = finalize_validation_matrix_error_sums(matrix_stats)

    assert metrics["validation/matrix_mae"] == pytest.approx(6.0 / 9.0)
    assert metrics["validation/matrix_mse"] == pytest.approx(14.0 / 9.0)
    assert metrics["validation/full_dense_matrix_mae"] == pytest.approx(1.0)
    assert metrics["validation/full_dense_matrix_mse"] == pytest.approx(
        19.0 / 9.0
    )
    assert metrics["validation/represented_block_matrix_mae"] == pytest.approx(
        6.0 / 5.0
    )
    assert metrics["validation/outside_cutoff_matrix_mae"] == pytest.approx(
        3.0 / 4.0
    )
    assert metrics[
        "validation/legacy_cutoff_truncated_matrix_mae"
    ] == pytest.approx(6.0 / 9.0)
    assert metrics["validation/represented_block_matrix_fraction"] == pytest.approx(
        5.0 / 9.0
    )
    assert metrics["validation/matrix_metrics_schema_version"] == 2.0

    variant_metrics = finalize_validation_matrix_error_sums(
        matrix_stats,
        metric_prefix="validation/example_variant",
    )
    assert variant_metrics[
        "validation/example_variant/full_dense_matrix_mae"
    ] == pytest.approx(1.0)
    assert set(variant_metrics) == {
        key.replace("validation/", "validation/example_variant/", 1)
        for key in metrics
    }


def test_default_validation_matrix_variant_hook_is_feature_neutral() -> None:
    calls = []

    class _Trainer(SplitTrainer):
        def compute_validation_matrix_error_sums(self, *args):
            calls.append(args)
            return tuple(float(index) for index in range(12))

    trainer = object.__new__(_Trainer)
    arguments = tuple(object() for _ in range(6))
    primary, variants = (
        trainer.compute_validation_matrix_error_sums_with_variants(*arguments)
    )

    assert calls == [arguments]
    assert primary == tuple(float(index) for index in range(12))
    assert variants == {}


def test_validation_matrix_metrics_preserve_small_antisymmetric_error(
    monkeypatch,
) -> None:
    target = object.__new__(Fock_Targets)
    target.scale_shift_data = None
    target.neighbour_list_list = [np.empty((2, 0), dtype=np.int64)]
    target.atomic_numbers_list = [np.array([1], dtype=np.int64)]
    target.orbitals_per_atom_list = [[2]]
    target.orbital_template = object()
    target.outside_cutoff_reference_error_sums_list = [((0.0,), (0.0,), 0)]

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
        matrix[0, 1] = labels[-1, 0]
        matrix[1, 0] = labels[-1, 1]

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
        node_output=torch.tensor([[1.0e-5, -1.0e-5]]),
        edge_output=torch.empty((0, 2)),
        node_target=torch.zeros((1, 2)),
        edge_target=torch.empty((0, 2)),
        basis_transform=IdentityBasis(),
    )

    # Preserve the historical tolerance before symmetric projection.
    assert stats[0] == pytest.approx(2.0e-5)
    assert stats[1] == pytest.approx(2.0e-10)
