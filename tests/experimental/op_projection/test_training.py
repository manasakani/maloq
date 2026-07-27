from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from maloq.experimental.op_projection.training import (
    OpProjectionTrainingConfig,
    deterministic_probe_seed,
    exact_matrix_sample_indices,
    identity_column_ranges,
    matrix_column_error_sums,
    molecule_probe_statistics,
    should_log_optimizer_step,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    PROJECT_ROOT
    / "_my_script/experiment/2026-07-28/nabladft_op_projection.yaml"
)


def test_molecule_probe_statistics_uses_equal_molecule_weighting() -> None:
    predicted = torch.zeros(5, 2)
    target = torch.ones(5, 2)
    molecule_ptr = torch.tensor([0, 2, 5])

    loss, numerator, denominator, count = molecule_probe_statistics(
        predicted,
        target,
        molecule_ptr,
    )

    expected = 0.5 * (4.0 / (2 * 2**2) + 6.0 / (2 * 3**2))
    torch.testing.assert_close(loss, torch.tensor(expected))
    torch.testing.assert_close(numerator, torch.tensor(10.0))
    torch.testing.assert_close(denominator, torch.tensor(10.0))
    assert count == 2


def test_deterministic_probe_seed_separates_streams_and_ranks() -> None:
    train_seed = deterministic_probe_seed(
        44,
        epoch=3,
        batch_index=9,
        rank=0,
        validation=False,
    )
    assert train_seed == deterministic_probe_seed(
        44,
        epoch=3,
        batch_index=9,
        rank=0,
        validation=False,
    )
    assert train_seed != deterministic_probe_seed(
        44,
        epoch=3,
        batch_index=9,
        rank=1,
        validation=False,
    )
    assert train_seed != deterministic_probe_seed(
        44,
        epoch=3,
        batch_index=9,
        rank=0,
        validation=True,
    )


def test_matrix_column_error_sums_match_dense_metrics_across_chunks() -> None:
    predicted = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    target = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [3.0, 7.0, 6.0],
            [7.0, 10.0, 8.0],
        ]
    )
    totals: dict[str, torch.Tensor | int] = {}
    for start, stop in ((0, 2), (2, 3)):
        chunk = matrix_column_error_sums(
            predicted[:, start:stop],
            target[:, start:stop],
            column_start=start,
            matrix_size=3,
        )
        for name, value in chunk.items():
            totals[name] = totals.get(name, 0) + value

    difference = predicted - target
    torch.testing.assert_close(
        totals["squared_error_sum"],
        difference.square().sum(),
    )
    torch.testing.assert_close(
        totals["absolute_error_sum"],
        difference.abs().sum(),
    )
    torch.testing.assert_close(
        totals["target_squared_sum"],
        target.square().sum(),
    )
    torch.testing.assert_close(
        totals["diagonal_absolute_error_sum"],
        difference.diagonal().abs().sum(),
    )
    torch.testing.assert_close(
        totals["off_diagonal_absolute_error_sum"],
        difference.abs().sum() - difference.diagonal().abs().sum(),
    )
    assert totals["entry_count"] == 9
    assert totals["diagonal_entry_count"] == 3
    assert totals["off_diagonal_entry_count"] == 6


def test_exact_matrix_validation_indices_cover_every_rank_local_sample() -> None:
    assert exact_matrix_sample_indices(32, split="validation") == list(range(32))
    assert exact_matrix_sample_indices(10, split="validation") == list(range(10))
    assert exact_matrix_sample_indices(
        6040,
        split="train",
        train_samples_per_rank=2,
    ) == [1510, 4530]


def test_identity_column_ranges_cover_all_columns_without_dense_chunk() -> None:
    for matrix_size in (1, 2, 17, 64, 65, 128):
        ranges = identity_column_ranges(matrix_size, configured_chunk_size=64)
        columns = [
            column
            for start, stop in ranges
            for column in range(start, stop)
        ]
        assert columns == list(range(matrix_size))
        if matrix_size > 1:
            assert all(stop - start < matrix_size for start, stop in ranges)


def test_matrix_metric_config_locks_validation_to_full_split() -> None:
    config = OpProjectionTrainingConfig.from_yaml(CONFIG_PATH)
    assert config.dataset.num_val == 64
    assert config.matrix_metrics.validation_scope == "full"
    assert config.for_scope("smoke").dataset.num_val == 20
    assert config.for_scope("smoke").matrix_metrics.validation_scope == "full"

    payload = config.model_dump(mode="python")
    payload["matrix_metrics"]["validation_scope"] = "subset"
    with pytest.raises(ValidationError):
        OpProjectionTrainingConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    payload["dataset"]["num_val"] = 63
    uneven = OpProjectionTrainingConfig.model_validate(payload)
    with pytest.raises(ValueError, match="validation rows must divide"):
        uneven.validate_contract()


def test_wandb_config_is_flat_and_records_full_validation_scope() -> None:
    config = OpProjectionTrainingConfig.from_yaml(CONFIG_PATH)
    payload = config.wandb_config(output_folder="/tmp/op-projection", scope="full")

    assert payload["output_folder"] == "/tmp/op-projection"
    assert payload["validation_matrix_metrics"] is True
    assert payload["validation_matrix_metrics_scope"] == "full"
    assert payload["num_val"] == 64
    assert all(not isinstance(value, dict) for value in payload.values())


def test_optimizer_step_logging_reserves_epoch_final_step() -> None:
    assert should_log_optimizer_step(
        optimizer_step=10,
        optimizer_step_in_epoch=10,
        optimizer_steps_per_epoch=604,
        every_n_steps=10,
    )
    assert not should_log_optimizer_step(
        optimizer_step=604,
        optimizer_step_in_epoch=604,
        optimizer_steps_per_epoch=604,
        every_n_steps=10,
    )
