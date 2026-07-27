from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from maloq.core.config import MaloqConfig
from maloq.dataset_utils import omol_csh_58k_dataset_utils as omol_module
from maloq.dataset_utils.omol_csh_58k_dataset_utils import (
    OMolCSH58kDatabase,
    OMolCSHGraphDataset,
    def2_tzvpd_basis_by_atomic_number,
    reinflate_symmetric_matrix,
    restore_orca_diffuse_order,
)
from maloq.fock_utils import utils_orca_out
from maloq.train_utils.training_workflow import TrainingWorkflow


def _pack_upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0])]


def test_reinflate_symmetric_matrix_round_trip() -> None:
    matrix = np.arange(25, dtype=np.float64).reshape(5, 5)
    matrix = matrix + matrix.T
    restored = reinflate_symmetric_matrix(_pack_upper(matrix))
    np.testing.assert_array_equal(restored, matrix)


def test_reinflate_symmetric_matrix_rejects_non_triangular_length() -> None:
    with pytest.raises(ValueError, match="triangular number"):
        reinflate_symmetric_matrix(np.arange(5, dtype=np.float32))


def test_restore_orca_diffuse_order_undoes_sorted_basis_transform() -> None:
    correct_basis = {1: [0, 1, 0]}
    sorted_basis = {1: sorted(correct_basis[1])}
    atomic_numbers = np.array([1], dtype=np.int64)
    matrix_size = sum(2 * degree + 1 for degree in correct_basis[1])
    raw_orca = np.arange(
        matrix_size * matrix_size,
        dtype=np.float64,
    ).reshape(matrix_size, matrix_size)
    raw_orca = raw_orca + raw_orca.T

    published = utils_orca_out.sort_by_m(
        raw_orca,
        sorted_basis,
        atomic_numbers,
        direction="orca_to_e3nn",
    )
    expected = utils_orca_out.sort_by_m(
        raw_orca,
        correct_basis,
        atomic_numbers,
        direction="orca_to_e3nn",
    )
    actual = restore_orca_diffuse_order(
        published,
        atomic_numbers,
        correct_basis_dict=correct_basis,
    )
    np.testing.assert_array_equal(actual, expected)

    torch_actual = restore_orca_diffuse_order(
        torch.tensor(published, dtype=torch.float32),
        torch.tensor(atomic_numbers),
        correct_basis_dict=correct_basis,
    )
    assert torch_actual.dtype == torch.float32
    torch.testing.assert_close(
        torch_actual,
        torch.tensor(expected, dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_h5_paper_contract_keeps_source_metadata_but_uses_csh_constants(
    tmp_path,
) -> None:
    database_path = tmp_path / "omol_csh.h5"
    key = "nested/example"
    basis = def2_tzvpd_basis_by_atomic_number()
    hydrogen_basis = basis[1]
    matrix_size = sum(2 * degree + 1 for degree in hydrogen_basis)
    raw_orca = np.arange(
        matrix_size * matrix_size,
        dtype=np.float64,
    ).reshape(matrix_size, matrix_size)
    raw_orca = raw_orca + raw_orca.T
    published = utils_orca_out.sort_by_m(
        raw_orca,
        {1: sorted(hydrogen_basis)},
        [1],
        direction="orca_to_e3nn",
    )
    expected = utils_orca_out.sort_by_m(
        raw_orca,
        {1: hydrogen_basis},
        [1],
        direction="orca_to_e3nn",
    )

    with h5py.File(database_path, "w") as handle:
        group = handle.create_group(key)
        group.create_dataset("elements", data=np.array([1], dtype=np.int32))
        group.create_dataset("coords", data=np.zeros((1, 3), dtype=np.float64))
        group.create_dataset("fock", data=_pack_upper(published))
        group.attrs["charge"] = -2
        group.attrs["spin"] = 17
    (tmp_path / "omol_csh.h5.keys.json").write_text(
        json.dumps([key]) + "\n"
    )

    paper_database = OMolCSH58kDatabase(
        database_path,
        metadata_policy="paper_contract",
    )
    sample = paper_database[0]
    assert sample["charge"] == 0
    assert sample["spin_multiplicity"] == 1
    assert sample["source_charge"] == -2
    assert sample["source_spin"] == 17
    assert sample["matrix_storage_convention"] == "maloq_e3nn"
    assert "overlap_matrix" not in sample
    np.testing.assert_array_equal(sample["fock_matrix"], expected)

    preserved_database = OMolCSH58kDatabase(
        database_path,
        metadata_policy="preserve",
    )
    preserved = preserved_database[0]
    assert preserved["charge"] == -2
    assert preserved["spin_multiplicity"] == 17


def test_streaming_graph_dataset_reuses_orbital_gpu_metadata(
    monkeypatch,
) -> None:
    calls = []
    basis_transform = object()
    device_cache = object()

    class FakeTargets:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            self.req_output_irreps = SimpleNamespace()
            self.basis_transformation = basis_transform
            self.orbital_template_device_cache = device_cache
            self.ls_list = torch.tensor([0])
            self.orbital_starts = {1: 0}
            self.orbital_template = [[(slice(0, 1),) * 3]]
            self.out_js_list = [0]
            self.orbital_basis = {1: [0]}

    monkeypatch.setattr(
        omol_module.fock_targets_batched,
        "Fock_Targets",
        FakeTargets,
    )

    sample = {
        "atomic_numbers": np.array([1]),
        "pos": np.zeros((1, 3)),
        "fock_matrix": np.zeros((1, 1)),
    }

    class FakeDatabase:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return sample

    dataset = OMolCSHGraphDataset(
        FakeDatabase(),
        start_idx=0,
        end_idx=1,
        cutoff=6.0,
        dtype=torch.float32,
    )
    dataset._make_targets(sample, reuse_orbital_metadata=True)

    assert len(calls) == 2
    assert "basis_transformation" not in calls[0]
    assert calls[1]["basis_transformation"] is basis_transform
    assert calls[1]["orbital_template_device_cache"] is device_cache
    assert calls[1]["verbose"] is False


def _omol_workflow_for_validation(**overrides) -> TrainingWorkflow:
    config = MaloqConfig(
        dataset={
            "dataset_name": "omol",
            "dataset_format": "omol_csh_h5",
            "omol_csh_metadata_policy": "paper_contract",
            "dbpath": "/not/opened/by-config-validation.h5",
        },
        execution={
            "compute_total_energy": False,
            "compute_eigenvalues": False,
        },
        splits={
            "num_train": 2,
            "num_val": 2,
            "num_test": 0,
        },
        model={
            "backbone_type": "esen",
            "atom_scalar_embedding_mode": "element_only",
        },
        loss={
            "loss_target": "fock_matrix",
            "compute_uncoupled_loss": True,
        },
    ).to_workflow_config()
    config.update(overrides)
    workflow = object.__new__(TrainingWorkflow)
    workflow.config = config
    workflow.rank = 1
    workflow.world_size = 2
    workflow.device = torch.device("cpu")
    return workflow


def test_omol_h5_validation_requires_metadata_independent_model() -> None:
    workflow = _omol_workflow_for_validation(
        atom_scalar_embedding_mode="element_charge_spin",
    )
    with pytest.raises(ValueError, match="audit-only"):
        workflow.check_input_config()


def test_omol_h5_validation_requires_equal_nonempty_rank_splits() -> None:
    workflow = _omol_workflow_for_validation(num_train=1)
    with pytest.raises(ValueError, match="multiple of world_size"):
        workflow.check_input_config()


def test_omol_h5_validation_rejects_overlap_dependent_metrics() -> None:
    with pytest.raises(ValueError, match="no overlap matrix"):
        _omol_workflow_for_validation(
            compute_eigenvalues=True,
        ).check_input_config()


def test_omol_h5_paper_contract_config_is_valid() -> None:
    workflow = _omol_workflow_for_validation()
    workflow.check_input_config()
    assert workflow.config["compute_uncoupled_loss"] is True
    assert workflow.config["compute_eigenvalues"] is False
