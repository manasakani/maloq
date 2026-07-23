from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from torch_geometric.data import Data

from maloq.dataset_utils.ASEDataset import ASEDataset
from maloq.dataset_utils.get_loader import _omol_matrix_target
from maloq.fock_utils import basis_sets, utils_orca_out, utils_tensor_decomp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_PATH = (
    PROJECT_ROOT
    / "_auto_script"
    / "omol_open_shell_process"
    / "process_omol_open_shell_to_ase.py"
)
SPEC = importlib.util.spec_from_file_location("omol_open_shell_processor", PROCESSOR_PATH)
assert SPEC is not None and SPEC.loader is not None
PROCESSOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESSOR)


def _printed_matrix_lines(matrix: np.ndarray, width: int = 2) -> list[str]:
    lines = []
    for start in range(0, matrix.shape[1], width):
        columns = list(range(start, min(start + width, matrix.shape[1])))
        lines.append(" ".join(str(column) for column in columns))
        for row in range(matrix.shape[0]):
            values = " ".join(f"{matrix[row, column]:.8f}" for column in columns)
            lines.append(f"{row} {values}")
        lines.append("")
    return lines


def test_streaming_orca_parser_reads_two_spin_matrices():
    alpha = np.arange(16, dtype=np.float64).reshape(4, 4)
    alpha = 0.5 * (alpha + alpha.T)
    beta = alpha + np.eye(4)
    lines = [
        "TOTAL SCF ENERGY",
        "Total Energy       :      -12.345678 Eh",
        "FOCK",
        "----",
        *_printed_matrix_lines(alpha),
        *_printed_matrix_lines(beta),
    ]
    payload = ("\n".join(lines) + "\n").encode()
    parsed, energy = PROCESSOR.parse_orca_output(
        io.BytesIO(payload), 4, np.dtype("float32")
    )
    np.testing.assert_allclose(parsed[0], alpha)
    np.testing.assert_allclose(parsed[1], beta)
    assert parsed.dtype == np.float32
    assert energy == -12.345678


def test_density_reconstruction_preserves_alpha_beta(tmp_path):
    total = np.asarray([4.0, 1.0, 6.0], dtype=np.float64)
    spin = np.asarray([2.0, 1.0, -2.0], dtype=np.float64)
    source = tmp_path / "density_mat.npz"
    np.savez(source, **{"orca.scfp": total, "orca.scfr": spin})

    density = PROCESSOR.load_open_shell_density(
        source, 2, np.dtype("float32")
    )
    expected_alpha = np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    expected_beta = np.asarray([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(density[0], expected_alpha)
    np.testing.assert_allclose(density[1], expected_beta)


def test_ase_dataset_loads_requested_open_shell_matrix(tmp_path):
    db_path = tmp_path / "open_shell.db"
    fock = np.stack((np.eye(2), 2.0 * np.eye(2)))
    density = np.stack((3.0 * np.eye(2), 4.0 * np.eye(2)))
    with connect(db_path) as database:
        database.write(
            Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]]),
            data={
                "fock_matrix": fock,
                "density_matrix": density,
                "total_energy [Eh]": -1.0,
                "is_open_shell": True,
                "charge": 0,
                "spin_multiplicity": 2,
                "folder_name": "sample",
                "matrix_storage_convention": "orca_real_spherical",
            },
        )

    dataset = ASEDataset(
        db_path,
        dtype=torch.float32,
        open_shell=True,
        matrix_target="density_matrix",
    )
    row = dataset[0]
    assert "density_matrix" in row
    assert "fock_matrix" not in row
    torch.testing.assert_close(row.density_matrix, torch.from_numpy(density).float())
    assert row.matrix_storage_convention == "orca_real_spherical"


def test_omol_target_converts_each_spin_from_orca_order():
    orbital_basis = {1: [0, 1]}
    alpha = np.arange(16, dtype=np.float64).reshape(4, 4)
    beta = alpha + 100.0
    matrix = np.stack((alpha, beta))
    row = Data(
        atomic_numbers=np.asarray([1]),
        density_matrix=torch.from_numpy(matrix),
        matrix_storage_convention="orca_real_spherical",
    )
    converted = _omol_matrix_target(row, "density_matrix", orbital_basis)
    expected = np.stack(
        [
            utils_orca_out.sort_by_m(spin, orbital_basis, np.asarray([1]))
            for spin in matrix
        ]
    )
    np.testing.assert_array_equal(converted, expected)


def test_3d_transition_metal_diffuse_f_shell_is_supported():
    orbital_basis = {
        22: list(basis_sets.def2_tzvpd["Ti"]),
        1: list(basis_sets.def2_tzvpd["H"]),
    }
    (
        targets,
        _,
        _,
        ls_list,
        out_js_list,
        _,
        interactions,
    ) = utils_tensor_decomp.make_output_irreps(orbital_basis)
    assert 13 in orbital_basis[22]
    blocks = utils_tensor_decomp.process_targets(
        orbital_basis, targets, ls_list, out_js_list, interactions
    )
    assert blocks
