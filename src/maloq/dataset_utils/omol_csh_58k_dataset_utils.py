# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Lazy support for the published OMol_CSH Hamiltonian HDF5 files.

The public files contain packed Fock matrices in the convention used during
their release.  That convention used an l-sorted def2-TZVPD shell list.  HELM
uses the actual, non-l-major def2-TZVPD shell order, so every matrix must first
be returned to ORCA order and then converted to the HELM/e3nn convention.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..fock_utils import basis_sets, fock_targets_batched, utils_orca_out


MetadataPolicy = Literal["preserve", "paper_contract"]


def def2_tzvpd_basis_by_atomic_number() -> dict[int, list[int]]:
    """Return a fresh def2-TZVPD shell dictionary keyed by atomic number."""
    return {
        int(utils_orca_out.periodic_table[element]): list(shells)
        for element, shells in basis_sets.def2_tzvpd.items()
    }


def reinflate_symmetric_matrix(flat_array: np.ndarray) -> np.ndarray:
    """Reconstruct a dense symmetric matrix from its packed upper triangle."""
    flat = np.asarray(flat_array)
    if flat.ndim != 1:
        raise ValueError(
            "Packed symmetric matrix must be one-dimensional, "
            f"got shape {flat.shape}."
        )

    length = int(flat.size)
    matrix_size = (math.isqrt(8 * length + 1) - 1) // 2
    if matrix_size * (matrix_size + 1) // 2 != length:
        raise ValueError(
            f"Packed matrix length {length} is not a triangular number."
        )

    matrix = np.zeros((matrix_size, matrix_size), dtype=flat.dtype)
    upper = np.triu_indices(matrix_size)
    matrix[upper] = flat
    matrix[(upper[1], upper[0])] = flat
    return matrix


def restore_orca_diffuse_order(
    matrix: np.ndarray | torch.Tensor,
    atomic_numbers: Sequence[int] | np.ndarray | torch.Tensor,
    correct_basis_dict: dict[int, list[int]] | None = None,
) -> np.ndarray | torch.Tensor:
    """Convert the released OMol_CSH matrix to HELM's e3nn AO convention.

    The released matrix is treated as if ``orca_to_e3nn`` had been applied
    with an l-sorted shell list.  We undo that transform and reapply it with
    HELM's actual def2-TZVPD shell ordering.
    """
    basis = (
        def2_tzvpd_basis_by_atomic_number()
        if correct_basis_dict is None
        else {int(key): list(value) for key, value in correct_basis_dict.items()}
    )
    if isinstance(atomic_numbers, torch.Tensor):
        numbers = atomic_numbers.detach().cpu().numpy()
    else:
        numbers = np.asarray(atomic_numbers)
    numbers = numbers.astype(np.int64, copy=False).reshape(-1)
    if numbers.size == 0:
        raise ValueError("OMol_CSH sample must contain at least one atom.")

    missing = sorted({int(number) for number in numbers} - set(basis))
    if missing:
        raise KeyError(
            "def2-TZVPD shell definitions are missing for atomic numbers "
            f"{missing}."
        )

    dynamic_basis = {
        int(number): sorted(basis[int(number)])
        for number in np.unique(numbers)
    }

    is_torch = isinstance(matrix, torch.Tensor)
    original_device = matrix.device if is_torch else None
    original_dtype = matrix.dtype if is_torch else None
    matrix_numpy = (
        matrix.detach().cpu().numpy() if is_torch else np.asarray(matrix)
    )
    if matrix_numpy.ndim == 1:
        matrix_numpy = reinflate_symmetric_matrix(matrix_numpy)
    elif (
        matrix_numpy.ndim != 2
        or matrix_numpy.shape[0] != matrix_numpy.shape[1]
    ):
        raise ValueError(
            "OMol_CSH Fock matrix must be square or packed upper-triangular, "
            f"got shape {matrix_numpy.shape}."
        )

    raw_orca = utils_orca_out.sort_by_m(
        matrix_numpy,
        dynamic_basis,
        numbers,
        direction="e3nn_to_orca",
    )
    corrected = utils_orca_out.sort_by_m(
        raw_orca,
        basis,
        numbers,
        direction="orca_to_e3nn",
    )
    if is_torch:
        return torch.from_numpy(corrected).to(
            device=original_device,
            dtype=original_dtype,
        )
    return corrected


def key_manifest_path(dbpath: str | Path) -> Path:
    return Path(f"{Path(dbpath)}.keys.json")


def load_key_manifest(dbpath: str | Path) -> list[str]:
    """Load the precomputed nested molecule keys for one public HDF5 file."""
    path = key_manifest_path(dbpath)
    if not path.is_file():
        raise FileNotFoundError(
            f"OMol_CSH key manifest is missing: {path}. "
            "Run _auto_script/omol_csh_download/download_omol_csh.sh verify "
            "before training."
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not all(
        isinstance(key, str) and key for key in payload
    ):
        raise ValueError(f"Invalid OMol_CSH key manifest: {path}")
    if len(payload) != len(set(payload)):
        raise ValueError(f"Duplicate molecule keys in OMol_CSH manifest: {path}")
    return payload


def _scalar_attribute(group: h5py.Group, name: str, default: float) -> float:
    value = group.attrs.get(name, default)
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"OMol_CSH attribute {name!r} must be scalar, got shape "
            f"{array.shape} in {group.name}."
        )
    return float(array.reshape(()))


def _integral_metadata(value: float, name: str, key: str) -> int:
    if not np.isfinite(value) or not float(value).is_integer():
        raise ValueError(
            f"OMol_CSH {name} must be an integer for {key}, got {value!r}."
        )
    return int(value)


class OMolCSH58kDatabase(Dataset):
    """Lazy, per-process reader for one published OMol_CSH HDF5 file."""

    def __init__(
        self,
        dbpath: str | Path,
        indices: Sequence[int] | None = None,
        metadata_policy: MetadataPolicy = "paper_contract",
    ) -> None:
        super().__init__()
        self.dbpath = str(Path(dbpath).resolve())
        if not Path(self.dbpath).is_file():
            raise FileNotFoundError(self.dbpath)
        if metadata_policy not in {"preserve", "paper_contract"}:
            raise ValueError(
                "metadata_policy must be 'preserve' or 'paper_contract', "
                f"got {metadata_policy!r}."
            )
        self.metadata_policy = metadata_policy

        all_keys = load_key_manifest(self.dbpath)
        self.global_size = len(all_keys)
        if indices is None:
            self.molecule_keys = all_keys
        else:
            selected = [int(index) for index in indices]
            invalid = [
                index
                for index in selected
                if index < 0 or index >= self.global_size
            ]
            if invalid:
                raise IndexError(
                    "OMol_CSH indices are outside "
                    f"[0, {self.global_size}): {invalid[:8]}"
                )
            self.molecule_keys = [all_keys[index] for index in selected]

        self._file: h5py.File | None = None

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(
                self.dbpath,
                "r",
                rdcc_nbytes=64 * 1024 * 1024,
                swmr=True,
            )
        return self._file

    def __len__(self) -> int:
        return len(self.molecule_keys)

    def __getitem__(self, index: int) -> dict[str, object]:
        key = self.molecule_keys[index]
        group = self._get_file()[key]
        required = {"coords", "elements", "fock"}
        missing = sorted(required - set(group))
        if missing:
            raise KeyError(
                f"OMol_CSH group {key!r} is missing datasets {missing}."
            )

        atomic_numbers = np.asarray(group["elements"][:], dtype=np.int64)
        positions = np.asarray(group["coords"][:])
        if positions.shape != (len(atomic_numbers), 3):
            raise ValueError(
                f"OMol_CSH coordinates for {key!r} have shape "
                f"{positions.shape}, expected ({len(atomic_numbers)}, 3)."
            )
        fock_matrix = restore_orca_diffuse_order(
            np.asarray(group["fock"][:]),
            atomic_numbers,
        )

        source_charge = _scalar_attribute(group, "charge", 0.0)
        source_spin = _scalar_attribute(group, "spin", 1.0)
        if self.metadata_policy == "paper_contract":
            charge = 0
            spin_multiplicity = 1
        else:
            charge = _integral_metadata(source_charge, "charge", key)
            spin_multiplicity = _integral_metadata(source_spin, "spin", key)

        return {
            "atomic_numbers": atomic_numbers,
            "pos": positions,
            "fock_matrix": fock_matrix,
            "charge": charge,
            "spin_multiplicity": spin_multiplicity,
            "source_charge": source_charge,
            "source_spin": source_spin,
            "name": key,
            "matrix_storage_convention": "maloq_e3nn",
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __del__(self) -> None:
        self.close()


class OMolCSHGraphDataset(Dataset):
    """Convert public HDF5 rows into HELM graph labels one sample at a time."""

    def __init__(
        self,
        database: OMolCSH58kDatabase,
        start_idx: int,
        end_idx: int,
        cutoff: float,
        dtype: torch.dtype,
        scale_shift_data: dict | None = None,
    ) -> None:
        super().__init__()
        if start_idx < 0 or end_idx > len(database) or start_idx >= end_idx:
            raise ValueError(
                "OMol_CSH graph slice must be non-empty and inside the "
                f"database: [{start_idx}, {end_idx}) of {len(database)}."
            )
        self.database = database
        self.sample_indices = tuple(range(start_idx, end_idx))
        self.cutoff = float(cutoff)
        self.dtype = dtype
        self.scale_shift_data = scale_shift_data
        self.orbital_basis = def2_tzvpd_basis_by_atomic_number()

        prototype_sample = self.database[self.sample_indices[0]]
        prototype = self._make_targets(prototype_sample)
        self.required_irreps = prototype.req_output_irreps
        self.basis_transformation = prototype.basis_transformation
        self._orbital_template_device_cache = (
            prototype.orbital_template_device_cache
        )
        self.ls_list = prototype.ls_list
        self._orbital_starts = prototype.orbital_starts
        self._orbital_template = prototype.orbital_template
        self._out_js_list = prototype.out_js_list
        self._target_orbital_basis = copy.deepcopy(prototype.orbital_basis)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _make_targets(
        self,
        sample: dict[str, object],
        *,
        reuse_orbital_metadata: bool = False,
    ) -> fock_targets_batched.Fock_Targets:
        kwargs = {}
        if reuse_orbital_metadata:
            kwargs = {
                "orbital_starts": self._orbital_starts,
                "orbital_template": self._orbital_template,
                "req_output_irreps": self.required_irreps,
                "out_js_list": self._out_js_list,
                "ls_list": self.ls_list,
                "basis_transformation": self.basis_transformation,
                "orbital_template_device_cache": (
                    self._orbital_template_device_cache
                ),
                "verbose": False,
            }
        return fock_targets_batched.Fock_Targets(
            [np.asarray(sample["atomic_numbers"])],
            [np.asarray(sample["pos"])],
            self.cutoff,
            copy.deepcopy(
                self._target_orbital_basis
                if reuse_orbital_metadata
                else self.orbital_basis
            ),
            [np.asarray(sample["fock_matrix"])],
            dataset_name="omol",
            dtype=self.dtype,
            scale_shift_data=self.scale_shift_data,
            distribute_graphs=False,
            **kwargs,
        )

    def __getitem__(self, index: int) -> Data:
        sample = self.database[self.sample_indices[index]]
        targets = self._make_targets(sample, reuse_orbital_metadata=True)
        atomic_numbers = np.asarray(targets.atomic_numbers_list[0])
        num_atoms = int(len(atomic_numbers))

        return Data(
            pos=torch.as_tensor(
                targets.atomic_positions_list[0],
                dtype=self.dtype,
            ),
            edge_index=torch.as_tensor(
                targets.neighbour_list_list[0],
                dtype=torch.long,
            ),
            edge_attr=torch.as_tensor(
                targets.edge_dist_list[0],
                dtype=self.dtype,
            ),
            y=torch.as_tensor(targets.edge_labels_list[0][0], dtype=self.dtype),
            node_y=torch.as_tensor(
                targets.node_labels_list[0][0],
                dtype=self.dtype,
            ),
            edge_padding_mask=torch.as_tensor(
                targets.edge_unpadding_mask_list[0][0],
                dtype=torch.bool,
            ),
            node_padding_mask=torch.as_tensor(
                targets.node_unpadding_mask_list[0][0],
                dtype=torch.bool,
            ),
            atomic_numbers=torch.as_tensor(
                atomic_numbers,
                dtype=torch.long,
            ),
            energies=torch.tensor(0.0, dtype=self.dtype),
            forces=torch.zeros((num_atoms, 3), dtype=self.dtype),
            num_atoms_in_molecule=num_atoms,
            fock_target_object=targets,
            fock_target_id=0,
            charge=torch.tensor(int(sample["charge"]), dtype=torch.long),
            spin_multiplicity=torch.tensor(
                int(sample["spin_multiplicity"]),
                dtype=torch.long,
            ),
            source_charge=torch.tensor(
                float(sample["source_charge"]),
                dtype=torch.float64,
            ),
            source_spin=torch.tensor(
                float(sample["source_spin"]),
                dtype=torch.float64,
            ),
            molecule_name=str(sample["name"]),
            distributed_graph_training=False,
        )


# Compatibility with the class name introduced by upstream's initial H5 path.
OMol_CSH_58k_Database = OMolCSH58kDatabase
