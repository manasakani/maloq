#!/usr/bin/env python3
"""Convert QH9Stable density matrices to MALOQ's ASE/QM7 contract.

The source LMDB stores def2-SVP matrices in PySCF real-spherical ordering.
Final/initial density and the initial Hamiltonian are written in the
pre-``orca_to_e3nn`` convention consumed by MALOQ's QM7 loader. The initial
Hamiltonian is conditioning-only for QHFlow3; this database deliberately does
not advertise a final Hamiltonian target. Overlap is written directly in
MALOQ/e3nn ordering because the loader does not transform it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import lmdb
import numpy as np
from ase import Atoms
from ase.db import connect


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maloq.dataset_utils.ASEDataset import ASEAtomsData  # noqa: E402
from maloq.fock_utils import basis_sets, utils_orca_out  # noqa: E402


PROPERTY_UNITS = {
    "energy": "Hartree",
    "forces": "Hartree/Angstrom",
    "density_matrix": "dimensionless",
    "initial_density_matrix": "dimensionless",
    "initial_hamiltonian": "Hartree",
    "overlap": "dimensionless",
}
VALID_SUBSETS = ("train", "val", "test")
ORBITAL_BASIS = basis_sets.orbital_basis_def2_svp_QM7


@dataclass(frozen=True)
class Selection:
    subset: str
    source_index: int


@dataclass
class MatrixRecord:
    source_index: int
    molecule_id: str
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    energy: float
    forces: np.ndarray
    density_pyscf: np.ndarray
    initial_density_pyscf: np.ndarray
    initial_hamiltonian_pyscf: np.ndarray
    overlap_pyscf: np.ndarray


class MatrixLmdb:
    """Minimal read-only reader for dft-dataset's split-key LMDB format."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.env = lmdb.open(
            str(self.path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=32,
            meminit=False,
        )
        with self.env.begin() as txn:
            length = txn.get(b"__len__")
            storage_format = txn.get(b"__format__")
            schema = txn.get(b"__schema__")
        if length is None:
            self.close()
            raise ValueError(f"Source LMDB has no __len__ marker: {self.path}")
        if storage_format != b"split_key":
            self.close()
            raise ValueError(
                f"Expected split_key LMDB, got {storage_format!r}: {self.path}"
            )
        self.length = int.from_bytes(length, byteorder="big")
        self.schema = pickle.loads(schema) if schema is not None else None
        self._validate_schema()

    def _validate_schema(self) -> None:
        if not isinstance(self.schema, dict):
            raise ValueError("QH9 matrix LMDB must contain schema metadata")
        expected = {
            "basis": "def2-svp",
            "xc": "b3lyp5",
            "convention": "pyscf",
        }
        for key, value in expected.items():
            if self.schema.get(key) != value:
                raise ValueError(
                    f"Source schema {key}={self.schema.get(key)!r}; expected {value!r}"
                )
        targets = self.schema.get("targets", {})
        for target in (
            "density_matrix",
            "initial_density_matrix",
            "initial_hamiltonian",
            "overlap",
        ):
            if targets.get(target) is not True:
                raise ValueError(f"Source LMDB does not advertise {target!r}")

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _field_key(index: int, name: str) -> bytes:
        return index.to_bytes(4, "big") + b":" + name.encode("ascii")

    @staticmethod
    def _unpack_matrix(meta: dict, raw: bytes | None, name: str) -> np.ndarray:
        if raw is None:
            raise KeyError(f"Source row is missing {name}_packed")
        nao = int(meta[f"{name}_nao"])
        dtype = np.dtype(meta.get(f"{name}_dtype", "float64"))
        packed = np.frombuffer(raw, dtype=dtype)
        expected = nao * (nao + 1) // 2
        if packed.size != expected:
            raise ValueError(
                f"{name} has {packed.size} packed values; expected {expected}"
            )
        matrix = np.empty((nao, nao), dtype=dtype)
        upper = np.triu_indices(nao)
        matrix[upper] = packed
        matrix[upper[1], upper[0]] = packed
        return matrix

    def read(self, index: int) -> MatrixRecord:
        if not 0 <= index < self.length:
            raise IndexError(f"Source index {index} is outside [0, {self.length})")
        key = index.to_bytes(4, "big")
        with self.env.begin() as txn:
            meta_raw = txn.get(key)
            if meta_raw is None:
                raise KeyError(f"Source index {index} was not found")
            meta = pickle.loads(meta_raw)
            atoms_raw = txn.get(self._field_key(index, "atomic_numbers"))
            positions_raw = txn.get(self._field_key(index, "positions"))
            forces_raw = txn.get(self._field_key(index, "forces"))
            density_raw = txn.get(self._field_key(index, "density_matrix_packed"))
            initial_density_raw = txn.get(
                self._field_key(index, "initial_density_matrix_packed")
            )
            initial_hamiltonian_raw = txn.get(
                self._field_key(index, "initial_hamiltonian_packed")
            )
            overlap_raw = txn.get(self._field_key(index, "overlap_packed"))

        if atoms_raw is None or positions_raw is None:
            raise KeyError(f"Source index {index} is missing atoms or positions")
        num_atoms = int(meta["num_atoms"])
        atomic_numbers = np.frombuffer(atoms_raw, dtype=np.int32).copy()
        positions = np.frombuffer(positions_raw, dtype=np.float64).copy()
        if atomic_numbers.size != num_atoms or positions.size != num_atoms * 3:
            raise ValueError(
                f"Source index {index}: inconsistent N/Z/R sizes "
                f"({num_atoms}, {atomic_numbers.size}, {positions.size})"
            )
        positions = positions.reshape(num_atoms, 3)
        if forces_raw is None:
            forces = np.zeros((num_atoms, 3), dtype=np.float64)
        else:
            force_dtype = np.dtype(meta.get("forces_dtype", "float64"))
            forces = np.frombuffer(forces_raw, dtype=force_dtype).copy()
            forces = forces.reshape(tuple(meta.get("forces_shape", (num_atoms, 3))))

        density = self._unpack_matrix(meta, density_raw, "density_matrix")
        initial_density = self._unpack_matrix(
            meta, initial_density_raw, "initial_density_matrix"
        )
        initial_hamiltonian = self._unpack_matrix(
            meta, initial_hamiltonian_raw, "initial_hamiltonian"
        )
        overlap = self._unpack_matrix(meta, overlap_raw, "overlap")
        return MatrixRecord(
            source_index=index,
            molecule_id=str(meta.get("mol_id", f"qh9_{index:06d}")),
            atomic_numbers=atomic_numbers,
            positions_angstrom=positions,
            energy=float(meta.get("energy", 0.0) or 0.0),
            forces=forces,
            density_pyscf=density,
            initial_density_pyscf=initial_density,
            initial_hamiltonian_pyscf=initial_hamiltonian,
            overlap_pyscf=overlap,
        )

    def close(self) -> None:
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None


def expected_nao(atomic_numbers: np.ndarray) -> int:
    total = 0
    for atomic_number in atomic_numbers:
        z = int(atomic_number)
        if z not in ORBITAL_BASIS:
            raise ValueError(f"Atomic number {z} is absent from the QM7 def2-SVP map")
        total += sum(
            2 * int(angular_momentum) + 1
            for angular_momentum in ORBITAL_BASIS[z]
        )
    return total


def parse_subset_limits(items: Sequence[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for item in items:
        try:
            subset, value = item.split("=", 1)
            count = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --subset-limit {item!r}; expected SUBSET=COUNT"
            ) from exc
        if subset not in VALID_SUBSETS or count < 0:
            raise argparse.ArgumentTypeError(f"Invalid subset limit: {item!r}")
        limits[subset] = count
    return limits


def load_selection(
    split_file: Path,
    subsets: Sequence[str],
    subset_limits: dict[str, int],
    total_count: int,
    slice_start: int,
    slice_stop: int | None,
) -> list[Selection]:
    with split_file.open() as handle:
        payload = json.load(handle)
    selected: list[Selection] = []
    for subset in subsets:
        if subset not in payload:
            raise KeyError(f"Split file {split_file} has no {subset!r} list")
        indices = [int(index) for index in payload[subset]]
        limit = subset_limits.get(subset)
        if limit is not None:
            indices = indices[:limit]
        selected.extend(Selection(subset, index) for index in indices)

    owners: dict[int, str] = {}
    for item in selected:
        if item.source_index in owners:
            raise ValueError(
                f"Source index {item.source_index} appears in both "
                f"{owners[item.source_index]!r} and {item.subset!r}"
            )
        if not 0 <= item.source_index < total_count:
            raise IndexError(
                f"Source index {item.source_index} is outside [0, {total_count})"
            )
        owners[item.source_index] = item.subset

    stop = len(selected) if slice_stop is None else min(slice_stop, len(selected))
    if slice_start < 0 or stop < slice_start:
        raise ValueError(f"Invalid flattened selection slice [{slice_start}, {stop})")
    result = selected[slice_start:stop]
    if not result:
        raise ValueError("Selection is empty")
    return result


def selection_metadata(selection: Sequence[Selection]) -> tuple[dict[str, int], str]:
    counts: dict[str, int] = {}
    digest = hashlib.sha256()
    for item in selection:
        counts[item.subset] = counts.get(item.subset, 0) + 1
        digest.update(f"{item.subset}:{item.source_index}\n".encode())
    return counts, digest.hexdigest()


def selection_segments(selection: Sequence[Selection]) -> list[dict[str, int | str]]:
    segments: list[dict[str, int | str]] = []
    start = 0
    for index in range(1, len(selection) + 1):
        at_end = index == len(selection)
        changed = not at_end and selection[index].subset != selection[start].subset
        if at_end or changed:
            segments.append(
                {"subset": selection[start].subset, "start": start, "stop": index}
            )
            start = index
    return segments


def validate_source_record(record: MatrixRecord) -> None:
    nao = expected_nao(record.atomic_numbers)
    for name, matrix in (
        ("density_matrix", record.density_pyscf),
        ("initial_density_matrix", record.initial_density_pyscf),
        ("initial_hamiltonian", record.initial_hamiltonian_pyscf),
        ("overlap", record.overlap_pyscf),
    ):
        if matrix.shape != (nao, nao):
            raise ValueError(
                f"Source index {record.source_index}: {name}={matrix.shape}, "
                f"expected {(nao, nao)}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Source index {record.source_index}: {name} has non-finite values"
            )
        if not np.allclose(matrix, matrix.T, atol=1.0e-10, rtol=1.0e-10):
            raise ValueError(
                f"Source index {record.source_index}: {name} is not symmetric"
            )


def transform_record(
    record: MatrixRecord,
    matrix_dtype: np.dtype,
) -> tuple[Atoms, dict[str, np.ndarray], dict[str, object]]:
    validate_source_record(record)
    def matrix_storage(matrix: np.ndarray, name: str) -> np.ndarray:
        matrix_e3nn = utils_orca_out.sort_by_m(
            matrix,
            ORBITAL_BASIS,
            record.atomic_numbers,
            direction="pyscf_to_e3nn",
        )
        storage = utils_orca_out.sort_by_m(
            matrix_e3nn,
            ORBITAL_BASIS,
            record.atomic_numbers,
            direction="e3nn_to_orca",
        )
        replay = utils_orca_out.sort_by_m(
            storage,
            ORBITAL_BASIS,
            record.atomic_numbers,
            direction="orca_to_e3nn",
        )
        if not np.allclose(replay, matrix_e3nn, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError(
                f"Source index {record.source_index}: {name} loader round trip failed"
            )
        return storage

    density_storage = matrix_storage(record.density_pyscf, "density_matrix")
    initial_density_storage = matrix_storage(
        record.initial_density_pyscf, "initial_density_matrix"
    )
    initial_hamiltonian_storage = matrix_storage(
        record.initial_hamiltonian_pyscf, "initial_hamiltonian"
    )
    overlap_e3nn = utils_orca_out.sort_by_m(
        record.overlap_pyscf,
        ORBITAL_BASIS,
        record.atomic_numbers,
        direction="pyscf_to_e3nn",
    )

    atoms = Atoms(
        numbers=record.atomic_numbers,
        positions=record.positions_angstrom,
        pbc=False,
    )
    properties = {
        "energy": np.asarray([record.energy], dtype=np.float64),
        "forces": np.asarray(record.forces, dtype=np.float64),
        "density_matrix": np.asarray(density_storage, dtype=matrix_dtype),
        "initial_density_matrix": np.asarray(
            initial_density_storage, dtype=matrix_dtype
        ),
        "initial_hamiltonian": np.asarray(
            initial_hamiltonian_storage, dtype=matrix_dtype
        ),
        "overlap": np.asarray(overlap_e3nn, dtype=matrix_dtype),
    }
    key_values = {
        "source_index": record.source_index,
        "source_molecule_id": record.molecule_id,
    }
    return atoms, properties, key_values


def create_output_metadata(
    args: argparse.Namespace,
    selection: Sequence[Selection],
    source_schema: dict,
    matrix_dtype: np.dtype,
) -> dict[str, object]:
    counts, digest = selection_metadata(selection)
    return {
        "_property_unit_dict": PROPERTY_UNITS,
        "_distance_unit": "Angstrom",
        "atomrefs": {},
        "dataset_name": "QH9StableMatrices",
        "source_database": str(args.input_lmdb.resolve()),
        "source_split_file": str(args.split_file.resolve()),
        "source_schema": source_schema,
        "source_selection_sha256": digest,
        "selected_subset_counts": counts,
        "selected_subset_segments": selection_segments(selection),
        "selected_count": len(selection),
        "target_properties": ["density_matrix"],
        "loss_targets_supported": ["density_matrix"],
        "delta_learning_supported": True,
        "delta_baseline_properties": {
            "density_matrix": "initial_density_matrix",
        },
        "conditioning_properties": [
            "initial_density_matrix",
            "initial_hamiltonian",
            "overlap",
        ],
        "delta_learning_scope": "density_only",
        "matrix_dtype": np.dtype(matrix_dtype).name,
        "basis": "def2-svp",
        "xc": "b3lyp5",
        "raw_density_convention": "pyscf_real_spherical_def2svp",
        "density_storage_convention": "maloq_qm7_pre_orca_to_e3nn",
        "initial_density_storage_convention": "maloq_qm7_pre_orca_to_e3nn",
        "initial_hamiltonian_storage_convention": "maloq_qm7_pre_orca_to_e3nn",
        "matrix_loader_output_convention": "maloq_e3nn_def2svp",
        "matrix_to_label_contract": "runtime_maloq_fock_targets_batched",
        "overlap_storage_convention": "maloq_e3nn_def2svp",
        "position_unit": "Angstrom",
        "maloq_loader_dataset_name": "QM7",
        "complete": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_output(path: Path, source: MatrixLmdb, selection: Sequence[Selection]) -> None:
    dataset = ASEAtomsData(str(path))
    if len(dataset) != len(selection):
        raise ValueError(f"Output contains {len(dataset)} rows; expected {len(selection)}")
    required = set(PROPERTY_UNITS)
    if not required.issubset(dataset.available_properties):
        raise ValueError(
            f"Output properties {dataset.available_properties} do not contain "
            f"{sorted(required)}"
        )
    indices = sorted({0, len(selection) // 2, len(selection) - 1})
    for output_index in indices:
        item = selection[output_index]
        raw = source.read(item.source_index)
        row = dataset[output_index]
        atomic_numbers = row["_atomic_numbers"].cpu().numpy()
        density_storage = row["density_matrix"].cpu().numpy()
        initial_density_storage = row["initial_density_matrix"].cpu().numpy()
        initial_hamiltonian_storage = row["initial_hamiltonian"].cpu().numpy()
        overlap_e3nn = row["overlap"].cpu().numpy()
        def loader_matrix(storage: np.ndarray) -> np.ndarray:
            return utils_orca_out.sort_by_m(
                storage,
                ORBITAL_BASIS,
                atomic_numbers,
                direction="orca_to_e3nn",
            )

        def source_matrix(matrix: np.ndarray) -> np.ndarray:
            return utils_orca_out.sort_by_m(
                matrix,
                ORBITAL_BASIS,
                raw.atomic_numbers,
                direction="pyscf_to_e3nn",
            )

        density_e3nn = loader_matrix(density_storage)
        initial_density_e3nn = loader_matrix(initial_density_storage)
        initial_hamiltonian_e3nn = loader_matrix(initial_hamiltonian_storage)
        expected_density = source_matrix(raw.density_pyscf)
        expected_initial_density = source_matrix(raw.initial_density_pyscf)
        expected_initial_hamiltonian = source_matrix(raw.initial_hamiltonian_pyscf)
        expected_overlap = utils_orca_out.sort_by_m(
            raw.overlap_pyscf,
            ORBITAL_BASIS,
            raw.atomic_numbers,
            direction="pyscf_to_e3nn",
        )
        tolerance = 1.0e-6 if density_storage.dtype == np.float32 else 1.0e-12
        if not np.allclose(density_e3nn, expected_density, atol=tolerance, rtol=tolerance):
            raise ValueError(f"Output row {output_index}: density reconstruction failed")
        if not np.allclose(
            initial_density_e3nn,
            expected_initial_density,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError(
                f"Output row {output_index}: initial density reconstruction failed"
            )
        if not np.allclose(
            initial_hamiltonian_e3nn,
            expected_initial_hamiltonian,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError(
                f"Output row {output_index}: initial Hamiltonian reconstruction failed"
            )
        if not np.allclose(overlap_e3nn, expected_overlap, atol=tolerance, rtol=tolerance):
            raise ValueError(f"Output row {output_index}: overlap reconstruction failed")
        electron_count = int(atomic_numbers.sum())
        traced = float(np.trace(density_e3nn @ overlap_e3nn))
        if not np.isclose(traced, electron_count, atol=1.0e-5, rtol=1.0e-6):
            raise ValueError(
                f"Output row {output_index}: trace(D S)={traced}, "
                f"expected {electron_count} electrons"
            )


def partial_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def process(args: argparse.Namespace) -> None:
    input_path = args.input_lmdb.resolve()
    split_path = args.split_file.resolve()
    output_path = args.output_db.resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"QH9 matrix LMDB not found: {input_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"QH9 split file not found: {split_path}")
    if output_path.suffix != ".db":
        raise ValueError("--output-db must end in .db")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    partial_path = partial_output_path(output_path)
    if partial_path.exists():
        raise FileExistsError(
            f"Partial output already exists: {partial_path}. Inspect or remove it first."
        )

    source = MatrixLmdb(input_path)
    try:
        selection = load_selection(
            split_path,
            tuple(args.subsets),
            parse_subset_limits(args.subset_limit),
            len(source),
            args.slice_start,
            args.slice_stop,
        )
        matrix_dtype = np.dtype(args.matrix_dtype)
        metadata = create_output_metadata(
            args, selection, source.schema, matrix_dtype
        )
        counts, digest = selection_metadata(selection)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Source: {input_path}")
        print(f"Dataset: QH9StableMatrices; source rows: {len(source)}")
        print(f"Selection: {len(selection)} rows {counts}; sha256={digest}")
        print(f"Output: {output_path} (staging at {partial_path})")

        database = connect(str(partial_path), use_lock_file=False)
        database.metadata = metadata
        started = time.perf_counter()
        with database:
            for output_index, item in enumerate(selection):
                record = source.read(item.source_index)
                atoms, properties, key_values = transform_record(record, matrix_dtype)
                key_values.update(qh9_subset=item.subset, qh9_variant="stable-matrices")
                database.write(atoms, key_value_pairs=key_values, data=properties)
                completed = output_index + 1
                if (
                    completed == 1
                    or completed % args.progress_every == 0
                    or completed == len(selection)
                ):
                    elapsed = time.perf_counter() - started
                    print(
                        f"Processed {completed}/{len(selection)} rows "
                        f"({completed / max(elapsed, 1.0e-9):.2f} rows/s)",
                        flush=True,
                    )

        validate_output(partial_path, source, selection)
        finalized = connect(str(partial_path), use_lock_file=False)
        final_metadata = dict(finalized.metadata)
        final_metadata["complete"] = True
        final_metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        finalized.metadata = final_metadata
        os.replace(partial_path, output_path)
        print(f"Validated and finalized {output_path}")
    finally:
        source.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lmdb", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument(
        "--subsets", nargs="+", choices=VALID_SUBSETS, default=list(VALID_SUBSETS)
    )
    parser.add_argument(
        "--subset-limit",
        action="append",
        default=[],
        metavar="SUBSET=COUNT",
    )
    parser.add_argument("--slice-start", type=int, default=0)
    parser.add_argument("--slice-stop", type=int, default=None)
    parser.add_argument("--matrix-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    process(args)


if __name__ == "__main__":
    main()
