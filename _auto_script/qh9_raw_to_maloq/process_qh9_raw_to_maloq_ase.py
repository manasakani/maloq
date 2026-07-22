#!/usr/bin/env python3
"""Convert raw QH9Stable SQLite data to MALOQ's native QM7 ASE format.

The resulting database can be opened by ``ASEAtomsData`` and consumed through
the unchanged ``dataset_name='QM7'`` branch. QH9 Hamiltonians begin in PySCF
real-spherical def2-SVP ordering. They are stored in the pre-conversion layout
expected by MALOQ's QM7 loader, which subsequently applies ``orca_to_e3nn``.
Overlap matrices are recomputed with PySCF and stored directly in e3nn order,
because the original loader does not transform overlap matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import os
import sqlite3
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
from pyscf import gto


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maloq.dataset_utils.ASEDataset import ASEAtomsData  # noqa: E402
from maloq.fock_utils import basis_sets, utils_orca_out  # noqa: E402


ABSOLUTE_PROPERTY_UNITS = {
    "energy": "Hartree",
    "forces": "Hartree/Angstrom",
    "hamiltonian": "Hartree",
    "overlap": "dimensionless",
}
DELTA_PROPERTY_UNITS = {
    **ABSOLUTE_PROPERTY_UNITS,
    "initial_hamiltonian": "Hartree",
    # QHFlow3 uses the opposite initial matrix as an auxiliary equivariant
    # conditioning input. It is not a density target in this database.
    "initial_density_matrix": "dimensionless",
}
VALID_SUBSETS = ("train", "val", "test")
ORBITAL_BASIS = basis_sets.orbital_basis_def2_svp_QM7


@dataclass(frozen=True)
class Selection:
    subset: str
    source_index: int


@dataclass
class RawRecord:
    source_index: int
    molecule_id: int
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    energy: float
    hamiltonian_pyscf: np.ndarray


@dataclass
class InitialMatrixRecord:
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    initial_hamiltonian_pyscf: np.ndarray
    initial_density_pyscf: np.ndarray


class InitialMatrixLmdb:
    """Read only the QH9 initial matrices needed for Hamiltonian delta learning."""

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
        if length is None or storage_format != b"split_key" or schema is None:
            self.close()
            raise ValueError(f"Unsupported QH9 matrix LMDB: {self.path}")
        self.length = int.from_bytes(length, byteorder="big")
        self.schema = pickle.loads(schema)
        expected = {
            "basis": "def2-svp",
            "xc": "b3lyp5",
            "convention": "pyscf",
        }
        for key, value in expected.items():
            if self.schema.get(key) != value:
                self.close()
                raise ValueError(
                    f"Initial-matrix schema {key}={self.schema.get(key)!r}; "
                    f"expected {value!r}"
                )
        targets = self.schema.get("targets", {})
        for target in ("initial_hamiltonian", "initial_density_matrix"):
            if targets.get(target) is not True:
                self.close()
                raise ValueError(f"Initial-matrix LMDB does not advertise {target!r}")

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

    def read(self, index: int) -> InitialMatrixRecord:
        if not 0 <= index < self.length:
            raise IndexError(f"Initial-matrix index {index} is out of range")
        row_key = index.to_bytes(4, "big")
        with self.env.begin() as txn:
            meta_raw = txn.get(row_key)
            if meta_raw is None:
                raise KeyError(f"Initial-matrix index {index} was not found")
            meta = pickle.loads(meta_raw)
            atoms_raw = txn.get(self._field_key(index, "atomic_numbers"))
            positions_raw = txn.get(self._field_key(index, "positions"))
            initial_hamiltonian_raw = txn.get(
                self._field_key(index, "initial_hamiltonian_packed")
            )
            initial_density_raw = txn.get(
                self._field_key(index, "initial_density_matrix_packed")
            )
        if atoms_raw is None or positions_raw is None:
            raise KeyError(f"Initial-matrix index {index} has no geometry")
        num_atoms = int(meta["num_atoms"])
        atomic_numbers = np.frombuffer(atoms_raw, dtype=np.int32).copy()
        positions = np.frombuffer(positions_raw, dtype=np.float64).copy()
        if atomic_numbers.size != num_atoms or positions.size != num_atoms * 3:
            raise ValueError(f"Initial-matrix index {index} has invalid geometry sizes")
        return InitialMatrixRecord(
            atomic_numbers=atomic_numbers,
            positions_angstrom=positions.reshape(num_atoms, 3),
            initial_hamiltonian_pyscf=self._unpack_matrix(
                meta, initial_hamiltonian_raw, "initial_hamiltonian"
            ),
            initial_density_pyscf=self._unpack_matrix(
                meta, initial_density_raw, "initial_density_matrix"
            ),
        )

    def close(self) -> None:
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None


def parse_subset_limits(items: Sequence[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for item in items:
        try:
            subset, value = item.split("=", 1)
            value_int = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --subset-limit {item!r}; expected SUBSET=COUNT"
            ) from exc
        if subset not in VALID_SUBSETS:
            raise argparse.ArgumentTypeError(
                f"Invalid subset {subset!r}; expected one of {VALID_SUBSETS}"
            )
        if value_int < 0:
            raise argparse.ArgumentTypeError("Subset limits must be non-negative")
        limits[subset] = value_int
    return limits


def decode_int(value) -> int:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) not in {1, 2, 4, 8}:
            raise ValueError(f"Cannot decode integer from {len(raw)} bytes")
        return int.from_bytes(raw, byteorder="little", signed=True)
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar integer, got array {value.shape}")
        return int(value.reshape(-1)[0])
    return int(value)


def open_raw_database(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(data)").fetchall()
    }
    required = {"id", "N", "Z", "pos", "Ham"}
    missing = required - columns
    if missing:
        connection.close()
        raise ValueError(f"Input is not QH9Stable raw data; missing columns: {sorted(missing)}")
    if "geo_id" in columns:
        connection.close()
        raise ValueError("QH9Dynamic input is out of scope; provide QH9Stable.db")
    return connection


def raw_count(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT MAX(id) FROM data").fetchone()
    return int(row[0]) + 1


def expected_nao(atomic_numbers: np.ndarray) -> int:
    total = 0
    for atomic_number in atomic_numbers:
        z = int(atomic_number)
        if z not in ORBITAL_BASIS:
            raise ValueError(f"Atomic number {z} is absent from the QM7 def2-SVP basis map")
        total += sum(
            2 * int(angular_momentum) + 1
            for angular_momentum in ORBITAL_BASIS[z]
        )
    return total


def read_raw_record(
    connection: sqlite3.Connection,
    source_index: int,
) -> RawRecord:
    row = connection.execute(
        "SELECT id, N, Z, pos, Ham FROM data WHERE id = ?",
        (int(source_index),),
    ).fetchone()
    if row is None:
        raise KeyError(f"QH9Stable source index {source_index} was not found")
    molecule_id, num_atoms, atoms_blob, pos_blob, ham_blob = row

    atomic_numbers = np.frombuffer(atoms_blob, dtype=np.int32).copy()
    num_atoms = int(num_atoms)
    if atomic_numbers.size != num_atoms:
        raise ValueError(
            f"Source index {source_index}: N={num_atoms}, Z has {atomic_numbers.size} entries"
        )
    positions = (
        np.frombuffer(pos_blob, dtype=np.float64)
        .copy()
        .reshape(num_atoms, 3)
    )
    nao = expected_nao(atomic_numbers)
    hamiltonian = np.frombuffer(ham_blob, dtype=np.float64).copy()
    if hamiltonian.size != nao * nao:
        raise ValueError(
            f"Source index {source_index}: Hamiltonian has {hamiltonian.size} entries, "
            f"expected {nao * nao} for {nao} AOs"
        )
    hamiltonian = hamiltonian.reshape(nao, nao)
    if not np.allclose(hamiltonian, hamiltonian.T, atol=1.0e-10, rtol=1.0e-10):
        error = float(np.max(np.abs(hamiltonian - hamiltonian.T)))
        raise ValueError(
            f"Source index {source_index}: Hamiltonian is not symmetric; max error={error}"
        )

    return RawRecord(
        source_index=int(source_index),
        molecule_id=decode_int(molecule_id),
        atomic_numbers=atomic_numbers,
        positions_angstrom=positions,
        energy=0.0,
        hamiltonian_pyscf=hamiltonian,
    )


def transform_record(
    record: RawRecord,
    *,
    matrix_dtype: np.dtype,
    initial_record: InitialMatrixRecord | None = None,
) -> tuple[Atoms, dict[str, np.ndarray], dict[str, object]]:
    atom_spec = [
        (int(z), tuple(float(x) for x in position))
        for z, position in zip(record.atomic_numbers, record.positions_angstrom)
    ]
    molecule = gto.M(
        atom=atom_spec,
        basis="def2-svp",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    nao = expected_nao(record.atomic_numbers)
    if molecule.nao_nr() != nao:
        raise ValueError(
            f"Source index {record.source_index}: PySCF reports {molecule.nao_nr()} AOs, "
            f"MALOQ basis map reports {nao}"
        )

    hamiltonian_e3nn = utils_orca_out.sort_by_m(
        record.hamiltonian_pyscf,
        ORBITAL_BASIS,
        record.atomic_numbers,
        direction="pyscf_to_e3nn",
    )
    hamiltonian_storage = utils_orca_out.sort_by_m(
        hamiltonian_e3nn,
        ORBITAL_BASIS,
        record.atomic_numbers,
        direction="e3nn_to_orca",
    )
    loader_replay = utils_orca_out.sort_by_m(
        hamiltonian_storage,
        ORBITAL_BASIS,
        record.atomic_numbers,
        direction="orca_to_e3nn",
    )
    if not np.allclose(loader_replay, hamiltonian_e3nn, atol=1.0e-12, rtol=1.0e-12):
        error = float(np.max(np.abs(loader_replay - hamiltonian_e3nn)))
        raise ValueError(
            f"Source index {record.source_index}: QM7 loader replay error={error}"
        )

    overlap_pyscf = molecule.intor("int1e_ovlp")
    overlap_e3nn = utils_orca_out.sort_by_m(
        overlap_pyscf,
        ORBITAL_BASIS,
        record.atomic_numbers,
        direction="pyscf_to_e3nn",
    )
    if not np.allclose(overlap_e3nn, overlap_e3nn.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(f"Source index {record.source_index}: overlap is not symmetric")

    atoms = Atoms(
        numbers=record.atomic_numbers,
        positions=record.positions_angstrom,
        pbc=False,
    )
    properties = {
        "energy": np.asarray([record.energy], dtype=np.float64),
        "forces": np.zeros((len(record.atomic_numbers), 3), dtype=np.float64),
        "hamiltonian": np.asarray(hamiltonian_storage, dtype=matrix_dtype),
        "overlap": np.asarray(overlap_e3nn, dtype=matrix_dtype),
    }
    if initial_record is not None:
        if not np.array_equal(record.atomic_numbers, initial_record.atomic_numbers):
            raise ValueError(
                f"Source index {record.source_index}: raw/initial atomic numbers differ"
            )
        if not np.allclose(
            record.positions_angstrom,
            initial_record.positions_angstrom,
            atol=1.0e-12,
            rtol=1.0e-12,
        ):
            raise ValueError(
                f"Source index {record.source_index}: raw/initial positions differ"
            )

        def initial_storage(matrix: np.ndarray, name: str) -> np.ndarray:
            if matrix.shape != (nao, nao) or not np.isfinite(matrix).all():
                raise ValueError(
                    f"Source index {record.source_index}: invalid {name} shape/values"
                )
            if not np.allclose(matrix, matrix.T, atol=1.0e-10, rtol=1.0e-10):
                raise ValueError(
                    f"Source index {record.source_index}: {name} is not symmetric"
                )
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
                    f"Source index {record.source_index}: {name} loader replay failed"
                )
            return storage

        properties.update(
            initial_hamiltonian=np.asarray(
                initial_storage(
                    initial_record.initial_hamiltonian_pyscf,
                    "initial_hamiltonian",
                ),
                dtype=matrix_dtype,
            ),
            initial_density_matrix=np.asarray(
                initial_storage(
                    initial_record.initial_density_pyscf,
                    "initial_density_matrix",
                ),
                dtype=matrix_dtype,
            ),
        )
    key_values: dict[str, object] = {
        "source_index": int(record.source_index),
        "source_molecule_id": int(record.molecule_id),
    }
    return atoms, properties, key_values


def load_selection(
    split_file: Path | None,
    subsets: Sequence[str],
    subset_limits: dict[str, int],
    total_count: int,
    slice_start: int,
    slice_stop: int | None,
) -> list[Selection]:
    if split_file is None:
        if subsets != ("train", "val", "test") and list(subsets) != list(VALID_SUBSETS):
            raise ValueError("--subsets requires --split-file")
        selected = [Selection("all", index) for index in range(total_count)]
    else:
        with split_file.open() as handle:
            payload = json.load(handle)
        selected = []
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
            previous = owners[item.source_index]
            raise ValueError(
                f"Source index {item.source_index} appears more than once "
                f"({previous!r}, {item.subset!r}); refusing split leakage"
            )
        owners[item.source_index] = item.subset

    stop = len(selected) if slice_stop is None else min(int(slice_stop), len(selected))
    start = int(slice_start)
    if start < 0 or stop < start:
        raise ValueError(f"Invalid flattened selection slice [{start}, {stop})")
    selected = selected[start:stop]
    if not selected:
        raise ValueError("Selection is empty")
    invalid = [item.source_index for item in selected if not 0 <= item.source_index < total_count]
    if invalid:
        raise IndexError(
            f"Selection contains source indices outside [0, {total_count}): {invalid[:5]}"
        )
    return selected


def selection_metadata(selection: Sequence[Selection]) -> tuple[dict[str, int], str]:
    counts: dict[str, int] = {}
    digest = hashlib.sha256()
    for item in selection:
        counts[item.subset] = counts.get(item.subset, 0) + 1
        digest.update(f"{item.subset}:{item.source_index}\n".encode())
    return counts, digest.hexdigest()


def selection_segments(selection: Sequence[Selection]) -> list[dict[str, int | str]]:
    segments: list[dict[str, int | str]] = []
    segment_start = 0
    for index in range(1, len(selection) + 1):
        at_end = index == len(selection)
        changed = not at_end and selection[index].subset != selection[segment_start].subset
        if at_end or changed:
            segments.append(
                {
                    "subset": selection[segment_start].subset,
                    "start": segment_start,
                    "stop": index,
                }
            )
            segment_start = index
    return segments


def create_output_metadata(
    *,
    args: argparse.Namespace,
    selection: Sequence[Selection],
    matrix_dtype: np.dtype,
    delta_learning: bool,
) -> dict[str, object]:
    counts, digest = selection_metadata(selection)
    property_units = DELTA_PROPERTY_UNITS if delta_learning else ABSOLUTE_PROPERTY_UNITS
    metadata = {
        "_property_unit_dict": property_units,
        "_distance_unit": "Angstrom",
        "atomrefs": {},
        "dataset_name": "QH9Stable",
        "source_database": str(args.input_db.resolve()),
        "source_split_file": None if args.split_file is None else str(args.split_file.resolve()),
        "source_selection_sha256": digest,
        "selected_subset_counts": counts,
        "selected_subset_segments": selection_segments(selection),
        "selected_count": len(selection),
        "matrix_dtype": np.dtype(matrix_dtype).name,
        "basis": "def2-svp",
        "raw_hamiltonian_convention": "pyscf_real_spherical_def2svp",
        "hamiltonian_storage_convention": "maloq_qm7_pre_orca_to_e3nn",
        "overlap_storage_convention": "maloq_e3nn_def2svp",
        "position_unit": "Angstrom",
        "energy_semantics": "zero placeholder; raw QH9Stable does not contain energies",
        "forces_semantics": "zero placeholder; raw QH9Stable does not contain forces",
        "maloq_loader_dataset_name": "QM7",
        "complete": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if delta_learning:
        metadata.update(
            source_initial_matrix_database=str(args.initial_matrix_lmdb.resolve()),
            source_initial_matrix_schema="split_key/pyscf/def2-svp/b3lyp5",
            target_properties=["hamiltonian"],
            loss_targets_supported=["fock_matrix"],
            delta_learning_supported=True,
            delta_baseline_properties={
                "fock_matrix": "initial_hamiltonian",
            },
            conditioning_properties=[
                "initial_hamiltonian",
                "initial_density_matrix",
                "overlap",
            ],
            initial_hamiltonian_storage_convention=(
                "maloq_qm7_pre_orca_to_e3nn"
            ),
            initial_density_storage_convention=(
                "maloq_qm7_pre_orca_to_e3nn"
            ),
            delta_learning_scope="hamiltonian_only",
            xc="b3lyp5",
        )
    return metadata


def validate_output(
    path: Path,
    selection: Sequence[Selection],
    initial_source: InitialMatrixLmdb | None,
) -> None:
    dataset = ASEAtomsData(str(path))
    expected_count = len(selection)
    if len(dataset) != expected_count:
        raise ValueError(f"Output contains {len(dataset)} rows; expected {expected_count}")
    required = set(
        DELTA_PROPERTY_UNITS
        if initial_source is not None
        else ABSOLUTE_PROPERTY_UNITS
    )
    if not required.issubset(dataset.available_properties):
        raise ValueError(
            f"Output properties {dataset.available_properties} do not contain {sorted(required)}"
        )
    indices = sorted({0, expected_count // 2, expected_count - 1})
    for index in indices:
        row = dataset[index]
        atomic_numbers = row["_atomic_numbers"].cpu().numpy()
        hamiltonian = row["hamiltonian"].cpu().numpy()
        overlap = row["overlap"].cpu().numpy()
        nao = expected_nao(atomic_numbers)
        if hamiltonian.shape != (nao, nao) or overlap.shape != (nao, nao):
            raise ValueError(
                f"Output row {index}: H={hamiltonian.shape}, S={overlap.shape}, expected {(nao, nao)}"
            )
        if not np.isfinite(hamiltonian).all() or not np.isfinite(overlap).all():
            raise ValueError(f"Output row {index} contains non-finite matrix entries")
        if not np.allclose(overlap, overlap.T, atol=1.0e-6, rtol=1.0e-6):
            raise ValueError(f"Output row {index} overlap is not symmetric")
        if initial_source is not None:
            initial = initial_source.read(selection[index].source_index)
            for property_name, source_matrix in (
                ("initial_hamiltonian", initial.initial_hamiltonian_pyscf),
                ("initial_density_matrix", initial.initial_density_pyscf),
            ):
                storage = row[property_name].cpu().numpy()
                replay = utils_orca_out.sort_by_m(
                    storage,
                    ORBITAL_BASIS,
                    atomic_numbers,
                    direction="orca_to_e3nn",
                )
                expected = utils_orca_out.sort_by_m(
                    source_matrix,
                    ORBITAL_BASIS,
                    atomic_numbers,
                    direction="pyscf_to_e3nn",
                )
                tolerance = 1.0e-6 if storage.dtype == np.float32 else 1.0e-12
                if not np.allclose(
                    replay, expected, atol=tolerance, rtol=tolerance
                ):
                    raise ValueError(
                        f"Output row {index}: {property_name} reconstruction failed"
                    )


def partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")


def process(args: argparse.Namespace) -> None:
    input_path = args.input_db.resolve()
    output_path = args.output_db.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw QH9 database not found: {input_path}")
    if output_path.suffix != ".db":
        raise ValueError("--output-db must end in .db")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    partial_path = partial_output_path(output_path)
    if partial_path.exists():
        raise FileExistsError(
            f"Partial output already exists: {partial_path}. Inspect or remove it before retrying."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_limits = parse_subset_limits(args.subset_limit)
    matrix_dtype = np.dtype(args.matrix_dtype)
    source = open_raw_database(input_path)
    total_count = raw_count(source)
    initial_source = None
    if args.initial_matrix_lmdb is not None:
        initial_path = args.initial_matrix_lmdb.resolve()
        if not initial_path.is_dir():
            source.close()
            raise FileNotFoundError(f"Initial-matrix LMDB not found: {initial_path}")
        initial_source = InitialMatrixLmdb(initial_path)
        if initial_source.length != total_count:
            source.close()
            initial_source.close()
            raise ValueError(
                f"Raw QH9 has {total_count} rows but initial LMDB has "
                f"{initial_source.length}"
            )
    selection = load_selection(
        args.split_file,
        tuple(args.subsets),
        subset_limits,
        total_count,
        args.slice_start,
        args.slice_stop,
    )
    metadata = create_output_metadata(
        args=args,
        selection=selection,
        matrix_dtype=matrix_dtype,
        delta_learning=initial_source is not None,
    )
    counts, digest = selection_metadata(selection)
    print(f"Source: {input_path}")
    print(f"Dataset: QH9Stable; source rows: {total_count}")
    print(f"Selection: {len(selection)} rows {counts}; sha256={digest}")
    print(f"Output: {output_path} (staging at {partial_path})")

    database = connect(str(partial_path), use_lock_file=False)
    database.metadata = metadata
    started = time.perf_counter()
    with database:
        for output_index, item in enumerate(selection):
            record = read_raw_record(
                source,
                item.source_index,
            )
            initial_record = (
                initial_source.read(item.source_index)
                if initial_source is not None
                else None
            )
            atoms, properties, key_values = transform_record(
                record,
                matrix_dtype=matrix_dtype,
                initial_record=initial_record,
            )
            key_values.update(qh9_subset=item.subset, qh9_variant="stable")
            database.write(atoms, key_value_pairs=key_values, data=properties)
            completed = output_index + 1
            if completed == 1 or completed % args.progress_every == 0 or completed == len(selection):
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1.0e-9)
                print(
                    f"Processed {completed}/{len(selection)} rows "
                    f"({rate:.2f} rows/s)",
                    flush=True,
                )
    source.close()

    validate_output(partial_path, selection, initial_source)
    finalized = connect(str(partial_path), use_lock_file=False)
    final_metadata = dict(finalized.metadata)
    final_metadata["complete"] = True
    final_metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    finalized.metadata = final_metadata
    os.replace(partial_path, output_path)
    print(f"Validated and finalized {output_path}")
    if initial_source is not None:
        initial_source.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-db", type=Path, required=True)
    parser.add_argument(
        "--initial-matrix-lmdb",
        type=Path,
        default=None,
        help=(
            "Optional aligned QH9 split-key LMDB providing initial_hamiltonian "
            "and initial_density_matrix for Hamiltonian delta learning."
        ),
    )
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Official QH9 JSON split file. Output is ordered train, val, test.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=VALID_SUBSETS,
        default=list(VALID_SUBSETS),
    )
    parser.add_argument(
        "--subset-limit",
        action="append",
        default=[],
        metavar="SUBSET=COUNT",
        help="Deterministic prefix limit; repeat for multiple subsets.",
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=0,
        help="Start offset in the flattened selected split; useful for output shards.",
    )
    parser.add_argument(
        "--slice-stop",
        type=int,
        default=None,
        help="Exclusive stop offset in the flattened selected split.",
    )
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
