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
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.db import connect
from pyscf import gto


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils.ASEDataset import ASEAtomsData  # noqa: E402
from fock_utils import basis_sets, utils_orca_out  # noqa: E402


PROPERTY_UNITS = {
    "energy": "Hartree",
    "forces": "Hartree/Angstrom",
    "hamiltonian": "Hartree",
    "overlap": "dimensionless",
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
        total += sum(2 * int(l) + 1 for l in ORBITAL_BASIS[z])
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
) -> dict[str, object]:
    counts, digest = selection_metadata(selection)
    return {
        "_property_unit_dict": PROPERTY_UNITS,
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


def validate_output(path: Path, expected_count: int) -> None:
    dataset = ASEAtomsData(str(path))
    if len(dataset) != expected_count:
        raise ValueError(f"Output contains {len(dataset)} rows; expected {expected_count}")
    required = set(PROPERTY_UNITS)
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
            atoms, properties, key_values = transform_record(
                record,
                matrix_dtype=matrix_dtype,
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

    validate_output(partial_path, len(selection))
    finalized = connect(str(partial_path), use_lock_file=False)
    final_metadata = dict(finalized.metadata)
    final_metadata["complete"] = True
    final_metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    finalized.metadata = final_metadata
    os.replace(partial_path, output_path)
    print(f"Validated and finalized {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-db", type=Path, required=True)
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
