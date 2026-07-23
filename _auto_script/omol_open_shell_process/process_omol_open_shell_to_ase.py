#!/usr/bin/env python3
"""Convert restored OMol25 open-shell sources into sharded MALOQ ASE DBs.

The converter preserves the official train/val/test split and writes both
alpha/beta Fock and density matrices.  Source matrices remain untouched; the
training databases default to float32 because MALOQ's CUDA matrix-to-label
kernel currently trains in float32.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as SYMBOL_TO_Z
from ase.db import connect

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - exercised on minimal processing hosts
    zstd = None


DEFAULT_SOURCE_ROOT = Path(
    "/home1/irteam/data-vol1/data/omol25/open_shell_restore"
)
DEFAULT_MANIFEST = Path(
    "/home1/irteam/data-vol1/datasets/omol25/manifests/ml_mo_v1/"
    "strict_transition_metal.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home1/irteam/data-vol1/data/omol25/open_shell_maloq_ase"
)
VALID_SPLITS = ("train", "val", "test")
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_selection(
    manifest: Path,
    splits: Sequence[str],
    limit_per_split: int,
) -> dict[str, list[dict]]:
    selected = {split: [] for split in splits}
    with manifest.open(encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            row = json.loads(line)
            split = row.get("split")
            if split not in selected or int(row.get("multiplicity", 1)) <= 1:
                continue
            row["_manifest_index"] = manifest_index
            selected[split].append(row)

    if limit_per_split > 0:
        for split in splits:
            # Small AO systems make deterministic smoke conversion practical.
            selected[split] = sorted(
                selected[split],
                key=lambda row: (
                    int(row["n_basis_orca"]), row["globus_relpath"]
                ),
            )[:limit_per_split]
    return selected


@contextlib.contextmanager
def open_streaming_tar(path: Path) -> Iterator[tarfile.TarFile]:
    """Open a .tar.zst sequentially without materializing the ORCA output."""
    if zstd is not None:
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    yield archive
        return

    executable = shutil.which("zstd") or shutil.which("unzstd")
    if executable is None:
        raise RuntimeError(
            "Reading orca.tar.zst requires the zstandard module or zstd CLI"
        )
    command = [executable, "-dc", str(path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if process.stdout is None:  # pragma: no cover
        process.kill()
        raise RuntimeError("Could not open zstd subprocess stdout")
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            yield archive
    finally:
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0 and sys.exc_info()[0] is None:
            raise RuntimeError(f"zstd exited with status {return_code}: {path}")


XYZ_START = re.compile(r"^\s*\*\s*xyz\s+(-?\d+)\s+(\d+)\s*$", re.I)


def parse_orca_input(text: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    lines = iter(text.splitlines())
    for line in lines:
        match = XYZ_START.match(line)
        if match is not None:
            charge = int(match.group(1))
            multiplicity = int(match.group(2))
            break
    else:
        raise ValueError("orca.inp does not contain an *xyz charge multiplicity block")

    numbers: list[int] = []
    positions: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "*":
            break
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 4 or fields[0] not in SYMBOL_TO_Z:
            raise ValueError(f"Malformed ORCA xyz line: {line!r}")
        numbers.append(SYMBOL_TO_Z[fields[0]])
        positions.append([float(value) for value in fields[1:4]])
    else:
        raise ValueError("ORCA xyz block is not terminated by '*'")

    return (
        np.asarray(numbers, dtype=np.int32),
        np.asarray(positions, dtype=np.float64),
        charge,
        multiplicity,
    )


FINAL_ENERGY = re.compile(
    r"FINAL SINGLE POINT ENERGY\s+([-+0-9.Ee]+)", re.I
)
TOTAL_ENERGY = re.compile(
    r"Total Energy\s*:\s*([-+0-9.Ee]+)\s+Eh", re.I
)


def _decode_lines(handle: BinaryIO) -> Iterator[str]:
    for raw_line in handle:
        yield raw_line.decode("utf-8", errors="replace").rstrip("\r\n")


def _parse_column_header(fields: list[str]) -> list[int] | None:
    if not fields:
        return None
    try:
        columns = [int(field) for field in fields]
    except ValueError:
        return None
    return columns


def read_printed_matrix(
    lines: Iterator[str],
    n_basis: int,
    dtype: np.dtype,
    label: str,
) -> np.ndarray:
    matrix = np.empty((n_basis, n_basis), dtype=dtype)
    next_column = 0
    columns: list[int] | None = None
    next_row = 0

    for line in lines:
        fields = line.split()
        if not fields:
            continue

        if columns is None:
            columns = _parse_column_header(fields)
            if columns is None:
                # The dashed line immediately below FOCK is intentionally
                # ignored; any other pre-header decoration is harmless.
                continue
            expected_columns = list(
                range(next_column, next_column + len(columns))
            )
            if columns != expected_columns or columns[-1] >= n_basis:
                raise ValueError(
                    f"{label}: expected columns {expected_columns}, got {columns}"
                )
            next_row = 0
            continue

        if len(fields) != len(columns) + 1:
            raise ValueError(
                f"{label}: row {next_row} has {len(fields) - 1} values; "
                f"expected {len(columns)}"
            )
        try:
            row = int(fields[0])
        except ValueError as error:
            raise ValueError(f"{label}: invalid row index {fields[0]!r}") from error
        if row != next_row:
            raise ValueError(f"{label}: expected row {next_row}, got {row}")
        values = np.fromstring(" ".join(fields[1:]), sep=" ", dtype=np.float64)
        if values.size != len(columns):
            raise ValueError(
                f"{label}: parsed {values.size} values for {len(columns)} columns"
            )
        matrix[row, columns] = values.astype(dtype, copy=False)
        next_row += 1

        if next_row == n_basis:
            next_column += len(columns)
            columns = None
            if next_column == n_basis:
                if not np.isfinite(matrix).all():
                    raise ValueError(f"{label}: matrix contains NaN or infinity")
                return matrix

    raise EOFError(
        f"{label}: ORCA output ended at column {next_column}, row {next_row}"
    )


def parse_orca_output(
    handle: BinaryIO,
    n_basis: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, float]:
    lines = _decode_lines(handle)
    energy: float | None = None
    for line in lines:
        if match := FINAL_ENERGY.search(line):
            energy = float(match.group(1))
        elif match := TOTAL_ENERGY.search(line):
            energy = float(match.group(1))
        if line.strip() == "FOCK":
            break
    else:
        raise ValueError("ORCA output does not contain a FOCK matrix")

    alpha = read_printed_matrix(lines, n_basis, dtype, "alpha Fock")
    beta = read_printed_matrix(lines, n_basis, dtype, "beta Fock")
    fock = np.stack((alpha, beta), axis=0)
    if energy is None or not math.isfinite(energy):
        raise ValueError("Could not parse a finite final ORCA energy")
    return fock, energy


def parse_archive(
    archive_path: Path,
    n_basis: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray, float]:
    input_record = None
    fock_record = None
    with open_streaming_tar(archive_path) as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile():
                continue
            if name == "orca.inp":
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not extract {member.name}")
                input_record = parse_orca_input(
                    extracted.read().decode("utf-8", errors="replace")
                )
            elif name == "orca.out":
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not extract {member.name}")
                fock_record = parse_orca_output(extracted, n_basis, dtype)
                if input_record is not None:
                    break

    if input_record is None or fock_record is None:
        raise ValueError(
            f"Archive must contain orca.inp and orca.out: {archive_path}"
        )
    numbers, positions, charge, multiplicity = input_record
    fock, energy = fock_record
    return numbers, positions, charge, multiplicity, fock, energy


def load_open_shell_density(
    path: Path,
    n_basis: int,
    dtype: np.dtype,
) -> np.ndarray:
    expected = n_basis * (n_basis + 1) // 2
    with np.load(path, allow_pickle=False) as source:
        missing = {"orca.scfp", "orca.scfr"} - set(source.files)
        if missing:
            raise KeyError(f"{path} is missing density arrays {sorted(missing)}")
        total = np.asarray(source["orca.scfp"]).reshape(-1)
        spin = np.asarray(source["orca.scfr"]).reshape(-1)
        if total.size != expected or spin.size != expected:
            raise ValueError(
                f"Packed density size mismatch for n_basis={n_basis}: "
                f"total={total.size}, spin={spin.size}, expected={expected}"
            )
        alpha_packed = (0.5 * (total + spin)).astype(dtype, copy=False)
        beta_packed = (0.5 * (total - spin)).astype(dtype, copy=False)

    upper = np.triu_indices(n_basis)
    density = np.empty((2, n_basis, n_basis), dtype=dtype)
    for target, packed in zip(density, (alpha_packed, beta_packed)):
        target[upper] = packed
        target[upper[1], upper[0]] = packed
    if not np.isfinite(density).all():
        raise ValueError(f"Density contains NaN or infinity: {path}")
    return density


def validate_record(
    row: dict,
    numbers: np.ndarray,
    positions: np.ndarray,
    charge: int,
    multiplicity: int,
    fock: np.ndarray,
    density: np.ndarray,
) -> None:
    expected_numbers = np.asarray(row["atomic_numbers"], dtype=np.int32)
    n_basis = int(row["n_basis_orca"])
    if not np.array_equal(numbers, expected_numbers):
        raise ValueError("ORCA input atom order differs from the official manifest")
    if positions.shape != (numbers.size, 3):
        raise ValueError(f"Position shape mismatch: {positions.shape}")
    if charge != int(row["charge"]):
        raise ValueError(f"Charge mismatch: input={charge}, manifest={row['charge']}")
    if multiplicity != int(row["multiplicity"]):
        raise ValueError(
            "Multiplicity mismatch: "
            f"input={multiplicity}, manifest={row['multiplicity']}"
        )
    expected_shape = (2, n_basis, n_basis)
    if fock.shape != expected_shape or density.shape != expected_shape:
        raise ValueError(
            f"Matrix shape mismatch: F={fock.shape}, D={density.shape}, "
            f"expected={expected_shape}"
        )
    if np.array_equal(fock[0], fock[1]):
        raise ValueError("Open-shell alpha and beta Fock matrices are identical")
    fock_symmetry_error = max(
        float(np.max(np.abs(spin - spin.T))) for spin in fock
    )
    if fock_symmetry_error > 2.0e-5:
        raise ValueError(
            f"Fock matrix is not symmetric; max error={fock_symmetry_error}"
        )


def process_row(row: dict, source_root: Path, dtype: np.dtype) -> tuple[Atoms, dict]:
    sample_root = source_root / row["globus_relpath"]
    archive_path = sample_root / "orca.tar.zst"
    density_path = sample_root / "density_mat.npz"
    if not archive_path.is_file() or not density_path.is_file():
        raise FileNotFoundError(f"Incomplete restored sample: {sample_root}")

    n_basis = int(row["n_basis_orca"])
    numbers, positions, charge, multiplicity, fock, energy = parse_archive(
        archive_path, n_basis, dtype
    )
    density = load_open_shell_density(density_path, n_basis, dtype)
    validate_record(
        row, numbers, positions, charge, multiplicity, fock, density
    )
    atoms = Atoms(numbers=numbers, positions=positions)
    data = {
        "charge": charge,
        "spin_multiplicity": multiplicity,
        "fock_matrix": fock,
        "density_matrix": density,
        "total_energy [Eh]": energy,
        "is_open_shell": True,
        "num_atoms_in_molecule": int(numbers.size),
        "folder_name": row["globus_relpath"],
        "split": row["split"],
        "configuration_id": row["configuration_id"],
        "property_id": row["property_id"],
        "source_manifest_index": int(row["_manifest_index"]),
        "matrix_storage_convention": "orca_real_spherical",
        "density_source_convention": "alpha_beta_from_total_spin",
        "source_matrix_dtype": "float64",
        "stored_matrix_dtype": dtype.name,
        "n_basis_orca": n_basis,
    }
    return atoms, data


def shard_paths(output_root: Path, split: str, shard_index: int) -> tuple[Path, Path]:
    stem = f"omol_open_shell_{split}_{shard_index:05d}"
    return (
        output_root / split / f"{stem}.db",
        output_root / "_state" / f"{stem}.json",
    )


def successful_shard_matches(db_path: Path, report_path: Path, count: int) -> bool:
    if not db_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "SUCCEEDED" or report.get("rows") != count:
            return False
        with connect(db_path) as database:
            return database.count() == count
    except Exception:
        return False


def process_shard(
    output_root_string: str,
    source_root_string: str,
    split: str,
    shard_index: int,
    rows: list[dict],
    dtype_name: str,
    attempt: int = 1,
) -> dict:
    output_root = Path(output_root_string)
    source_root = Path(source_root_string)
    dtype = np.dtype(dtype_name)
    db_path, report_path = shard_paths(output_root, split, shard_index)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_db = db_path.with_name(
        f".{db_path.stem}.partial-{os.getpid()}.db"
    )
    started = time.monotonic()
    report = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "shard_index": shard_index,
        "rows": len(rows),
        "attempt": attempt,
        "first_manifest_index": int(rows[0]["_manifest_index"]),
        "last_manifest_index": int(rows[-1]["_manifest_index"]),
        "started_at": utc_now(),
        "status": "RUNNING",
    }
    write_json_atomic(report_path, report)

    if temporary_db.exists():
        temporary_db.unlink()
    try:
        with connect(temporary_db) as database:
            database.metadata = {
                "schema": "maloq_omol_open_shell_v1",
                "split": split,
                "matrix_storage_convention": "orca_real_spherical",
                "stored_matrix_dtype": dtype.name,
                "spin_channels": ["alpha", "beta"],
                "density_source_arrays": ["orca.scfp", "orca.scfr"],
            }
            for row in rows:
                atoms, data = process_row(row, source_root, dtype)
                database.write(atoms, data=data)
        with connect(temporary_db) as database:
            actual_count = database.count()
        if actual_count != len(rows):
            raise RuntimeError(
                f"Shard contains {actual_count} rows; expected {len(rows)}"
            )
        os.replace(temporary_db, db_path)
        report.update(
            status="SUCCEEDED",
            completed_at=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            db_bytes=db_path.stat().st_size,
        )
        write_json_atomic(report_path, report)
        return report
    except Exception as error:
        if temporary_db.exists():
            temporary_db.unlink()
        report.update(
            status="FAILED",
            completed_at=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        write_json_atomic(report_path, report)
        return report


def process_shard_with_retries(
    output_root_string: str,
    source_root_string: str,
    split: str,
    shard_index: int,
    rows: list[dict],
    dtype_name: str,
    max_attempts: int,
) -> dict:
    report = None
    for attempt in range(1, max_attempts + 1):
        report = process_shard(
            output_root_string,
            source_root_string,
            split,
            shard_index,
            rows,
            dtype_name,
            attempt,
        )
        if report["status"] == "SUCCEEDED":
            return report
    assert report is not None
    return report


def make_jobs(
    selection: dict[str, list[dict]], shard_size: int
) -> list[tuple[str, int, list[dict]]]:
    jobs = []
    for split in selection:
        rows = selection[split]
        for start in range(0, len(rows), shard_size):
            jobs.append((split, start // shard_size, rows[start : start + shard_size]))
    return jobs


def summarize_reports(output_root: Path) -> dict:
    statuses = Counter()
    rows = Counter()
    bytes_written = 0
    state_root = output_root / "_state"
    if state_root.is_dir():
        for path in sorted(state_root.glob("omol_open_shell_*.json")):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                statuses["UNREADABLE"] += 1
                continue
            status = report.get("status", "UNKNOWN")
            statuses[status] += 1
            if status == "SUCCEEDED":
                rows[report["split"]] += int(report["rows"])
                bytes_written += int(report.get("db_bytes", 0))
    return {
        "shards": dict(sorted(statuses.items())),
        "successful_rows": dict(sorted(rows.items())),
        "database_bytes": bytes_written,
        "database_GB": round(bytes_written / 1.0e9, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS)
    )
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--matrix-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Deterministic smallest-AO subset for smoke conversion; 0 means all.",
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        print(json.dumps(summarize_reports(args.output_root), indent=2))
        return 0
    if (
        args.shard_size <= 0
        or args.workers <= 0
        or args.max_attempts <= 0
        or args.limit_per_split < 0
    ):
        raise ValueError(
            "shard-size/workers/max-attempts must be positive; "
            "limit must be nonnegative"
        )
    if not args.manifest.is_file() or not args.source_root.is_dir():
        raise FileNotFoundError("Manifest or restored source root does not exist")

    selection = load_selection(args.manifest, args.splits, args.limit_per_split)
    expected_counts = {split: len(rows) for split, rows in selection.items()}
    jobs = make_jobs(selection, args.shard_size)
    manifest_digest = sha256_file(args.manifest)
    metadata = {
        "schema": "maloq_omol_open_shell_v1",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "PLANNED" if args.dry_run else "PROCESSING",
        "source_root": str(args.source_root.resolve()),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": manifest_digest,
        "selection": "strict_transition_metal AND multiplicity > 1",
        "official_split_counts": expected_counts,
        "matrix_targets": ["fock_matrix", "density_matrix"],
        "spin_channels": ["alpha", "beta"],
        "matrix_storage_convention": "orca_real_spherical",
        "source_matrix_dtype": "float64",
        "stored_matrix_dtype": args.matrix_dtype,
        "shard_size": args.shard_size,
        "planned_shards": len(jobs),
        "max_attempts": args.max_attempts,
        "limit_per_split": args.limit_per_split,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_root / "dataset_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    if args.dry_run:
        return 0

    pending = []
    for split, shard_index, rows in jobs:
        db_path, report_path = shard_paths(args.output_root, split, shard_index)
        if successful_shard_matches(db_path, report_path, len(rows)):
            continue
        pending.append((split, shard_index, rows))
    print(
        f"Planned {len(jobs)} shards; {len(jobs) - len(pending)} complete; "
        f"{len(pending)} pending",
        flush=True,
    )

    failures = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                process_shard_with_retries,
                str(args.output_root),
                str(args.source_root),
                split,
                shard_index,
                rows,
                args.matrix_dtype,
                args.max_attempts,
            ): (split, shard_index)
            for split, shard_index, rows in pending
        }
        completed = len(jobs) - len(pending)
        for future in concurrent.futures.as_completed(futures):
            split, shard_index = futures[future]
            report = future.result()
            completed += 1
            if report["status"] != "SUCCEEDED":
                failures.append(report)
            print(
                f"[{completed}/{len(jobs)}] {split}-{shard_index:05d}: "
                f"{report['status']} ({report.get('elapsed_seconds', 0):.1f}s)",
                flush=True,
            )

    summary = summarize_reports(args.output_root)
    metadata.update(
        status="FAILED" if failures else "SUCCEEDED",
        completed_at=utc_now(),
        result=summary,
        failed_shards=len(failures),
    )
    write_json_atomic(args.output_root / "dataset_metadata.json", metadata)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
