#!/usr/bin/env python3
"""Download OMol25 open-shell metal-organic density and Hamiltonian sources.

The selected population is the strict C + transition-metal ``ml_mo`` manifest
with multiplicity greater than one.  For every selected calculation this
script transfers:

* ``density_mat.npz``: original fp64 ``orca.scfp`` and ``orca.scfr`` arrays.
* ``orca.tar.zst``: ORCA output containing the alpha and beta Fock matrices.

Files are restored to a separate tree.  The repacked fp32 ``electronic`` tree
is never modified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_COLLECTION = "0b73865a-ff20-4f57-a1d7-573d86b54624"
LEGACY_QUASAR_DESTINATION_ENDPOINT = "6ac548e4-31bd-11f1-ae94-0ea3589134b3"

DEFAULT_MANIFEST = Path(
    "/dataset/seongsu/shared-home/datasets/omol25_open_shell_maloq_ase/"
    "manifests/strict_transition_metal.jsonl"
)
DEFAULT_DESTINATION_ROOT = Path(
    "/dataset/seongsu/shared-home/datasets/omol25_open_shell_source"
)

TRANSFER_FILES = ("density_mat.npz", "orca.tar.zst")
LABEL_PREFIX = "omol25-ml-mo-open-shell"
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
MAX_ATTEMPTS = 3
POLL_SECONDS = 15
TRANSITION_METALS = frozenset(
    (*range(21, 31), *range(39, 49), *range(72, 81))
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_open_shell_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            atoms = {int(z) for z in item["atomic_numbers"]}
            multiplicity = int(item["multiplicity"])
            source_rel = str(item.get("globus_relpath") or item["source_rel"]).strip("/")
            if multiplicity <= 1:
                continue
            if 6 not in atoms or not (atoms & TRANSITION_METALS):
                raise ValueError(
                    f"manifest line {line_number} violates strict metal-organic selection"
                )
            if source_rel in seen:
                raise ValueError(f"duplicate source path in manifest: {source_rel}")
            seen.add(source_rel)
            copied = dict(item)
            copied["globus_relpath"] = source_rel
            copied["multiplicity"] = multiplicity
            rows.append(copied)
    return sorted(rows, key=lambda row: (row["globus_relpath"], row["property_id"]))


def _metal_period(row: dict[str, Any]) -> int:
    metals = sorted(set(int(z) for z in row["atomic_numbers"]) & TRANSITION_METALS)
    z = metals[0]
    if z <= 30:
        return 3
    if z <= 48:
        return 4
    return 5


def select_preflight(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Choose small systems while covering multiplicities and metal periods."""
    if count < 8:
        raise ValueError("preflight count must be at least 8")
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row.get("n_basis_orca") or 10**9),
            int(row["multiplicity"]),
            row["globus_relpath"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        path = row["globus_relpath"]
        if path not in selected_paths:
            selected.append(row)
            selected_paths.add(path)

    for multiplicity in sorted({int(row["multiplicity"]) for row in ranked}):
        add(next(row for row in ranked if int(row["multiplicity"]) == multiplicity))
    for period in (3, 4, 5):
        candidates = [row for row in ranked if _metal_period(row) == period]
        if candidates:
            add(candidates[0])
    for row in ranked:
        if len(selected) >= count:
            break
        add(row)
    return sorted(selected, key=lambda row: row["globus_relpath"])


def _globus_command() -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    direct = shutil.which("globus")
    if direct:
        return [direct], env

    uvx = shutil.which("uvx")
    if uvx is None:
        raise RuntimeError("Neither globus nor uvx is available.")
    env.setdefault("UV_CACHE_DIR", "/tmp/omol_open_shell_globus_uv_cache")
    return [uvx, "--from", "globus-cli", "globus"], env


def run_globus(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    base, env = _globus_command()
    result = subprocess.run(
        [*base, *args],
        text=True,
        capture_output=True,
        env=env,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Globus command failed ({result.returncode}): "
            f"{shlex.join([*base, *args])}\n{result.stderr.strip()}"
        )
    return result


def batch_digest(rows: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['globus_relpath']}|{'|'.join(TRANSFER_FILES)}" for row in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def make_batches(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    batch_size: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [
        (f"{kind}-{index:05d}", rows[start : start + batch_size])
        for index, start in enumerate(range(0, len(rows), batch_size))
    ]


class TransferState:
    def __init__(self, root: Path, destination_endpoint: str | None = None) -> None:
        self.root = root
        self.destination_endpoint = destination_endpoint
        self.control_dir = root / "_transfer"
        self.batch_dir = self.control_dir / "batches"
        self.state_log = self.control_dir / "state.jsonl"
        self.report_dir = self.control_dir / "reports"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def append(self, **entry: Any) -> None:
        with self.state_log.open("a") as handle:
            handle.write(json.dumps({"time": now(), **entry}, sort_keys=True) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.state_log.exists():
            return []
        rows = []
        with self.state_log.open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid state log line {line_number}: {exc}"
                    ) from exc
        return rows

    def latest_by_batch(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for entry in self.read():
            if "batch" in entry:
                latest[entry["batch"]] = entry
        return latest

    def attempts_by_batch(self) -> Counter[str]:
        return Counter(
            entry["batch"]
            for entry in self.read()
            if entry.get("event") == "SUBMITTED"
        )

    def write_batch_file(
        self,
        batch_key: str,
        rows: list[dict[str, Any]],
    ) -> Path:
        path = self.batch_dir / f"{batch_key}.txt"
        with path.open("w") as handle:
            for row in rows:
                source_rel = row["globus_relpath"]
                for filename in TRANSFER_FILES:
                    source = f"/{source_rel}/{filename}"
                    destination = str(self.root / source_rel / filename)
                    handle.write(f"{shlex.quote(source)} {shlex.quote(destination)}\n")
        return path


def task_info(task_id: str) -> dict[str, Any] | None:
    result = run_globus("task", "show", task_id, "-F", "json", check=False)
    if result.returncode != 0:
        print(f"WARNING: cannot inspect task {task_id}: {result.stderr.strip()}")
        return None
    return json.loads(result.stdout)


def append_terminal(
    state: TransferState,
    batch_key: str,
    rows: list[dict[str, Any]],
    attempt: int,
    task_id: str,
    info: dict[str, Any],
) -> None:
    state.append(
        event="TERMINAL",
        status=info["status"],
        batch=batch_key,
        attempt=attempt,
        n_samples=len(rows),
        n_files=len(rows) * len(TRANSFER_FILES),
        digest=batch_digest(rows),
        task_id=task_id,
        bytes_transferred=int(info.get("bytes_transferred", 0)),
        files_transferred=int(info.get("files_transferred", 0)),
        faults=int(info.get("faults", 0)),
    )


def submit_batch(
    state: TransferState,
    batch_key: str,
    rows: list[dict[str, Any]],
    attempt: int,
) -> str:
    if not state.destination_endpoint:
        raise RuntimeError("SC26 Globus destination endpoint is not configured")
    batch_file = state.write_batch_file(batch_key, rows)
    label = f"{LABEL_PREFIX}-{batch_key}-a{attempt}"
    result = run_globus(
        "transfer",
        SOURCE_COLLECTION,
        state.destination_endpoint,
        "--batch",
        str(batch_file),
        "--label",
        label,
        "--sync-level",
        "checksum",
        "--verify-checksum",
        "-F",
        "json",
    )
    task_id = json.loads(result.stdout)["task_id"]
    state.append(
        event="SUBMITTED",
        status="ACTIVE",
        batch=batch_key,
        attempt=attempt,
        n_samples=len(rows),
        n_files=len(rows) * len(TRANSFER_FILES),
        digest=batch_digest(rows),
        task_id=task_id,
        label=label,
    )
    return task_id


def run_batches(
    state: TransferState,
    items: list[tuple[str, list[dict[str, Any]]]],
    *,
    parallel: int,
) -> None:
    if parallel < 1:
        raise ValueError("parallel must be positive")
    latest = state.latest_by_batch()
    attempts = state.attempts_by_batch()
    pending: deque[tuple[str, list[dict[str, Any]]]] = deque()
    active: dict[str, tuple[str, list[dict[str, Any]], int]] = {}
    succeeded = 0

    for batch_key, rows in items:
        prior = latest.get(batch_key)
        digest = batch_digest(rows)
        if prior and prior.get("digest") != digest:
            raise RuntimeError(f"manifest changed for recorded batch {batch_key}")
        if prior and prior.get("status") == "SUCCEEDED":
            succeeded += 1
            continue
        if prior and prior.get("event") == "SUBMITTED" and prior.get("task_id"):
            info = task_info(prior["task_id"])
            if info and info.get("status") not in TERMINAL:
                active[prior["task_id"]] = (
                    batch_key,
                    rows,
                    int(prior.get("attempt", 1)),
                )
                continue
            if info and info.get("status") == "SUCCEEDED":
                append_terminal(
                    state,
                    batch_key,
                    rows,
                    int(prior.get("attempt", 1)),
                    prior["task_id"],
                    info,
                )
                succeeded += 1
                continue
        if attempts[batch_key] >= MAX_ATTEMPTS:
            raise RuntimeError(f"{batch_key} exhausted {MAX_ATTEMPTS} attempts")
        pending.append((batch_key, rows))

    print(
        f"Batches: {len(items)} total, {succeeded} succeeded, "
        f"{len(active)} active, {len(pending)} pending",
        flush=True,
    )
    failures: list[str] = []
    while pending or active:
        while pending and len(active) < parallel:
            batch_key, rows = pending.popleft()
            attempts[batch_key] += 1
            task_id = submit_batch(state, batch_key, rows, attempts[batch_key])
            active[task_id] = (batch_key, rows, attempts[batch_key])
            print(
                f"Submitted {batch_key}: {len(rows):,} samples / "
                f"{len(rows) * len(TRANSFER_FILES):,} files; task {task_id}",
                flush=True,
            )

        if not active:
            break
        time.sleep(POLL_SECONDS)
        for task_id, (batch_key, rows, attempt) in list(active.items()):
            info = task_info(task_id)
            if not info or info.get("status") not in TERMINAL:
                continue
            append_terminal(state, batch_key, rows, attempt, task_id, info)
            del active[task_id]
            status = info["status"]
            print(
                f"Finished {batch_key}: {status}; "
                f"{int(info.get('bytes_transferred', 0)) / 1e9:.3f} GB; "
                f"faults={int(info.get('faults', 0))}; task {task_id}",
                flush=True,
            )
            if status == "SUCCEEDED":
                succeeded += 1
            elif attempts[batch_key] < MAX_ATTEMPTS:
                pending.append((batch_key, rows))
            else:
                failures.append(batch_key)
        print(
            f"Progress: {succeeded}/{len(items)} batches succeeded; "
            f"{len(active)} active; {len(pending)} pending",
            flush=True,
        )
    if failures:
        raise RuntimeError(
            f"failed after {MAX_ATTEMPTS} attempts: {', '.join(failures)}"
        )


def _npy_header(stream: Any) -> dict[str, Any]:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major, minor = stream.read(2)
    if (major, minor) == (1, 0):
        header_length = struct.unpack("<H", stream.read(2))[0]
    else:
        header_length = struct.unpack("<I", stream.read(4))[0]
    return ast.literal_eval(stream.read(header_length).decode("latin1").strip())


def verify_density(path: Path, n_basis: int) -> dict[str, Any]:
    expected_length = n_basis * (n_basis + 1) // 2
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failure in {bad_member}")
        names = set(archive.namelist())
        expected_names = {"orca.scfp.npy", "orca.scfr.npy"}
        if names != expected_names:
            raise ValueError(f"density keys {sorted(names)} != {sorted(expected_names)}")
        headers = {}
        for name in sorted(names):
            with archive.open(name) as stream:
                header = _npy_header(stream)
            if header["descr"] not in {"<f8", "=f8", "|f8"}:
                raise ValueError(f"{name} dtype {header['descr']} is not float64")
            if tuple(header["shape"]) != (expected_length,):
                raise ValueError(
                    f"{name} shape {header['shape']} != ({expected_length},)"
                )
            headers[name] = header
    return {"keys": sorted(names), "packed_length": expected_length}


def _is_int(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def inspect_uhf_fock(text: str, n_basis: int) -> dict[str, Any]:
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "FOCK")
    except StopIteration as exc:
        raise ValueError("FOCK section is absent from orca.out") from exc

    matrix_count = 0
    columns_seen: set[int] = set()
    digest = hashlib.sha256()
    matrix_digests: list[str] = []
    index = start + 1
    while index < len(lines) and matrix_count < 2:
        parts = lines[index].split()
        if parts and all(_is_int(token) for token in parts):
            columns = [int(token) for token in parts]
            if not columns or min(columns) < 0 or max(columns) >= n_basis:
                index += 1
                continue
            columns_seen.update(columns)
            row_count = 0
            index += 1
            while index < len(lines):
                row = lines[index].split()
                if not row or not _is_int(row[0]) or len(row) != len(columns) + 1:
                    break
                row_index = int(row[0])
                if row_index < 0 or row_index >= n_basis:
                    break
                digest.update((" ".join(row) + "\n").encode())
                row_count += 1
                index += 1
            if row_count != n_basis:
                raise ValueError(
                    f"FOCK column chunk {columns[0]}..{columns[-1]} has "
                    f"{row_count} rows, expected {n_basis}"
                )
            if len(columns_seen) == n_basis:
                matrix_count += 1
                matrix_digests.append(digest.hexdigest())
                columns_seen = set()
                digest = hashlib.sha256()
            continue
        index += 1

    if matrix_count != 2:
        raise ValueError(f"expected two UHF Fock matrices, found {matrix_count}")
    if matrix_digests[0] == matrix_digests[1]:
        raise ValueError("alpha and beta Fock matrices are unexpectedly identical")
    return {"matrix_count": matrix_count, "matrix_digests": matrix_digests}


def verify_hamiltonian_archive(path: Path, n_basis: int) -> dict[str, Any]:
    decompressed = subprocess.run(
        ["zstd", "-d", "-c", str(path)],
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r:") as archive:
        names = set(archive.getnames())
        required = {"orca.out", "orca.engrad", "orca.inp"}
        if not required.issubset(names):
            raise ValueError(f"ORCA archive is missing {sorted(required - names)}")
        stream = archive.extractfile("orca.out")
        if stream is None:
            raise ValueError("cannot extract orca.out")
        text = stream.read().decode("utf-8", errors="replace")
    if "HFTyp           .... UHF" not in text:
        raise ValueError("orca.out is not marked as UHF")
    fock = inspect_uhf_fock(text, n_basis)
    return {"members": sorted(names), **fock}


def verify_rows(
    state: TransferState,
    rows: list[dict[str, Any]],
    *,
    report_name: str,
) -> dict[str, Any]:
    failures = []
    verified = []
    for index, row in enumerate(rows, 1):
        source_rel = row["globus_relpath"]
        root = state.root / source_rel
        try:
            density = verify_density(
                root / "density_mat.npz", int(row["n_basis_orca"])
            )
            hamiltonian = verify_hamiltonian_archive(
                root / "orca.tar.zst", int(row["n_basis_orca"])
            )
            verified.append(
                {
                    "globus_relpath": source_rel,
                    "property_id": row["property_id"],
                    "multiplicity": int(row["multiplicity"]),
                    "n_basis_orca": int(row["n_basis_orca"]),
                    "density": density,
                    "hamiltonian": hamiltonian,
                }
            )
        except Exception as exc:
            failures.append({"globus_relpath": source_rel, "error": repr(exc)})
        if len(rows) > 100 and index % 100 == 0:
            print(f"Verified {index:,}/{len(rows):,}", flush=True)
    report = {
        "generated_at": now(),
        "selection": "strict_transition_metal and multiplicity > 1",
        "expected": len(rows),
        "verified": len(verified),
        "failed": len(failures),
        "failures": failures[:100],
        "samples": verified,
    }
    report_path = state.report_dir / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("expected", "verified", "failed")}, indent=2))
    print(f"Report: {report_path}")
    if failures:
        raise RuntimeError(f"verification failed for {len(failures)} samples")
    return report


def show_status(
    state: TransferState,
    items: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    latest = state.latest_by_batch()
    counts: Counter[str] = Counter()
    expected_files = 0
    present_files = 0
    for batch_key, rows in items:
        entry = latest.get(batch_key)
        status = str(entry.get("status", "PENDING")) if entry else "PENDING"
        if entry and entry.get("event") == "SUBMITTED" and entry.get("task_id"):
            info = task_info(entry["task_id"])
            if info:
                status = str(info.get("status", status))
        counts[status] += 1
        for row in rows:
            for filename in TRANSFER_FILES:
                expected_files += 1
                if (state.root / row["globus_relpath"] / filename).is_file():
                    present_files += 1
    print(f"Destination: {state.root}")
    print(f"Samples: {sum(len(rows) for _, rows in items):,}")
    print(f"Files present: {present_files:,}/{expected_files:,}")
    print(f"Batch status: {dict(counts)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--all", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--verify-preflight", action="store_true")
    action.add_argument("--verify-all", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT
    )
    parser.add_argument(
        "--destination-endpoint",
        default=os.environ.get("OMOL_GLOBUS_DESTINATION_ENDPOINT"),
        help=(
            "SC26 Globus collection UUID; defaults to "
            "OMOL_GLOBUS_DESTINATION_ENDPOINT"
        ),
    )
    parser.add_argument("--preflight-count", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--parallel", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight or args.all:
        if not args.destination_endpoint:
            raise SystemExit(
                "Set OMOL_GLOBUS_DESTINATION_ENDPOINT or pass "
                "--destination-endpoint with the SC26 collection UUID."
            )
        if args.destination_endpoint == LEGACY_QUASAR_DESTINATION_ENDPOINT:
            raise SystemExit(
                "Refusing the deprecated Quasar destination endpoint. "
                "Configure an SC26 collection rooted under /dataset."
            )
    rows = load_open_shell_rows(args.manifest)
    preflight_rows = select_preflight(rows, args.preflight_count)
    state = TransferState(args.destination_root, args.destination_endpoint)
    preflight_batches = make_batches(
        preflight_rows,
        kind=f"preflight{args.preflight_count}",
        batch_size=args.preflight_count,
    )
    full_batches = make_batches(rows, kind="full", batch_size=args.batch_size)

    if args.list:
        counts = Counter(int(row["multiplicity"]) for row in rows)
        print(f"Manifest: {args.manifest}")
        print(f"Selection: {len(rows):,} samples; multiplicities={dict(sorted(counts.items()))}")
        print(f"Transfer files per sample: {TRANSFER_FILES}")
        print(f"Full batches: {len(full_batches)} at {args.batch_size:,} samples")
        print(f"Destination: {state.root}")
        print("Preflight:")
        for row in preflight_rows:
            print(
                f"  M={row['multiplicity']} nao={row['n_basis_orca']} "
                f"{row['formula']} {row['globus_relpath']}"
            )
    elif args.preflight:
        run_batches(state, preflight_batches, parallel=1)
        verify_rows(
            state,
            preflight_rows,
            report_name=f"preflight{args.preflight_count}_verification.json",
        )
    elif args.all:
        run_batches(state, full_batches, parallel=args.parallel)
    elif args.status:
        show_status(state, full_batches)
    elif args.verify_preflight:
        verify_rows(
            state,
            preflight_rows,
            report_name=f"preflight{args.preflight_count}_verification.json",
        )
    elif args.verify_all:
        verify_rows(state, rows, report_name="full_verification.json")


if __name__ == "__main__":
    main()
