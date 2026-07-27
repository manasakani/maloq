#!/usr/bin/env python3
"""Verify a full-v2 electrolyte index, its shards, and packed matrix physics."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lmdb
import numpy as np

from full_v2_common import (
    ContractError,
    EXPECTED_SCHEMA,
    SOURCE_V2,
    SOURCE_VERSIONS,
    SPLITS,
    atomic_write_json,
    atomic_write_text,
    load_index_rows,
    load_manifest_catalog,
    manifest_key,
    manifest_mol_id,
    path_is_lexically_within,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=None,
        help="When supplied, prove every available v2 sample wins precedence.",
    )
    parser.add_argument(
        "--view-root",
        type=Path,
        default=None,
        help="Require index LMDB/summary paths to be symlinks inside this view.",
    )
    parser.add_argument(
        "--mode",
        choices=("metadata", "sampled", "full"),
        default="sampled",
    )
    parser.add_argument("--records-per-shard", type=int, default=1)
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="Testing only. Zero checks every referenced shard.",
    )
    parser.add_argument("--max-density-trace-error", type=float, default=5.0e-2)
    parser.add_argument("--max-initial-trace-error", type=float, default=1.0e-3)
    parser.add_argument("--max-summary-trace-delta", type=float, default=1.0e-3)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--mark-complete",
        type=Path,
        default=None,
        metavar="VIEW_ROOT",
        help="Write verification.json and COMPLETE only after an all-record check.",
    )
    return parser.parse_args()


def _decode_lmdb_int(raw: bytes | memoryview | None, label: str) -> int:
    if raw is None:
        raise ContractError(f"LMDB missing {label}")
    value = bytes(raw)
    if not value:
        raise ContractError(f"LMDB has empty {label}")
    return int.from_bytes(value, "big")


def _packed_array(sample: dict[str, Any], name: str) -> tuple[np.ndarray, int]:
    packed_key = f"{name}_packed"
    nao_key = f"{name}_nao"
    dtype_key = f"{name}_dtype"
    for key in (packed_key, nao_key, dtype_key):
        if key not in sample:
            raise ContractError(f"sample {sample.get('mol_id')}: missing {key}")
    nao = int(sample[nao_key])
    if nao <= 0:
        raise ContractError(f"sample {sample.get('mol_id')}: invalid {nao_key}={nao}")
    dtype = np.dtype(sample[dtype_key])
    if dtype != np.dtype("float32"):
        raise ContractError(
            f"sample {sample.get('mol_id')}: {dtype_key}={dtype}, expected float32"
        )
    raw = sample[packed_key]
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ContractError(f"sample {sample.get('mol_id')}: {packed_key} is not bytes")
    expected_values = nao * (nao + 1) // 2
    expected_bytes = expected_values * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ContractError(
            f"sample {sample.get('mol_id')}: {packed_key} bytes={len(raw)}, "
            f"expected={expected_bytes}"
        )
    values = np.frombuffer(raw, dtype=dtype)
    if not np.isfinite(values).all():
        raise ContractError(f"sample {sample.get('mol_id')}: non-finite {name}")
    return values, nao


def _packed_symmetric_trace(left: np.ndarray, right: np.ndarray, nao: int) -> float:
    if left.shape != right.shape:
        raise ContractError("packed matrices have different shapes")
    # Packed upper triangle is row-major. 2*dot(all)-dot(diagonal) counts
    # off-diagonal pairs twice and diagonal pairs once, equal to Tr(left*right).
    dot_all = float(np.einsum("i,i->", left, right, dtype=np.float64))
    diagonal_positions = (
        np.arange(nao, dtype=np.int64) * nao
        - np.arange(nao, dtype=np.int64)
        * (np.arange(nao, dtype=np.int64) - 1)
        // 2
    )
    dot_diagonal = float(
        np.einsum(
            "i,i->",
            left[diagonal_positions],
            right[diagonal_positions],
            dtype=np.float64,
        )
    )
    return 2.0 * dot_all - dot_diagonal


def _summary_samples_by_index(summary: dict[str, Any], path: Path) -> dict[int, dict]:
    samples = summary.get("samples")
    if not isinstance(samples, list):
        raise ContractError(f"{path}: samples is not a list")
    result: dict[int, dict] = {}
    for sample in samples:
        local_index = int(sample["local_index"])
        if local_index in result:
            raise ContractError(f"{path}: duplicate local_index={local_index}")
        result[local_index] = sample
    return result


def _validate_schema(schema: dict[str, Any], path: Path, split: str) -> None:
    for key, expected in EXPECTED_SCHEMA.items():
        if schema.get(key) != expected:
            raise ContractError(
                f"{path}: schema[{key!r}]={schema.get(key)!r}, expected={expected!r}"
            )
    if schema.get("split") != split:
        raise ContractError(
            f"{path}: schema split={schema.get('split')!r}, expected={split!r}"
        )


def _validate_record(
    sample: dict[str, Any],
    *,
    expected_mol_id: str,
    summary_sample: dict[str, Any],
    density_tolerance: float,
    initial_tolerance: float,
    summary_tolerance: float,
) -> dict[str, float]:
    mol_id = str(sample.get("mol_id") or "")
    if mol_id != expected_mol_id:
        raise ContractError(
            f"index mol_id={expected_mol_id}, LMDB mol_id={mol_id!r}"
        )
    if not sample.get("_packed", False):
        raise ContractError(f"{mol_id}: sample is not marked packed")

    atomic_raw = sample.get("atomic_numbers")
    position_raw = sample.get("positions")
    if not isinstance(atomic_raw, (bytes, bytearray, memoryview)):
        raise ContractError(f"{mol_id}: atomic_numbers is not bytes")
    if not isinstance(position_raw, (bytes, bytearray, memoryview)):
        raise ContractError(f"{mol_id}: positions is not bytes")
    atoms = np.frombuffer(atomic_raw, dtype=np.int32)
    num_atoms = int(sample.get("num_atoms", -1))
    if len(atoms) != num_atoms:
        raise ContractError(
            f"{mol_id}: num_atoms={num_atoms}, atomic_numbers={len(atoms)}"
        )
    positions = np.frombuffer(position_raw, dtype=np.float64)
    if positions.size != num_atoms * 3 or not np.isfinite(positions).all():
        raise ContractError(f"{mol_id}: invalid positions payload")

    density, density_nao = _packed_array(sample, "density_matrix")
    overlap, overlap_nao = _packed_array(sample, "overlap")
    initial, initial_nao = _packed_array(sample, "initial_density_matrix")
    if len({density_nao, overlap_nao, initial_nao}) != 1:
        raise ContractError(f"{mol_id}: matrix nao values disagree")
    nao = density_nao

    diagonal_positions = (
        np.arange(nao, dtype=np.int64) * nao
        - np.arange(nao, dtype=np.int64)
        * (np.arange(nao, dtype=np.int64) - 1)
        // 2
    )
    if np.any(overlap[diagonal_positions] <= 0):
        raise ContractError(f"{mol_id}: overlap has non-positive diagonal")

    density_trace = _packed_symmetric_trace(density, overlap, nao)
    initial_trace = _packed_symmetric_trace(initial, overlap, nao)
    expected_electrons = (
        int(atoms.astype(np.int64).sum())
        - int(sample.get("charge", 0))
        - int(sample.get("num_ecp_electrons", 0))
    )
    density_error = density_trace - expected_electrons
    initial_error = initial_trace - expected_electrons
    if abs(density_error) > density_tolerance:
        raise ContractError(
            f"{mol_id}: density trace error={density_error:.8g}, "
            f"limit={density_tolerance}"
        )
    if abs(initial_error) > initial_tolerance:
        raise ContractError(
            f"{mol_id}: initial-density trace error={initial_error:.8g}, "
            f"limit={initial_tolerance}"
        )

    for summary_key, computed in (
        ("trace_target", density_trace),
        ("trace_initial_density", initial_trace),
    ):
        if summary_key in summary_sample:
            delta = computed - float(summary_sample[summary_key])
            if abs(delta) > summary_tolerance:
                raise ContractError(
                    f"{mol_id}: {summary_key} delta={delta:.8g}, "
                    f"limit={summary_tolerance}"
                )
    if int(summary_sample.get("nao", nao)) != nao:
        raise ContractError(f"{mol_id}: summary nao disagrees with LMDB")
    if int(summary_sample.get("n_atoms", num_atoms)) != num_atoms:
        raise ContractError(f"{mol_id}: summary n_atoms disagrees with LMDB")
    return {
        "density_trace_error": density_error,
        "initial_density_trace_error": initial_error,
    }


def _discover_v2_mol_ids(v2_root: Path) -> set[str]:
    result: set[str] = set()
    for split in SPLITS:
        split_dir = v2_root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            if int(summary.get("failure_count", 0)):
                raise ContractError(f"{path}: failed v2 shard cannot establish precedence")
            for sample in summary.get("samples", []):
                manifest = sample.get("manifest")
                if not isinstance(manifest, dict):
                    raise ContractError(f"{path}: sample lacks manifest")
                mol_id = manifest_mol_id(manifest)
                if mol_id in result:
                    raise ContractError(f"duplicate v2 sample while checking precedence: {mol_id}")
                result.add(mol_id)
    return result


def main() -> int:
    args = parse_args()
    if args.records_per_shard < 1 and args.mode != "metadata":
        raise ContractError("--records-per-shard must be positive")
    if args.max_shards < 0:
        raise ContractError("--max-shards may not be negative")
    if args.mark_complete is not None:
        if args.mode != "full":
            raise ContractError("--mark-complete requires --mode full")
        if args.max_shards:
            raise ContractError("--mark-complete cannot be used with --max-shards")

    manifest_dir = args.manifest_dir.resolve()
    index_root = args.index_root.resolve()
    view_root = args.view_root.absolute() if args.view_root is not None else None
    catalog = load_manifest_catalog(manifest_dir)
    rows_by_split = load_index_rows(index_root)

    indexed: dict[str, tuple[str, dict[str, Any]]] = {}
    source_counts: Counter[str] = Counter()
    for split in SPLITS:
        if len(rows_by_split[split]) != catalog.counts[split]:
            raise ContractError(
                f"{split}: index={len(rows_by_split[split])}, "
                f"manifest={catalog.counts[split]}"
            )
        for ordinal, row in enumerate(rows_by_split[split]):
            mol_id = str(row.get("mol_id") or "")
            manifest = catalog.by_mol_id.get(mol_id)
            if manifest is None:
                raise ContractError(f"{split}:{ordinal}: unknown mol_id={mol_id!r}")
            if manifest.split != split:
                raise ContractError(f"{mol_id}: index split={split}, manifest={manifest.split}")
            if row.get("key") != manifest.key:
                raise ContractError(f"{mol_id}: canonical manifest key mismatch")
            if int(row.get("manifest_ordinal", -1)) != manifest.ordinal:
                raise ContractError(f"{mol_id}: manifest ordinal mismatch")
            if mol_id in indexed:
                raise ContractError(f"duplicate full-v2 index sample: {mol_id}")
            source_version = str(row.get("source_version") or "")
            if source_version not in SOURCE_VERSIONS:
                raise ContractError(f"{mol_id}: unknown source_version={source_version!r}")
            indexed[mol_id] = (split, row)
            source_counts[source_version] += 1

    missing = set(catalog.by_mol_id) - set(indexed)
    extra = set(indexed) - set(catalog.by_mol_id)
    if missing or extra:
        raise ContractError(
            f"index coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    precedence_checked = False
    if args.v2_root is not None:
        v2_ids = _discover_v2_mol_ids(args.v2_root.resolve())
        precedence_violations = [
            mol_id
            for mol_id in sorted(v2_ids)
            if indexed.get(mol_id, ("", {}))[1].get("source_version") != SOURCE_V2
        ]
        if precedence_violations:
            raise ContractError(
                f"{len(precedence_violations)} rebuilt-v2 samples did not win "
                f"precedence; first={precedence_violations[:5]}"
            )
        precedence_checked = True

    grouped: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for split in SPLITS:
        for row in rows_by_split[split]:
            lmdb_path = Path(row["lmdb"]).absolute()
            summary_path = Path(row["summary"]).absolute()
            if view_root is not None:
                if not path_is_lexically_within(lmdb_path, view_root):
                    raise ContractError(f"index LMDB is outside view: {lmdb_path}")
                if not path_is_lexically_within(summary_path, view_root):
                    raise ContractError(f"index summary is outside view: {summary_path}")
                if not lmdb_path.is_symlink():
                    raise ContractError(f"view LMDB must be a symlink: {lmdb_path}")
                if not summary_path.is_symlink():
                    raise ContractError(f"view summary must be a symlink: {summary_path}")
            if not lmdb_path.is_dir() or not (lmdb_path / "data.mdb").is_file():
                raise ContractError(f"missing LMDB/data.mdb: {lmdb_path}")
            if not summary_path.is_file():
                raise ContractError(f"missing shard summary: {summary_path}")
            grouped[(split, str(lmdb_path), str(summary_path))].append(row)

    shard_items = sorted(grouped.items(), key=lambda item: item[0])
    if args.max_shards:
        shard_items = shard_items[: args.max_shards]
    checked_records = 0
    checked_shards = 0
    max_density_error = 0.0
    max_initial_error = 0.0
    shard_counts: Counter[str] = Counter()

    for (split, lmdb_raw, summary_raw), rows in shard_items:
        lmdb_path = Path(lmdb_raw)
        summary_path = Path(summary_raw)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("split", split)) != split:
            raise ContractError(f"{summary_path}: summary split mismatch")
        if int(summary.get("failure_count", 0)):
            raise ContractError(f"{summary_path}: referenced shard contains failures")
        samples_by_index = _summary_samples_by_index(summary, summary_path)
        written_count = int(summary.get("written_count", len(samples_by_index)))
        if written_count != len(samples_by_index):
            raise ContractError(
                f"{summary_path}: written_count={written_count}, "
                f"summary samples={len(samples_by_index)}"
            )
        if set(samples_by_index) != set(range(written_count)):
            raise ContractError(f"{summary_path}: non-contiguous summary local indices")

        environment = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            subdir=True,
        )
        try:
            with environment.begin(buffers=True) as transaction:
                lmdb_length = _decode_lmdb_int(transaction.get(b"__len__"), "__len__")
                if lmdb_length != written_count:
                    raise ContractError(
                        f"{lmdb_path}: __len__={lmdb_length}, summary={written_count}"
                    )
                raw_format = transaction.get(b"__format__")
                if raw_format is None or bytes(raw_format) != b"pickle":
                    raise ContractError(f"{lmdb_path}: __format__ is not pickle")
                raw_schema = transaction.get(b"__schema__")
                if raw_schema is None:
                    raise ContractError(f"{lmdb_path}: missing __schema__")
                schema = pickle.loads(raw_schema)
                if not isinstance(schema, dict):
                    raise ContractError(f"{lmdb_path}: schema is not a dictionary")
                _validate_schema(schema, lmdb_path, split)

                referenced_indices: list[int] = []
                for row in rows:
                    local_index = int(row["local_index"])
                    if local_index not in samples_by_index:
                        raise ContractError(
                            f"{lmdb_path}: indexed local_index={local_index} absent "
                            "from summary"
                        )
                    summary_manifest = samples_by_index[local_index].get("manifest")
                    if not isinstance(summary_manifest, dict):
                        raise ContractError(
                            f"{summary_path}: local_index={local_index} has no manifest"
                        )
                    if manifest_mol_id(summary_manifest) != row["mol_id"]:
                        raise ContractError(
                            f"{summary_path}: local_index={local_index} mol_id mismatch"
                        )
                    if manifest_key(summary_manifest) != row["key"]:
                        raise ContractError(
                            f"{summary_path}: local_index={local_index} key mismatch"
                        )
                    if transaction.get(local_index.to_bytes(4, "big")) is None:
                        raise ContractError(
                            f"{lmdb_path}: missing record local_index={local_index}"
                        )
                    referenced_indices.append(local_index)

                if args.mode == "metadata":
                    chosen_indices: list[int] = []
                elif args.mode == "full":
                    chosen_indices = list(range(lmdb_length))
                else:
                    unique = sorted(set(referenced_indices))
                    if len(unique) <= args.records_per_shard:
                        chosen_indices = unique
                    else:
                        # Cover both ends of each shard for records_per_shard > 1.
                        positions = np.linspace(
                            0, len(unique) - 1, args.records_per_shard, dtype=int
                        )
                        chosen_indices = [unique[int(position)] for position in positions]

                for local_index in chosen_indices:
                    raw = transaction.get(local_index.to_bytes(4, "big"))
                    if raw is None:
                        raise ContractError(
                            f"{lmdb_path}: missing record local_index={local_index}"
                        )
                    sample = pickle.loads(raw)
                    summary_sample = samples_by_index[local_index]
                    expected_mol_id = manifest_mol_id(summary_sample["manifest"])
                    trace_report = _validate_record(
                        sample,
                        expected_mol_id=expected_mol_id,
                        summary_sample=summary_sample,
                        density_tolerance=args.max_density_trace_error,
                        initial_tolerance=args.max_initial_trace_error,
                        summary_tolerance=args.max_summary_trace_delta,
                    )
                    max_density_error = max(
                        max_density_error,
                        abs(trace_report["density_trace_error"]),
                    )
                    max_initial_error = max(
                        max_initial_error,
                        abs(trace_report["initial_density_trace_error"]),
                    )
                    checked_records += 1
        finally:
            environment.close()
        checked_shards += 1
        shard_counts[split] += 1

    report = {
        "status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "schema": "omol_electrolyte_full_v2_verification_v1",
        "mode": args.mode,
        "manifest_dir": str(manifest_dir),
        "manifest_sha256": catalog.file_sha256,
        "manifest_counts": catalog.counts,
        "index_root": str(index_root),
        "index_sha256": {
            f"{split}.index.jsonl": sha256_file(
                index_root / f"{split}.index.jsonl"
            )
            for split in SPLITS
        },
        "indexed_total": len(indexed),
        "source_counts": dict(sorted(source_counts.items())),
        "precedence_checked": precedence_checked,
        "view_root": str(view_root) if view_root is not None else None,
        "referenced_shards": len(grouped),
        "checked_shards": checked_shards,
        "checked_shards_by_split": dict(shard_counts),
        "checked_records": checked_records,
        "max_abs_density_trace_error": max_density_error,
        "max_abs_initial_density_trace_error": max_initial_error,
        "limits": {
            "density_trace_error": args.max_density_trace_error,
            "initial_density_trace_error": args.max_initial_trace_error,
            "summary_trace_delta": args.max_summary_trace_delta,
        },
        "all_shards_checked": not args.max_shards,
        "all_records_checked": args.mode == "full" and not args.max_shards,
    }
    if args.report_path is not None:
        atomic_write_json(args.report_path.absolute(), report)
    if args.mark_complete is not None:
        complete_root = args.mark_complete.absolute()
        if view_root is None or complete_root != view_root:
            raise ContractError("--mark-complete must equal --view-root")
        if index_root != (view_root / "_index").resolve():
            raise ContractError(
                "--mark-complete requires --index-root VIEW_ROOT/_index"
            )
        atomic_write_json(view_root / "verification.json", report)
        atomic_write_text(
            view_root / "COMPLETE",
            "verified full-v2 view\n"
            f"verified_at={report['verified_at']}\n"
            f"indexed_total={report['indexed_total']}\n"
            f"index_summary_sha256={sha256_file(view_root / '_index' / 'summary.json')}\n",
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, lmdb.Error, pickle.UnpicklingError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
