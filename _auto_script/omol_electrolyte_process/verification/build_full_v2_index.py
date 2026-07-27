#!/usr/bin/env python3
"""Build an immutable 100%-coverage electrolyte index with v2 precedence.

The accepted manifest is authoritative. A real rebuilt-v2 sample replaces the
same v1 sample; samples not rebuilt in v2 continue to reference the immutable
v1 shard. The destination is published atomically only when coverage is exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from full_v2_common import (
    ContractError,
    SOURCE_V1,
    SOURCE_V2,
    SPLITS,
    assert_no_symlink_ancestor,
    compact_manifest,
    discard_staging_directory,
    fsync_file,
    json_dump_line,
    lmdb_for_summary,
    load_manifest_catalog,
    manifest_key,
    manifest_mol_id,
    new_staging_directory,
    path_is_within,
    publish_staging_directory,
    read_jsonl,
    resolve_lmdb_reference,
    resolve_summary_reference,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument(
        "--v1-index-root",
        type=Path,
        default=None,
        help="Defaults to V1_ROOT/_index.",
    )
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--out-index-root", type=Path, required=True)
    parser.add_argument(
        "--expected-total",
        type=int,
        default=0,
        help="Zero derives the total from the accepted manifest summary/scan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check coverage and print the plan without creating output.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Only valid with --dry-run; report missing samples without failure.",
    )
    return parser.parse_args()


def load_v1_sources(
    *,
    v1_root: Path,
    v1_index_root: Path,
    catalog,
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        index_path = v1_index_root / f"{split}.index.jsonl"
        if not index_path.is_file():
            raise ContractError(f"missing v1 index: {index_path}")
        for line_number, row in enumerate(read_jsonl(index_path), start=1):
            mol_id = str(row.get("mol_id") or "")
            if not mol_id:
                raise ContractError(f"{index_path}:{line_number}: missing mol_id")
            manifest = catalog.by_mol_id.get(mol_id)
            if manifest is None:
                raise ContractError(
                    f"{index_path}:{line_number}: {mol_id} is not accepted"
                )
            if manifest.split != split:
                raise ContractError(
                    f"{mol_id}: v1 index split={split}, manifest={manifest.split}"
                )
            if mol_id in sources:
                raise ContractError(f"duplicate v1 index sample: {mol_id}")
            lmdb_path = resolve_lmdb_reference(row["lmdb"], v1_root, split)
            summary_path = resolve_summary_reference(row["summary"], v1_root, split)
            if not path_is_within(lmdb_path, v1_root):
                raise ContractError(f"v1 LMDB escaped declared root: {lmdb_path}")
            if not path_is_within(summary_path, v1_root):
                raise ContractError(f"v1 summary escaped declared root: {summary_path}")
            sources[mol_id] = {
                "lmdb": lmdb_path,
                "summary": summary_path,
                "local_index": int(row["local_index"]),
                "source_version": SOURCE_V1,
                "source_split": split,
                "source_shard_index": _shard_index(row, summary_path),
                "source_stats": {
                    key: row[key]
                    for key in ("nao", "n_atoms", "n_electrons")
                    if key in row
                },
            }
    return sources


def _shard_index(row: dict[str, Any], path: Path) -> int:
    if "source_shard_index" in row:
        return int(row["source_shard_index"])
    stem = path.name.split(".")[0]
    if stem.startswith("shard_"):
        try:
            return int(stem[len("shard_") :])
        except ValueError:
            pass
    return -1


def load_v2_sources(
    *,
    v2_root: Path,
    v1_root: Path,
    catalog,
) -> dict[str, dict[str, Any]]:
    if not v2_root.is_dir():
        raise ContractError(f"v2 root does not exist: {v2_root}")
    sources: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_dir = v2_root / split
        if not split_dir.is_dir():
            continue
        for summary_path in sorted(split_dir.glob("*.summary.json")):
            assert_no_symlink_ancestor(summary_path, v2_root)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_split = str(summary.get("split", split))
            if summary_split != split:
                raise ContractError(
                    f"{summary_path}: split={summary_split}, directory={split}"
                )
            failure_count = int(summary.get("failure_count", 0))
            if failure_count:
                raise ContractError(
                    f"{summary_path}: rebuilt shard has {failure_count} failures"
                )
            samples = summary.get("samples")
            if not isinstance(samples, list):
                raise ContractError(f"{summary_path}: samples is not a list")
            written_count = int(summary.get("written_count", len(samples)))
            if written_count != len(samples):
                raise ContractError(
                    f"{summary_path}: written_count={written_count}, "
                    f"samples={len(samples)}"
                )
            lmdb_path = lmdb_for_summary(summary_path, summary, v2_root)
            assert_no_symlink_ancestor(lmdb_path, v2_root)
            if path_is_within(lmdb_path, v1_root):
                raise ContractError(
                    f"{summary_path}: v2 shard resolves inside immutable v1 root"
                )

            local_indices: set[int] = set()
            for sample in samples:
                manifest_row = sample.get("manifest")
                if not isinstance(manifest_row, dict):
                    raise ContractError(
                        f"{summary_path}: v2 sample lacks embedded manifest"
                    )
                mol_id = manifest_mol_id(manifest_row)
                accepted = catalog.by_mol_id.get(mol_id)
                if accepted is None:
                    raise ContractError(
                        f"{summary_path}: rebuilt sample {mol_id} is not accepted"
                    )
                if accepted.split != split:
                    raise ContractError(
                        f"{summary_path}: {mol_id} split={split}, "
                        f"manifest={accepted.split}"
                    )
                if manifest_key(manifest_row) != accepted.key:
                    raise ContractError(
                        f"{summary_path}: accepted identity mismatch for {mol_id}"
                    )
                if mol_id in sources:
                    raise ContractError(f"duplicate rebuilt-v2 sample: {mol_id}")
                local_index = int(sample["local_index"])
                if local_index in local_indices:
                    raise ContractError(
                        f"{summary_path}: duplicate local_index={local_index}"
                    )
                local_indices.add(local_index)
                sources[mol_id] = {
                    "lmdb": lmdb_path,
                    "summary": summary_path.absolute(),
                    "local_index": local_index,
                    "source_version": SOURCE_V2,
                    "source_split": split,
                    "source_shard_index": int(summary.get("shard_index", -1)),
                    "source_stats": {
                        key: sample[key]
                        for key in (
                            "n_atoms",
                            "nao",
                            "n_electrons",
                            "trace_target",
                            "trace_error",
                            "trace_initial_density",
                            "trace_initial_error",
                            "storage_dtype",
                        )
                        if key in sample
                    },
                }
            if local_indices != set(range(len(samples))):
                raise ContractError(
                    f"{summary_path}: failure-free rebuilt local indices are not "
                    "contiguous from zero"
                )
    return sources


def index_record(manifest, source: dict[str, Any]) -> dict[str, Any]:
    stats = {
        "n_atoms": manifest.n_atoms,
        "nao": manifest.nao,
        "n_electrons": manifest.n_electrons,
    }
    stats.update(source.get("source_stats", {}))
    return {
        "key": manifest.key,
        "mol_id": manifest.mol_id,
        "manifest_split": manifest.split,
        "manifest_ordinal": manifest.ordinal,
        "lmdb": str(source["lmdb"]),
        "summary": str(source["summary"]),
        "local_index": int(source["local_index"]),
        "source_version": source["source_version"],
        "source_split": source["source_split"],
        "source_shard_index": int(source["source_shard_index"]),
        "manifest": compact_manifest(manifest.row),
        "stats": stats,
    }


def main() -> int:
    args = parse_args()
    if args.allow_incomplete and not args.dry_run:
        raise ContractError("--allow-incomplete is permitted only with --dry-run")

    manifest_dir = args.manifest_dir.resolve()
    v1_root = args.v1_root.resolve()
    v1_index_root = (
        args.v1_index_root.resolve()
        if args.v1_index_root is not None
        else v1_root / "_index"
    )
    v2_root = args.v2_root.absolute()
    out_root = args.out_index_root.absolute()
    if path_is_within(v2_root, v1_root) or path_is_within(v1_root, v2_root):
        raise ContractError("v1 and rebuilt-v2 roots must be disjoint")
    if path_is_within(out_root, v1_root):
        raise ContractError("index destination may not be inside immutable v1 root")
    if path_is_within(out_root, manifest_dir):
        raise ContractError("index destination may not be inside accepted manifest")

    catalog = load_manifest_catalog(manifest_dir)
    expected_total = args.expected_total or catalog.expected_total
    if expected_total != catalog.expected_total:
        raise ContractError(
            f"--expected-total={expected_total}, "
            f"accepted manifest={catalog.expected_total}"
        )

    v1_sources = load_v1_sources(
        v1_root=v1_root,
        v1_index_root=v1_index_root,
        catalog=catalog,
    )
    v2_sources = load_v2_sources(
        v2_root=v2_root,
        v1_root=v1_root,
        catalog=catalog,
    )

    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    missing: list[str] = []
    source_counts: Counter[str] = Counter()
    overlap_replacements = 0
    for split in SPLITS:
        for manifest in catalog.ordered[split]:
            source = v2_sources.get(manifest.mol_id)
            if source is not None:
                if manifest.mol_id in v1_sources:
                    overlap_replacements += 1
            else:
                source = v1_sources.get(manifest.mol_id)
            if source is None:
                missing.append(manifest.mol_id)
                continue
            selected[split].append(index_record(manifest, source))
            source_counts[source["source_version"]] += 1

    selected_total = sum(len(rows) for rows in selected.values())
    status = "complete" if not missing and selected_total == expected_total else "incomplete"
    summary = {
        "schema": "omol_electrolyte_full_v2_index_v1",
        "status": status,
        "manifest_dir": str(manifest_dir),
        "manifest_sha256": catalog.file_sha256,
        "manifest_counts": catalog.counts,
        "expected_total": expected_total,
        "v1_root": str(v1_root),
        "v1_index_root": str(v1_index_root),
        "v1_available_samples": len(v1_sources),
        "v2_root": str(v2_root),
        "v2_rebuilt_samples": len(v2_sources),
        "v2_overlap_replacements": overlap_replacements,
        "source_counts": dict(sorted(source_counts.items())),
        "index_counts": {split: len(selected[split]) for split in SPLITS},
        "indexed_total": selected_total,
        "missing_count": len(missing),
        "missing_preview": missing[:20],
        "precedence": [SOURCE_V2, SOURCE_V1],
        "sources_are_references": True,
        "v1_modified": False,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if status == "complete" or args.allow_incomplete else 2
    if status != "complete":
        raise ContractError(
            f"refusing to publish incomplete full-v2 index: "
            f"indexed={selected_total}/{expected_total}, missing={len(missing)}; "
            f"first={missing[:5]}"
        )

    staging: Path | None = None
    try:
        staging = new_staging_directory(out_root)
        hashes: dict[str, str] = {}
        for split in SPLITS:
            path = staging / f"{split}.index.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in selected[split]:
                    handle.write(json_dump_line(row))
                handle.flush()
            fsync_file(path)
            hashes[path.name] = sha256_file(path)
        summary["index_sha256"] = hashes
        write_json(staging / "summary.json", summary)
        fsync_file(staging / "summary.json")
        publish_staging_directory(staging, out_root)
        staging = None
    finally:
        discard_staging_directory(staging)

    print(json.dumps({**summary, "out_index_root": str(out_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
