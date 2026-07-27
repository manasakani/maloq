#!/usr/bin/env python3
"""Atomically build an immutable symlink view for a full-v2 index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from full_v2_common import (
    ContractError,
    SPLITS,
    discard_staging_directory,
    fsync_file,
    json_dump_line,
    load_index_rows,
    new_staging_directory,
    path_is_within,
    publish_staging_directory,
    sha256_file,
    source_shard_token,
    summary_for_lmdb,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out-view-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_root = args.index_root.resolve()
    manifest_dir = args.manifest_dir.resolve()
    out_root = args.out_view_root.absolute()
    if not index_root.is_dir():
        raise ContractError(f"index root does not exist: {index_root}")
    if not manifest_dir.is_dir():
        raise ContractError(f"manifest directory does not exist: {manifest_dir}")
    if out_root.exists() or out_root.is_symlink():
        raise ContractError(f"refusing to replace existing view: {out_root}")
    if path_is_within(out_root, index_root) or path_is_within(out_root, manifest_dir):
        raise ContractError("view destination may not be inside index or manifest source")

    source_summary_path = index_root / "summary.json"
    if not source_summary_path.is_file():
        raise ContractError(f"missing index summary: {source_summary_path}")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete":
        raise ContractError("refusing to build view from a non-complete index")
    rows_by_split = load_index_rows(index_root)

    unique_sources: dict[
        tuple[str, str, str, str], tuple[Path, Path, str]
    ] = {}
    source_counts: Counter[str] = Counter()
    for split in SPLITS:
        for row in rows_by_split[split]:
            source_version = str(row["source_version"])
            lmdb_path = Path(row["lmdb"]).absolute()
            summary_path = Path(row["summary"]).absolute()
            if not lmdb_path.is_dir() or not (lmdb_path / "data.mdb").is_file():
                raise ContractError(f"dangling LMDB reference: {lmdb_path}")
            if not summary_path.is_file():
                raise ContractError(f"dangling summary reference: {summary_path}")
            source_root = lmdb_path.parent.parent
            if path_is_within(out_root, source_root):
                raise ContractError(
                    f"view destination would modify a source root: {source_root}"
                )
            token = source_shard_token(source_version, lmdb_path)
            key = (split, source_version, str(lmdb_path.resolve()), str(summary_path.resolve()))
            unique_sources.setdefault(key, (lmdb_path, summary_path, token))
            source_counts[source_version] += 1

    plan = {
        "schema": "omol_electrolyte_full_v2_symlink_view_v1",
        "status": "view_built_not_fully_verified",
        "index_root": str(index_root),
        "index_summary_sha256": sha256_file(source_summary_path),
        "manifest_dir": str(manifest_dir),
        "out_view_root": str(out_root),
        "sample_counts": {
            split: len(rows_by_split[split]) for split in SPLITS
        },
        "source_sample_counts": dict(sorted(source_counts.items())),
        "unique_shards": len(unique_sources),
        "symlink_only": True,
        "source_data_modified": False,
        "complete_marker_created": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    staging: Path | None = None
    try:
        staging = new_staging_directory(out_root)
        (staging / "_index").mkdir()
        for split in SPLITS:
            (staging / split).mkdir()
        manifests_dir = staging / "manifests"
        manifests_dir.mkdir()
        os.symlink(
            str(manifest_dir),
            manifests_dir / manifest_dir.name,
            target_is_directory=True,
        )

        mapping: dict[tuple[str, str, str, str], tuple[Path, Path]] = {}
        for key, (lmdb_path, summary_path, token) in sorted(
            unique_sources.items(),
            key=lambda item: item[0],
        ):
            split = key[0]
            lmdb_link = staging / split / token
            summary_link = summary_for_lmdb(lmdb_link)
            if lmdb_link.exists() or lmdb_link.is_symlink():
                raise ContractError(f"generated LMDB link collision: {lmdb_link}")
            if summary_link.exists() or summary_link.is_symlink():
                raise ContractError(f"generated summary link collision: {summary_link}")
            os.symlink(str(lmdb_path.resolve()), lmdb_link, target_is_directory=True)
            os.symlink(str(summary_path.resolve()), summary_link)
            mapping[key] = (
                out_root / split / lmdb_link.name,
                out_root / split / summary_link.name,
            )

        index_hashes: dict[str, str] = {}
        for split in SPLITS:
            destination = staging / "_index" / f"{split}.index.jsonl"
            with destination.open("w", encoding="utf-8") as handle:
                for row in rows_by_split[split]:
                    original_lmdb = Path(row["lmdb"]).absolute()
                    original_summary = Path(row["summary"]).absolute()
                    key = (
                        split,
                        str(row["source_version"]),
                        str(original_lmdb.resolve()),
                        str(original_summary.resolve()),
                    )
                    view_lmdb, view_summary = mapping[key]
                    rewritten: dict[str, Any] = dict(row)
                    rewritten["source_lmdb"] = str(original_lmdb)
                    rewritten["source_summary"] = str(original_summary)
                    rewritten["lmdb"] = str(view_lmdb)
                    rewritten["summary"] = str(view_summary)
                    handle.write(json_dump_line(rewritten))
                handle.flush()
            fsync_file(destination)
            index_hashes[destination.name] = sha256_file(destination)
            os.symlink(
                str(Path("_index") / destination.name),
                staging / destination.name,
            )

        view_index_summary = dict(source_summary)
        view_index_summary.update(
            {
                "schema": "omol_electrolyte_full_v2_view_index_v1",
                "source_index_root": str(index_root),
                "view_root": str(out_root),
                "view_index_sha256": index_hashes,
            }
        )
        write_json(staging / "_index" / "summary.json", view_index_summary)
        fsync_file(staging / "_index" / "summary.json")
        plan["view_index_sha256"] = index_hashes
        write_json(staging / "summary.json", plan)
        fsync_file(staging / "summary.json")

        for split in SPLITS:
            index_link = staging / f"{split}.index.jsonl"
            if not index_link.is_symlink():
                raise ContractError(f"expected index symlink was not created: {index_link}")
        publish_staging_directory(staging, out_root)
        staging = None
    finally:
        discard_staging_directory(staging)

    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
