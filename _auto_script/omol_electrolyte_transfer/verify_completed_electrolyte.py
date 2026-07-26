#!/usr/bin/env python3
"""Verify the completed OMol25 electrolyte MALOQ LMDB snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import lmdb


EXPECTED = {
    "train": {"samples": 43_252, "shards": 10_813, "bytes": 2_364_694_531_929},
    "val": {"samples": 5_028, "shards": 1_257, "bytes": 271_495_935_094},
    "test": {"samples": 9_620, "shards": 2_405, "bytes": 524_981_775_073},
}
EXPECTED_DATASET_BYTES = 3_161_180_105_435
INDEX_SHA256 = {
    "summary.json": "1dc9d1370813b8ccbb321f26ae4cab565f3b9f757ac4391ecfac60d4a3b83c11",
    "train.index.jsonl": "236bb0b57e8cf90719ad9ebd111208bb54876b103dd96d84cde764fee2f4ab4d",
    "val.index.jsonl": "c48efb5261f835d5542fbbbf2d1e2ebb3e51fcc70a9527ec141d355c0484e4fb",
    "test.index.jsonl": "5bfdeb1feb57f9f1d7c368862c2fef68258ae997e0d5c9ecd487ca4bfe4017e5",
}
MANIFEST_SHA256 = {
    "summary.json": "17a119ed99d27e381be557ee581e7ad211fd6380dde3baadbae40923a4411934",
    "train.jsonl": "ecaf4e86a4e4b34149c2c91895153188b4d32bbbe640f0dde2eed7b2dd6c1d8b",
    "val.jsonl": "2943dc74e90335e9f3a487c403356573e2115142300260960badbeedbbd91ab3",
    "test.jsonl": "31014434fae781cf94311d5e0d2267f0f29a4b6c16151ce8c87f07dca7a7254c",
    "rejected.jsonl": "cbffed041b545e1c1fd17ed9f45534de9568fdb0731777303a49576b167f4e2d",
}
MANIFEST_DIR = (
    "unsolvated_electrolytes_all_supported_elements_85_5_10_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hashes(root: Path, expected: dict[str, str]) -> dict[str, dict]:
    result = {}
    for name, expected_digest in expected.items():
        path = root / name
        actual = sha256(path)
        if actual != expected_digest:
            raise RuntimeError(f"{path}: SHA256={actual}, expected={expected_digest}")
        result[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return result


def load_index(root: Path, split: str) -> tuple[int, dict[Path, Path]]:
    samples = 0
    shards = {}
    with (root / "_index" / f"{split}.index.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            samples += 1
            lmdb_path = Path(row["lmdb"])
            summary_path = Path(row["summary"])
            shards[Path(split) / lmdb_path.name] = (
                Path(split) / summary_path.name
            )
    return samples, shards


def sample_lmdb(path: Path) -> dict:
    environment = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=True,
    )
    try:
        with environment.begin() as transaction:
            stats = transaction.stat()
            cursor = transaction.cursor()
            first = cursor.first()
            if not first:
                raise RuntimeError(f"empty LMDB: {path}")
            first_key = cursor.key().decode(errors="replace")
        return {"entries": stats["entries"], "first_key": first_key}
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    index_hashes = check_hashes(root / "_index", INDEX_SHA256)
    manifest_hashes = check_hashes(
        root / "manifests" / MANIFEST_DIR,
        MANIFEST_SHA256,
    )

    split_reports = {}
    total_bytes = 0
    for split, expected in EXPECTED.items():
        samples, shards = load_index(root, split)
        if samples != expected["samples"]:
            raise RuntimeError(
                f"{split}: index samples={samples}, expected={expected['samples']}"
            )
        if len(shards) != expected["shards"]:
            raise RuntimeError(
                f"{split}: index shards={len(shards)}, expected={expected['shards']}"
            )

        split_bytes = 0
        missing = []
        for lmdb_relative, summary_relative in shards.items():
            paths = (
                root / lmdb_relative / "data.mdb",
                root / lmdb_relative / "lock.mdb",
                root / summary_relative,
            )
            for path in paths:
                if not path.is_file():
                    missing.append(str(path))
                else:
                    split_bytes += path.stat().st_size
        if missing:
            raise RuntimeError(
                f"{split}: {len(missing)} completed files are missing; "
                f"first={missing[0]}"
            )
        if split_bytes != expected["bytes"]:
            raise RuntimeError(
                f"{split}: bytes={split_bytes}, expected={expected['bytes']}"
            )

        first_lmdb = root / sorted(shards)[0]
        split_reports[split] = {
            "samples": samples,
            "shards": len(shards),
            "bytes": split_bytes,
            "sample_lmdb": sample_lmdb(first_lmdb),
        }
        total_bytes += split_bytes

    for name in ("summary.json", "train.shard_lengths.json", "val.shard_lengths.json"):
        path = root / name
        if path.is_file():
            total_bytes += path.stat().st_size
    if total_bytes != EXPECTED_DATASET_BYTES:
        raise RuntimeError(
            f"completed dataset bytes={total_bytes}, "
            f"expected={EXPECTED_DATASET_BYTES}"
        )

    report = {
        "status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "schema": "OMol25 electrolyte density/overlap/initial-density LMDB",
        "storage_dtype": "float32",
        "basis": "def2-tzvpd",
        "matrix_convention": "MALOQ/e3nn",
        "dataset_bytes": total_bytes,
        "splits": split_reports,
        "index_files": index_hashes,
        "manifest_files": manifest_hashes,
    }
    report_path = root / "verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
