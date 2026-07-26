#!/usr/bin/env python3
"""Verify the published OMol_CSH HDF5 files without loading matrices."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py


EXPECTED = {
    "omol_csh_58k_train.h5": {
        "bytes": 276_516_996_009,
        "entries": 57_559,
        "etag": "803ced5b6c77f3ca40e7f9b8710f874a-8241",
    },
    "omol_csh_5k_test_all.h5": {
        "bytes": 33_350_917_699,
        "entries": 4_986,
        "etag": "5ce268166f25f3ab09f362c5f745a9fc-3976",
    },
    "omol_csh_1k_test_common.h5": {
        "bytes": 8_203_349_480,
        "entries": 1_008,
        "etag": "44888394300ed4d457251815d05c7b1f-978",
    },
}


def object_summary(obj: h5py.Group | h5py.Dataset) -> dict:
    if isinstance(obj, h5py.Dataset):
        return {
            "kind": "dataset",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    return {
        "kind": "group",
        "child_count": len(obj),
        "children": sorted(obj.keys())[:32],
    }


def jsonable(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def verify_file(path: Path, expected: dict) -> dict:
    actual_bytes = path.stat().st_size
    if actual_bytes != expected["bytes"]:
        raise RuntimeError(
            f"{path.name}: bytes={actual_bytes}, expected={expected['bytes']}"
        )

    with h5py.File(path, "r") as handle:
        entry_paths = []

        def collect_entry(name, obj):
            if isinstance(obj, h5py.Group) and {
                "coords",
                "elements",
                "fock",
            }.issubset(obj.keys()):
                entry_paths.append(name)

        handle.visititems(collect_entry)
        entry_paths.sort()
        entry_count = len(entry_paths)
        if entry_count != expected["entries"]:
            raise RuntimeError(
                f"{path.name}: Hamiltonian entries={entry_count}, "
                f"expected={expected['entries']}"
            )
        samples = {
            key: object_summary(handle[key])
            for key in [entry_paths[0], entry_paths[-1]]
        }
        attributes = {
            key: jsonable(value)
            for key, value in handle.attrs.items()
        }

    return {
        "bytes": actual_bytes,
        "entries": entry_count,
        "etag_at_download": expected["etag"],
        "first_key": entry_paths[0],
        "last_key": entry_paths[-1],
        "sample_objects": samples,
        "root_attributes": attributes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    files = {}
    for name, expected in EXPECTED.items():
        path = args.root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = verify_file(path, expected)

    report = {
        "status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "format": "OMol_CSH HDF5",
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "total_entries": sum(item["entries"] for item in files.values()),
        "files": files,
    }
    report_path = args.root / "verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
