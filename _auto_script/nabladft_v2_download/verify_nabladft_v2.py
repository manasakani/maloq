#!/usr/bin/env python3
"""Verify native NablaDFT v2 Hamiltonian SQLite databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import apsw
import numpy as np


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase


ARTIFACTS = {
    "train_2k": {
        "filename": "train_2k.db",
        "bytes": 15_118_426_112,
        "etag": "068975858201ae70743d3d4427c55e47-1803",
        "official_registry_name": "dataset_train_tiny",
    },
    "train_10k": {
        "filename": "train_10k.db",
        "bytes": 68_388_278_272,
        "etag": "41f03a745d88afa8689f7a41e0afb54f-8153",
        "official_registry_name": "dataset_train_medium",
    },
    "test_2k_conformers": {
        "filename": "test_2k_conformers.db",
        "bytes": 3_099_738_112,
        "etag": "0b9a02f0e3d1dee44bb4d40353845a1c-370",
        "official_registry_name": "dataset_test_conformations_tiny",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sample_summary(database: HamiltonianDatabase, index: int) -> dict:
    (
        atomic_numbers,
        positions,
        energy,
        forces,
        hamiltonian,
        overlap,
        core,
        moses_id,
        conformer_id,
    ) = database[index]
    matrices = {
        "hamiltonian": hamiltonian,
        "overlap": overlap,
        "core": core,
    }
    matrix_shapes = {name: list(value.shape) for name, value in matrices.items()}
    if len(set(map(tuple, matrix_shapes.values()))) != 1:
        raise RuntimeError(f"row {index}: matrix shapes disagree: {matrix_shapes}")
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise RuntimeError(
            f"row {index}: expected square matrices, got {hamiltonian.shape}"
        )
    arrays = {
        "positions": positions,
        "energy": energy,
        "forces": forces,
        **matrices,
    }
    non_finite = {
        name: int(np.size(value) - np.count_nonzero(np.isfinite(value)))
        for name, value in arrays.items()
    }
    if any(non_finite.values()):
        raise RuntimeError(f"row {index}: non-finite values: {non_finite}")
    return {
        "index": index,
        "moses_id": int(moses_id),
        "conformer_id": int(conformer_id),
        "atoms": int(len(atomic_numbers)),
        "elements": sorted({int(value) for value in atomic_numbers}),
        "matrix_shapes": matrix_shapes,
        "non_finite": non_finite,
    }


def verify_file(path: Path, expected: dict, include_hash: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected["bytes"]:
        raise RuntimeError(
            f"{path.name}: bytes={actual_bytes}, expected={expected['bytes']}"
        )

    connection = apsw.Connection(
        str(path),
        flags=apsw.SQLITE_OPEN_READONLY,
    )
    tables = {
        row[0]
        for row in connection.cursor().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    required_tables = {
        "metadata",
        "dataset_ids",
        "data",
        "basisset",
        "nuclear_charges",
    }
    missing_tables = required_tables - tables
    if missing_tables:
        raise RuntimeError(
            f"{path.name}: missing SQLite tables {sorted(missing_tables)}"
        )

    database = HamiltonianDatabase(str(path))
    rows = len(database)
    if rows <= 0:
        raise RuntimeError(f"{path.name}: metadata declares no rows")
    elements = sorted({int(value) for value in database.Z})
    orbitals = {
        str(element): [int(value) for value in database.get_orbitals(element)]
        for element in elements
    }
    samples = [
        sample_summary(database, 0),
        sample_summary(database, rows - 1),
    ]
    report = {
        "status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "path": str(path),
        "bytes": actual_bytes,
        "rows": rows,
        "official_registry_name": expected["official_registry_name"],
        "official_etag": expected["etag"],
        "format": "NablaDFT v2 native Hamiltonian SQLite",
        "position_unit": "bohr",
        "matrix_convention": "native_nabladft_psi4",
        "elements": elements,
        "orbital_basis_l_values": orbitals,
        "samples": samples,
    }
    if include_hash:
        report["sha256"] = sha256_file(path)
    return report


def write_report(path: Path, report: dict) -> None:
    target = Path(f"{path}.verification.json")
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        choices=tuple(ARTIFACTS),
        required=True,
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Also compute a local SHA-256 digest (slow for train_10k).",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write an atomic <database>.verification.json sidecar.",
    )
    parser.add_argument(
        "--print-rows",
        action="store_true",
        help="For one artifact, print only the database row count.",
    )
    args = parser.parse_args()

    selected = tuple(dict.fromkeys(args.artifact))
    if args.print_rows and len(selected) != 1:
        raise SystemExit("--print-rows requires exactly one --artifact")

    reports = {}
    for artifact in selected:
        expected = ARTIFACTS[artifact]
        path = args.root / expected["filename"]
        report = verify_file(path, expected, include_hash=args.hash)
        if args.write_report:
            write_report(path, report)
        reports[artifact] = report

    if args.print_rows:
        print(reports[selected[0]]["rows"])
    else:
        print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
