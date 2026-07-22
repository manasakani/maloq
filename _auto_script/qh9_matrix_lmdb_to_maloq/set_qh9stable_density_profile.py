#!/usr/bin/env python3
"""Restrict an existing combined QH9 ASE DB to the density-delta contract.

Older conversions physically stored final H together with D/D0/H0/S and
advertised both Hamiltonian and density delta learning. This migration changes
only ASE metadata: density remains the sole target, while H0 remains available
as QHFlow3 conditioning. The legacy final-H blob is left untouched and hidden
from ``ASEAtomsData.available_properties`` so the migration is fast and
recoverable from the emitted metadata audit files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ase.db import connect


DENSITY_PROPERTY_UNITS = {
    "energy": "Hartree",
    "forces": "Hartree/Angstrom",
    "density_matrix": "dimensionless",
    "initial_density_matrix": "dimensionless",
    "initial_hamiltonian": "Hartree",
    "overlap": "dimensionless",
}
REQUIRED_PHYSICAL_PROPERTIES = set(DENSITY_PROPERTY_UNITS) | {"hamiltonian"}


def density_metadata(metadata: dict) -> dict:
    updated = dict(metadata)
    updated.pop("raw_hamiltonian_convention", None)
    updated.pop("hamiltonian_storage_convention", None)
    updated.update(
        {
            "_property_unit_dict": dict(DENSITY_PROPERTY_UNITS),
            "target_properties": ["density_matrix"],
            "loss_targets_supported": ["density_matrix"],
            "delta_learning_supported": True,
            "delta_baseline_properties": {
                "density_matrix": "initial_density_matrix",
            },
            "conditioning_properties": [
                "initial_density_matrix",
                "initial_hamiltonian",
                "overlap",
            ],
            "delta_learning_scope": "density_only",
            "legacy_unadvertised_properties": ["hamiltonian"],
            "metadata_profile_updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )
    return updated


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the metadata update. Without this flag, perform a dry run.",
    )
    args = parser.parse_args()

    db_path = args.db.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    database = connect(str(db_path), use_lock_file=False)
    if database.count() <= 0:
        raise ValueError(f"Database is empty: {db_path}")
    before = dict(database.metadata)
    if before.get("dataset_name") != "QH9StableMatrices":
        raise ValueError(
            f"Expected QH9StableMatrices metadata, got "
            f"{before.get('dataset_name')!r}"
        )
    first_row_properties = set(database.get(1).data)
    missing = REQUIRED_PHYSICAL_PROPERTIES - first_row_properties
    if missing:
        raise ValueError(f"First row is missing physical properties: {sorted(missing)}")

    after = density_metadata(before)
    audit_dir = args.audit_dir.resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    stem = db_path.stem
    before_path = audit_dir / f"{stem}.metadata-before.json"
    if not before_path.exists():
        write_json(before_path, before)
    write_json(audit_dir / f"{stem}.metadata-proposed.json", after)

    if not args.apply:
        print(f"Dry run passed for {db_path}; add --apply to update metadata")
        return

    database.metadata = after
    observed = dict(connect(str(db_path), use_lock_file=False).metadata)
    for key in (
        "_property_unit_dict",
        "target_properties",
        "loss_targets_supported",
        "delta_baseline_properties",
        "delta_learning_scope",
    ):
        if observed.get(key) != after[key]:
            raise RuntimeError(f"Metadata verification failed for {key}")
    write_json(audit_dir / f"{stem}.metadata-after.json", observed)
    print(f"Applied density-only metadata profile to {db_path}")


if __name__ == "__main__":
    main()
