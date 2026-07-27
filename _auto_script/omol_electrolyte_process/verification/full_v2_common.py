#!/usr/bin/env python3
"""Shared, read-only source helpers for the OMol electrolyte full-v2 tools."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


SPLITS = ("train", "val", "test")
SOURCE_V1 = "v1_reference"
SOURCE_V2 = "v2_rebuilt"
SOURCE_VERSIONS = (SOURCE_V1, SOURCE_V2)

EXPECTED_SCHEMA = {
    "dataset": "omol_unsolvated_electrolyte_raw_density",
    "targets": {
        "density_matrix": True,
        "overlap": True,
        "initial_density_matrix": True,
    },
    "basis": "def2-tzvpd",
    "convention": "e3nn",
    "xc": "omol-orca-raw",
    "initial_density": "sad",
    "initial_density_charge_correction": "trace-scale",
    "initial_density_convention": "orca_raw_density_e3nn",
    "overlap_source": "pyscf-orca-raw-density-sign",
    "pyscf_overlap_deprecated": False,
    "storage_dtype": "float32",
}


class ContractError(RuntimeError):
    """A source or generated artifact violates the full-v2 contract."""


@dataclass(frozen=True)
class ManifestRecord:
    split: str
    ordinal: int
    key: str
    mol_id: str
    row: dict[str, Any]

    @property
    def n_atoms(self) -> int:
        if "nsites" in self.row:
            return int(self.row["nsites"])
        return len(self.row["atomic_numbers"])

    @property
    def nao(self) -> int:
        return int(self.row["n_basis_orca"])

    @property
    def n_electrons(self) -> int:
        if "num_electrons_meta" in self.row:
            return int(self.row["num_electrons_meta"])
        return (
            sum(int(value) for value in self.row["atomic_numbers"])
            - int(self.row.get("charge", 0))
        )


@dataclass
class ManifestCatalog:
    by_mol_id: dict[str, ManifestRecord]
    ordered: dict[str, list[ManifestRecord]]
    counts: dict[str, int]
    expected_total: int
    file_sha256: dict[str, str]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{line_number}: JSONL row is not an object")
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_rel(source: str) -> str:
    if source.endswith("/orca.tar.zst"):
        return source[: -len("/orca.tar.zst")]
    path = Path(source)
    if path.name.startswith("orca."):
        return str(path.parent)
    return source.rstrip("/")


def manifest_mol_id(row: dict[str, Any]) -> str:
    value = row.get("configuration_id") or row.get("mol_id")
    if not value:
        raise ContractError("manifest row has no configuration_id/mol_id")
    return str(value)


def manifest_key(row: dict[str, Any]) -> str:
    configuration_id = manifest_mol_id(row)
    property_id = str(row.get("property_id") or "")
    source = source_rel(str(row.get("source") or ""))
    if not property_id or not source:
        raise ContractError(
            f"{configuration_id}: manifest row lacks property_id or source"
        )
    return f"{configuration_id}|{property_id}|{source}"


def load_manifest_catalog(manifest_dir: Path) -> ManifestCatalog:
    manifest_dir = manifest_dir.resolve()
    if not manifest_dir.is_dir():
        raise ContractError(f"manifest directory does not exist: {manifest_dir}")

    by_mol_id: dict[str, ManifestRecord] = {}
    ordered: dict[str, list[ManifestRecord]] = {}
    counts: dict[str, int] = {}
    file_sha256: dict[str, str] = {}

    for split in SPLITS:
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise ContractError(f"missing accepted manifest: {path}")
        file_sha256[path.name] = sha256_file(path)
        records: list[ManifestRecord] = []
        for ordinal, row in enumerate(read_jsonl(path)):
            row_split = str(row.get("split", split))
            if row_split != split:
                raise ContractError(
                    f"{path}: row {ordinal} declares split={row_split!r}"
                )
            mol_id = manifest_mol_id(row)
            key = manifest_key(row)
            if mol_id in by_mol_id:
                other = by_mol_id[mol_id]
                raise ContractError(
                    f"duplicate configuration_id {mol_id}: "
                    f"{other.split}:{other.ordinal}, {split}:{ordinal}"
                )
            record = ManifestRecord(
                split=split,
                ordinal=ordinal,
                key=key,
                mol_id=mol_id,
                row=row,
            )
            by_mol_id[mol_id] = record
            records.append(record)
        ordered[split] = records
        counts[split] = len(records)

    summary_path = manifest_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        file_sha256[summary_path.name] = sha256_file(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        declared_counts = summary.get("by_split", {})
        for split in SPLITS:
            if split in declared_counts and int(declared_counts[split]) != counts[split]:
                raise ContractError(
                    f"manifest summary {split}={declared_counts[split]}, "
                    f"scanned={counts[split]}"
                )
        declared_total = summary.get("counts", {}).get("accepted")
        if declared_total is not None and int(declared_total) != len(by_mol_id):
            raise ContractError(
                f"manifest summary accepted={declared_total}, "
                f"scanned={len(by_mol_id)}"
            )

    return ManifestCatalog(
        by_mol_id=by_mol_id,
        ordered=ordered,
        counts=counts,
        expected_total=len(by_mol_id),
        file_sha256=file_sha256,
    )


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def path_is_lexically_within(path: Path, parent: Path) -> bool:
    """Check path placement without resolving the final symlink target."""
    try:
        path.absolute().relative_to(parent.absolute())
        return True
    except ValueError:
        return False


def resolve_lmdb_reference(raw: str | Path, root: Path, split: str) -> Path:
    raw_path = Path(raw)
    candidates = [
        raw_path,
        root / split / raw_path.name,
        root / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "data.mdb").is_file():
            return candidate.absolute()
    raise ContractError(
        f"cannot resolve LMDB {raw_path} below {root} for split={split}"
    )


def resolve_summary_reference(raw: str | Path, root: Path, split: str) -> Path:
    raw_path = Path(raw)
    candidates = [
        raw_path,
        root / split / raw_path.name,
        root / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    raise ContractError(
        f"cannot resolve summary {raw_path} below {root} for split={split}"
    )


def lmdb_for_summary(summary_path: Path, summary: dict[str, Any], root: Path) -> Path:
    name = summary_path.name
    if not name.endswith(".summary.json"):
        raise ContractError(f"unexpected shard summary name: {summary_path}")
    sibling = summary_path.with_name(name[: -len(".summary.json")] + ".lmdb")
    if sibling.is_dir() and (sibling / "data.mdb").is_file():
        return sibling.absolute()
    raw = summary.get("lmdb")
    if not raw:
        raise ContractError(f"{summary_path}: no sibling LMDB and no lmdb field")
    return resolve_lmdb_reference(str(raw), root, summary_path.parent.name)


def summary_for_lmdb(lmdb_path: Path) -> Path:
    if lmdb_path.name.endswith(".lmdb"):
        return lmdb_path.with_name(
            lmdb_path.name[: -len(".lmdb")] + ".summary.json"
        )
    return lmdb_path.with_suffix(".summary.json")


def json_dump_line(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def new_staging_directory(destination: Path) -> Path:
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ContractError(
            f"refusing to replace existing destination: {destination}"
        )
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=str(destination.parent),
        )
    )


def publish_staging_directory(staging: Path, destination: Path) -> None:
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ContractError(
            f"refusing to replace existing destination: {destination}"
        )
    fsync_directory(staging)
    os.replace(staging, destination)
    fsync_directory(destination.parent)


def discard_staging_directory(staging: Path | None) -> None:
    if staging is not None and staging.exists():
        shutil.rmtree(staging)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def compact_manifest(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "configuration_id",
        "property_id",
        "source",
        "source_group",
        "split",
        "charge",
        "spin",
        "multiplicity",
        "unrestricted",
        "nsites",
        "n_basis_orca",
        "num_electrons_meta",
    )
    return {key: row[key] for key in keys if key in row}


def source_shard_token(source_version: str, lmdb_path: Path) -> str:
    if source_version not in SOURCE_VERSIONS:
        raise ContractError(f"unknown source_version={source_version!r}")
    digest = hashlib.sha256(str(lmdb_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{source_version}__{digest}__{lmdb_path.name}"


def assert_no_symlink_ancestor(path: Path, stop: Path) -> None:
    """Reject a source whose path from stop downward traverses a symlink."""
    path = path.absolute()
    stop = stop.absolute()
    try:
        relative = path.relative_to(stop)
    except ValueError as exc:
        raise ContractError(f"{path} is outside declared root {stop}") from exc
    current = stop
    if current.is_symlink():
        raise ContractError(f"declared root is a symlink: {current}")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ContractError(f"v2 rebuilt source must be real, not symlink: {current}")


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def load_index_rows(index_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = index_root / f"{split}.index.jsonl"
        if not path.is_file():
            raise ContractError(f"missing index file: {path}")
        result[split] = list(read_jsonl(path))
    return result


def iter_unique_shards(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> Iterable[tuple[str, str, Path, Path]]:
    seen: set[tuple[str, str]] = set()
    for split in SPLITS:
        for row in rows_by_split[split]:
            lmdb_path = Path(row["lmdb"]).absolute()
            summary_path = Path(row["summary"]).absolute()
            key = (str(lmdb_path), str(summary_path))
            if key in seen:
                continue
            seen.add(key)
            yield split, str(row["source_version"]), lmdb_path, summary_path
