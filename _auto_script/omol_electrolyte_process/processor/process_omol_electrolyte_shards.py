#!/usr/bin/env python3
"""Safely materialize selected OMol electrolyte density LMDB shards on SC26.

The numerical conversion is delegated to a provenance-pinned hybrid snapshot:
corrected working-tree conventions plus the HEAD molecule serialization schema.
This wrapper adds SC26-specific path mapping, explicit shard selection, a bounded
per-process parquet cache, strong resume validation, per-shard locks, and
recoverable atomic publication.

No shard is selected implicitly.  A run must provide ``--shard-list`` and/or
one or more ``--shard`` selectors.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROCESSOR_VERSION = "sc26-omol-electrolyte-hybrid-v2"
PROCESSOR_ROOT = Path(__file__).resolve().parent
STAGE_ROOT = PROCESSOR_ROOT.parent
SNAPSHOT_ROOT = STAGE_ROOT / "source_snapshot"
WORKING_ROOT = SNAPSHOT_ROOT / "working"
ML_DFT_ROOT = WORKING_ROOT / "ml_dft"
DFT_DATASET_ROOT = SNAPSHOT_ROOT / "hybrid" / "dft-dataset"

EXPECTED_SOURCE_HASHES = {
    "working/ml_dft/scripts/build_omol_density_shards.py":
        "345715eb111cbf3c96833a485f69b5988323f877d28f5e05811e1100af603036",
    "working/ml_dft/scripts/build_omol_density_pilot.py":
        "e9dc402bc38341951eb2348b3df63d5ec944b3cabc13d63f8953b6c9e43a8cfc",
    "working/dft-dataset/dft_dataset/conventions.py":
        "12d43caaa71d674e9c7d4a5b7475f2c0eb090582c59661f8efd342b30f0f8e60",
    "hybrid/dft-dataset/dft_dataset/conventions.py":
        "12d43caaa71d674e9c7d4a5b7475f2c0eb090582c59661f8efd342b30f0f8e60",
    "hybrid/dft-dataset/dft_dataset/lmdb_dataset.py":
        "66c05b52d40df85db18e06477c8e8df562a4dc0b2f230695d77a05eed58937e8",
    "head/dft-dataset/dft_dataset/molecule.py":
        "d9795864d2aa575844e6583855983a71b5a4f0e21b508e1ea6822413692b6a8d",
    "hybrid/dft-dataset/dft_dataset/molecule.py":
        "d9795864d2aa575844e6583855983a71b5a4f0e21b508e1ea6822413692b6a8d",
    "hybrid/dft-dataset/dft_dataset/storage_formats.py":
        "b3b7a23a39588b2095ca59882312bd260b1ff8e9e1fe6fe05a0ef4fd8005b169",
}

SOURCE_PROVENANCE = {
    "ml_dft_head": "cea0dbd9a80227cefea60afee07ae7c93616d668",
    "dft_dataset_head": "9606b07bd5b2cbc631dd944ab807d04092ac2366",
    "ml_dft_dirty_patch_sha256":
        "0b2b003faeea7b80948005f9d7a5d7ef2ba9d9a50d001cea69d020c873a93a40",
    "dft_dataset_dirty_patch_sha256":
        "4b9d1d51b68208fb441a9511ab24f126203bed7649e3829ea3a27d1c8b087b59",
    "runtime_source_composition": (
        "ml_dft-working-builders+dft_dataset-working-corrected-conventions+"
        "dft_dataset-head-molecule-schema"
    ),
    "be_raw_density_correction":
        "def2-tzvpd-be-signed-2p-3p-swap-from-captured-working-source",
    "molecule_serialization":
        "dft_dataset-head-9606b07-no-num_ecp_electrons-key",
}

SHARD_SELECTOR_RE = re.compile(
    r"^(?P<split>[A-Za-z0-9_.-]+):"
    r"(?P<start>[0-9]+)"
    r"(?:-(?P<end>[0-9]+)(?::(?P<step>[0-9]+))?)?$"
)
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

REQUIRED_SAMPLE_KEYS = frozenset({
    "_basis_info",
    "_packed",
    "atomic_numbers",
    "charge",
    "density_matrix_packed",
    "initial_density_matrix_packed",
    "mol_id",
    "overlap_packed",
    "positions",
    "source",
    "spin",
    "unrestricted",
    "xc",
})
FORBIDDEN_SAMPLE_KEYS = frozenset({"num_ecp_electrons"})


@dataclass(frozen=True, order=True)
class ShardSpec:
    split: str
    index: int


@dataclass
class ValidationResult:
    valid: bool
    reasons: list[str]
    split: str
    shard_index: int
    lmdb_path: str
    summary_path: str
    expected_count: int
    actual_count: int | None = None
    has_be: bool = False


@dataclass(frozen=True)
class Contract:
    basis: str
    initial_density: str
    initial_density_charge_correction: str
    overlap_source: str
    storage_dtype: str
    max_trace_error: float
    max_initial_trace_error: float
    require_corrected_be: bool


@dataclass(frozen=True)
class PathMapping:
    density_root: Path
    density_prefix: PurePosixPath
    parquet_root: Path
    parquet_prefix: PurePosixPath


_SNAPSHOT_MODULES: dict[str, Any] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_snapshot() -> dict[str, str]:
    actual: dict[str, str] = {}
    failures = []
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        path = SNAPSHOT_ROOT / relative
        if not path.is_file():
            failures.append(f"missing source: {path}")
            continue
        digest = sha256(path)
        actual[relative] = digest
        if digest != expected:
            failures.append(
                f"source hash mismatch: {relative}: {digest} != {expected}"
            )
    if failures:
        raise RuntimeError("\n".join(failures))
    return actual


def load_snapshot_modules() -> dict[str, Any]:
    global _SNAPSHOT_MODULES
    if _SNAPSHOT_MODULES is not None:
        return _SNAPSHOT_MODULES

    verify_source_snapshot()
    for path in (ML_DFT_ROOT, DFT_DATASET_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    import lmdb  # type: ignore
    import numpy as np  # type: ignore
    from dft_dataset.lmdb_dataset import LMDBDataset  # type: ignore
    from dft_dataset.molecule import Molecule  # type: ignore
    from dft_dataset import conventions  # type: ignore
    from scripts import build_omol_density_pilot as pilot  # type: ignore

    expected_module_files = {
        "conventions": (Path(conventions.__file__), DFT_DATASET_ROOT / "dft_dataset/conventions.py"),
        "lmdb_dataset": (Path(sys.modules[LMDBDataset.__module__].__file__), DFT_DATASET_ROOT / "dft_dataset/lmdb_dataset.py"),
        "molecule": (Path(sys.modules[Molecule.__module__].__file__), DFT_DATASET_ROOT / "dft_dataset/molecule.py"),
        "pilot": (Path(pilot.__file__), ML_DFT_ROOT / "scripts/build_omol_density_pilot.py"),
    }
    for label, (actual_path, expected_path) in expected_module_files.items():
        if actual_path.resolve() != expected_path.resolve():
            raise RuntimeError(
                f"captured module collision for {label}: "
                f"{actual_path.resolve()} != {expected_path.resolve()}"
            )
    _SNAPSHOT_MODULES = {
        "lmdb": lmdb,
        "np": np,
        "LMDBDataset": LMDBDataset,
        "Molecule": Molecule,
        "conventions": conventions,
        "pilot": pilot,
    }
    return _SNAPSHOT_MODULES


def parse_selector(selector: str) -> list[ShardSpec]:
    value = selector.strip()
    match = SHARD_SELECTOR_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"invalid shard selector {selector!r}; expected "
            "SPLIT:INDEX or SPLIT:START-END[:STEP]"
        )
    split = match.group("split")
    if split not in SPLIT_ORDER:
        raise ValueError(
            f"unsupported split {split!r}; expected one of "
            f"{', '.join(SPLIT_ORDER)}"
        )
    start = int(match.group("start"))
    end_raw = match.group("end")
    if end_raw is None:
        return [ShardSpec(split, start)]
    end = int(end_raw)
    step = int(match.group("step") or 1)
    if end < start:
        raise ValueError(f"selector end {end} is smaller than start {start}")
    if step < 1:
        raise ValueError("selector step must be positive")
    return [ShardSpec(split, index) for index in range(start, end + 1, step)]


def load_shard_specs(
    shard_list: Path | None,
    inline_selectors: Iterable[str],
) -> list[ShardSpec]:
    selectors = list(inline_selectors)
    if shard_list is not None:
        with shard_list.open() as handle:
            for line_number, raw in enumerate(handle, start=1):
                value = raw.split("#", 1)[0].strip()
                if not value:
                    continue
                if ":" not in value:
                    fields = value.replace(",", " ").split()
                    if len(fields) != 2:
                        raise ValueError(
                            f"{shard_list}:{line_number}: expected "
                            "SPLIT:INDEX/RANGE or two columns"
                        )
                    value = f"{fields[0]}:{fields[1]}"
                selectors.append(value)
    if not selectors:
        raise ValueError("no shards selected; use --shard-list and/or --shard")

    specs = []
    for selector in selectors:
        specs.extend(parse_selector(selector))
    duplicates = [
        spec for spec, count in _counts(specs).items() if count > 1
    ]
    if duplicates:
        first = sorted(duplicates)[0]
        raise ValueError(
            f"duplicate shard selection: {first.split}:{first.index}"
        )
    return sorted(specs, key=lambda item: (SPLIT_ORDER[item.split], item.index))


def _counts(items: Iterable[ShardSpec]) -> dict[ShardSpec, int]:
    result: dict[ShardSpec, int] = {}
    for item in items:
        result[item] = result.get(item, 0) + 1
    return result


def _abs_float_or_inf(value: Any) -> float:
    try:
        result = abs(float(value))
    except (TypeError, ValueError, OverflowError):
        return float("inf")
    if not math.isfinite(result):
        return float("inf")
    return result


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen_ids = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            mol_id = str(row.get("configuration_id") or row.get("property_id"))
            if not mol_id or mol_id == "None":
                raise ValueError(f"{path}:{line_number}: missing molecule ID")
            if mol_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate ID {mol_id}")
            seen_ids.add(mol_id)
            rows.append(row)
    return rows


class ManifestCache:
    def __init__(self, manifest_dir: Path, shard_size: int):
        self.manifest_dir = manifest_dir.resolve()
        self.shard_size = shard_size
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._hashes: dict[str, str] = {}

    def rows(self, split: str) -> list[dict[str, Any]]:
        if split not in self._rows:
            path = self.manifest_dir / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"manifest split not found: {path}")
            self._rows[split] = read_manifest(path)
            self._hashes[split] = sha256(path)
        return self._rows[split]

    def shard(self, spec: ShardSpec) -> list[dict[str, Any]]:
        rows = self.rows(spec.split)
        start = spec.index * self.shard_size
        if start >= len(rows):
            max_index = (len(rows) - 1) // self.shard_size
            raise IndexError(
                f"{spec.split}:{spec.index} is outside manifest; "
                f"maximum shard index is {max_index}"
            )
        return rows[start : start + self.shard_size]

    def split_hash(self, split: str) -> str:
        self.rows(split)
        return self._hashes[split]


def _resolve_recorded_path(
    root: Path,
    prefix: PurePosixPath,
    recorded: str,
) -> Path:
    relative = PurePosixPath(recorded)
    if relative.is_absolute():
        raise ValueError(f"manifest path must be relative, got {recorded!r}")
    if ".." in relative.parts:
        raise ValueError(f"manifest path escapes its root: {recorded!r}")
    if prefix.parts and prefix != PurePosixPath("."):
        try:
            relative = relative.relative_to(prefix)
        except ValueError as exc:
            raise ValueError(
                f"manifest path {recorded!r} does not start with "
                f"configured prefix {str(prefix)!r}"
            ) from exc
    root_resolved = root.resolve()
    candidate = (root_resolved / Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"resolved path escaped root: {candidate}")
    return candidate


class ParquetCache:
    """Small per-process LRU cache for full row tables."""

    def __init__(self, max_files: int, row_columns: list[str], pandas_module: Any):
        if max_files < 1:
            raise ValueError("--parquet-cache-files must be at least 1")
        self.max_files = max_files
        self.row_columns = row_columns
        self.pandas = pandas_module
        self._tables: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path) -> Any:
        if path in self._tables:
            table = self._tables.pop(path)
            self._tables[path] = table
            return table
        if not path.is_file():
            raise FileNotFoundError(f"parquet file not found: {path}")
        table = self.pandas.read_parquet(
            path,
            engine="pyarrow",
            columns=self.row_columns,
        )
        self._tables[path] = table
        while len(self._tables) > self.max_files:
            self._tables.popitem(last=False)
        return table


def shard_paths(root: Path, spec: ShardSpec) -> tuple[Path, Path]:
    base = root / spec.split / f"shard_{spec.index:06d}"
    return base.with_suffix(".lmdb"), base.with_suffix(".summary.json")


def _mol_id(row: dict[str, Any]) -> str:
    return str(row.get("configuration_id") or row.get("property_id"))


def _has_be(items: list[dict[str, Any]]) -> bool:
    return any(4 in [int(value) for value in item.get("elements", [])] for item in items)


def expected_core_schema(
    contract: Contract,
    spec: ShardSpec,
) -> dict[str, Any]:
    return {
        "dataset": "omol_unsolvated_electrolyte_raw_density",
        "targets": {
            "density_matrix": True,
            "overlap": True,
            "initial_density_matrix": contract.initial_density != "none",
        },
        "basis": contract.basis,
        "convention": "e3nn",
        "xc": "omol-orca-raw",
        "initial_density": contract.initial_density,
        "initial_density_charge_correction":
            contract.initial_density_charge_correction,
        "overlap_source": contract.overlap_source,
        "storage_dtype": contract.storage_dtype,
        "split": spec.split,
        "shard_index": spec.index,
    }


def validate_shard(
    *,
    lmdb_path: Path,
    summary_path: Path,
    spec: ShardSpec,
    items: list[dict[str, Any]],
    contract: Contract,
    require_current_provenance: bool,
) -> ValidationResult:
    reasons: list[str] = []
    expected_count = len(items)
    has_be = _has_be(items)
    actual_count: int | None = None

    if not lmdb_path.is_dir():
        reasons.append(f"LMDB directory missing: {lmdb_path}")
    else:
        data_path = lmdb_path / "data.mdb"
        if not data_path.is_file() or data_path.stat().st_size == 0:
            reasons.append(f"nonempty data.mdb missing: {data_path}")
        lock_path = lmdb_path / "lock.mdb"
        if not lock_path.is_file():
            reasons.append(f"lock.mdb missing: {lock_path}")

    summary: dict[str, Any] | None = None
    if not summary_path.is_file():
        reasons.append(f"summary missing: {summary_path}")
    else:
        try:
            loaded_summary = json.loads(summary_path.read_text())
            if isinstance(loaded_summary, dict):
                summary = loaded_summary
            else:
                reasons.append("summary root is not a dictionary")
        except Exception as exc:
            reasons.append(f"summary is unreadable: {type(exc).__name__}: {exc}")

    expected_ids = [_mol_id(item) for item in items]
    if summary is not None:
        for field, expected in (
            ("split", spec.split),
            ("shard_index", spec.index),
            ("manifest_count", expected_count),
            ("written_count", expected_count),
            ("failure_count", 0),
        ):
            if summary.get(field) != expected:
                reasons.append(
                    f"summary {field}={summary.get(field)!r}, expected {expected!r}"
                )
        samples = summary.get("samples")
        if not isinstance(samples, list):
            reasons.append("summary samples is not a list")
        else:
            for index, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    reasons.append(f"summary sample {index} is not a dictionary")
                    samples[index] = {"manifest": {}}
                    continue
                manifest_item = sample.get("manifest")
                if not isinstance(manifest_item, dict):
                    reasons.append(f"summary sample {index} manifest is not a dictionary")
                    samples[index] = {**sample, "manifest": {}}
            actual_ids = [
                _mol_id(sample.get("manifest", {})) for sample in samples
            ]
            if actual_ids != expected_ids:
                reasons.append("summary sample IDs do not match manifest shard")
            for index, sample in enumerate(samples):
                if sample.get("local_index") != index:
                    reasons.append(
                        f"summary sample {index} has local_index="
                        f"{sample.get('local_index')!r}"
                    )
                trace_error = _abs_float_or_inf(
                    sample.get("trace_error", float("inf"))
                )
                if trace_error > contract.max_trace_error:
                    reasons.append(
                        f"sample {index} trace error {trace_error:.8g} exceeds "
                        f"{contract.max_trace_error:.8g}"
                    )
                if contract.initial_density != "none":
                    initial_error = _abs_float_or_inf(
                        sample.get("trace_initial_error", float("inf"))
                    )
                    if initial_error > contract.max_initial_trace_error:
                        reasons.append(
                            f"sample {index} initial trace error "
                            f"{initial_error:.8g} exceeds "
                            f"{contract.max_initial_trace_error:.8g}"
                        )
        provenance = summary.get("processor_provenance")
        if require_current_provenance and not _current_provenance(provenance):
            reasons.append("summary lacks current processor provenance")
        if (
            has_be
            and contract.require_corrected_be
            and not _corrected_be_provenance(provenance)
        ):
            reasons.append("Be shard lacks corrected raw-density provenance")

    if lmdb_path.is_dir() and (lmdb_path / "data.mdb").is_file():
        try:
            modules = load_snapshot_modules()
            lmdb = modules["lmdb"]
            environment = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=1,
                subdir=True,
            )
            try:
                with environment.begin() as transaction:
                    raw_len = transaction.get(b"__len__")
                    if raw_len is None:
                        reasons.append("LMDB __len__ key is missing")
                    else:
                        actual_count = int.from_bytes(raw_len, "big")
                        if actual_count != expected_count:
                            reasons.append(
                                f"LMDB length {actual_count}, expected {expected_count}"
                            )
                    if transaction.get(b"__format__") != b"pickle":
                        reasons.append("LMDB format is not pickle")
                    raw_schema = transaction.get(b"__schema__")
                    if raw_schema is None:
                        reasons.append("LMDB schema key is missing")
                    else:
                        schema = pickle.loads(raw_schema)
                        for field, expected in expected_core_schema(
                            contract, spec
                        ).items():
                            if schema.get(field) != expected:
                                reasons.append(
                                    f"schema {field}={schema.get(field)!r}, "
                                    f"expected {expected!r}"
                                )
                        if (
                            require_current_provenance
                            and not _current_provenance(
                                schema.get("processor_provenance")
                            )
                        ):
                            reasons.append(
                                "LMDB schema lacks current processor provenance"
                            )
                        if (
                            has_be
                            and contract.require_corrected_be
                            and not _corrected_be_provenance(
                                schema.get("processor_provenance")
                            )
                        ):
                            reasons.append("LMDB schema lacks corrected Be provenance")
                    for index in range(expected_count):
                        key = index.to_bytes(4, "big")
                        raw_sample = transaction.get(key)
                        if raw_sample is None:
                            reasons.append(f"LMDB sample key {index} is missing")
                            continue
                        try:
                            sample = pickle.loads(raw_sample)
                        except Exception as exc:
                            reasons.append(
                                f"LMDB sample {index} cannot be unpickled: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            continue
                        if not isinstance(sample, dict):
                            reasons.append(
                                f"LMDB sample {index} is not a dictionary"
                            )
                            continue
                        missing_keys = REQUIRED_SAMPLE_KEYS - sample.keys()
                        if missing_keys:
                            reasons.append(
                                f"LMDB sample {index} lacks required keys: "
                                f"{sorted(missing_keys)}"
                            )
                        forbidden_keys = FORBIDDEN_SAMPLE_KEYS & sample.keys()
                        if forbidden_keys:
                            reasons.append(
                                f"LMDB sample {index} has incompatible keys: "
                                f"{sorted(forbidden_keys)}"
                            )
                        if str(sample.get("mol_id")) != expected_ids[index]:
                            reasons.append(
                                f"LMDB sample {index} mol_id="
                                f"{sample.get('mol_id')!r}, expected "
                                f"{expected_ids[index]!r}"
                            )
                        try:
                            _validate_stored_payload(sample, items[index], contract)
                        except Exception as exc:
                            reasons.append(
                                f"LMDB sample {index} payload is invalid: "
                                f"{type(exc).__name__}: {exc}"
                            )
                    expected_entries = expected_count + 3
                    entries = transaction.stat()["entries"]
                    if entries != expected_entries:
                        reasons.append(
                            f"LMDB entries={entries}, expected {expected_entries}"
                        )
            finally:
                environment.close()
        except Exception as exc:
            reasons.append(f"LMDB validation failed: {type(exc).__name__}: {exc}")

    return ValidationResult(
        valid=not reasons,
        reasons=reasons,
        split=spec.split,
        shard_index=spec.index,
        lmdb_path=str(lmdb_path),
        summary_path=str(summary_path),
        expected_count=expected_count,
        actual_count=actual_count,
        has_be=has_be,
    )


def _current_provenance(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("processor_version") == PROCESSOR_VERSION
        and value.get("source_hashes") == EXPECTED_SOURCE_HASHES
    )


def _corrected_be_provenance(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("be_raw_density_correction")
        == SOURCE_PROVENANCE["be_raw_density_correction"]
    )


def processor_provenance(
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "processor_version": PROCESSOR_VERSION,
        "processor_path": str(Path(__file__).resolve()),
        "processor_sha256": sha256(Path(__file__).resolve()),
        "source_snapshot": str(SNAPSHOT_ROOT.resolve()),
        "source_hashes": EXPECTED_SOURCE_HASHES,
        **SOURCE_PROVENANCE,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
    }


def _selected_sample(
    item: dict[str, Any],
    mapping: PathMapping,
    pilot: Any,
) -> tuple[Any, Path]:
    density_path = _resolve_recorded_path(
        mapping.density_root,
        mapping.density_prefix,
        str(item["density_path"]),
    )
    parquet_path = _resolve_recorded_path(
        mapping.parquet_root,
        mapping.parquet_prefix,
        str(item["parquet_file"]),
    )
    if not density_path.is_file():
        raise FileNotFoundError(f"density source not found: {density_path}")
    selected = pilot.SelectedSample(
        parquet_file=str(parquet_path),
        row_in_file=int(item["row_in_file"]),
        mol_id=_mol_id(item),
        n_atoms=int(item["nsites"]),
        charge=int(item.get("charge", 0)),
        spin=int(item.get("spin", 0)),
        formula=item.get("formula"),
        n_basis_orca=int(item["n_basis_orca"]),
        density_path=str(density_path),
    )
    return selected, parquet_path


def _validate_row_and_molecule_identity(
    row: dict[str, Any],
    molecule: Any,
    item: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    expected_id = _mol_id(item)
    row_id = str(row.get("configuration_id") or row.get("property_id") or "")
    if row_id != expected_id:
        raise ValueError(f"parquet row ID {row_id!r} != manifest ID {expected_id!r}")
    expected_z = [int(value) for value in item["atomic_numbers"]]
    row_z = [int(value) for value in row["atomic_numbers"]]
    if row_z != expected_z:
        raise ValueError(f"{expected_id}: parquet atomic_numbers != manifest")
    molecule_z = [int(value) for value in molecule.atomic_numbers]
    if molecule_z != expected_z:
        raise ValueError(f"{expected_id}: molecule atomic_numbers != manifest")
    expected_atoms = int(item["nsites"])
    if int(stats["n_atoms"]) != expected_atoms or molecule.num_atoms != expected_atoms:
        raise ValueError(f"{expected_id}: atom count != manifest {expected_atoms}")
    if int(molecule.charge) != int(item.get("charge", 0)):
        raise ValueError(f"{expected_id}: molecule charge != manifest")
    if int(molecule.spin) != int(item.get("spin", 0)):
        raise ValueError(f"{expected_id}: molecule spin != manifest")


def _validate_matrix_fields(molecule: Any, stats: dict[str, Any]) -> None:
    modules = load_snapshot_modules()
    np = modules["np"]
    nao = int(stats["nao"])
    for field in ("density_matrix", "overlap", "initial_density_matrix"):
        matrix = getattr(molecule, field, None)
        if matrix is None:
            raise ValueError(f"converted molecule lacks {field}")
        array = np.asarray(matrix)
        if array.shape != (nao, nao):
            raise ValueError(
                f"{field} shape {array.shape}, expected {(nao, nao)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{field} contains non-finite values")
        if not np.allclose(array, array.T, atol=2.0e-6, rtol=0.0):
            maximum = float(np.max(np.abs(array - array.T)))
            raise ValueError(f"{field} is not symmetric; max error={maximum}")



def _validate_stored_payload(
    sample: dict[str, Any],
    item: dict[str, Any],
    contract: Contract,
) -> None:
    modules = load_snapshot_modules()
    np = modules["np"]
    expected_z = [int(value) for value in item["atomic_numbers"]]
    expected_atoms = int(item["nsites"])
    if int(sample["num_atoms"]) != expected_atoms:
        raise ValueError("stored num_atoms does not match manifest")
    stored_z = np.frombuffer(sample["atomic_numbers"], dtype=np.int32).tolist()
    if stored_z != expected_z:
        raise ValueError("stored atomic_numbers do not match manifest")
    positions = np.frombuffer(sample["positions"], dtype=np.float64)
    if positions.size != expected_atoms * 3:
        raise ValueError(f"stored positions size {positions.size} is invalid")
    positions = positions.reshape(expected_atoms, 3)
    if not np.isfinite(positions).all():
        raise ValueError("stored positions contain non-finite values")
    if int(sample.get("charge", 0)) != int(item.get("charge", 0)):
        raise ValueError("stored charge does not match manifest")
    if int(sample.get("spin", 0)) != int(item.get("spin", 0)):
        raise ValueError("stored spin does not match manifest")
    matrices: dict[str, Any] = {}
    nao: int | None = None
    for field in ("density_matrix", "overlap", "initial_density_matrix"):
        field_dtype = np.dtype(sample[f"{field}_dtype"])
        if field_dtype != np.dtype(contract.storage_dtype):
            raise ValueError(f"stored {field} dtype {field_dtype} is invalid")
        field_nao = int(sample[f"{field}_nao"])
        if nao is not None and field_nao != nao:
            raise ValueError(f"stored {field} nao {field_nao} != {nao}")
        nao = field_nao
        packed = sample[f"{field}_packed"]
        values = np.frombuffer(packed, dtype=field_dtype)
        expected_size = field_nao * (field_nao + 1) // 2
        if values.size != expected_size:
            raise ValueError(f"stored {field} size {values.size} != {expected_size}")
        if not np.isfinite(values).all():
            raise ValueError(f"stored {field} contains non-finite values")
        matrices[field] = values
    assert nao is not None
    diagonal = np.arange(nao, dtype=np.int64)
    diagonal = diagonal * nao - diagonal * (diagonal - 1) // 2
    density = matrices["density_matrix"]
    overlap = matrices["overlap"]
    target_trace = float(
        2.0 * np.sum(density * overlap, dtype=np.float64)
        - np.sum(density[diagonal] * overlap[diagonal], dtype=np.float64)
    )
    expected_electrons = sum(expected_z) - int(item.get("charge", 0))
    if (
        _abs_float_or_inf(target_trace - expected_electrons)
        > contract.max_trace_error
    ):
        raise ValueError("stored target density trace exceeds tolerance")
    initial = matrices["initial_density_matrix"]
    initial_trace = float(
        2.0 * np.sum(initial * overlap, dtype=np.float64)
        - np.sum(initial[diagonal] * overlap[diagonal], dtype=np.float64)
    )
    if (
        _abs_float_or_inf(initial_trace - expected_electrons)
        > contract.max_initial_trace_error
    ):
        raise ValueError("stored initial density trace exceeds tolerance")
def build_shard(
    *,
    spec: ShardSpec,
    items: list[dict[str, Any]],
    temp_lmdb: Path,
    temp_summary: Path,
    mapping: PathMapping,
    contract: Contract,
    parquet_cache: ParquetCache,
    manifest_path: Path,
    manifest_sha256: str,
    lmdb_map_size_gb: float,
    require_density_dtype: str,
) -> ValidationResult:
    modules = load_snapshot_modules()
    pilot = modules["pilot"]
    LMDBDataset = modules["LMDBDataset"]
    molecules = []
    sample_summaries = []
    started = time.perf_counter()

    for local_index, item in enumerate(items):
        selected, parquet_path = _selected_sample(item, mapping, pilot)
        table = parquet_cache.get(parquet_path)
        row_index = selected.row_in_file
        if row_index < 0 or row_index >= len(table):
            raise IndexError(
                f"{parquet_path}: row {row_index} outside [0, {len(table)})"
            )
        row = table.iloc[row_index].to_dict()
        molecule, stats = pilot._build_molecule(
            row,
            selected=selected,
            basis=contract.basis,
            initial_density=contract.initial_density,
            initial_density_charge_correction=
                contract.initial_density_charge_correction,
            storage_dtype=contract.storage_dtype,
            overlap_source=contract.overlap_source,
            orca_bin=Path("/nonexistent/orca-not-used-by-fast-path"),
            orca_work_dir=None,
            orca_timeout_seconds=1800.0,
            keep_orca_overlap_files=False,
            orca_wait_for_completion=False,
            required_density_dtype=(
                None if require_density_dtype == "any"
                else require_density_dtype
            ),
        )
        _validate_row_and_molecule_identity(row, molecule, item, stats)
        _validate_matrix_fields(molecule, stats)
        trace_error = _abs_float_or_inf(stats["trace_error"])
        if trace_error > contract.max_trace_error:
            raise ValueError(
                f"{selected.mol_id}: abs target trace error "
                f"{trace_error:.8g} exceeds {contract.max_trace_error:.8g}"
            )
        initial_error = _abs_float_or_inf(stats["trace_initial_error"])
        if initial_error > contract.max_initial_trace_error:
            raise ValueError(
                f"{selected.mol_id}: abs initial trace error "
                f"{initial_error:.8g} exceeds "
                f"{contract.max_initial_trace_error:.8g}"
            )
        molecules.append(molecule)
        sample_summaries.append(
            {
                "local_index": local_index,
                "manifest": item,
                **stats,
            }
        )

    provenance = processor_provenance(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    schema = {
        **expected_core_schema(contract, spec),
        "initial_density_convention": pilot._initial_density_convention(
            contract.initial_density,
            contract.overlap_source,
        ),
        "pyscf_overlap_deprecated": False,
        "source_density_dtype_requirement": require_density_dtype,
        "processor_provenance": provenance,
    }
    temp_lmdb.parent.mkdir(parents=True, exist_ok=True)
    count = LMDBDataset.write(
        molecules,
        str(temp_lmdb),
        packed=True,
        schema=schema,
        format="pickle",
        map_size_per_sample_mb=(
            lmdb_map_size_gb * 1024.0 / max(len(molecules), 1)
        ),
    )
    summary = {
        "split": spec.split,
        "shard_index": spec.index,
        "lmdb": str(shard_paths(Path("."), spec)[0]),
        "manifest_count": len(items),
        "written_count": count,
        "failure_count": 0,
        "failures": [],
        "seconds": time.perf_counter() - started,
        "samples": sample_summaries,
        "processor_provenance": provenance,
    }
    temp_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return validate_shard(
        lmdb_path=temp_lmdb,
        summary_path=temp_summary,
        spec=spec,
        items=items,
        contract=contract,
        require_current_provenance=True,
    )


class ShardLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "ShardLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(
                self.handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            self.handle.close()
            raise RuntimeError(f"shard is locked by another process: {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": os.uname().nodename,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )
        self.handle.flush()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def publish_shard(
    *,
    temp_lmdb: Path,
    temp_summary: Path,
    final_lmdb: Path,
    final_summary: Path,
    backup_root: Path,
) -> dict[str, str]:
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_lmdb = backup_root / final_lmdb.name
    backup_summary = backup_root / final_summary.name
    if backup_lmdb.exists() or backup_summary.exists():
        raise FileExistsError(f"backup target already exists under {backup_root}")

    moved_lmdb = False
    moved_summary = False
    installed_lmdb = False
    installed_summary = False
    try:
        if final_lmdb.exists():
            os.replace(final_lmdb, backup_lmdb)
            moved_lmdb = True
        if final_summary.exists():
            os.replace(final_summary, backup_summary)
            moved_summary = True
        final_lmdb.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_lmdb, final_lmdb)
        installed_lmdb = True
        os.replace(temp_summary, final_summary)
        installed_summary = True
    except Exception:
        if installed_summary and final_summary.exists():
            os.replace(final_summary, temp_summary)
        if installed_lmdb and final_lmdb.exists():
            os.replace(final_lmdb, temp_lmdb)
        if moved_lmdb and backup_lmdb.exists():
            os.replace(backup_lmdb, final_lmdb)
        if moved_summary and backup_summary.exists():
            os.replace(backup_summary, final_summary)
        raise
    return {
        "backup_lmdb": str(backup_lmdb) if moved_lmdb else "",
        "backup_summary": str(backup_summary) if moved_summary else "",
    }


def rollback_published_shard(
    *,
    temp_lmdb: Path,
    temp_summary: Path,
    final_lmdb: Path,
    final_summary: Path,
    publish_record: dict[str, str],
) -> None:
    if final_summary.exists():
        os.replace(final_summary, temp_summary)
    if final_lmdb.exists():
        os.replace(final_lmdb, temp_lmdb)
    for key, final_path in (
        ("backup_lmdb", final_lmdb),
        ("backup_summary", final_summary),
    ):
        value = publish_record.get(key)
        if value:
            backup_path = Path(value)
            if backup_path.exists():
                os.replace(backup_path, final_path)


def contract_from_args(args: argparse.Namespace) -> Contract:
    return Contract(
        basis=args.basis,
        initial_density=args.initial_density,
        initial_density_charge_correction=
            args.initial_density_charge_correction,
        overlap_source=args.overlap_source,
        storage_dtype=args.storage_dtype,
        max_trace_error=args.max_trace_error,
        max_initial_trace_error=args.max_initial_trace_error,
        require_corrected_be=not args.allow_legacy_be,
    )


def run_command(args: argparse.Namespace) -> int:
    modules = load_snapshot_modules()
    pilot = modules["pilot"]
    specs = load_shard_specs(args.shard_list, args.shard)
    manifest = ManifestCache(args.manifest_dir, args.shard_size)
    contract = contract_from_args(args)
    mapping = PathMapping(
        density_root=args.density_root,
        density_prefix=PurePosixPath(args.density_prefix),
        parquet_root=args.parquet_root,
        parquet_prefix=PurePosixPath(args.parquet_prefix),
    )
    cache = ParquetCache(
        args.parquet_cache_files,
        list(pilot.ROW_COLUMNS),
        pilot.pd,
    )
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{os.uname().nodename}-p{os.getpid()}"
    )
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(
            "--run-id must be a 1-128 character alphanumeric slug using ._-"
        )
    results: list[dict[str, Any]] = []
    failures = 0

    for spec in specs:
        items = manifest.shard(spec)
        final_lmdb, final_summary = shard_paths(args.out, spec)
        lock_path = args.out / "_locks" / (
            f"{spec.split}-shard_{spec.index:06d}.lock"
        )
        with ShardLock(lock_path):
            existing = validate_shard(
                lmdb_path=final_lmdb,
                summary_path=final_summary,
                spec=spec,
                items=items,
                contract=contract,
                require_current_provenance=False,
            )
            if existing.valid and not args.replace_valid_existing:
                results.append(
                    {
                        "split": spec.split,
                        "shard_index": spec.index,
                        "status": "skipped_valid",
                        "validation": asdict(existing),
                    }
                )
                print(
                    f"[skip-valid] {spec.split}:{spec.index}",
                    flush=True,
                )
                continue
            exists_at_all = final_lmdb.exists() or final_summary.exists()
            if (
                exists_at_all
                and not existing.valid
                and not args.replace_invalid_existing
            ):
                failures += 1
                results.append(
                    {
                        "split": spec.split,
                        "shard_index": spec.index,
                        "status": "refused_invalid_existing",
                        "validation": asdict(existing),
                    }
                )
                print(
                    f"[refuse-invalid] {spec.split}:{spec.index}: "
                    + "; ".join(existing.reasons[:3]),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if (
                existing.valid
                and args.replace_valid_existing is False
            ):
                raise AssertionError("valid shard should already have been skipped")

            incomplete_root = args.out / "_incomplete" / run_id / spec.split
            temp_lmdb = incomplete_root / (
                f"shard_{spec.index:06d}.lmdb.incomplete"
            )
            temp_summary = incomplete_root / (
                f"shard_{spec.index:06d}.summary.json.incomplete"
            )
            if temp_lmdb.exists() or temp_summary.exists():
                failures += 1
                results.append(
                    {
                        "split": spec.split,
                        "shard_index": spec.index,
                        "status": "refused_stale_incomplete",
                        "temp_lmdb": str(temp_lmdb),
                        "temp_summary": str(temp_summary),
                    }
                )
                continue

            print(f"[build] {spec.split}:{spec.index}", flush=True)
            try:
                built = build_shard(
                    spec=spec,
                    items=items,
                    temp_lmdb=temp_lmdb,
                    temp_summary=temp_summary,
                    mapping=mapping,
                    contract=contract,
                    parquet_cache=cache,
                    manifest_path=(
                        args.manifest_dir / f"{spec.split}.jsonl"
                    ),
                    manifest_sha256=manifest.split_hash(spec.split),
                    lmdb_map_size_gb=args.lmdb_map_size_gb,
                    require_density_dtype=args.require_density_dtype,
                )
                if not built.valid:
                    raise RuntimeError(
                        "temporary shard failed validation: "
                        + "; ".join(built.reasons)
                    )
                backup = publish_shard(
                    temp_lmdb=temp_lmdb,
                    temp_summary=temp_summary,
                    final_lmdb=final_lmdb,
                    final_summary=final_summary,
                    backup_root=(
                        args.out / "_replaced" / run_id / spec.split
                    ),
                )
                final_check = validate_shard(
                    lmdb_path=final_lmdb,
                    summary_path=final_summary,
                    spec=spec,
                    items=items,
                    contract=contract,
                    require_current_provenance=True,
                )
                if not final_check.valid:
                    rollback_published_shard(
                        temp_lmdb=temp_lmdb,
                        temp_summary=temp_summary,
                        final_lmdb=final_lmdb,
                        final_summary=final_summary,
                        publish_record=backup,
                    )
                    raise RuntimeError(
                        "published shard failed validation: "
                        + "; ".join(final_check.reasons)
                    )
                results.append(
                    {
                        "split": spec.split,
                        "shard_index": spec.index,
                        "status": "published",
                        "validation": asdict(final_check),
                        **backup,
                    }
                )
            except Exception as exc:
                failures += 1
                failure_path = incomplete_root / (
                    f"shard_{spec.index:06d}.failure.json"
                )
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(
                    json.dumps(
                        {
                            "split": spec.split,
                            "shard_index": spec.index,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "temp_lmdb": str(temp_lmdb),
                            "temp_summary": str(temp_summary),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                results.append(
                    {
                        "split": spec.split,
                        "shard_index": spec.index,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "failure_report": str(failure_path),
                    }
                )
                print(
                    f"[failed] {spec.split}:{spec.index}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    report = {
        "processor_version": PROCESSOR_VERSION,
        "run_id": run_id,
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "selected_shards": len(specs),
        "failures": failures,
        "contract": asdict(contract),
        "source_provenance": SOURCE_PROVENANCE,
        "source_hashes": EXPECTED_SOURCE_HASHES,
        "results": results,
    }
    report_path = args.out / "_runs" / f"{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            "run_report": str(report_path),
            "selected_shards": len(specs),
            "failures": failures,
            "status_counts": _status_counts(results),
        },
        indent=2,
        sort_keys=True,
    ))
    return 1 if failures else 0


def _status_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def validate_command(args: argparse.Namespace) -> int:
    load_snapshot_modules()
    specs = load_shard_specs(args.shard_list, args.shard)
    manifest = ManifestCache(args.manifest_dir, args.shard_size)
    contract = contract_from_args(args)
    reports = []
    for spec in specs:
        lmdb_path, summary_path = shard_paths(args.out, spec)
        reports.append(
            validate_shard(
                lmdb_path=lmdb_path,
                summary_path=summary_path,
                spec=spec,
                items=manifest.shard(spec),
                contract=contract,
                require_current_provenance=args.require_current_provenance,
            )
        )
    invalid = [report for report in reports if not report.valid]
    print(json.dumps(
        {
            "checked": len(reports),
            "valid": len(reports) - len(invalid),
            "invalid": len(invalid),
            "reports": [asdict(report) for report in reports],
        },
        indent=2,
        sort_keys=True,
    ))
    return 1 if invalid else 0


def source_info_command(_args: argparse.Namespace) -> int:
    hashes = verify_source_snapshot()
    print(json.dumps(
        {
            "processor_version": PROCESSOR_VERSION,
            "snapshot_root": str(SNAPSHOT_ROOT),
            "source_hashes": hashes,
            "source_provenance": SOURCE_PROVENANCE,
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


def self_test_command(_args: argparse.Namespace) -> int:
    modules = load_snapshot_modules()
    np = modules["np"]
    lmdb = modules["lmdb"]
    conventions = modules["conventions"]
    from pyscf import gto  # type: ignore

    specs = parse_selector("test:1781-1797:8")
    if specs != [
        ShardSpec("test", 1781),
        ShardSpec("test", 1789),
        ShardSpec("test", 1797),
    ]:
        raise AssertionError("shard selector parser self-test failed")

    be = gto.M(
        atom="Be 0 0 0",
        basis="def2-tzvpd",
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )
    indices, signs = conventions.build_orca_raw_density_layout_transform(
        be,
        "def2-tzvpd",
        src_convention="pyscf",
        dst_convention="e3nn",
    )
    if len(indices) != be.nao or sorted(indices.tolist()) != list(range(be.nao)):
        raise AssertionError("Be raw-density transform is not a permutation")
    if not np.isin(signs, (-1.0, 1.0)).all() or not (signs < 0).any():
        raise AssertionError("Be raw-density transform lacks signed correction")

    with tempfile.TemporaryDirectory(prefix="sc26_omol_processor_selftest_") as tmp:
        root = Path(tmp)
        spec = ShardSpec("train", 0)
        items = [
            {
                "configuration_id": "synthetic-0",
                "property_id": "synthetic-property-0",
                "elements": [4],
                "atomic_numbers": [4],
                "nsites": 1,
                "charge": 0,
                "spin": 0,
            }
        ]
        contract = Contract(
            basis="def2-tzvpd",
            initial_density="sad",
            initial_density_charge_correction="trace-scale",
            overlap_source="pyscf-orca-raw-density-sign",
            storage_dtype="float32",
            max_trace_error=0.05,
            max_initial_trace_error=1.0e-5,
            require_corrected_be=True,
        )
        lmdb_path, summary_path = shard_paths(root, spec)
        lmdb_path.parent.mkdir(parents=True)
        environment = lmdb.open(
            str(lmdb_path),
            map_size=1 << 20,
            subdir=True,
        )
        provenance = {
            "processor_version": PROCESSOR_VERSION,
            "source_hashes": EXPECTED_SOURCE_HASHES,
            "be_raw_density_correction":
                SOURCE_PROVENANCE["be_raw_density_correction"],
        }
        schema = {
            **expected_core_schema(contract, spec),
            "processor_provenance": provenance,
        }
        with environment.begin(write=True) as transaction:
            transaction.put(
                (0).to_bytes(4, "big"),
                pickle.dumps({
                    **dict.fromkeys(REQUIRED_SAMPLE_KEYS),
                    "mol_id": "synthetic-0",
                    "_packed": True,
                    "atomic_numbers": np.array([4], dtype=np.int32).tobytes(),
                    "positions": np.zeros((1, 3), dtype=np.float64).tobytes(),
                    "charge": 0,
                    "spin": 0,
                    "num_atoms": 1,
                    "density_matrix_dtype": "float32",
                    "density_matrix_nao": 1,
                    "density_matrix_packed": np.array([4.0], dtype=np.float32).tobytes(),
                    "overlap_dtype": "float32",
                    "overlap_nao": 1,
                    "overlap_packed": np.array([1.0], dtype=np.float32).tobytes(),
                    "initial_density_matrix_dtype": "float32",
                    "initial_density_matrix_nao": 1,
                    "initial_density_matrix_packed": np.array([4.0], dtype=np.float32).tobytes(),
                }),
            )
            transaction.put(b"__len__", (1).to_bytes(4, "big"))
            transaction.put(b"__format__", b"pickle")
            transaction.put(b"__schema__", pickle.dumps(schema))
        environment.close()
        summary_path.write_text(json.dumps({
            "split": "train",
            "shard_index": 0,
            "manifest_count": 1,
            "written_count": 1,
            "failure_count": 0,
            "samples": [{
                "local_index": 0,
                "manifest": items[0],
                "trace_error": 0.0,
                "trace_initial_error": 0.0,
            }],
            "processor_provenance": provenance,
        }))
        result = validate_shard(
            lmdb_path=lmdb_path,
            summary_path=summary_path,
            spec=spec,
            items=items,
            contract=contract,
            require_current_provenance=True,
        )
        if not result.valid:
            raise AssertionError(
                "synthetic strong-resume validation failed: "
                + "; ".join(result.reasons)
            )

    print(json.dumps(
        {
            "status": "ok",
            "processor_version": PROCESSOR_VERSION,
            "source_hashes_verified": len(EXPECTED_SOURCE_HASHES),
            "be_nao": int(be.nao),
            "be_negative_signs": int((signs < 0).sum()),
            "selector_count": len(specs),
            "synthetic_resume_validation": "passed",
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shard-list",
        type=Path,
        help=(
            "Text file containing SPLIT:INDEX or "
            "SPLIT:START-END[:STEP] selectors."
        ),
    )
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        help=(
            "Inline explicit selector; repeat as needed. "
            "Examples: train:10566, test:1781-2493:8."
        ),
    )


def add_contract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=4)
    parser.add_argument("--basis", choices=("def2-tzvpd",), default="def2-tzvpd")
    parser.add_argument(
        "--overlap-source",
        choices=("pyscf-orca-raw-density-sign",),
        default="pyscf-orca-raw-density-sign",
    )
    parser.add_argument(
        "--initial-density",
        choices=("sad",),
        default="sad",
    )
    parser.add_argument(
        "--initial-density-charge-correction",
        choices=("trace-scale",),
        default="trace-scale",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32",),
        default="float32",
    )
    parser.add_argument("--max-trace-error", type=float, default=0.05)
    parser.add_argument(
        "--max-initial-trace-error",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--allow-legacy-be",
        action="store_true",
        help=(
            "Accept structurally valid legacy Be shards without corrected "
            "signed-layout provenance. Not recommended for the full-v2 build."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Build only explicitly selected shards.",
    )
    add_selection_args(run)
    add_contract_args(run)
    run.add_argument("--density-root", type=Path, required=True)
    run.add_argument(
        "--density-prefix",
        default="data/omol25/electronic",
        help="Prefix removed from each manifest density_path.",
    )
    run.add_argument("--parquet-root", type=Path, required=True)
    run.add_argument(
        "--parquet-prefix",
        default="datasets/omol25_train_4M",
        help="Prefix removed from each manifest parquet_file.",
    )
    run.add_argument("--parquet-cache-files", type=int, default=1)
    run.add_argument("--lmdb-map-size-gb", type=float, default=8.0)
    run.add_argument(
        "--require-density-dtype",
        choices=("any", "float32", "float64"),
        default="any",
    )
    run.add_argument(
        "--replace-invalid-existing",
        action="store_true",
        help=(
            "Back up and atomically replace an existing shard that fails "
            "validation. Required for legacy Be shard correction."
        ),
    )
    run.add_argument(
        "--replace-valid-existing",
        action="store_true",
        help="Back up and rebuild even a valid existing shard.",
    )
    run.add_argument("--run-id")
    run.set_defaults(func=run_command)

    validate = subparsers.add_parser(
        "validate",
        help="Strongly validate explicitly selected existing shards.",
    )
    add_selection_args(validate)
    add_contract_args(validate)
    validate.add_argument(
        "--require-current-provenance",
        action="store_true",
    )
    validate.set_defaults(func=validate_command)

    source_info = subparsers.add_parser(
        "source-info",
        help="Verify and print captured source provenance.",
    )
    source_info.set_defaults(func=source_info_command)

    self_test = subparsers.add_parser(
        "self-test",
        help="Run source, Be-convention, selector, and resume-validation smoke tests.",
    )
    self_test.set_defaults(func=self_test_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "shard_size", 4) < 1:
        parser.error("--shard-size must be positive")
    if getattr(args, "max_trace_error", 0.05) <= 0:
        parser.error("--max-trace-error must be positive")
    if getattr(args, "max_initial_trace_error", 1.0e-5) <= 0:
        parser.error("--max-initial-trace-error must be positive")
    if getattr(args, "lmdb_map_size_gb", 8.0) <= 0:
        parser.error("--lmdb-map-size-gb must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
