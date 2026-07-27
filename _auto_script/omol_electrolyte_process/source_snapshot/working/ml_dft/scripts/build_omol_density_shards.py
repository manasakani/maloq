#!/usr/bin/env python3
"""Convert OMol density manifests into sharded e3nn LMDBs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parents[1]
DFT_DATASET_ROOT = DATA_ROOT / "projects" / "dft-dataset"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DFT_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DFT_DATASET_ROOT))

from dft_dataset.lmdb_dataset import LMDBDataset  # noqa: E402
from scripts.build_omol_density_pilot import (  # noqa: E402
    SelectedSample,
    _build_molecule,
    _initial_density_convention,
    _load_row_tables,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DATA_ROOT / "datasets" / "omol25" / "manifests"
        / "unsolvated_electrolytes_v1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "datasets" / "omol25" / "lmdb"
        / "omol_dm_unsolvated_electrolytes_e3nn_orca_overlap_v1",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--basis", default="def2-tzvpd")
    parser.add_argument(
        "--overlap-source",
        choices=(
            "orca",
            "pyscf",
            "pyscf-orca-raw-density-sign",
            "pyscf-orca-raw-density-sign-orca-be",
        ),
        default="orca",
        help=(
            "Source for the stored overlap matrix. Default 'orca' stores "
            "ORCA-native overlap reordered to e3nn. "
            "'pyscf-orca-raw-density-sign' computes PySCF overlap and applies "
            "the ORCA raw-density sign convention without running ORCA. "
            "'pyscf-orca-raw-density-sign-orca-be' uses that fast path except "
            "for Be-containing molecules, where ORCA S.tmp is used. "
            "'pyscf' is deprecated and kept only for legacy comparisons."
        ),
    )
    parser.add_argument(
        "--orca-bin",
        type=Path,
        default=Path(os.environ.get(
            "ORCA_BIN",
            str(DATA_ROOT / "tools" / "orca" / "6.1.1" / "orca"),
        )),
        help="ORCA executable used when --overlap-source=orca.",
    )
    parser.add_argument(
        "--orca-work-dir",
        type=Path,
        default=None,
        help="Optional directory for temporary ORCA overlap extraction runs.",
    )
    parser.add_argument(
        "--orca-timeout-seconds",
        type=float,
        default=1800.0,
        help="Timeout for one ORCA overlap extraction.",
    )
    parser.add_argument(
        "--orca-wait-for-completion",
        action="store_true",
        help=(
            "Wait for ORCA to finish after orca.S.tmp is written. By default, "
            "terminate ORCA once the overlap file reaches the expected size."
        ),
    )
    parser.add_argument(
        "--keep-orca-overlap-files",
        action="store_true",
        help="Keep per-sample ORCA overlap work directories for debugging.",
    )
    parser.add_argument("--initial-density", choices=("none", "sad"), default="none")
    parser.add_argument(
        "--initial-density-charge-correction",
        choices=("none", "trace-scale"),
        default="none",
        help=(
            "How to make initial_density_matrix charge-consistent. "
            "'trace-scale' rescales SAD so Tr(D_init S) matches the molecular "
            "electron count."
        ),
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Matrix dtype stored in LMDB. Raw OMol density may be fp32/fp64; "
        "float32 keeps converted LMDBs compact, while float64 preserves the "
        "previous conversion behavior.",
    )
    parser.add_argument(
        "--require-density-dtype",
        choices=("any", "float32", "float64"),
        default="any",
        help="Reject a source density whose orca.scfp dtype does not match.",
    )
    parser.add_argument(
        "--max-trace-error",
        type=float,
        default=0.0,
        help=(
            "Reject a converted sample when abs(Tr(D S) - n_electrons) exceeds "
            "this value. Zero disables the gate."
        ),
    )
    parser.add_argument(
        "--lmdb-map-size-gb",
        type=float,
        default=0.0,
        help=(
            "Optional fixed LMDB map size per output shard in GiB. Use this "
            "for very large def2-TZVPD shards that can exceed the writer's "
            "default 1 GiB minimum map."
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=16,
        help="Manifest rows per output shard. Keep this modest because each "
        "row materializes full density and overlap matrices before LMDB write.",
    )
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument(
        "--shard-stride",
        type=int,
        default=1,
        help="Process every Nth output shard. Use with --shard-offset for parallel runs.",
    )
    parser.add_argument(
        "--shard-offset",
        type=int,
        default=0,
        help="Output shard offset modulo --shard-stride for parallel runs.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _selected(item: dict[str, Any]) -> SelectedSample:
    return SelectedSample(
        parquet_file=item["parquet_file"],
        row_in_file=int(item["row_in_file"]),
        mol_id=str(item.get("configuration_id") or item.get("property_id")),
        n_atoms=int(item["nsites"]),
        charge=int(item.get("charge", 0)),
        spin=int(item.get("spin", 0)),
        formula=item.get("formula"),
        n_basis_orca=int(item["n_basis_orca"]),
        density_path=item["density_path"],
    )


def _chunks(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield start // size, items[start : start + size]


def _write_shard(
    *,
    split: str,
    shard_idx: int,
    items: list[dict[str, Any]],
    out_dir: Path,
    basis: str,
    initial_density: str,
    initial_density_charge_correction: str,
    storage_dtype: str,
    lmdb_map_size_gb: float,
    overlap_source: str,
    orca_bin: Path,
    orca_work_dir: Path | None,
    orca_timeout_seconds: float,
    keep_orca_overlap_files: bool,
    orca_wait_for_completion: bool,
    required_density_dtype: str | None,
    max_trace_error: float,
) -> dict[str, Any]:
    shard_path = out_dir / split / f"shard_{shard_idx:06d}.lmdb"
    row_tables = _load_row_tables(items)
    molecules = []
    samples = []
    failures = []
    t0 = time.perf_counter()

    for local_idx, item in enumerate(items):
        selected = _selected(item)
        try:
            parquet_path = (DATA_ROOT / selected.parquet_file).resolve()
            row = row_tables[parquet_path].iloc[selected.row_in_file].to_dict()
            mol, stats = _build_molecule(
                row,
                selected=selected,
                basis=basis,
                initial_density=initial_density,
                initial_density_charge_correction=initial_density_charge_correction,
                storage_dtype=storage_dtype,
                overlap_source=overlap_source,
                orca_bin=orca_bin,
                orca_work_dir=orca_work_dir,
                orca_timeout_seconds=orca_timeout_seconds,
                keep_orca_overlap_files=keep_orca_overlap_files,
                orca_wait_for_completion=orca_wait_for_completion,
                required_density_dtype=required_density_dtype,
            )
            if (
                max_trace_error > 0.0
                and abs(float(stats["trace_error"])) > max_trace_error
            ):
                raise ValueError(
                    f"abs trace error {abs(float(stats['trace_error'])):.6g} exceeds "
                    f"--max-trace-error={max_trace_error:.6g}"
                )
        except Exception as exc:  # keep the full job moving, but record loudly
            failed = dict(item)
            failed["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(failed)
            continue
        molecules.append(mol)
        samples.append({
            "local_index": local_idx,
            "manifest": item,
            **stats,
        })

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "dataset": "omol_unsolvated_electrolyte_raw_density",
        "targets": {
            "density_matrix": True,
            "overlap": True,
            "initial_density_matrix": initial_density != "none",
        },
        "basis": basis,
        "convention": "e3nn",
        "xc": "omol-orca-raw",
        "initial_density": initial_density,
        "initial_density_charge_correction": initial_density_charge_correction,
        "initial_density_convention": _initial_density_convention(
            initial_density,
            overlap_source,
        ),
        "overlap_source": overlap_source,
        "pyscf_overlap_deprecated": overlap_source == "pyscf",
        "storage_dtype": storage_dtype,
        "source_density_dtype_requirement": required_density_dtype or "any",
        "split": split,
        "shard_index": shard_idx,
    }
    count = LMDBDataset.write(
        molecules,
        str(shard_path),
        packed=True,
        schema=schema,
        format="pickle",
        map_size_per_sample_mb=(
            float(lmdb_map_size_gb) * 1024.0 / max(len(molecules), 1)
            if lmdb_map_size_gb > 0
            else 2.0
        ),
    )
    elapsed = time.perf_counter() - t0
    summary = {
        "split": split,
        "shard_index": shard_idx,
        "lmdb": str(shard_path),
        "manifest_count": len(items),
        "written_count": count,
        "failure_count": len(failures),
        "seconds": elapsed,
        "samples": samples,
        "failures": failures,
    }
    summary_path = shard_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if failures:
        fail_path = shard_path.with_suffix(".failures.jsonl")
        with fail_path.open("w") as f:
            for failure in failures:
                f.write(json.dumps(failure, sort_keys=True) + "\n")
    return summary


def main() -> int:
    args = _parse_args()
    if args.shard_stride < 1:
        raise ValueError("--shard-stride must be >= 1")
    if not (0 <= args.shard_offset < args.shard_stride):
        raise ValueError("--shard-offset must satisfy 0 <= offset < stride")
    args.out.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    root_summary = {
        "manifest_dir": str(args.manifest_dir),
        "out": str(args.out),
        "basis": args.basis,
        "initial_density": args.initial_density,
        "initial_density_charge_correction": args.initial_density_charge_correction,
        "overlap_source": args.overlap_source,
        "pyscf_overlap_deprecated": args.overlap_source == "pyscf",
        "storage_dtype": args.storage_dtype,
        "source_density_dtype_requirement": args.require_density_dtype,
        "max_trace_error": args.max_trace_error,
        "lmdb_map_size_gb": args.lmdb_map_size_gb,
        "shard_size": args.shard_size,
        "shard_stride": args.shard_stride,
        "shard_offset": args.shard_offset,
        "splits": {},
    }

    for split in splits:
        manifest_path = args.manifest_dir / f"{split}.jsonl"
        items = _read_jsonl(manifest_path, limit=args.limit_per_split)
        split_summaries = []
        for shard_idx, shard_items in _chunks(items, args.shard_size):
            if shard_idx % args.shard_stride != args.shard_offset:
                continue
            shard_path = args.out / split / f"shard_{shard_idx:06d}.lmdb"
            if args.skip_existing and shard_path.exists():
                split_summaries.append({
                    "split": split,
                    "shard_index": shard_idx,
                    "lmdb": str(shard_path),
                    "skipped_existing": True,
                })
                continue
            print(
                f"[omol-density-shards] {split} shard {shard_idx:06d} "
                f"({len(shard_items)} manifest rows)",
                flush=True,
            )
            split_summaries.append(
                _write_shard(
                    split=split,
                    shard_idx=shard_idx,
                    items=shard_items,
                    out_dir=args.out,
                    basis=args.basis,
                    initial_density=args.initial_density,
                    initial_density_charge_correction=args.initial_density_charge_correction,
                    storage_dtype=args.storage_dtype,
                    lmdb_map_size_gb=args.lmdb_map_size_gb,
                    overlap_source=args.overlap_source,
                    orca_bin=args.orca_bin,
                    orca_work_dir=args.orca_work_dir,
                    orca_timeout_seconds=args.orca_timeout_seconds,
                    keep_orca_overlap_files=args.keep_orca_overlap_files,
                    orca_wait_for_completion=args.orca_wait_for_completion,
                    required_density_dtype=(
                        None
                        if args.require_density_dtype == "any"
                        else args.require_density_dtype
                    ),
                    max_trace_error=args.max_trace_error,
                )
            )
        root_summary["splits"][split] = split_summaries

    summary_name = (
        "summary.json"
        if args.shard_stride == 1
        else f"summary.offset_{args.shard_offset:03d}_of_{args.shard_stride:03d}.json"
    )
    root_summary["failure_count"] = sum(
        int(summary.get("failure_count", 0))
        for summaries in root_summary["splits"].values()
        for summary in summaries
    )
    (args.out / summary_name).write_text(
        json.dumps(root_summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(root_summary, indent=2, sort_keys=True))
    return 1 if root_summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
