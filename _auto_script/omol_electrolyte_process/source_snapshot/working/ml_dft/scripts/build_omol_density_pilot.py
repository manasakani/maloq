#!/usr/bin/env python3
"""Build a tiny OMol raw-density pilot LMDB for ml_dft.

This script ingests OMol's stored ORCA ``density_mat.npz`` files directly,
converts the raw ORCA AO layout to the e3nn AO layout with a basis/element
shell-order cache, and writes block-trainable density/overlap matrices.

The important convention step is:

    ORCA raw `.densities` order
      -> cached ORCA shell order + ORCA m-order
      -> e3nn shell/m order used by ml_dft block heads

Example:

  /home1/irteam/data-vol1/conda/envs/proj-dft-dataset/bin/python \
    scripts/build_omol_density_pilot.py \
    --out /home1/irteam/data-vol1/datasets/omol25/lmdb/omol_dm_orca_raw_e3nn_orca_overlap_pilot_v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import subprocess
import time
import tempfile
import warnings
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from pyscf import gto


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parents[1]
DFT_DATASET_ROOT = DATA_ROOT / "projects" / "dft-dataset"
if str(DFT_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DFT_DATASET_ROOT))

from dft_dataset.conventions import (  # noqa: E402
    build_ao_shell_specs_from_mol,
    build_cached_orca_to_e3nn_indices,
    build_layout_reorder_indices,
    layout_matrix_to_orca_raw_density_layout,
    pyscf_overlap_to_orca_raw_density_layout,
    reorder_matrix,
)
from dft_dataset.lmdb_dataset import LMDBDataset  # noqa: E402
from dft_dataset.molecule import BasisInfo, Molecule  # noqa: E402


ROW_COLUMNS = [
    "configuration_id",
    "property_id",
    "atomic_numbers",
    "positions",
    "energy",
    "atomic_forces",
    "method",
    "multiplicity",
    "property_metadata",
    "chemical_formula_hill",
]

DEFAULT_ORCA_BIN = DATA_ROOT / "tools" / "orca" / "6.1.1" / "orca"
ORCA_BASIS_KEYWORDS = {
    "def2-svp": "def2-SVP",
    "def2-svp-nabla": "def2-SVP",
    "def2-tzvp": "def2-TZVP",
    "def2-tzvpd": "def2-TZVPD",
}


@dataclass
class SelectedSample:
    parquet_file: str
    row_in_file: int
    mol_id: str
    n_atoms: int
    charge: int
    spin: int
    formula: str | None
    n_basis_orca: int
    density_path: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_ROOT / "datasets" / "omol25" / "pilots"
        / "omol_density_pilot_train_128_supported_le60.jsonl",
        help="Input pilot manifest jsonl.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "datasets" / "omol25" / "lmdb"
        / "omol_dm_orca_raw_e3nn_orca_overlap_pilot_v1",
        help="Output directory containing train.lmdb / val.lmdb / test.lmdb.",
    )
    parser.add_argument("--basis", type=str, default="def2-tzvpd")
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
            "Source for the stored overlap matrix. Default 'orca' runs ORCA "
            "once per molecule and stores ORCA-native overlap in e3nn order. "
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
        default=Path(os.environ.get("ORCA_BIN", str(DEFAULT_ORCA_BIN))),
        help="ORCA executable used when --overlap-source=orca.",
    )
    parser.add_argument(
        "--orca-work-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for temporary ORCA overlap runs. By default, "
            "a system temporary directory is used and deleted after parsing."
        ),
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
            "Wait for the ORCA process to finish after orca.S.tmp is written. "
            "By default, terminate ORCA as soon as the overlap file reaches "
            "the expected size."
        ),
    )
    parser.add_argument(
        "--keep-orca-overlap-files",
        action="store_true",
        help="Keep per-sample ORCA overlap work directories for debugging.",
    )
    parser.add_argument(
        "--initial-density",
        choices=("none", "sad"),
        default="none",
        help="Optional initial density baseline to store as initial_density_matrix.",
    )
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
        "--max-atoms",
        type=int,
        default=10,
        help="Restrict to tiny molecules for a fast pilot.",
    )
    parser.add_argument("--train-count", type=int, default=2)
    parser.add_argument("--val-count", type=int, default=1)
    parser.add_argument("--test-count", type=int, default=1)
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_row_tables(items: Iterable[dict]) -> dict[Path, pd.DataFrame]:
    grouped: dict[Path, list[dict]] = defaultdict(list)
    for item in items:
        grouped[(DATA_ROOT / item["parquet_file"]).resolve()].append(item)

    tables: dict[Path, pd.DataFrame] = {}
    for parquet_path in sorted(grouped):
        tables[parquet_path] = pd.read_parquet(
            parquet_path,
            engine="pyarrow",
            columns=ROW_COLUMNS,
        )
    return tables


def _filter_candidates(items: list[dict], max_atoms: int) -> tuple[list[SelectedSample], dict]:
    """Choose a tiny H/C/N/O/F closed-shell neutral subset with raw densities."""
    allowed_z = {1, 6, 7, 8, 9}
    stats = {
        "seen": 0,
        "allowed_elements": 0,
        "closed_shell_neutral": 0,
        "within_atom_limit": 0,
        "density_exists": 0,
    }

    tables = _load_row_tables(items)
    selected: list[SelectedSample] = []
    for item in items:
        stats["seen"] += 1
        parquet_path = (DATA_ROOT / item["parquet_file"]).resolve()
        row = tables[parquet_path].iloc[item["row_in_file"]].to_dict()
        mol = Molecule.from_omol25_row(row)
        if not set(int(z) for z in mol.atomic_numbers).issubset(allowed_z):
            continue
        stats["allowed_elements"] += 1
        if mol.charge != 0 or mol.spin != 0:
            continue
        stats["closed_shell_neutral"] += 1
        if mol.num_atoms > max_atoms:
            continue
        stats["within_atom_limit"] += 1
        density_path = item.get("density_path")
        if not density_path or not (DATA_ROOT / density_path).exists():
            continue
        stats["density_exists"] += 1
        selected.append(
            SelectedSample(
                parquet_file=item["parquet_file"],
                row_in_file=int(item["row_in_file"]),
                mol_id=mol.mol_id or f"{Path(item['parquet_file']).stem}:{item['row_in_file']}",
                n_atoms=mol.num_atoms,
                charge=mol.charge,
                spin=mol.spin,
                formula=mol.formula,
                n_basis_orca=int(item["n_basis_orca"]),
                density_path=density_path,
            )
        )

    selected.sort(key=lambda s: (s.n_atoms, s.mol_id))
    return selected, stats


def _atom_string(mol: Molecule) -> str:
    return "; ".join(
        f"{gto.elements.ELEMENTS[int(z)]} {x} {y} {zv}"
        for z, (x, y, zv) in zip(mol.atomic_numbers, mol.positions)
    )


def _pyscf_mol(mol: Molecule, basis: str, *, with_ecp: bool = False):
    ecp = None
    if with_ecp:
        ecp = {}
        for z in sorted(set(int(value) for value in mol.atomic_numbers)):
            symbol = gto.elements.ELEMENTS[z]
            ecp_data = gto.basis.load_ecp(basis, symbol)
            if ecp_data:
                ecp[symbol] = ecp_data
    return gto.M(
        atom=_atom_string(mol),
        basis=basis,
        ecp=ecp,
        charge=mol.charge,
        spin=mol.spin,
        unit="Angstrom",
        verbose=0,
    )


def _unpack_upper_triangle(packed: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=packed.dtype)
    idx = np.triu_indices(n)
    out[idx] = packed
    out[(idx[1], idx[0])] = packed
    return out


def _load_raw_orca_total_density(
    path: Path,
    n: int,
    required_dtype: str | np.dtype | None = None,
) -> np.ndarray:
    with np.load(path) as data:
        if "orca.scfp" not in data:
            raise KeyError(f"{path} does not contain 'orca.scfp'; keys={list(data.files)}")
        packed = data["orca.scfp"]
    if required_dtype is not None and packed.dtype != np.dtype(required_dtype):
        raise ValueError(
            f"{path} orca.scfp dtype {packed.dtype} != required {np.dtype(required_dtype)}"
        )
    expected = n * (n + 1) // 2
    if packed.shape[0] != expected:
        raise ValueError(f"{path} packed size {packed.shape[0]} != expected {expected} for nao={n}")
    return _unpack_upper_triangle(packed, n).astype(np.float64, copy=False)


def _pyscf_to_e3nn_indices(pmol) -> np.ndarray:
    pyscf_shells = build_ao_shell_specs_from_mol(pmol)
    return build_layout_reorder_indices(
        pyscf_shells,
        pyscf_shells,
        src_convention="pyscf",
        dst_convention="e3nn",
    )


def _orca_basis_keyword(basis: str) -> str:
    return ORCA_BASIS_KEYWORDS.get(basis.lower(), basis)


def _orca_atom_lines(mol: Molecule) -> str:
    lines = []
    for atomic_number, (x, y, z_coord) in zip(mol.atomic_numbers, mol.positions):
        symbol = gto.elements.ELEMENTS[int(atomic_number)]
        lines.append(f"{symbol} {x:.15f} {y:.15f} {z_coord:.15f}")
    return "\n".join(lines)


def _write_orca_overlap_input(path: Path, mol: Molecule, basis: str) -> None:
    multiplicity = int(mol.spin) + 1
    path.write_text(
        "\n".join([
            f"! HF {_orca_basis_keyword(basis)} NoUseSym",
            "%scf MaxIter 1 end",
            f"*xyz {int(mol.charge)} {multiplicity}",
            _orca_atom_lines(mol),
            "*",
            "",
        ])
    )


def _read_orca_s_tmp(path: Path, n: int) -> np.ndarray:
    """Read ORCA's binary packed overlap file in raw ORCA AO order.

    ORCA 6.1.1 writes ``orca.S.tmp`` as a 24-byte header followed by the
    lower triangle of S in little-endian float64. This avoids parsing huge
    text overlap matrices for large OMol systems.
    """
    expected_packed = n * (n + 1) // 2
    expected_size = 24 + expected_packed * 8
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path} size {actual_size} != expected {expected_size} for nao={n}"
        )
    packed = np.fromfile(path, dtype="<f8", offset=24)
    if packed.shape[0] != expected_packed:
        raise ValueError(
            f"{path} packed length {packed.shape[0]} != expected {expected_packed}"
        )
    overlap = np.zeros((n, n), dtype=np.float64)
    lower = np.tril_indices(n)
    overlap[lower] = packed
    overlap[(lower[1], lower[0])] = packed
    return overlap


def _orca_s_tmp_expected_size(n: int) -> int:
    return 24 + (n * (n + 1) // 2) * 8


def _safe_work_name(selected: SelectedSample) -> str:
    raw = f"{selected.mol_id}_{selected.density_path}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)[:180]


@contextmanager
def _orca_workdir(
    selected: SelectedSample,
    work_root: Path | None,
    keep_files: bool,
) -> Iterator[Path]:
    if keep_files:
        root = work_root or (PROJECT_ROOT / "runs" / "orca_overlap_work")
        path = root / _safe_work_name(selected)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return

    with tempfile.TemporaryDirectory(
        prefix="omol_orca_overlap_",
        dir=None if work_root is None else str(work_root),
    ) as tmp:
        yield Path(tmp)


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


def _orca_overlap_matrix(
    mol: Molecule,
    selected: SelectedSample,
    basis: str,
    n: int,
    orca_bin: Path,
    work_root: Path | None = None,
    timeout_seconds: float = 1800.0,
    keep_files: bool = False,
    wait_for_completion: bool = False,
) -> np.ndarray:
    if not orca_bin.exists():
        raise FileNotFoundError(f"ORCA executable not found: {orca_bin}")

    with _orca_workdir(selected, work_root, keep_files) as workdir:
        inp = workdir / "orca.inp"
        out = workdir / "orca.out"
        err = workdir / "orca.err"
        _write_orca_overlap_input(inp, mol, basis)
        s_tmp = workdir / "orca.S.tmp"
        expected_size = _orca_s_tmp_expected_size(n)
        start = time.monotonic()
        proc = None
        try:
            with out.open("w") as stdout, err.open("w") as stderr:
                proc = subprocess.Popen(
                    [str(orca_bin), "orca.inp"],
                    cwd=workdir,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                while True:
                    if s_tmp.exists() and s_tmp.stat().st_size == expected_size:
                        if wait_for_completion:
                            proc.wait(timeout=max(timeout_seconds - (time.monotonic() - start), 1.0))
                        else:
                            time.sleep(0.05)
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=5.0)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                                proc.wait(timeout=5.0)
                        break
                    rc = proc.poll()
                    if rc is not None:
                        break
                    if time.monotonic() - start > timeout_seconds:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait(timeout=5.0)
                        raise TimeoutError(
                            f"ORCA overlap extraction timed out after {timeout_seconds}s "
                            f"for {selected.mol_id}"
                        )
                    time.sleep(0.05)
        except ProcessLookupError:
            pass

        if not s_tmp.exists():
            raise RuntimeError(
                f"ORCA did not produce orca.S.tmp for {selected.mol_id}.\n"
                f"orca.out tail:\n{_tail_text(out)}\n"
                f"orca.err tail:\n{_tail_text(err)}"
            )
        if s_tmp.stat().st_size != expected_size:
            raise RuntimeError(
                f"ORCA produced incomplete orca.S.tmp for {selected.mol_id}: "
                f"{s_tmp.stat().st_size} bytes, expected {expected_size}.\n"
                f"orca.out tail:\n{_tail_text(out)}\n"
                f"orca.err tail:\n{_tail_text(err)}"
            )
        return _read_orca_s_tmp(s_tmp, n)


def _sad_density_pyscf(pmol) -> np.ndarray:
    """Return PySCF's SAD/atom initial total density in PySCF AO order."""
    from pyscf import scf

    mf = scf.RHF(pmol) if pmol.spin == 0 else scf.UHF(pmol)
    dm = mf.get_init_guess(pmol, key="atom")
    dm_arr = np.asarray(dm)
    if dm_arr.ndim == 3:
        dm_arr = dm_arr.sum(axis=0)
    return 0.5 * (dm_arr + dm_arr.T)


def _initial_density_convention(initial_density: str, overlap_source: str) -> str | None:
    if initial_density == "none":
        return None
    if overlap_source == "pyscf":
        return "pyscf_e3nn"
    return "orca_raw_density_e3nn"


def _build_molecule(
    row: dict,
    selected: SelectedSample,
    basis: str,
    initial_density: str,
    initial_density_charge_correction: str = "none",
    storage_dtype: str | np.dtype = np.float64,
    overlap_source: str = "orca",
    orca_bin: Path = DEFAULT_ORCA_BIN,
    orca_work_dir: Path | None = None,
    orca_timeout_seconds: float = 1800.0,
    keep_orca_overlap_files: bool = False,
    orca_wait_for_completion: bool = False,
    required_density_dtype: str | np.dtype | None = None,
) -> tuple[Molecule, dict[str, float | int | bool]]:
    mol = Molecule.from_omol25_row(row)
    t0 = time.perf_counter()
    pmol = _pyscf_mol(mol, basis)
    pmol_ecp = _pyscf_mol(mol, basis, with_ecp=True)
    if int(pmol_ecp.nelectron) != int(mol.num_electrons):
        raise ValueError(
            f"PySCF ECP nelectron={pmol_ecp.nelectron} but OMol metadata gives "
            f"{mol.num_electrons} for {selected.mol_id}"
        )
    if int(pmol.nao) != int(selected.n_basis_orca):
        raise ValueError(
            f"PySCF nao={pmol.nao} but manifest ORCA n_basis={selected.n_basis_orca} "
            f"for {selected.mol_id}"
        )

    density_raw = _load_raw_orca_total_density(
        DATA_ROOT / selected.density_path,
        pmol.nao,
        required_dtype=required_density_dtype,
    )
    idx_orca_to_e3nn = build_cached_orca_to_e3nn_indices(pmol, basis)
    idx_pyscf_to_e3nn = _pyscf_to_e3nn_indices(pmol)

    dm = reorder_matrix(density_raw, idx_orca_to_e3nn)
    use_orca_overlap = overlap_source == "orca" or (
        overlap_source == "pyscf-orca-raw-density-sign-orca-be"
        and any(int(z) == 4 for z in mol.atomic_numbers)
    )
    effective_overlap_source = overlap_source
    if use_orca_overlap:
        overlap_raw = _orca_overlap_matrix(
            mol,
            selected=selected,
            basis=basis,
            n=pmol.nao,
            orca_bin=orca_bin,
            work_root=orca_work_dir,
            timeout_seconds=orca_timeout_seconds,
            keep_files=keep_orca_overlap_files,
            wait_for_completion=orca_wait_for_completion,
        )
        overlap = reorder_matrix(overlap_raw, idx_orca_to_e3nn)
        if overlap_source == "pyscf-orca-raw-density-sign-orca-be":
            effective_overlap_source = "orca-be-fallback"
    elif overlap_source == "pyscf":
        warnings.warn(
            "overlap_source='pyscf' is deprecated for OMol ORCA raw density "
            "lanes because Tr(D S_pyscf) is not physically exact. Use "
            "overlap_source='orca' for new datasets.",
            DeprecationWarning,
            stacklevel=2,
        )
        overlap = reorder_matrix(pmol.intor("int1e_ovlp"), idx_pyscf_to_e3nn)
    elif overlap_source == "pyscf-orca-raw-density-sign":
        overlap = pyscf_overlap_to_orca_raw_density_layout(
            pmol,
            basis,
            dst_convention="e3nn",
        )
    elif overlap_source == "pyscf-orca-raw-density-sign-orca-be":
        overlap = pyscf_overlap_to_orca_raw_density_layout(
            pmol,
            basis,
            dst_convention="e3nn",
        )
        effective_overlap_source = "pyscf-orca-raw-density-sign"
    else:
        raise ValueError(f"Unsupported overlap_source={overlap_source!r}")

    n_electrons = int(pmol_ecp.nelectron)
    init_dm = None
    trace_initial_uncorrected = None
    initial_density_charge_scale = 1.0
    if initial_density == "sad":
        sad_pyscf = _sad_density_pyscf(pmol_ecp)
        if overlap_source == "pyscf":
            init_dm = reorder_matrix(sad_pyscf, idx_pyscf_to_e3nn)
        else:
            init_dm = layout_matrix_to_orca_raw_density_layout(
                sad_pyscf,
                pmol,
                basis,
                src_convention="pyscf",
                dst_convention="e3nn",
            )
        trace_initial_uncorrected = float((init_dm @ overlap).trace())
        if initial_density_charge_correction == "trace-scale":
            if abs(trace_initial_uncorrected) < 1.0e-12:
                raise ValueError(
                    f"Cannot trace-scale SAD for {selected.mol_id}: "
                    f"Tr(D_init S)={trace_initial_uncorrected}"
                )
            initial_density_charge_scale = float(n_electrons) / trace_initial_uncorrected
            init_dm = 0.5 * (init_dm * initial_density_charge_scale + (init_dm * initial_density_charge_scale).T)
        elif initial_density_charge_correction != "none":
            raise ValueError(
                "initial_density_charge_correction must be 'none' or "
                f"'trace-scale', got {initial_density_charge_correction!r}"
            )
    elapsed = time.perf_counter() - t0
    basis_info = BasisInfo.from_pyscf_basis(basis, mol.atomic_numbers, convention="e3nn")
    store_dtype = np.dtype(storage_dtype)

    out = Molecule.from_omol25_row(row)
    out.xc = f"{mol.xc}-orca-raw-density"
    out.source = "omol25-orca-raw-density"
    out.overlap = overlap.astype(store_dtype, copy=False)
    out.density_matrix = dm.astype(store_dtype, copy=False)
    out.initial_density_matrix = (
        None if init_dm is None else init_dm.astype(store_dtype, copy=False)
    )
    out.basis_info = basis_info
    # Store total alpha+beta density as a single matrix target.
    out.unrestricted = False

    trace_target = float((dm @ overlap).trace())
    stats = {
        "n_atoms": int(mol.num_atoms),
        "nao": int(pmol.nao),
        "seconds": elapsed,
        "trace_target": trace_target,
        "trace_error": trace_target - n_electrons,
        "n_electrons": n_electrons,
        "storage_dtype": str(store_dtype),
        "source_density_dtype_requirement": (
            str(np.dtype(required_density_dtype))
            if required_density_dtype is not None
            else "unchecked"
        ),
        "overlap_source": overlap_source,
        "overlap_effective_source": effective_overlap_source,
    }
    if init_dm is not None:
        trace_initial_density = float((init_dm @ overlap).trace())
        diff = init_dm - dm
        stats.update({
            "trace_initial_density": trace_initial_density,
            "trace_initial_error": trace_initial_density - n_electrons,
            "trace_initial_density_uncorrected": trace_initial_uncorrected,
            "initial_density_charge_correction": initial_density_charge_correction,
            "initial_density_charge_scale": initial_density_charge_scale,
            "initial_density_full_mae": float(np.mean(np.abs(diff))),
            "initial_density_rel_fro": float(
                np.linalg.norm(diff) / max(np.linalg.norm(dm), 1e-30)
            ),
            "initial_density_convention": _initial_density_convention(
                initial_density,
                overlap_source,
            ),
        })
    return out, stats


def _materialize_split(
    split_name: str,
    selected: list[SelectedSample],
    row_tables: dict[Path, pd.DataFrame],
    out_dir: Path,
    basis: str,
    initial_density: str,
    initial_density_charge_correction: str,
    overlap_source: str,
    orca_bin: Path,
    orca_work_dir: Path | None,
    orca_timeout_seconds: float,
    keep_orca_overlap_files: bool,
    orca_wait_for_completion: bool,
) -> dict:
    molecules: list[Molecule] = []
    per_sample: list[dict] = []
    for idx, item in enumerate(selected, start=1):
        print(
            f"[omol-raw-density] {split_name} {idx}/{len(selected)} "
            f"{item.mol_id} ({item.n_atoms} atoms)",
            flush=True,
        )
        parquet_path = (DATA_ROOT / item.parquet_file).resolve()
        row = row_tables[parquet_path].iloc[item.row_in_file].to_dict()
        mol, stats = _build_molecule(
            row,
            selected=item,
            basis=basis,
            initial_density=initial_density,
            initial_density_charge_correction=initial_density_charge_correction,
            overlap_source=overlap_source,
            orca_bin=orca_bin,
            orca_work_dir=orca_work_dir,
            orca_timeout_seconds=orca_timeout_seconds,
            keep_orca_overlap_files=keep_orca_overlap_files,
            orca_wait_for_completion=orca_wait_for_completion,
        )
        molecules.append(mol)
        per_sample.append({"mol_id": item.mol_id, "density_path": item.density_path, **stats})

    schema = {
        "dataset": "omol_orca_raw_density_pilot",
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
        "split": split_name,
    }
    lmdb_path = out_dir / f"{split_name}.lmdb"
    count = LMDBDataset.write(
        molecules,
        str(lmdb_path),
        packed=True,
        schema=schema,
        format="pickle",
    )
    total_seconds = sum(float(s["seconds"]) for s in per_sample)
    return {
        "lmdb": str(lmdb_path),
        "count": count,
        "samples": per_sample,
        "total_seconds": total_seconds,
        "mean_seconds": total_seconds / max(count, 1),
    }


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_items = _load_manifest(args.manifest)
    candidates, filter_stats = _filter_candidates(manifest_items, max_atoms=args.max_atoms)

    need = args.train_count + args.val_count + args.test_count
    if len(candidates) < need:
        raise SystemExit(
            f"Need {need} filtered candidates but found {len(candidates)}. "
            f"Try raising --max-atoms."
        )

    selected = candidates[:need]
    train = selected[: args.train_count]
    val = selected[args.train_count : args.train_count + args.val_count]
    test = selected[args.train_count + args.val_count :]

    selected_path = args.out / "selected_samples.jsonl"
    with selected_path.open("w") as f:
        for item in selected:
            f.write(json.dumps(asdict(item)) + "\n")

    row_tables = _load_row_tables(
        [{"parquet_file": s.parquet_file, "row_in_file": s.row_in_file} for s in selected]
    )
    summary = {
        "manifest": str(args.manifest),
        "basis": args.basis,
        "convention": "e3nn",
        "initial_density": args.initial_density,
        "initial_density_charge_correction": args.initial_density_charge_correction,
        "overlap_source": args.overlap_source,
        "pyscf_overlap_deprecated": args.overlap_source == "pyscf",
        "max_atoms": args.max_atoms,
        "filter_stats": filter_stats,
        "selected_samples": len(selected),
        "selected_manifest": str(selected_path),
        "splits": {},
    }
    for split_name, split_items in (("train", train), ("val", val), ("test", test)):
        summary["splits"][split_name] = _materialize_split(
            split_name,
            split_items,
            row_tables,
            args.out,
            basis=args.basis,
            initial_density=args.initial_density,
            initial_density_charge_correction=args.initial_density_charge_correction,
            overlap_source=args.overlap_source,
            orca_bin=args.orca_bin,
            orca_work_dir=args.orca_work_dir,
            orca_timeout_seconds=args.orca_timeout_seconds,
            keep_orca_overlap_files=args.keep_orca_overlap_files,
            orca_wait_for_completion=args.orca_wait_for_completion,
        )

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
