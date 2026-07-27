#!/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
"""Export full MALOQ matrices to the DFT monitor sample/prediction contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
WEB_DATA_DIR = Path("/dataset/seongsu/shared-home/projects/dft-monitor/data")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maloq.fock_utils import basis_sets, utils_orca_out  # noqa: E402

SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
    "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
]


def scalar(data: Any, key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def formula_from_atomic_numbers(atomic_numbers: np.ndarray) -> str:
    counts = Counter(int(z) for z in atomic_numbers)
    order = []
    if 6 in counts:
        order.append(6)
    if 1 in counts:
        order.append(1)
    order.extend(sorted(z for z in counts if z not in {1, 6}))
    parts = []
    for z in order:
        symbol = SYMBOLS[z] if z < len(SYMBOLS) else f"Z{z}"
        parts.append(symbol)
        if counts[z] != 1:
            parts.append(str(counts[z]))
    return "".join(parts)


def orbital_basis(atomic_numbers: np.ndarray, basis: str) -> dict[int, list[int]]:
    if basis.lower().replace("_", "-") == "def2-svp":
        known = basis_sets.orbital_basis_def2_svp_QM7
        if all(int(z) in known for z in atomic_numbers):
            return known

    from pyscf import gto

    l_by_atom: dict[int, list[int]] = {}
    for z in sorted(set(int(x) for x in atomic_numbers)):
        symbol = gto.mole._symbol(z)
        spin = z % 2
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)
        l_by_atom[z] = [int(mol.bas_angular(i)) for i in range(mol.nbas)]
    return l_by_atom


def transform_matrix(
    matrix: np.ndarray | None,
    convention: str,
    atomic_numbers: np.ndarray,
    basis: str,
) -> np.ndarray | None:
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    if convention == "pyscf":
        return array
    basis_map = orbital_basis(atomic_numbers, basis)

    def convert_one(item: np.ndarray) -> np.ndarray:
        if convention == "maloq-e3nn":
            return utils_orca_out.sort_by_m(
                item, basis_map, atomic_numbers, direction="e3nn_to_pyscf"
            )
        if convention == "maloq-storage":
            e3nn = utils_orca_out.sort_by_m(
                item, basis_map, atomic_numbers, direction="orca_to_e3nn"
            )
            return utils_orca_out.sort_by_m(
                e3nn, basis_map, atomic_numbers, direction="e3nn_to_pyscf"
            )
        raise ValueError(f"Unsupported convention: {convention}")

    if array.ndim == 2:
        return convert_one(array)
    if array.ndim == 3 and array.shape[0] == 2:
        return np.stack([convert_one(array[0]), convert_one(array[1])])
    raise ValueError(f"Expected (nao, nao) or (2, nao, nao), got {array.shape}")


def compute_overlap(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    basis: str,
    charge: int,
    spin: int,
) -> np.ndarray:
    from pyscf import gto

    atoms = [
        (gto.mole._symbol(int(z)), tuple(float(v) for v in pos))
        for z, pos in zip(atomic_numbers, positions)
    ]
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    return mol.intor("int1e_ovlp")


def check_square(name: str, matrix: np.ndarray, nao: int) -> None:
    if matrix.ndim == 3 and matrix.shape[0] == 2:
        expected = (2, nao, nao)
    else:
        expected = (nao, nao)
    if matrix.shape != expected:
        raise ValueError(f"{name} has shape {matrix.shape}; expected {expected}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix, matrix.swapaxes(-1, -2), atol=1.0e-8, rtol=1.0e-8):
        raise ValueError(f"{name} is not symmetric")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def load_reference(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    with np.load(path, allow_pickle=True) as data:
        return {
            "atomic_numbers": np.asarray(data["atomic_numbers"], dtype=np.int32),
            "basis": str(scalar(data, "basis", "def2-svp")),
            "mol_id": str(scalar(data, "mol_id", path.stem)),
        }


def export_sample(args: argparse.Namespace) -> None:
    with np.load(args.input, allow_pickle=True) as data:
        atomic_numbers = np.asarray(data[args.atomic_numbers_key], dtype=np.int32)
        positions = np.asarray(data[args.positions_key], dtype=np.float64)
        basis = str(args.basis or scalar(data, "basis", "def2-svp"))
        charge = int(args.charge if args.charge is not None else scalar(data, "charge", 0))
        spin = int(args.spin if args.spin is not None else scalar(data, "spin", 0))
        sample_id = str(args.sample_id or scalar(data, "mol_id", args.input.stem))
        formula = str(args.formula or scalar(data, "formula", formula_from_atomic_numbers(atomic_numbers)))
        xc = str(args.xc or scalar(data, "xc", "unknown"))
        hamiltonian = transform_matrix(
            np.asarray(data[args.hamiltonian_key]),
            args.matrix_convention,
            atomic_numbers,
            basis,
        )
        density = (
            transform_matrix(
                np.asarray(data[args.density_key]),
                args.matrix_convention,
                atomic_numbers,
                basis,
            )
            if args.density_key in data
            else None
        )
        if args.overlap_key in data:
            overlap_convention = args.overlap_convention
            if overlap_convention == "auto":
                overlap_convention = (
                    "pyscf" if args.matrix_convention == "pyscf" else "maloq-e3nn"
                )
            overlap = transform_matrix(
                np.asarray(data[args.overlap_key]),
                overlap_convention,
                atomic_numbers,
                basis,
            )
        else:
            overlap = compute_overlap(atomic_numbers, positions, basis, charge, spin)

    if positions.shape != (len(atomic_numbers), 3):
        raise ValueError(f"positions has shape {positions.shape}")
    nao = int(hamiltonian.shape[-1])
    check_square("hamiltonian", hamiltonian, nao)
    check_square("overlap", overlap, nao)
    if density is not None:
        check_square("density_matrix", density, nao)

    payload = {
        "mol_id": sample_id,
        "formula": formula,
        "n_atoms": int(len(atomic_numbers)),
        "nao": nao,
        "charge": charge,
        "spin": spin,
        "basis": basis,
        "xc": xc,
        "unrestricted": bool(hamiltonian.ndim == 3),
        "atomic_numbers": atomic_numbers.tolist(),
        "positions": positions.tolist(),
        "hamiltonian": hamiltonian.tolist(),
        "overlap": overlap.tolist(),
        "matrix_convention": "pyscf",
        "source": str(args.input.resolve()),
    }
    if density is not None:
        payload["density_matrix"] = density.tolist()
        payload["density_source"] = "provided"

    output = args.output or (WEB_DATA_DIR / "samples" / f"{sample_id}.json")
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
    atomic_write_json(output, payload)

    if not args.no_register:
        index_path = WEB_DATA_DIR / "index.json"
        index = json.loads(index_path.read_text())
        entry = {k: payload[k] for k in (
            "n_atoms", "mol_id", "formula", "nao", "charge", "spin", "basis",
            "xc", "atomic_numbers", "positions",
        )}
        existing = [item for item in index["samples"] if item["mol_id"] == sample_id]
        if existing and not args.force:
            raise FileExistsError(
                f"Sample {sample_id} is already registered; pass --force"
            )
        index["samples"] = [
            item for item in index["samples"] if item["mol_id"] != sample_id
        ] + [entry]
        index["samples"].sort(key=lambda item: (item["n_atoms"], item["mol_id"]))
        index["n_samples"] = len(index["samples"])
        atomic_write_json(index_path, index)

    total_density = density.sum(axis=0) if density is not None and density.ndim == 3 else density
    trace = float(np.trace(total_density @ overlap)) if total_density is not None else None
    print(f"wrote sample: {output}")
    print(f"shape: H={hamiltonian.shape} D={None if density is None else density.shape} S={overlap.shape}")
    if trace is not None:
        print(f"trace(D@S): {trace:.8f}")


def export_prediction(args: argparse.Namespace) -> None:
    reference = load_reference(args.reference)
    atomic_numbers = np.asarray(reference["atomic_numbers"], dtype=np.int32)
    basis = str(args.basis or reference.get("basis", "def2-svp"))
    with np.load(args.input, allow_pickle=True) as data:
        matrix = transform_matrix(
            np.asarray(data[args.hamiltonian_key]),
            args.matrix_convention,
            atomic_numbers,
            basis,
        )
        model_name = str(args.model_name or scalar(data, "model_name", args.input.stem))
    nao = int(matrix.shape[-1])
    check_square("hamiltonian", matrix, nao)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force")
    np.savez_compressed(
        args.output,
        hamiltonian=matrix,
        model_name=np.array(model_name),
        convention=np.array("pyscf"),
        unrestricted=np.array(matrix.ndim == 3),
        mol_id=np.array(reference.get("mol_id", args.reference.stem)),
    )
    print(f"wrote prediction: {args.output}")
    print(f"shape: {matrix.shape}; convention=pyscf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="register H/S and optional direct D")
    sample.add_argument("input", type=Path)
    sample.add_argument("--output", type=Path)
    sample.add_argument("--sample-id")
    sample.add_argument("--formula")
    sample.add_argument("--basis")
    sample.add_argument("--xc")
    sample.add_argument("--charge", type=int)
    sample.add_argument("--spin", type=int)
    sample.add_argument("--atomic-numbers-key", default="atomic_numbers")
    sample.add_argument("--positions-key", default="positions")
    sample.add_argument("--hamiltonian-key", default="hamiltonian")
    sample.add_argument("--density-key", default="density_matrix")
    sample.add_argument("--overlap-key", default="overlap")
    sample.add_argument(
        "--matrix-convention",
        choices=("pyscf", "maloq-e3nn", "maloq-storage"),
        default="pyscf",
    )
    sample.add_argument(
        "--overlap-convention",
        choices=("auto", "pyscf", "maloq-e3nn", "maloq-storage"),
        default="auto",
    )
    sample.add_argument("--no-register", action="store_true")
    sample.add_argument("--force", action="store_true")
    sample.set_defaults(func=export_sample)

    prediction = sub.add_parser("prediction", help="convert a predicted H for Comparison")
    prediction.add_argument("input", type=Path)
    prediction.add_argument("--reference", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--basis")
    prediction.add_argument("--model-name")
    prediction.add_argument("--hamiltonian-key", default="hamiltonian")
    prediction.add_argument(
        "--matrix-convention",
        choices=("pyscf", "maloq-e3nn", "maloq-storage"),
        default="maloq-e3nn",
    )
    prediction.add_argument("--force", action="store_true")
    prediction.set_defaults(func=export_prediction)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
