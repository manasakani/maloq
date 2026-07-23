#!/usr/bin/env python3
"""Compute train-only NablaDFT node-label scale/shift statistics."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
OUTPUT_ROOT = (PROJECT_ROOT / "outputs").resolve()
DEFAULT_DB = Path(
    "/dataset/seongsu/shared-home/datasets/nablaDFT/"
    "hamiltonian_databases/train_2k.db"
)
DEFAULT_OUTPUT = (
    OUTPUT_ROOT
    / "scale-shift-statistics"
    / "nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbpath", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-train", type=int, default=12081)
    parser.add_argument("--rcut", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact instead of validating and reusing it.",
    )
    return parser


def _resolve_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output != OUTPUT_ROOT and OUTPUT_ROOT not in output.parents:
        raise SystemExit(f"--output must be below {OUTPUT_ROOT}")
    return output


def _provenance(
    dbpath: Path,
    database_rows: int,
    num_train: int,
    rcut: float,
    dtype_name: str,
) -> dict[str, object]:
    stat = dbpath.stat()
    return {
        "schema_version": 1,
        "dataset_name": "nablaDFT",
        "database_path": str(dbpath),
        "database_size_bytes": stat.st_size,
        "database_mtime_ns": stat.st_mtime_ns,
        "database_rows": database_rows,
        "split_name": "nablaDFT-2k fixed ordered split",
        "training_index_start": 0,
        "training_index_end_exclusive": num_train,
        "num_train": num_train,
        "validation_rows_in_statistics": 0,
        "test_rows_in_statistics": 0,
        "loss_target": "fock_matrix",
        "matrix_convention": "native_nabladft_psi4",
        "basis": "def2-svp-nabla",
        "rcut_orbitals": rcut,
        "dtype": dtype_name,
        "normalization": "elementwise_standardize_l0_node_labels",
    }


def _validate_existing(
    output: Path,
    expected_provenance: dict[str, object],
) -> bool:
    import torch

    if not output.is_file():
        return False
    payload = torch.load(output, map_location="cpu", weights_only=False)
    required = {
        "element_scalar_means",
        "element_scalar_stds",
        "scalar_irrep_indices",
        "provenance",
    }
    missing = required.difference(payload)
    if missing:
        raise SystemExit(
            f"Existing scale-shift artifact is missing {sorted(missing)}: {output}"
        )
    actual_provenance = payload["provenance"]
    mismatches = {
        key: (actual_provenance.get(key), value)
        for key, value in expected_provenance.items()
        if actual_provenance.get(key) != value
    }
    if mismatches:
        details = "\n".join(
            f"  {key}: existing={actual!r}, expected={expected!r}"
            for key, (actual, expected) in mismatches.items()
        )
        raise SystemExit(
            f"Existing artifact provenance does not match:\n{details}\n"
            "Use --overwrite only after confirming the intended replacement."
        )
    print(f"Validated existing train-only scale-shift artifact: {output}")
    return True


def _scalar_indices(irreps) -> list[int]:
    indices: list[int] = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = int(multiplicity) * int(irrep.dim)
        if irrep.l == 0:
            indices.extend(range(offset, offset + width))
        offset += width
    return indices


def compute_statistics(
    database,
    num_train: int,
    rcut: float,
    dtype,
    batch_size: int,
) -> dict[str, object]:
    import torch

    from maloq.fock_utils import basis_sets
    from maloq.fock_utils.fock_targets_batched import Fock_Targets

    sums: dict[int, torch.Tensor] = {}
    squared_sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    template: dict[str, object] | None = None
    scalar_indices: list[int] | None = None
    started = time.perf_counter()

    for start in range(0, num_train, batch_size):
        end = min(start + batch_size, num_train)
        rows = [database[index] for index in range(start, end)]
        atomic_numbers = [row[0] for row in rows]
        positions = [row[1] for row in rows]
        hamiltonians = [row[4] for row in rows]
        target_kwargs = {} if template is None else template
        targets = Fock_Targets(
            atomic_numbers,
            positions,
            rcut,
            copy.deepcopy(basis_sets.orbital_basis_def2_svp_nabla),
            hamiltonians,
            dataset_name="nablaDFT",
            dtype=dtype,
            scale_shift_data=None,
            **target_kwargs,
        )

        if template is None:
            scalar_indices = _scalar_indices(targets.req_output_irreps)
            if not scalar_indices:
                raise RuntimeError("NablaDFT target irreps contain no l=0 components")
            template = {
                "orbital_starts": targets.orbital_starts,
                "orbital_template": targets.orbital_template,
                "req_output_irreps": targets.req_output_irreps,
                "out_js_list": targets.out_js_list,
                "ls_list": targets.ls_list,
            }

        assert scalar_indices is not None
        scalar_index_tensor = torch.as_tensor(
            scalar_indices,
            dtype=torch.long,
            device=targets.node_labels_list[0].device,
        )
        for numbers, node_labels in zip(
            atomic_numbers,
            targets.node_labels_list,
            strict=True,
        ):
            values = (
                node_labels[0]
                .index_select(1, scalar_index_tensor)
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
            numbers_tensor = torch.as_tensor(numbers, dtype=torch.long)
            for element in torch.unique(numbers_tensor).tolist():
                element = int(element)
                element_values = values[numbers_tensor == element]
                if element not in sums:
                    sums[element] = torch.zeros(
                        len(scalar_indices), dtype=torch.float64
                    )
                    squared_sums[element] = torch.zeros_like(sums[element])
                    counts[element] = 0
                sums[element] += element_values.sum(dim=0)
                squared_sums[element] += element_values.square().sum(dim=0)
                counts[element] += int(element_values.shape[0])

        del targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        print(
            f"Processed train rows [{start}, {end}) / {num_train} "
            f"in {elapsed:.1f}s",
            flush=True,
        )

    means: dict[int, list[float]] = {}
    stds: dict[int, list[float]] = {}
    for element in sorted(sums):
        mean = sums[element] / counts[element]
        variance = squared_sums[element] / counts[element] - mean.square()
        std = variance.clamp_min(0.0).sqrt()
        std[std < 1.0e-4] = 1.0
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise RuntimeError(f"Non-finite statistics for element Z={element}")
        means[element] = mean.tolist()
        stds[element] = std.tolist()

    return {
        "element_scalar_means": means,
        "element_scalar_stds": stds,
        "scalar_irrep_indices": scalar_indices,
        "element_atom_counts": counts,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.num_train <= 0:
        raise SystemExit("--num-train must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.rcut <= 0:
        raise SystemExit("--rcut must be positive")

    dbpath = args.dbpath.expanduser().resolve()
    output = _resolve_output(args.output)
    if not dbpath.is_file():
        raise SystemExit(f"NablaDFT database not found: {dbpath}")

    sys.path.insert(0, str(SOURCE_ROOT))
    import torch

    from maloq.dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

    if not torch.cuda.is_available():
        raise SystemExit(
            "NablaDFT matrix-to-label conversion requires a visible CUDA GPU"
        )
    torch.cuda.set_device(0)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    database = HamiltonianDatabase(str(dbpath))
    if args.num_train > len(database):
        raise SystemExit(
            f"--num-train={args.num_train} exceeds database rows={len(database)}"
        )
    provenance = _provenance(
        dbpath,
        len(database),
        args.num_train,
        args.rcut,
        args.dtype,
    )
    if output.exists() and not args.overwrite:
        if _validate_existing(output, provenance):
            return

    payload = compute_statistics(
        database,
        args.num_train,
        args.rcut,
        dtype,
        args.batch_size,
    )
    payload["provenance"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    _validate_existing(output, provenance)
    print(f"Saved train-only scale-shift statistics: {output}")


if __name__ == "__main__":
    main()
