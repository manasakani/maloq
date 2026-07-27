from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import lmdb
import numpy as np


TOOLS = Path(__file__).resolve().parents[1]


def _manifest_row(mol_id: str, split: str) -> dict:
    return {
        "atomic_numbers": [1, 1],
        "charge": 0,
        "configuration_id": mol_id,
        "density_path": f"data/omol25/electronic/toy/{mol_id}/density_mat.npz",
        "elements": [1],
        "formula": "H2",
        "multiplicity": 1,
        "n_basis_orca": 18,
        "nsites": 2,
        "num_electrons_from_atoms": 2,
        "num_electrons_meta": 2,
        "parquet_file": "datasets/toy.parquet",
        "property_id": f"P_{mol_id}",
        "row_in_file": 0,
        "source": f"toy/{mol_id}/orca.tar.zst",
        "source_group": "toy",
        "spin": 0,
        "split": split,
        "unrestricted": False,
    }


def _pack(matrix: np.ndarray) -> bytes:
    upper = np.triu_indices(matrix.shape[0])
    return np.asarray(matrix[upper], dtype=np.float32).tobytes()


def _lmdb_sample(mol_id: str, marker: float = 1.0) -> dict:
    overlap = np.eye(18, dtype=np.float32)
    density = overlap / np.float32(9.0)
    return {
        "_basis_info": {
            "name": "def2-tzvpd",
            "angular_type": "spherical",
            "convention": "e3nn",
        },
        "_packed": True,
        "atomic_numbers": np.asarray([1, 1], dtype=np.int32).tobytes(),
        "charge": 0,
        "density_matrix_dtype": "float32",
        "density_matrix_nao": 18,
        "density_matrix_packed": _pack(density),
        "energy": marker,
        "formula": "H2",
        "initial_density_matrix_dtype": "float32",
        "initial_density_matrix_nao": 18,
        "initial_density_matrix_packed": _pack(density),
        "mol_id": mol_id,
        "num_atoms": 2,
        "overlap_dtype": "float32",
        "overlap_nao": 18,
        "overlap_packed": _pack(overlap),
        "positions": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            dtype=np.float64,
        ).tobytes(),
        "source": "toy",
        "spin": 0,
        "unrestricted": False,
        "xc": "toy",
    }


def _write_shard(
    root: Path,
    split: str,
    shard_index: int,
    rows: list[dict],
    *,
    marker: float,
) -> tuple[Path, Path]:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = split_dir / f"shard_{shard_index:06d}.lmdb"
    summary_path = split_dir / f"shard_{shard_index:06d}.summary.json"
    schema = {
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
        "split": split,
        "shard_index": shard_index,
    }
    environment = lmdb.open(str(lmdb_path), map_size=1 << 24)
    try:
        with environment.begin(write=True) as transaction:
            for local_index, row in enumerate(rows):
                transaction.put(
                    local_index.to_bytes(4, "big"),
                    pickle.dumps(_lmdb_sample(row["configuration_id"], marker)),
                )
            transaction.put(b"__len__", len(rows).to_bytes(4, "big"))
            transaction.put(b"__format__", b"pickle")
            transaction.put(b"__schema__", pickle.dumps(schema))
    finally:
        environment.close()

    samples = []
    for local_index, row in enumerate(rows):
        samples.append(
            {
                "local_index": local_index,
                "manifest": row,
                "n_atoms": 2,
                "n_electrons": 2,
                "nao": 18,
                "storage_dtype": "float32",
                "trace_error": 0.0,
                "trace_initial_density": 2.0,
                "trace_initial_error": 0.0,
                "trace_target": 2.0,
            }
        )
    summary_path.write_text(
        json.dumps(
            {
                "failure_count": 0,
                "failures": [],
                "lmdb": str(lmdb_path),
                "manifest_count": len(rows),
                "samples": samples,
                "shard_index": shard_index,
                "split": split,
                "written_count": len(rows),
            }
        )
        + "\n"
    )
    return lmdb_path, summary_path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class FullV2ToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_dir = self.root / "manifest"
        self.v1_root = self.root / "v1"
        self.v2_root = (
            self.root / "omol25_electrolyte_maloq_lmdb_corrected_v2_overlay"
        )
        self.manifest_dir.mkdir()
        (self.v1_root / "_index").mkdir(parents=True)

        self.a = _manifest_row("C_A", "train")
        self.b = _manifest_row("C_B", "train")
        self.c = _manifest_row("C_C", "val")
        by_split = {
            "train": [self.a, self.b],
            "val": [self.c],
            "test": [],
        }
        for split, rows in by_split.items():
            with (self.manifest_dir / f"{split}.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        (self.manifest_dir / "summary.json").write_text(
            json.dumps(
                {
                    "by_split": {"train": 2, "val": 1, "test": 0},
                    "counts": {"accepted": 3},
                }
            )
            + "\n"
        )

        v1_train_lmdb, v1_train_summary = _write_shard(
            self.v1_root, "train", 0, [self.a], marker=1.0
        )
        v1_val_lmdb, v1_val_summary = _write_shard(
            self.v1_root, "val", 0, [self.c], marker=1.0
        )
        v1_rows = {
            "train": [
                {
                    # Deliberately stale source paths exercise local basename mapping.
                    "lmdb": "/quasar/stale/train/shard_000000.lmdb",
                    "local_index": 0,
                    "mol_id": "C_A",
                    "nao": 18,
                    "summary": "/quasar/stale/train/shard_000000.summary.json",
                }
            ],
            "val": [
                {
                    "lmdb": str(v1_val_lmdb),
                    "local_index": 0,
                    "mol_id": "C_C",
                    "nao": 18,
                    "summary": str(v1_val_summary),
                }
            ],
            "test": [],
        }
        # Ensure stale basename resolution has a real local counterpart.
        self.assertEqual(
            v1_train_lmdb.name,
            Path(v1_rows["train"][0]["lmdb"]).name,
        )
        self.assertEqual(
            v1_train_summary.name,
            Path(v1_rows["train"][0]["summary"]).name,
        )
        for split, rows in v1_rows.items():
            with (self.v1_root / "_index" / f"{split}.index.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

        # Rebuild A as a corrected overlap and build the previously missing B.
        _write_shard(self.v2_root, "train", 10, [self.a, self.b], marker=2.0)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_end_to_end_precedence_view_and_full_verification(self) -> None:
        index_root = self.v2_root / "_full_index"
        view_root = (
            self.root / "omol25_electrolyte_maloq_lmdb_corrected_full_v2"
        )

        dry = _run(
            str(TOOLS / "build_full_v2_index.py"),
            "--manifest-dir",
            str(self.manifest_dir),
            "--v1-root",
            str(self.v1_root),
            "--v2-root",
            str(self.v2_root),
            "--out-index-root",
            str(index_root),
            "--dry-run",
        )
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertFalse(index_root.exists())
        dry_report = json.loads(dry.stdout)
        self.assertEqual(dry_report["indexed_total"], 3)
        self.assertEqual(dry_report["v2_overlap_replacements"], 1)

        built = _run(
            str(TOOLS / "build_full_v2_index.py"),
            "--manifest-dir",
            str(self.manifest_dir),
            "--v1-root",
            str(self.v1_root),
            "--v2-root",
            str(self.v2_root),
            "--out-index-root",
            str(index_root),
        )
        self.assertEqual(built.returncode, 0, built.stderr)
        train_rows = [
            json.loads(line)
            for line in (index_root / "train.index.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [row["source_version"] for row in train_rows],
            ["v2_rebuilt", "v2_rebuilt"],
        )
        val_row = json.loads((index_root / "val.index.jsonl").read_text())
        self.assertEqual(val_row["source_version"], "v1_reference")

        view = _run(
            str(TOOLS / "build_full_v2_view.py"),
            "--index-root",
            str(index_root),
            "--manifest-dir",
            str(self.manifest_dir),
            "--out-view-root",
            str(view_root),
        )
        self.assertEqual(view.returncode, 0, view.stderr)
        self.assertTrue((view_root / "train.index.jsonl").is_symlink())
        view_train_rows = [
            json.loads(line)
            for line in (view_root / "_index" / "train.index.jsonl")
            .read_text()
            .splitlines()
        ]
        for row in view_train_rows:
            self.assertTrue(Path(row["lmdb"]).is_symlink())
            self.assertTrue(Path(row["summary"]).is_symlink())
            self.assertTrue(str(row["lmdb"]).startswith(str(view_root)))

        ml_dft_source = Path("/dataset/seongsu/shared-home/projects/ml_dft/src")
        sys.path.insert(0, str(ml_dft_source))
        from ml_dft.data.datasets.omol_density import OMolDensityDataset

        dataset = OMolDensityDataset(
            root=str(view_root),
            split="train",
            basis="def2-tzvpd",
            elements=[1],
        )
        try:
            self.assertEqual(len(dataset), 2)
            loaded = dataset[0]
            self.assertEqual(int(loaded.num_nodes), 2)
            self.assertEqual(loaded.diagonal_dm.shape[0], 2)
        finally:
            dataset.close()

        verified = _run(
            str(TOOLS / "verify_full_v2.py"),
            "--manifest-dir",
            str(self.manifest_dir),
            "--index-root",
            str(view_root / "_index"),
            "--v2-root",
            str(self.v2_root),
            "--view-root",
            str(view_root),
            "--mode",
            "full",
            "--mark-complete",
            str(view_root),
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)
        report = json.loads(verified.stdout)
        self.assertEqual(report["indexed_total"], 3)
        self.assertEqual(report["checked_records"], 3)
        self.assertEqual(report["limits"]["density_trace_error"], 0.05)
        self.assertTrue(report["all_records_checked"])
        self.assertTrue((view_root / "COMPLETE").is_file())
        self.assertTrue((view_root / "verification.json").is_file())

    def test_existing_destination_is_never_replaced(self) -> None:
        index_root = self.v2_root / "_full_index"
        index_root.mkdir()
        sentinel = index_root / "sentinel"
        sentinel.write_text("keep")
        result = _run(
            str(TOOLS / "build_full_v2_index.py"),
            "--manifest-dir",
            str(self.manifest_dir),
            "--v1-root",
            str(self.v1_root),
            "--v2-root",
            str(self.v2_root),
            "--out-index-root",
            str(index_root),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), "keep")

        forbidden = self.v1_root / "_forbidden_full_index"
        inside_v1 = _run(
            str(TOOLS / "build_full_v2_index.py"),
            "--manifest-dir",
            str(self.manifest_dir),
            "--v1-root",
            str(self.v1_root),
            "--v2-root",
            str(self.v2_root),
            "--out-index-root",
            str(forbidden),
        )
        self.assertNotEqual(inside_v1.returncode, 0)
        self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
