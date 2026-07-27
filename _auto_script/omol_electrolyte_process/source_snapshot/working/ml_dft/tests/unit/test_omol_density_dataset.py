from __future__ import annotations

import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _write_lmdb_shard(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=1 << 24, readonly=False, lock=True)
    try:
        with env.begin(write=True) as txn:
            for idx, row in enumerate(rows):
                txn.put(int(idx).to_bytes(4, "big"), pickle.dumps(row))
            txn.put(b"__len__", int(len(rows)).to_bytes(4, "big"))
    finally:
        env.close()
    path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "local_index": i,
                        "n_atoms": row.get("n_atoms", i + 1),
                        "nao": row.get("nao", 100 * (i + 1)),
                    }
                    for i, row in enumerate(rows)
                ],
            }
        )
        + "\n"
    )


def _pack_symmetric(matrix: np.ndarray) -> bytes:
    rows, cols = np.triu_indices(matrix.shape[0])
    return np.asarray(matrix[rows, cols], dtype=matrix.dtype).tobytes()


def _density_sample(
    atoms: list[int],
    positions: list[list[float]],
    *,
    dtype: np.dtype = np.dtype("float64"),
) -> dict:
    atoms_arr = np.asarray(atoms, dtype=np.int32)
    pos_arr = np.asarray(positions, dtype=np.float64)
    # H in def2-TZVPD has 9 active AOs in this loader's basis metadata.
    nao = 9 * len(atoms)
    density = np.eye(nao, dtype=dtype)
    overlap = np.eye(nao, dtype=dtype)
    return {
        "atomic_numbers": atoms_arr.tobytes(),
        "positions": pos_arr.tobytes(),
        "num_atoms": len(atoms),
        "charge": 0,
        "spin": 0,
        "density_matrix_packed": _pack_symmetric(density),
        "density_matrix_nao": nao,
        "density_matrix_dtype": str(dtype),
        "overlap_packed": _pack_symmetric(overlap),
        "overlap_nao": nao,
        "overlap_dtype": str(dtype),
    }


def test_omol_density_sharded_loader_uses_summary_lengths_and_lazy_lmdb(tmp_path):
    from ml_dft.data.datasets.omol_density import OMolDensityDataset

    split_dir = tmp_path / "train"
    _write_lmdb_shard(split_dir / "shard_000000.lmdb", [{"x": 0}, {"x": 1}])
    _write_lmdb_shard(split_dir / "shard_000001.lmdb", [{"x": 2}])

    ds = OMolDensityDataset(root=str(tmp_path), split="train", max_open_envs=1)
    try:
        assert len(ds) == 3
        assert len(ds._env_by_path) == 0
        assert (tmp_path / "train.shard_lengths.json").exists()

        assert ds.raw_sample(0)["x"] == 0
        assert len(ds._env_by_path) == 1

        assert ds.raw_sample(2)["x"] == 2
        assert len(ds._env_by_path) == 1
        assert next(iter(ds._env_by_path)).name == "shard_000001.lmdb"
        assert ds.shard_index_ranges() == [(0, 2), (2, 3)]
    finally:
        ds.close()


def test_omol_density_raw_sample_unpickles_from_zero_copy_lmdb_buffer(
    tmp_path,
    monkeypatch,
):
    from ml_dft.data.datasets.omol_density import OMolDensityDataset

    split_dir = tmp_path / "train"
    _write_lmdb_shard(split_dir / "shard_000000.lmdb", [{"x": 7}])
    ds = OMolDensityDataset(root=str(tmp_path), split="train")
    calls = []
    payload = pickle.dumps({"x": 7})

    class _Txn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def get(self, key):
            assert key == (0).to_bytes(4, "big")
            return memoryview(payload)

    class _Env:
        def begin(self, *, buffers=False):
            calls.append(buffers)
            return _Txn()

    try:
        monkeypatch.setattr(ds, "_open_env", lambda path: _Env())
        assert ds.raw_sample(0) == {"x": 7}
        assert calls == [True]
    finally:
        ds.close()


def test_omol_density_edge_cutoff_builds_sparse_transpose_closed_edges(tmp_path):
    from ml_dft.data.datasets.omol_density import OMolDensityDataset

    split_dir = tmp_path / "train"
    _write_lmdb_shard(
        split_dir / "shard_000000.lmdb",
        [
            _density_sample(
                [1, 1, 1],
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                ],
            ),
        ],
    )

    ds = OMolDensityDataset(
        root=str(tmp_path),
        split="train",
        basis="def2-tzvpd",
        elements=[1],
        edge_cutoff_A=2.0,
    )
    try:
        data = ds[0]
        assert data.num_unfiltered_edges.item() == 6
        assert data.edge_cutoff_A.item() == pytest.approx(2.0)
        assert data.edge_index_full.tolist() == [[0, 1], [1, 0]]
        assert data.non_diagonal_dm.shape[0] == 2
        assert data.non_diagonal_overlap.shape[0] == 2
    finally:
        ds.close()


def test_omol_density_size_filter_builds_lazy_index_from_summaries(tmp_path):
    from ml_dft.data.datasets.omol_density import OMolDensityDataset

    split_dir = tmp_path / "train"
    _write_lmdb_shard(
        split_dir / "shard_000000.lmdb",
        [
            {"x": 0, "n_atoms": 4, "nao": 100},
            {"x": 1, "n_atoms": 12, "nao": 500},
        ],
    )
    _write_lmdb_shard(
        split_dir / "shard_000001.lmdb",
        [
            {"x": 2, "n_atoms": 6, "nao": 200},
            {"x": 3, "n_atoms": 20, "nao": 700},
        ],
    )

    ds = OMolDensityDataset(
        root=str(tmp_path),
        split="train",
        max_atoms=10,
        max_nao=300,
        max_open_envs=1,
    )
    try:
        assert len(ds) == 2
        assert len(ds._env_by_path) == 0
        assert ds.shard_index_ranges() == [(0, 1), (1, 2)]
        assert ds.raw_sample(0)["x"] == 0
        assert ds.raw_sample(1)["x"] == 2
        assert len(ds._env_by_path) == 1
    finally:
        ds.close()
