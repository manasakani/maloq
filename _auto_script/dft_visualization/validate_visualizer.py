#!/usr/bin/env python3
"""End-to-end validation for the migrated DFT visualization backend."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

DFT_DATASET_DIR = Path("/dataset/seongsu/shared-home/projects/dft-dataset")
DFT_MONITOR_DIR = Path("/dataset/seongsu/shared-home/projects/dft-monitor")
SAMPLE_ID = "qh9_000002"
SMOKE_ID = "__sc26_density_smoke__"

os.environ.setdefault("DFT_SHARED_ROOT", "/dataset/seongsu/shared-home")
os.environ.setdefault("DFT_MONITOR_WEB_DIR", str(DFT_MONITOR_DIR))
os.environ.setdefault(
    "DFT_PREDICTIONS_ROOT",
    "/dataset/seongsu/shared-home/workspace/project/outputs/dft-visualization/predictions",
)
os.environ.setdefault(
    "DFT_CACHE_DIR",
    "/dataset/seongsu/shared-home/workspace/project/outputs/dft-visualization/cache",
)
sys.path.insert(0, str(DFT_DATASET_DIR))
os.chdir(DFT_DATASET_DIR)
server = importlib.import_module("server")


def main() -> None:
    index = json.loads((DFT_MONITOR_DIR / "data" / "index.json").read_text())
    assert index["n_samples"] == len(index["samples"])
    assert any(item["mol_id"] == SAMPLE_ID for item in index["samples"])

    sample = server._load_sample_dict(SAMPLE_ID)
    hamiltonian = np.asarray(sample["hamiltonian"], dtype=np.float64)
    overlap = np.asarray(sample["overlap"], dtype=np.float64)
    assert hamiltonian.shape == overlap.shape
    assert np.allclose(hamiltonian, hamiltonian.T, atol=1.0e-10)
    assert np.allclose(overlap, overlap.T, atol=1.0e-10)

    density = asyncio.run(server.get_density_matrix(SAMPLE_ID))
    assert density["shape"] == list(hamiltonian.shape)
    assert density["density_source"] == "hamiltonian"
    assert abs(density["trace_DS"] - density["n_electrons"]) < 1.0e-5

    grid = asyncio.run(server.get_density_grid(SAMPLE_ID, resolution=10))
    assert grid["shape"] == [10, 10, 10]
    assert "Cube file from DFT Dataset" in grid["cube_string"]
    assert grid["max_value"] > 0.0

    comparison = asyncio.run(
        server.compare_prediction(
            SAMPLE_ID,
            {"hamiltonian": hamiltonian.tolist(), "model_name": "self-check"},
        )
    )
    assert comparison["pred_eigenvalues"] == comparison["gt_eigenvalues"]

    smoke_sample_path = DFT_MONITOR_DIR / "data" / "samples" / f"{SMOKE_ID}.json"
    smoke_cache_path = Path(os.environ["DFT_CACHE_DIR"]) / f"{SMOKE_ID}_eigen.npz"
    try:
        supplied_density = 0.5 * np.asarray(density["density_matrix"], dtype=np.float64)
        smoke = dict(sample)
        smoke["mol_id"] = SMOKE_ID
        smoke["density_matrix"] = supplied_density.tolist()
        smoke_sample_path.write_text(json.dumps(smoke))
        direct = asyncio.run(server.get_density_matrix(SMOKE_ID))
        assert direct["density_source"] == "sample"
        assert np.allclose(np.asarray(direct["density_matrix"]), supplied_density)
    finally:
        smoke_sample_path.unlink(missing_ok=True)
        smoke_cache_path.unlink(missing_ok=True)

    print(
        "validated:",
        f"samples={index['n_samples']}",
        f"nao={hamiltonian.shape[0]}",
        f"trace(DS)={density['trace_DS']:.8f}",
        "grid=10x10x10",
        "direct-density=ok",
    )


if __name__ == "__main__":
    main()
