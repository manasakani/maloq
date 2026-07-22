# NablaDFT and QH9 Hamiltonian/density training

## Verified source datasets

- NablaDFT: `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
  contains 12,145 rows and already exactly matches MALOQ's native
  `HamiltonianDatabase` schema. It should not be copied or converted to ASE.
- QH9 matrices: `/dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/data.lmdb`
  contains 130,831 rows in split-key LMDB format. Its schema advertises
  B3LYP5/def2-SVP PySCF-convention H/S/D/H0/D0 matrices.
- QH9Stable split: the existing official random split has 104,664 train,
  13,083 validation, and 13,084 test indices.

## Commands

Inspect and copy exactly one complete block from:

```bash
cd /dataset/seongsu/shared-home/workspace/project
sed -n '1,260p' \
  _my_script/experiment/2026-07-22/02_nabladft_qh9_density_script.sh
```

- `(a)` validates NablaDFT without training.
- `(b)` runs one small CUDA epoch on NablaDFT for all three models.
- `(c)` launches the corresponding three-model production comparison.
- `(d)` creates the density-target QH9 2/1/1 converted smoke DB once.
- `(e1)` and `(e2)` run one small CUDA delta-learning epoch for Hamiltonian
  and density respectively, using all three models.
- `(f)` performs the full density-target matrix conversion once.
- `(g1)` and `(g2)` launch the full Hamiltonian and density delta-learning
  comparisons after `(f)` completes.

All run artifacts are written below
`outputs/<dataset>-<absolute|delta>-<scope>-seed44-<timestamp>/`.
Successful smoke output directories are deleted automatically; failed smoke
directories remain. Use `--keep-smoke-output` only for explicit debugging.

## Model scope

NablaDFT uses the native loader and supports the MALOQ, MALOQ-NTE, and
QHFlow3 lanes. QHFlow3 selects the `def2-svp-nabla` layout, including S, Cl,
and Br, with up to 32 padded AOs per atom. Its pair trunk uses a fully connected
molecular graph and currently supports molecule data parallelism rather than
the separate distributed-graph mode.

The two QH9 tasks now use separate contracts. `QH9Stable_random.db` exposes
`H/H0/S` as the Hamiltonian-delta target and retains D0 only as QHFlow3
conditioning. `QH9StableMatrices_random.db` exposes `D/D0/S` as the
density-delta target and retains H0 only as QHFlow3 conditioning. The native
loader applies `Fock_Targets`/`matrix2labels` to the final and matching initial
matrix. Since this map is linear, delta labels are exactly
`label(M_final) - label(M_initial) = label(M_final - M_initial)`.

With `--delta-learning`, the trainer evaluates either
`H_pred = H0 + delta_H` or `D_pred = D0 + delta_D` against the final matrix.
QHFlow3 conditions on both initial matrices and overlap: the initial matrix
matching the target is primary and the other one is auxiliary. MALOQ and
MALOQ-NTE retain their original geometry trunks and use the matching initial
matrix as the residual baseline. Final H/D never enters a trunk. Omitting
`--delta-learning` keeps absolute-matrix comparisons available.

## Status

- NablaDFT native schema: verified.
- QH9 density LMDB schema, sample H/S/D shapes, and `trace(D S)=N_e`: verified.
- Hamiltonian-delta QH9 2/1/1 conversion: completed and validated at
  `/dataset_tmp/qh9_maloq_ase_verification/QH9StableHamiltonianDelta_random_2_1_1.db`.
- Density-delta QH9 2/1/1 conversion: completed and validated at
  `/dataset_tmp/qh9_matrix_maloq_ase/QH9StableMatrices_random_2_1_1.db`.
- NablaDFT MALOQ/MALOQ-NTE CUDA train+validation smoke: passed; summary at
  `outputs/nabladft-smoke-verification-20260722-v2/comparison.json`.
- NablaDFT QHFlow3 2-GPU train+validation smoke: passed; summary at
  `outputs/nabladft-qhflow3-2gpu-eb20-smoke-seed44-20260722-052320/comparison.json`.
- QH9 Hamiltonian delta-learning three-model CUDA smoke: passed; summary at
  `outputs/qh9-hamiltonian-delta-smoke-verification-20260722/comparison.json`.
- QH9 density delta-learning three-model CUDA smoke: passed; summary at
  `outputs/qh9-density-delta-matrixdb-smoke-verification-20260722/comparison.json`.
- `QH9StableMatrices_random.db` now advertises density delta only; the metadata
  audit is in `outputs/qh9stable-density-profile-20260722/`.
- The Hamiltonian-delta replacement for `QH9Stable_random.db` completed all
  130,831 rows and was promoted after validation. The former absolute-only DB
  is archived as `QH9Stable_random.absolute-only-20260722.db`.
- The scp-gpu-2 six-lane full-size batch-32 smoke passed for H/D ×
  MALOQ/MALOQ-NTE/QHFlow3; successful smoke outputs were removed.
- Production training: not launched.

Monitor the full matrix conversion with:

```bash
cd /dataset/seongsu/shared-home/workspace/project
tail -f outputs/qh9stable-matrices-full-conversion-20260722/conversion.log
pgrep -af process_qh9_matrix_lmdb_to_maloq_ase.py
ls -lh /dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9StableMatrices_random*
```

For the current batch-32, six-lane `scp-gpu-2` workflow, use
`06_qh9stable_delta_scp_gpu2_bs32.sh`; the older `(g1)`/`(g2)` blocks are kept
only as serial local examples.
