# NablaDFT QHFlow3 overlap and NTE grid ablations

These two experiments isolate candidate causes of the validation gap between
the completed QHFlow3 V2 run (`zqs1eohc`) and NTE-64/2 V1 run (`loaiifgp`).
They use the native NablaDFT database at
`/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
with 12,081 training molecules, 64 validation molecules, and no test split.

Both full runs use the SC26 environment
`/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26`,
two data-parallel GPUs, micro-batch 5 per rank, gradient accumulation 2,
effective batch 20, Muon, seed 44, 20 epochs, SS0, and W&B
`kaist-korea/maloq-nablaDFT` logging every 10 optimizer steps.

## QHFlow3 V2 without overlap

This changes only `qhflow3_use_overlap` from `true` to `false`. The source
NablaDFT row still contains an overlap matrix, but the QHFlow3 backbone neither
extracts nor contracts it.

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/01_nabladft_qhflow3_v2_no_overlap_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/01_nabladft_qhflow3_v2_no_overlap_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/01_nabladft_qhflow3_v2_no_overlap_2gpu.sh full 0,1
```

Full outputs go to
`/dataset/seongsu/shared-home/workspace/project/outputs/nabla-qhf3-muon-ss0-ov0-v2/`.
The W&B display name is
`NablaDFT | QHFlow3 | Muon | SS0 | OV0 | V2`.

## NTE-64/2 with a 48x48 grid

This changes only `esen_grid_resolution` from `null`, which gives the lmax=4
default 10x11 grid, to `48`. The model width, NodeBlock x3, EdgeBlock x2,
bounded-degree LayerScale, head, optimizer, and training schedule remain the
same as NTE-64/2 V1.

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/02_nabladft_nte64e2_grid48_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/02_nabladft_nte64e2_grid48_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/02_nabladft_nte64e2_grid48_2gpu.sh full 0,1
```

Full outputs go to
`/dataset/seongsu/shared-home/workspace/project/outputs/nabla-nte64e2-muon-ss0-g48-v1/`.
The W&B display name is
`NablaDFT | NTE-64/2 | Muon | SS0 | Grid48 | V1`.

Successful smoke artifacts are removed. Failed smoke artifacts are retained.
The launchers reject missing/busy GPUs and refuse to overwrite an existing
compact output directory.

## NTE-64/2 with overlap-only conditioning

`NTE-Scond` keeps NTE's existing atom/charge/spin scalar and adds only the
basis-aware `ParamContraction(S)` plus its channel-linear projection before
edge-degree embedding. Downstream NodeBlock x3, EdgeBlock x2, and the
MALOQ-Muon head are unchanged.

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/03_nabladft_nte64e2_scond_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/03_nabladft_nte64e2_scond_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/03_nabladft_nte64e2_scond_2gpu.sh full 0,1
```

Full outputs go to
`/dataset/seongsu/shared-home/workspace/project/outputs/nabla-nte64e2-muon-ss0-scond-v1/`.
The W&B display name is
`NablaDFT | NTE-64/2 | Muon | SS0 | Scond | V1`.

## NTE-64/2 with QHFlow3 input mixing

`NTE-QHFcond` replaces NTE's initial scalar with QHFlow3's active
zero-H/S/time/atom/zero-charge/zero-spin mixing. It does not copy QHFlow3's
later FiLM layers or pair-xy2 blocks; all downstream computation remains NTE.
For this absolute NablaDFT run, H is an explicit zero on-site block and S is
the sample overlap.

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/04_nabladft_nte64e2_qhflow3_conditioning_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/04_nabladft_nte64e2_qhflow3_conditioning_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/04_nabladft_nte64e2_qhflow3_conditioning_2gpu.sh full 0,1
```

Full outputs go to
`/dataset/seongsu/shared-home/workspace/project/outputs/nabla-nte64e2-muon-ss0-qcond-v1/`.
The W&B display name is
`NablaDFT | NTE-64/2 | Muon | SS0 | QHFcond | V1`.

## Comparison map

| Lane | Changed axis | Existing/reference run |
|---|---|---|
| NTE-base | none | `loaiifgp` |
| NTE-Scond | overlap-only input | new lane 03 |
| NTE-QHFcond | QHFlow3 input mixing | new lane 04 |
| NTE-Grid48 | SO(3) grid only | new lane 02 |
| QHF-base | none | `zqs1eohc` |
| QHF-OV0 | overlap disabled | new lane 01 |

Full training has not been launched.

## Validation status

On 2026-07-25 (Asia/Seoul), all four launchers passed `validate`. Each new
model path also passed a reduced single-GPU 2-train/1-validation integration
smoke through data loading, forward/backward, a Muon update, and validation
matrix reconstruction; successful smoke outputs were discarded. The focused
QH9/Nabla regression suite passed 30 tests. Full-size two-GPU smoke and full
20-epoch training have not been launched.
