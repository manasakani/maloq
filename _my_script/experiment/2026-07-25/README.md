# NablaDFT QHFlow3 overlap and NTE grid ablations

These four experiments isolate candidate causes of the validation gap between
the completed QHFlow3 V2 run (`zqs1eohc`) and NTE-64/2 V1 run (`loaiifgp`).
They use the native NablaDFT database at
`/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
with 12,081 training molecules, 64 validation molecules, and no test split.

All four full runs use the SC26 environment
`/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26`,
two data-parallel GPUs, micro-batch 5 per rank, gradient accumulation 2,
effective batch 20, Muon, seed 44, 20 epochs, RAW targets, and W&B
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
`NablaDFT | QHFlow3 | Muon | RAW | OV0 | V2`.

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
`NablaDFT | NTE-64/2 | Muon | RAW | Grid48 | V1`.

The original `MB5 x 2 GPUs x GA2 = EB20` full run failed with CUDA OOM in the
grid MLP. W&B run `c9yy08ci` was deleted on request; its local output is
retained. The first retry (`MB2 x 2 GPUs x GA5`) reached 78.2 GiB peak on a
real full-data batch and left the two ranks out of sync, so it was stopped.
The replacement preserves EB20 as `MB1 x 2 GPUs x GA10`:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/07_nabladft_nte64e2_grid48_oom_retry_mb1_ga10_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/07_nabladft_nte64e2_grid48_oom_retry_mb1_ga10_2gpu.sh smoke 4,5
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/07_nabladft_nte64e2_grid48_oom_retry_mb1_ga10_2gpu.sh full 4,5
```

The retry output is
`/dataset/seongsu/shared-home/workspace/project/outputs/nabla-nte64e2-muon-ss0-g48-mb1ga10-r2/`.

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
`NablaDFT | NTE-64/2 | Muon | RAW | Scond | V1`.

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
`NablaDFT | NTE-64/2 | Muon | RAW | QHFcond | V1`.

## Comparison map

| Lane | Changed axis | Existing/reference run |
|---|---|---|
| NTE-base | none | `loaiifgp` |
| NTE-Scond | overlap-only input | new lane 03 |
| NTE-QHFcond | QHFlow3 input mixing | new lane 04 |
| NTE-Grid48 | SO(3) grid only | new lane 02 |
| QHF-base | none | `zqs1eohc` |
| QHF-OV0 | overlap disabled | new lane 01 |
| QHF-OV0-NTEGrid | QHFlow3 with NTE default 10x11 grid | new lane 06 |

Full training was launched in parallel at 2026-07-25 05:50 KST:

| Lane | Host | GPUs | tmux | W&B |
|---|---|---|---|---|
| QHF-OV0 | `usr310-gpumngc-01` | `0,1` | `sc26-ablation-01-qhf-ov0` | `g0l50g72` |
| NTE-Grid48 A1 (OOM) | `usr310-gpumngc-01` | `4,5` | completed/failed | `c9yy08ci` (deleted) |
| NTE-Grid48 R1 (MB2 GA5) | `usr310-gpumngc-01` | `4,5` | stopped (unsafe peak) | `7gupb927` (deleted) |
| NTE-Grid48 R2 (MB1 GA10) | `usr310-gpumngc-01` | `4,5` | stopped; replaced by Grid24 | `40swc0ck` (deleted) |
| QHF-OV0-NTEGrid | `usr310-gpumngc-01` | `6,7` | `sc26-ablation-06-qhf-ov0-ntegrid` | `80sa5m4j` |
| NTE-Scond | `usr310-gpumngc-02` | `0,1` | `sc26-ablation-03-nte-scond` | `xwekzbsw` |
| NTE-QHFcond | `usr310-gpumngc-02` | `3,4` | `sc26-ablation-04-nte-qcond` | `fao0946w` |

The remaining runs use separate output directories and execute concurrently.

## Validation status

On 2026-07-25 (Asia/Seoul), all four launchers passed `validate`. Each new
model path also passed a reduced single-GPU 2-train/1-validation integration
smoke through data loading, forward/backward, a Muon update, and validation
matrix reconstruction; successful smoke outputs were discarded. The focused
QH9/Nabla regression suite passed 30 tests. Full-size two-GPU smoke was not
run; the explicitly requested full 20-epoch runs are now active in the
parallel allocation above.

## Fixed-workflow checkpoint and resume experiment

`TrainingWorkflowFixed` is an opt-in continuation path. It leaves the legacy
workflow as the default and writes one atomic `training_state.pt` containing
both models, optimizer, scheduler, completed epoch, loss history, semantic
configuration signature, W&B run ID, and one RNG state per distributed rank.
The prior valid generation is retained as `training_state.prev.pt`.

The two-GPU smoke performs a complete interruption-boundary test: stage 1
trains one epoch and saves, while stage 2 starts in a fresh output directory,
loads stage 1, and trains only the remaining epoch of a two-epoch target. It
checks checkpoint lineage and optimizer-step continuity, then removes both
successful smoke outputs. Failed evidence is retained.

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/08_nabladft_nte_fixed_workflow_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/08_nabladft_nte_fixed_workflow_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/08_nabladft_nte_fixed_workflow_2gpu.sh full 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/08_nabladft_nte_fixed_workflow_2gpu.sh resume /absolute/path/to/run 0,1
```

Full and resumed outputs use timestamped directories below
`/dataset/seongsu/shared-home/workspace/project/outputs/`. Resume requires the
same model, dataset, optimizer, batch/accumulation settings, and world size as
the source checkpoint. It resumes the original W&B run when the source
metadata contains its ID.

The implementation resumes only at completed epoch boundaries. An abrupt stop
during an epoch rolls back to the previous completed epoch; it does not restore
a partial gradient-accumulation window.

Validation on 2026-07-25 passed 37 focused tests, including a two-rank Gloo
checkpoint collective and rank-specific RNG collection. A single-GPU
NablaDFT integration run also completed a planned two-epoch schedule as
one epoch, clean stop, and one resumed epoch; optimizer steps advanced from 2
to 4, scheduler state advanced from 2 to 4, and the two-entry loss history was
preserved. Its temporary outputs were discarded. The launcher passed a real
two-rank MPI configuration validation with world size 2 and effective batch
20. The CUDA/NCCL two-GPU smoke and full run have not been launched.

## NablaDFT l=0 shift-only baselines

These four runs restore the original MALOQ target treatment: subtract the
element-specific mean from node-block `l=0` components, without dividing by
their standard deviations. Node `l>0` components and all edge components are
unchanged. The configuration is recorded as
`scale_and_shift: true` plus `scale_shift_mode: shift_only`.

The historical runs used the ambiguous `SS0` display label with
`scale_and_shift: false`, so neither shift nor scale was applied. Their W&B
display names are now `RAW`, while these four mean-subtraction runs are
`SHIFT`. Metrics, checkpoints, and legacy output paths are preserved. The new
internal output names include `shift-only` to prevent checkpoint collisions.

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/09_nabladft_shift_only_baselines_2gpu.sh validate qhflow3
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/09_nabladft_shift_only_baselines_2gpu.sh smoke qhflow3 2,3
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/09_nabladft_shift_only_baselines_2gpu.sh full qhflow3 2,3
```

Replace `qhflow3` with `nte128`, `nte64`, or `maloq` for the other models.
All runs use NablaDFT rows 0–12080 for training, rows 12081–12144 for
validation, two data-parallel GPUs, micro-batch 5 per rank, gradient
accumulation 2, effective batch 20, Muon, seed 44, and 20 epochs.

All four configs passed validation and a two-GPU, one-epoch, 20-train /
20-validation full-model smoke. Each smoke completed forward, backward, Muon
update, validation loss, and matrix reconstruction; successful smoke
checkpoint directories were removed.

Full launch allocation at 2026-07-25 06:42 KST:

| Model | Host | GPUs | tmux | W&B / queue |
|---|---|---|---|---|
| QHFlow3 V2 | `usr310-gpumngc-01` | `2,3` | `sc26-shift-qhf3-v2` | `wrsptkz2` |
| NTE-128/3 V1 | `usr310-gpumngc-01` | `4,5` | `sc26-shift-nte128-v1` | `olqbt8jr` |
| NTE-64/2 V1 | `usr310-gpumngc-02` | `6,7` | `sc26-shift-nte64-then-maloq` | `9b9we9ts` |
| MALOQ V1 | `usr310-gpumngc-02` | `6,7` | same sequential queue | queued after NTE-64/2 |

The durable queue status is
`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-shift-only-usr310-gpumngc-02-g6-7/status.tsv`.
