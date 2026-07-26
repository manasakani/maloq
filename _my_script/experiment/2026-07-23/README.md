# QH9Stable OOM recovery with two-step gradient accumulation

This experiment restarts only the four 2026-07-22 QH9Stable lanes that failed
before completing epoch 1:

- Hamiltonian: MALOQ on GPU 0, MALOQ-NTE on GPU 1, QHFlow3 on GPU 2
- Density: MALOQ-NTE on GPU 4

The surviving density MALOQ (GPU 3) and density QHFlow3 (GPU 5) runs are not
modified. The recovery uses micro-batch 16 and two-step gradient accumulation,
for effective batch 32. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set
to reduce allocator fragmentation.

Environment:

```text
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
```

Dataset and split:

- QH9Stable Hamiltonian DB:
  `/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db`
- QH9Stable density DB:
  `/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9StableMatrices_random.db`
- official ordered split: 104664 train / 13083 validation / 13084 test
- delta learning from the matching initial matrix

Commands on `scp-gpu-2`:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/01_qh9stable_oom_recovery_mb16_ga2.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/01_qh9stable_oom_recovery_mb16_ga2.sh smoke
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/01_qh9stable_oom_recovery_mb16_ga2.sh full
```

Full outputs are written below:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-full-seed44-<timestamp>/
```

W&B uses entity `kaist-korea`, project `maloq-qh9`, and logs every 10 optimizer
steps. Smoke output is deleted after all four lanes pass; failed smoke output is
retained for diagnosis.

Initial GA8 attempt on 2026-07-23 (Asia/Seoul):

- configuration validation passed for all four lanes
- production-size one-epoch smoke passed for all four lanes; temporary outputs
  were removed
- full 80-epoch recovery launched at `20260723-040732`, then intentionally
  stopped during DB loading to apply the requested `grad_acc=2`
- full output:
  `/dataset/seongsu/shared-home/workspace/project/outputs/qh9stable-oom-recovery-four-lane-mb4-ga8-eb32-full-seed44-20260723-040732/`
- full launcher PID: `2769245`; worker PIDs: `2769252` through `2769255`
- initial online W&B connection passed for all four lanes

GA2 status on 2026-07-23 (Asia/Seoul):

- configuration validation passed for all four lanes
- production-size one-epoch smoke passed for all four lanes; temporary outputs
  were removed
- full 80-epoch recovery launched at `20260723-041305`
- full output:
  `/dataset/seongsu/shared-home/workspace/project/outputs/qh9stable-oom-recovery-four-lane-mb4-ga2-eb8-full-seed44-20260723-041305/`
- full launcher PID: `2771071`; worker PIDs: `2771078` through `2771081`
- initial online W&B connection passed for all four lanes
- this effective-batch-8 run was intentionally stopped when the requested
  effective batch was clarified to be 32

Corrected GA2/effective-batch-32 configuration:

- micro-batch 16
- gradient accumulation 2
- effective batch 32
- output prefix:
  `/dataset/seongsu/shared-home/workspace/project/outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-<scope>-seed44-<timestamp>/`
- configuration validation passed for all four lanes
- production-size one-epoch CUDA smoke passed for all four lanes; temporary
  outputs were removed
- full 80-epoch recovery launched at `20260723-133856`
- full output:
  `/dataset/seongsu/shared-home/workspace/project/outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-full-seed44-20260723-133856/`
- full launcher PID: `2847531`; worker PIDs: `2847538` through `2847541`

## NablaDFT QHFlow3 local matrix objective

QHFlow3 now consumes the same directed `rcut_orbitals=8.0` pair graph as the
MALOQ baselines. Pair blocks outside the cutoff are not predicted or included
in the local loss, and both prediction and label reconstruction leave those
blocks at zero.

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/02_nabladft_qhflow3_local_muon_head_2gpu_mb5_ga2.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/02_nabladft_qhflow3_local_muon_head_2gpu_mb5_ga2.sh smoke
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/02_nabladft_qhflow3_local_muon_head_2gpu_mb5_ga2.sh full
```

The full run uses two GPUs, micro-batch 5, gradient accumulation 2, effective
batch 20, 20 epochs, seed 44, the QHFlow3-clean trunk, and the MUON-compatible
MALOQ head. Outputs are written below:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-<scope>-seed44-<timestamp>/
```

Loader-fix status on 2026-07-23 (Asia/Seoul):

- the first full attempt at `20260723-151840` stopped during loader creation
  because the NablaDFT branch did not initialize optional QHFlow3 initial
  matrices
- failed output was preserved at
  `/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-151840/`
- the shared loader now explicitly leaves those optional matrices absent for
  non-delta NablaDFT instead of reading undefined branch-local variables
- a single-GPU end-to-end smoke with 2 training and 1 validation molecule
  passed through data loading, QHFlow3, the MUON head, optimizer updates, and
  validation matrix metrics
- smoke output:
  `/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-qhflow3-local-loader-fix-smoke-seed44-after-20260723-151840/`
- the two-GPU full run was not restarted automatically

## MALOQ-NTE do=128 / Le=3 head comparison

This controlled NablaDFT experiment increases only MALOQ-NTE's output width
and edge depth to match MALOQ (`dt/do=128/128`, `Ln/Le=3/3`). All other NTE
choices remain unchanged: node-then-edge scheduling, grid MLPs, sigmoid gates,
edge envelope/scalar modulation, and bounded-degree LayerScale initialized to
1/64.

The two lanes differ only in the prediction head:

| Lane | `head_type` | Corrected Muon head |
| --- | --- | --- |
| `native-head` | `maloq` | No |
| `muon-head` | `maloq_muon` | Yes |

Both use the fixed optimizer routing rule: every trainable parameter with
`ndim >= 2` uses Muon and the remaining one-dimensional parameters use AdamW.
The corrected head converts the final node/edge contraction weights to
semantic 2D matrices; the native head intentionally retains its original
flattened representation. This isolates the corrected-head effect.

Dataset and schedule:

- native NablaDFT `train_2k.db`
- 12,081 train / 64 validation / 0 test
- 20 epochs, seed 44
- native head: two data-parallel ranks on GPUs `0,1`
- corrected Muon head: two data-parallel ranks on GPUs `2,3`
- both lanes start concurrently with separate distributed master ports
- micro-batch 5, gradient accumulation 2, effective batch 20
- W&B `kaist-korea/maloq-nablaDFT`, every 10 optimizer steps

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/03_nabladft_maloq_nte_do128_le3_head_comparison.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/03_nabladft_maloq_nte_do128_le3_head_comparison.sh smoke
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/03_nabladft_maloq_nte_do128_le3_head_comparison.sh full
```

The two lanes run concurrently. Override their GPU pairs with `NATIVE_GPUS`
and `MUON_GPUS`, or their ports with `NATIVE_MASTER_PORT` and
`MUON_MASTER_PORT`. Full outputs, per-lane logs/status, and the merged
`comparison.csv` are written below
`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-maloq-nte-do128-le3-head-comparison-parallel-2x2gpu-eb20-mb5-ga2-<scope>-seed44-<timestamp>/`.
Successful smoke output is deleted; failed smoke output is retained. Config
validation is complete. Exact full-size single-GPU 2-train/1-validation smoke
passed for both heads, and its temporary outputs were deleted. Both models have
33,750,157 parameters. The native head routes 33,572,480 parameters through
Muon; the corrected head routes 33,623,168, moving the 50,688 semantic output
weights from AdamW to Muon without changing model size. The parallel four-GPU
comparison smoke and full training have not been launched automatically.

## Independent scale-shift factorial comparison

The head and scale-shift factors can also be compared as four independent
two-GPU runs. These scripts do not launch one another and stay in the
foreground. The GPU pair is either the second command-line argument or the
editable `DEFAULT_GPUS` value near the top of each script. Use scripts
`04`-`07` for this requested independent workflow; script `03` remains only as
the earlier combined-launch record.

| Script | Head | Scale-shift | Default GPUs |
| --- | --- | --- | --- |
| `04_nabladft_nte_native_head_no_scale_shift_2gpu.sh` | native | off | `6,7` |
| `05_nabladft_nte_native_head_scale_shift_2gpu.sh` | native | on | `6,7` |
| `06_nabladft_nte_muon_head_no_scale_shift_2gpu.sh` | corrected Muon | off | `6,7` |
| `07_nabladft_nte_muon_head_scale_shift_2gpu.sh` | corrected Muon | on | `6,7` |

The scale-shift artifact is computed once from only ordered training indices
`[0, 12081)`. For each element and each equivariant `l=0` node-label component,
training uses `(value - mean) / std`; validation matrix metrics undo the
transformation before reporting MAE. Edge labels and all `l>0` components are
unchanged. The saved artifact includes its database, split, target, basis,
cutoff, dtype, and normalization provenance.

Prepare the shared statistics once on an available GPU:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/05_nabladft_nte_native_head_scale_shift_2gpu.sh prepare 6,7
```

Then run each experiment independently, choosing its GPU pair:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/04_nabladft_nte_native_head_no_scale_shift_2gpu.sh full 6,7
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/05_nabladft_nte_native_head_scale_shift_2gpu.sh full 6,7
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/06_nabladft_nte_muon_head_no_scale_shift_2gpu.sh full 6,7
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/07_nabladft_nte_muon_head_scale_shift_2gpu.sh full 6,7
```

Replace `full` with `validate` or `smoke` as needed. Every training run uses
micro-batch 5, two data-parallel ranks, gradient accumulation 2, effective
batch 20, 20 epochs, seed 44, and W&B
`kaist-korea/maloq-nablaDFT` logging every 10 optimizer steps. Full outputs are
written to compact, configuration-specific directories:

| Head | Normalization | Legacy output directory |
| --- | --- | --- |
| native | RAW | `nabla-nte128e3-native-ss0-v1` |
| native | SHIFT+STD | `nabla-nte128e3-native-ss1-v1` |
| corrected Muon | RAW | `nabla-nte128e3-muon-ss0-v1` |
| corrected Muon | SHIFT+STD | `nabla-nte128e3-muon-ss1-v1` |

Each directory lives directly below
`/dataset/seongsu/shared-home/workspace/project/outputs/`. Repeated launches
refuse to overwrite an existing result.
Successful smoke outputs are removed; failed outputs are retained. The
train-12,081 statistics and multi-GPU smoke/full runs are not launched
automatically. A two-training/one-validation, full-size corrected-head CUDA
smoke passed with a temporary two-row scale-shift artifact; both the smoke
output and temporary artifact were discarded.

Scripts `04`-`07` run on any host by default and still validate that the
selected GPU indices exist and have at most 1024 MiB allocated. Set
`EXPECTED_HOST=<hostname>` only when a run should be pinned to one host.
All project, config, environment, dataset, statistics, and output-base paths
are stored as canonical absolute paths for the shared SC26 layout.

## QHFlow3 scale-shift ablation

Added on 2026-07-24 (Asia/Seoul). This pair keeps the QHFlow3-clean trunk,
corrected `maloq_muon` head, local `rcut_orbitals=8.0` matrix objective,
optimizer, seed, and training schedule fixed. The only intended training
difference is whether train-only `l=0` node-label standardization is enabled.

| Script | Scale-shift | Default GPUs |
| --- | --- | --- |
| `08_nabladft_qhflow3_muon_head_no_scale_shift_2gpu.sh` | off | `0,1` |
| `09_nabladft_qhflow3_muon_head_scale_shift_2gpu.sh` | on | `0,1` |

The scale-shift lane reuses
`/dataset/seongsu/shared-home/workspace/project/outputs/scale-shift-statistics/nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt`.
This artifact was computed only from ordered training indices `[0, 12081)`.
Validation matrix MAE is reported after undoing the transformation, so the two
lanes remain comparable in the original atomic units.

Prepare or validate the shared statistics, then validate both configurations:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/09_nabladft_qhflow3_muon_head_scale_shift_2gpu.sh prepare 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/08_nabladft_qhflow3_muon_head_no_scale_shift_2gpu.sh validate 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/09_nabladft_qhflow3_muon_head_scale_shift_2gpu.sh validate 0,1
```

Run the two experiments independently on the same GPU pair:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/08_nabladft_qhflow3_muon_head_no_scale_shift_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/09_nabladft_qhflow3_muon_head_scale_shift_2gpu.sh smoke 0,1

/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/08_nabladft_qhflow3_muon_head_no_scale_shift_2gpu.sh full 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/09_nabladft_qhflow3_muon_head_scale_shift_2gpu.sh full 0,1
```

Each full run uses two data-parallel ranks, micro-batch 5, gradient
accumulation 2, effective batch 20, 20 epochs, seed 44, and W&B
`kaist-korea/maloq-nablaDFT` logging every 10 optimizer steps. Outputs are
written to
`/dataset/seongsu/shared-home/workspace/project/outputs/nabla-qhf3-muon-ss0-v2/`
and
`/dataset/seongsu/shared-home/workspace/project/outputs/nabla-qhf3-muon-ss1-v1/`.
Repeated launches refuse to overwrite an existing result.
Successful smoke outputs are removed; failed smoke outputs are retained. Full
training is not launched automatically.

Validation status on 2026-07-24:

- shell syntax and both configuration-validation paths passed
- the two configs match except for descriptive output/model names and the
  intended `scale_and_shift`/`scale_shift_path` fields
- both lanes passed a full-size, single-GPU 2-training/1-validation smoke
  through loader creation, QHFlow3 forward/backward, Muon updates, inverse
  scale-shift, and validation matrix reconstruction
- the off/on smoke validation matrix MAEs were approximately `0.22493` and
  `0.22085`, respectively, but this one-molecule validation smoke is only an
  execution check and is not evidence of a performance improvement
- successful verification outputs were discarded; the 20-epoch comparison
  has not been launched

## Compact NablaDFT naming

New NablaDFT experiment IDs follow
`nabla-<model>-<head>-<raw|shift|shift-std>-v<N>`. W&B uses the readable
display name
`NablaDFT | <model> | <head> | <RAW|SHIFT|SHIFT+STD> | V<N>`.
Existing `ss0`/`ss1` output directories remain unchanged as provenance paths.

- `nte128e3` means MALOQ-NTE output width 128 and three edge layers.
- `qhf3` means the local-objective QHFlow3-clean trunk.
- `native`, `muon`, and `staticte` identify the matrix head.
- `RAW` applies no target transform.
- `SHIFT` subtracts the element-specific `l=0` node mean only.
- `SHIFT+STD` subtracts that mean and divides by the standard deviation.
- `V1`, `V2`, and later values distinguish revisions of the same model/head/
  scale-shift lineage. A materially different head or scale-shift setting
  starts its own version sequence.

W&B groups comparison-compatible runs as `nabla-nte128e3-head-ss` and
`nabla-qhflow3-ss`. Dataset, model, head, scale-shift, scope, seed, version,
and `sc26-seongsu` are also stored as tags. Earlier valid results remain
visible under their model names and versions; `superseded` is a tag rather
than a display-name prefix.
