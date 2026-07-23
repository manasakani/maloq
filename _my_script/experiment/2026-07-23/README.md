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
cd /dataset/seongsu/shared-home/workspace/project
./_my_script/experiment/2026-07-23/01_qh9stable_oom_recovery_mb16_ga2.sh validate
./_my_script/experiment/2026-07-23/01_qh9stable_oom_recovery_mb16_ga2.sh smoke
./_my_script/experiment/2026-07-23/01_qh9stable_oom_recovery_mb16_ga2.sh full
```

Full outputs are written below:

```text
outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-full-seed44-<timestamp>/
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
  `outputs/qh9stable-oom-recovery-four-lane-mb4-ga8-eb32-full-seed44-20260723-040732/`
- full launcher PID: `2769245`; worker PIDs: `2769252` through `2769255`
- initial online W&B connection passed for all four lanes

GA2 status on 2026-07-23 (Asia/Seoul):

- configuration validation passed for all four lanes
- production-size one-epoch smoke passed for all four lanes; temporary outputs
  were removed
- full 80-epoch recovery launched at `20260723-041305`
- full output:
  `outputs/qh9stable-oom-recovery-four-lane-mb4-ga2-eb8-full-seed44-20260723-041305/`
- full launcher PID: `2771071`; worker PIDs: `2771078` through `2771081`
- initial online W&B connection passed for all four lanes
- this effective-batch-8 run was intentionally stopped when the requested
  effective batch was clarified to be 32

Corrected GA2/effective-batch-32 configuration:

- micro-batch 16
- gradient accumulation 2
- effective batch 32
- output prefix:
  `outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-<scope>-seed44-<timestamp>/`
- configuration validation passed for all four lanes
- production-size one-epoch CUDA smoke passed for all four lanes; temporary
  outputs were removed
- full 80-epoch recovery launched at `20260723-133856`
- full output:
  `outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-full-seed44-20260723-133856/`
- full launcher PID: `2847531`; worker PIDs: `2847538` through `2847541`

## NablaDFT QHFlow3 local matrix objective

QHFlow3 now consumes the same directed `rcut_orbitals=8.0` pair graph as the
MALOQ baselines. Pair blocks outside the cutoff are not predicted or included
in the local loss, and both prediction and label reconstruction leave those
blocks at zero.

```bash
cd /dataset/seongsu/shared-home/workspace/project
./_my_script/experiment/2026-07-23/02_nabladft_qhflow3_local_muon_head_2gpu_mb5_ga2.sh validate
./_my_script/experiment/2026-07-23/02_nabladft_qhflow3_local_muon_head_2gpu_mb5_ga2.sh smoke
./_my_script/experiment/2026-07-23/02_nabladft_qhflow3_local_muon_head_2gpu_mb5_ga2.sh full
```

The full run uses two GPUs, micro-batch 5, gradient accumulation 2, effective
batch 20, 20 epochs, seed 44, the QHFlow3-clean trunk, and the MUON-compatible
MALOQ head. Outputs are written below:

```text
outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-<scope>-seed44-<timestamp>/
```

Loader-fix status on 2026-07-23 (Asia/Seoul):

- the first full attempt at `20260723-151840` stopped during loader creation
  because the NablaDFT branch did not initialize optional QHFlow3 initial
  matrices
- failed output was preserved at
  `outputs/nabladft-qhflow3-local-muon-head-2gpu-eb20-mb5-ga2-full-e20-seed44-20260723-151840/`
- the shared loader now explicitly leaves those optional matrices absent for
  non-delta NablaDFT instead of reading undefined branch-local variables
- a single-GPU end-to-end smoke with 2 training and 1 validation molecule
  passed through data loading, QHFlow3, the MUON head, optimizer updates, and
  validation matrix metrics
- smoke output:
  `outputs/nabladft-qhflow3-local-loader-fix-smoke-seed44-after-20260723-151840/`
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
cd /dataset/seongsu/shared-home/workspace/project
./_my_script/experiment/2026-07-23/03_nabladft_maloq_nte_do128_le3_head_comparison.sh validate
./_my_script/experiment/2026-07-23/03_nabladft_maloq_nte_do128_le3_head_comparison.sh smoke
./_my_script/experiment/2026-07-23/03_nabladft_maloq_nte_do128_le3_head_comparison.sh full
```

The two lanes run concurrently. Override their GPU pairs with `NATIVE_GPUS`
and `MUON_GPUS`, or their ports with `NATIVE_MASTER_PORT` and
`MUON_MASTER_PORT`. Full outputs, per-lane logs/status, and the merged
`comparison.csv` are written below
`outputs/nabladft-maloq-nte-do128-le3-head-comparison-parallel-2x2gpu-eb20-mb5-ga2-<scope>-seed44-<timestamp>/`.
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
cd /dataset/seongsu/shared-home/workspace/project
./_my_script/experiment/2026-07-23/05_nabladft_nte_native_head_scale_shift_2gpu.sh prepare 6,7
```

Then run each experiment independently, choosing its GPU pair:

```bash
./_my_script/experiment/2026-07-23/04_nabladft_nte_native_head_no_scale_shift_2gpu.sh full 6,7
./_my_script/experiment/2026-07-23/05_nabladft_nte_native_head_scale_shift_2gpu.sh full 6,7
./_my_script/experiment/2026-07-23/06_nabladft_nte_muon_head_no_scale_shift_2gpu.sh full 6,7
./_my_script/experiment/2026-07-23/07_nabladft_nte_muon_head_scale_shift_2gpu.sh full 6,7
```

Replace `full` with `validate` or `smoke` as needed. Every training run uses
micro-batch 5, two data-parallel ranks, gradient accumulation 2, effective
batch 20, 20 epochs, seed 44, and W&B
`kaist-korea/maloq-nablaDFT` logging every 10 optimizer steps. Full outputs are
written below
`outputs/<experiment>-2gpu-eb20-mb5-ga2-full-e20-seed44-<timestamp>/`.
Successful smoke outputs are removed; failed outputs are retained. The
train-12,081 statistics and multi-GPU smoke/full runs are not launched
automatically. A two-training/one-validation, full-size corrected-head CUDA
smoke passed with a temporary two-row scale-shift artifact; both the smoke
output and temporary artifact were discarded.

Scripts `04`-`07` run on any host by default and still validate that the
selected GPU indices exist and have at most 1024 MiB allocated. Set
`EXPECTED_HOST=<hostname>` only when a run should be pinned to one host.
Script `05` also accepts `SC26_PROJECT_ROOT`, `SC26_ENV_ROOT`, and `NABLA_DB`
to override its derived project root, shared conda environment, and NablaDFT
database path when a host uses a different mount layout.
