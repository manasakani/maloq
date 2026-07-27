# NablaDFT MALOQ-E3 + Muon + RAW control

This isolated lane completes the RAW structure comparison next to the
2026-07-27 NTEV2-E3 and QHFlow3-E3 Muon+RAW lanes. It does not modify or
import the active 2026-07-27 experiment artifacts. The runner explicitly uses
`maloq.train_utils.training_workflow_v2.TrainingWorkflowV2Fixed`.

## Matched contract

- architecture: canonical MALOQ `esen`, three interleaved node/edge updates
- head: `maloq_muon`
- optimizer: Muon LR `0.02` plus auxiliary AdamW LR `5e-4`
- targets: RAW absolute Fock matrix, with `scale_and_shift: false` and
  `scale_shift_path: null`
- fixed ordered split: 12,081 train / 64 validation / 0 test
- two data-parallel GPUs; micro-batch 5 per rank; accumulation 2; effective
  batch 20
- 20 epochs, float32, seed 44, shuffle off
- W&B project: `kaist-korea/MALOQ-nablaDFT-v2`
- W&B display name: `NablaDFT | MALOQ-E3 | Muon | RAW | V2`
- database:
  `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
- environment:
  `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26`

This lane is a normalization control for
`NablaDFT | MALOQ-E3 | Muon | SHIFT | V2`. It is not a full-factorial
extension.

## Files

- typed config:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/nabladft_maloq_e3_muon_raw.yaml`
- dedicated runner:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/run_nabladft_maloq_e3_muon_raw.py`
- two-GPU launcher:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/01_nabladft_maloq_e3_muon_raw_2gpu.sh`
- server-1 queue manifest:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/queue_nabladft_maloq_e3_muon_raw_server1.yaml`
- full-run output prefix:
  `/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-v2-ofat-maloq-e3-muon-raw-2gpu-eb20-mb5-ga2-full-e20-`

## Commands

Preparation and validation are read-only and do not initialize training:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/01_nabladft_maloq_e3_muon_raw_2gpu.sh prepare
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/01_nabladft_maloq_e3_muon_raw_2gpu.sh validate
```

Run the disposable two-GPU CUDA/DDP smoke before enqueueing:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/01_nabladft_maloq_e3_muon_raw_2gpu.sh smoke 0,1
```

A successful smoke removes only its collision-resistant temporary output.
Failed evidence is retained. The launcher requires exactly two distinct GPU
indices and refuses a GPU with active compute PIDs, more than 1,024 MiB in
use, or more than 10% utilization. `EXPECTED_HOST` is an optional host guard.

Manual full-run form:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/01_nabladft_maloq_e3_muon_raw_2gpu.sh full 0,1
```

Queue ID:
`nabla-v2-ofat-maloq-e3-muon-raw-20260728a`. The manifest is pinned to
worker label `server-1` and contains exactly one `{gpus}` placeholder.

Before enqueueing, run the smoke and inspect both hosts, workers, queue locks,
live GPU allocations, and shared `/dataset` usage:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/experiment_queue/sc26_queue.py \
  enqueue \
  /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/queue_nabladft_maloq_e3_muon_raw_server1.yaml
```

Status: artifacts prepared and verified. `bash -n`, Ruff, Python compilation,
queue schema, `prepare`, `validate`, and a disposable two-GPU CUDA/DDP smoke
passed on 2026-07-28. The durable queue state for the job ID above is the
authoritative source for enqueue and full-run status.

---

# NablaDFT-2k NTEV2 operator projection

This lane trains the experimental node-only NTEV2 backbone and matrix-free
operator-projection head on the canonical NablaDFT-2k database. The prediction
path applies the learned Hamiltonian to probe vectors without materializing a
global dense matrix. The loader converts each reference Hamiltonian once into
cutoff-local coupled onsite/pair labels; training applies those labels to the
probes with the same packed block-matvec primitive, without retaining or
reconstructing a dense target matrix.

## Training contract

- fixed ordered split: 12,081 train / 64 validation / 0 test
- matched local graph radius `8.0` bohr and operator radius `16.0` bohr
- node trunk channels `128`, hidden channels `128`, projected output/head
  channels `64`, pair hidden/edge channels `64`, 512 distance bases, three
  layers, basis-inferred `lmax`
- pair-projection chunk size `2048`
- two Rademacher probes for training and eight for validation
- full-split probe estimates logged as `train/probe_matrix_*` and
  `validation/probe_matrix_*`
- streamed exact cutoff-label matrix metrics every epoch on a fixed 2-molecule
  global train diagnostic and all 64 validation molecules (32/rank). The
  configured callback width is at most 64 identity columns and is reduced below
  `M` for an `M x M` operator, so no callback assembles a full predicted/target
  matrix
- two GPUs; micro-batch five molecules per rank; accumulation 2; effective
  batch 20
- AdamW at `5e-4` with weight decay `1e-4`, gradient clipping at `1.0`, and
  1,000-step polynomial warmup
- 20 epochs, float32, seed 44
- W&B project: `kaist-korea/MALOQ-nablaDFT-v2`
- W&B display name:
  `NablaDFT | NTEV2-OpProjection | AdamW | RAW | V4`
- canonical W&B namespaces: `train_step/total_loss`, `train/total_loss`,
  `validation/total_loss`, `validation/matrix_mae`,
  `validation/matrix_mse`, `optimizer/learning_rate`, `time/*`, and
  `system/gpu_peak_memory_mb`

## Files and commands

- config:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/nabladft_op_projection.yaml`
- runner:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/run_nabladft_op_projection.py`
- launcher:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/02_nabladft_op_projection_2gpu.sh`
- queue manifest:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/queue_nabladft_op_projection.yaml`
- full-run output prefix:
  `/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-ntev2-op-projection-full-val-matrix-metrics-v4-2gpu-eb20-mb5-ga2-full-e20-`

Preparation and validation do not reserve GPUs:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/02_nabladft_op_projection_2gpu.sh prepare
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/02_nabladft_op_projection_2gpu.sh validate
```

Run the disposable two-GPU CUDA/DDP smoke before enqueueing:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/02_nabladft_op_projection_2gpu.sh smoke 2,3
```

A successful smoke removes only its collision-resistant temporary output.
Failed evidence is retained for diagnosis. The launcher requires exactly two
distinct, idle GPU indices and supports `EXPECTED_HOST` as an optional host
guard. Smoke mode overrides the runner to 20 train rows, 20 validation rows,
one probe, one epoch, one exact train molecule per rank, all 20 validation
molecules, and disabled W&B while retaining the production
micro-batch/accumulation profile.

The V4 smoke passed on server-1 GPUs 6,7: all 20 validation molecules were
evaluated (`coverage_fraction=1.0`), the canonical matrix aliases matched the
authoritative micro metrics, checkpoint reload passed, and peak allocated
memory was 7.68 GB/GPU. The complete feature unit suite passed 13 tests.

Exact keys use `train_subset_exact/*` and `validation_exact/*`. They include
macro and micro MAE/MSE/RMSE, relative Frobenius error, AO
diagonal/off-diagonal MAE, evaluated/expected molecule coverage, entry counts,
and mean AO dimension. `validation/matrix_mae` and
`validation/matrix_mse` alias the full-validation AO-entry micro values for the
canonical dashboard. These are raw directed cutoff-operator errors; unlike the
canonical dense evaluator, they are not symmetrized when the learned operator
is asymmetric. The detailed `validation_exact/*` names are authoritative.

"Matrix-free" in this lane refers to the predicted Hamiltonian and supervised
operator action. Loader preprocessing still reads dense reference H/S, and the
current QHFlow3-exact conditioner consumes each molecule's dense overlap S.

Manual full-run form:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/02_nabladft_op_projection_2gpu.sh full 2,3
```

Queue ID: `nabla-ntev2-op-projection-full-val-matrix-metrics-2k-20260728c`.
The manifest allows only `server-1`, because server-2 GPUs 4--7 are under a
hard reservation and the queue cannot enforce per-GPU exclusions. It requests
two GPUs, has priority 10, and contains exactly one `{gpus}` placeholder. The
durable queue state is the authoritative source for enqueue and run status.
