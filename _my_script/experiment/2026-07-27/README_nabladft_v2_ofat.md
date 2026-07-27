# NablaDFT V2 matched OFAT suite

This suite compares canonical MALOQ, clean NTEV2, and QHFlow3 with a compact
one-factor-at-a-time design. It uses the new W&B project
`kaist-korea/MALOQ-nablaDFT-v2` and the dedicated
`TrainingWorkflowV2Fixed`; it does not invoke the legacy 2026-07-22 runner or
the branch-heavy experimental NTE workflow.

## Why these settings

The common settings come from the best completed 20-epoch NablaDFT run:
QHFlow3-E2, Muon head, and element-wise `l=0` mean subtraction. Its final
physical validation matrix MAE was `4.1153792e-5` (W&B run `wrsptkz2`).
The experiment label `SHIFT` means mean subtraction only:

```yaml
scale_and_shift: true
scale_shift_mode: shift_only
```

It is not `SHIFT+STD`. RAW lanes set `scale_and_shift: false`. Both are
reported in physical AO units by the current validation path.

The fixed statistics artifact is:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/scale-shift-statistics/nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt
SHA-256 375167ad551fb0b60dbe9cd049a4995276b54ce075e09906639ef3daa4f79475
```

Its provenance records NablaDFT training rows 0–12080 only, with zero
validation or test rows included.

## Nine unique lanes

Shared baselines participate in more than one contrast, so the three OFAT
axes require nine unique training runs rather than a 20-run full factorial.

| Lane ID | Architecture | Edge/pair blocks | Head | Target treatment | OFAT axis |
|---|---|---:|---|---|---|
| `maloq-e3-muon-shift` | MALOQ | 3 | Muon | SHIFT | structure |
| `ntev2-e3-muon-shift` | NTEV2 | 3 | Muon | SHIFT | structure, head, normalization |
| `qhflow3-e3-muon-shift` | QHFlow3 | 3 | Muon | SHIFT | structure, head, normalization |
| `ntev2-e2-muon-shift` | NTEV2 | 2 | Muon | SHIFT | structure |
| `qhflow3-e2-muon-shift` | QHFlow3 | 2 | Muon | SHIFT | structure |
| `ntev2-e3-native-shift` | NTEV2 | 3 | native | SHIFT | head |
| `qhflow3-e3-native-shift` | QHFlow3 | 3 | native | SHIFT | head |
| `ntev2-e3-muon-raw` | NTEV2 | 3 | Muon | RAW | normalization |
| `qhflow3-e3-muon-raw` | QHFlow3 | 3 | Muon | RAW | normalization |

The head contrast keeps `optimizer_type: muon` in both lanes. The native
head's flat e3nn contractions therefore stay in auxiliary AdamW, while the
semantically materialized matrices in `maloq_muon` are routed to Muon. With
the same seed, the two heads have identical initial forward values.

## Matched training contract

- database:
  `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
- fixed ordered split: 12,081 train / 64 validation / 0 test
- two data-parallel GPUs
- micro-batch 5 per rank, accumulation 2, effective batch 20
- 20 epochs, float32, seed 44, shuffle off
- absolute Fock target, no delta learning
- 3 node blocks, trunk/hidden channels 128, distance basis 512
- orbital cutoff 8, eSEN Gaussian cutoff 16
- Muon LR 0.02, momentum 0.95, Nesterov, 5 Newton–Schulz steps
- auxiliary AdamW LR `5e-4`, betas `(0.9, 0.95)`, epsilon `1e-10`
- weight decay `1e-4`, gradient clipping 1.0
- 1,000-step polynomial warmup
- validation and physical matrix metrics every epoch
- W&B group `nabladft-v2-ofat-seed44`

The E3 structure lanes are close in capacity:

| Architecture | Expected parameters |
|---|---:|
| MALOQ-E3 | 34,489,297 |
| NTEV2-E3 | 33,891,021 |
| QHFlow3-E3 | 34,382,227 |
| NTEV2-E2 | 28,278,851 |
| QHFlow3-E2 | 28,654,483 |

The matching boundary is deliberate. Canonical MALOQ keeps its original
interleaved 3-node/3-edge spectral stack and 128-channel head input. NTEV2
runs all node blocks before a recurrent edge stack, uses a 64-channel output,
and consumes QHFlow3-compatible zero-primary-matrix/overlap conditioning.
For E3 it applies one `InitialEdgeBlock` followed by two independently
parameterized `EdgeRefinementBlock` instances. QHFlow3 uses three node blocks
followed by two or
three independent `xy` pair blocks with residual-sum aggregation, Grid48,
chunk size 512, and overlap conditioning. Thus block count and capacity are
matched; internal operations remain architecture-native.

## Files

- common typed config:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/nabladft_v2_ofat_common.yaml`
- dedicated runner:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/run_nabladft_v2_ofat.py`
- two-GPU launcher:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/04_nabladft_v2_ofat_2gpu.sh`
- server-1 queue manifest:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/queue_nabladft_v2_ofat_server1.yaml`
- environment:
  `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26`

## Commands

Preparation validates the typed base, all nine derived lane configs, the
database schema and fixed row count, and the SHIFT artifact hash/provenance.
It does not train.

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/04_nabladft_v2_ofat_2gpu.sh prepare all
```

Validate one lane without initializing distributed training:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/04_nabladft_v2_ofat_2gpu.sh validate ntev2-e3-muon-shift
```

Run the required disposable smoke before enqueueing:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/04_nabladft_v2_ofat_2gpu.sh smoke ntev2-e3-muon-shift 0,1
```

A successful smoke is removed; failed evidence is retained. The launcher
requires exactly two distinct GPU indices and refuses GPUs with an active
compute PID, more than 1,024 MiB in use, or more than 10% utilization.
`EXPECTED_HOST` remains an optional host guard.

Manual full-run form:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/04_nabladft_v2_ofat_2gpu.sh full ntev2-e3-muon-shift 0,1
```

Full and failed-smoke outputs use collision-resistant directories below:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-v2-ofat-<lane>-2gpu-eb20-mb5-ga2-<scope>-<timestamp>-<pid>/
```

The queue manifest contains all nine jobs, pins them to worker label
`server-1`, requests two GPUs each, and gives every job exactly one `{gpus}`
placeholder. Use the actual SSH alias `scp-gpu-1`, not `scp-gpu-v1`.

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python \
  /dataset/seongsu/shared-home/workspace/project/_auto_script/experiment_queue/sc26_queue.py \
  enqueue \
  /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/queue_nabladft_v2_ofat_server1.yaml
```

Status: artifacts prepared. The queue manifest has not been enqueued, and no
smoke or full lane has been launched from this suite.
