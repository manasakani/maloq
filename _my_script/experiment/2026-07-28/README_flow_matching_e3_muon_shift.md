# NablaDFT E3/Muon/SHIFT FlowMatching

Status: all three canonical-MALOQ-loss lanes passed fresh server-1 two-rank CUDA
smokes with endpoint matrix metrics enabled. Earlier runs and their
logs/checkpoints remain preserved under distinct identities.

## Purpose

This suite applies the experimental full-matrix clean-endpoint flow objective
to the matched NablaDFT V2 structure comparison:

| Lane | Backbone | Edge layers | Head | Normalization | SO(3) grid |
| --- | --- | ---: | --- | --- | --- |
| `maloq-e3-muon-shift` | canonical MALOQ (`esen`) | 3 | MatrixMuon | SHIFT | native |
| `ntev2-e3-muon-shift` | `maloq_nte_v2` | 3 | MatrixMuon | SHIFT | 10x11 default |
| `qhflow3-e3-muon-shift` | `qhflow3` | 3 | MatrixMuon | SHIFT | 10x11 default |

All non-architecture settings match the completed 2026-07-27 direct baselines:
20 epochs, train/validation/test `12081/64/0`, micro-batch 5 per rank, two
ranks, gradient accumulation 2, effective batch 20, Muon plus AuxAdamW,
warmup-polynomial scheduling, float32, and seed 44.

The only intended modeling change is FlowMatching. Node and directed-edge
coupled Hamiltonian blocks share one graph time, are both corrupted, and are
both integrated with three endpoint-parameterized Euler steps. The trainer
delegates by default to the canonical MALOQ `rmse_mse_padded_loss` after
canonical node/edge padding masks are removed. A separate NTEV2-only
`rmse-mse-mae` route composes the isolated `matrix_composite_loss` feature and
optimizes `RMSE + MSE + componentwise MAE` over those same filtered coupled
coordinates. It does not change the immutable typed YAML or either feature's
default. Validation loss uses the selected random-time endpoint-matching
objective for scheduling. Every epoch, the existing matrix and node/edge
MAE/MSE metrics are still reported
from joint Euler3 inference with a fixed rank-and-batch prior, then delegated
to canonical dense-AO accounting in physical SHIFT-restored units.

SHIFT subtracts the frozen elementwise mean from node `l=0` targets only. Edge
targets remain unshifted, matching the completed direct baselines. The artifact
is:

```text
/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/scale-shift-statistics/nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt
SHA-256: 375167ad551fb0b60dbe9cd049a4995276b54ce075e09906639ef3daa4f79475
```

## Identity and outputs

W&B project: `MALOQ-nablaDFT-v2`

Display names deliberately include `FlowMatching` so they cannot be confused
with the completed direct V2 runs:

- `NablaDFT | MALOQ-E3 | Muon | SHIFT | FlowMatching | MALOQ-RMSE+MSE | V2`
- `NablaDFT | NTEV2-E3 | Muon | SHIFT | FlowMatching | MALOQ-RMSE+MSE | V2`
- `NablaDFT | QHFlow3-E3 | Muon | SHIFT | FlowMatching | MALOQ-RMSE+MSE | V2`
- `NablaDFT | NTEV2-E3 | Muon | SHIFT | FlowMatching | RMSE+MSE+MAE | V2`

The last identity is a loss-axis ablation and is deliberately NTEV2-only.
Its componentwise MAE term is coordinate dependent for `l>0`; the model forward
remains SO(3)-equivariant, but the composite objective is not rotation
invariant.

Full outputs use collision-resistant directories below:

```text
/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/nabladft-flow-matching-maloq-loss-<lane>-v2-2gpu-eb20-mb5-ga2-full-e20-<timestamp>-<pid>/
/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/nabladft-flow-matching-rmse-mse-mae-ntev2-e3-muon-shift-v2-2gpu-eb20-mb5-ga2-full-e20-<timestamp>-<pid>/
```

Each rank-shared run directory records `resolved_flow_matching_config.json`
with the selected lane, typed config, base-config hash, database metadata, and
SHIFT artifact hash. The resolved provenance also records the effective QHFlow3
grid shape as `[10, 11]`.

## Commands

Validate the immutable inputs and all three typed lane configs without
training:

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/04_nabladft_flow_matching_e3_muon_shift_2gpu.sh validate all
```

Run one full-size, one-epoch, two-GPU smoke:

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/04_nabladft_flow_matching_e3_muon_shift_2gpu.sh smoke qhflow3-e3-muon-shift 6,7
```

Run one 20-epoch lane directly only after its smoke passes:

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/04_nabladft_flow_matching_e3_muon_shift_2gpu.sh full qhflow3-e3-muon-shift 6,7
```

Validate the NTEV2 composite route without training:

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/04_nabladft_flow_matching_e3_muon_shift_2gpu.sh validate ntev2-e3-muon-shift '' rmse-mse-mae
```

The queue uses a guarded wrapper that runs an exact two-rank, one-epoch smoke
before the 20-epoch job on the same claimed GPU pair:

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/10_nabladft_ntev2_flow_matching_rmse_mse_mae_2gpu.sh 0,1
```

The launcher rejects duplicate, invalid, unavailable, or materially busy GPUs.
`EXPECTED_HOST` may optionally pin an interactive invocation to a hostname.

## Queue manifest

The server-1 smoke manifest used for the three passed CUDA smokes is:

```text
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/queue_nabladft_flow_matching_e3_muon_shift_smoke_server1.yaml
```

The canonical-MALOQ-loss full-run manifest is:

```text
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/queue_nabladft_flow_matching_e3_muon_shift_maloq_loss.yaml
```

The NTEV2 `RMSE+MSE+MAE` guarded full-run manifest is:

```text
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-07-28/queue_nabladft_ntev2_flow_matching_rmse_mse_mae.yaml
```

It contains one server-1-only two-GPU job and exactly one `{gpus}` placeholder.
The wrapper refuses to start the full run unless its exact composite-loss CUDA
smoke succeeds. The canonical manifest defines exactly three two-GPU jobs, permits only `server-1` so server-2 GPUs
4-7 remain reserved, contains exactly one `{gpus}` placeholder per job, and
pins the launcher, runner, base config, and SHIFT artifact as immutable inputs.
All three CUDA smokes passed on server-1 GPU 6,7 at 04:21-04:22 KST. Initial
full attempts were stopped on request so endpoint-inference metrics could be
added before the comparison continued. The preserved original-loss job records
remain blocked because their immutable launchers point to the pre-relocation
repository root. The MALOQ-loss suite uses distinct queue IDs, W&B identities,
and fresh timestamped output directories; historical requests and outputs are
not rewritten or reused. Fresh MALOQ-loss smokes passed on 2026-07-28 at about
21:27 KST for MALOQ on GPU 0-1, NTEV2 on GPU 2-3, and QHFlow3 on GPU 4-5.

## Source contract

The runner imports
`maloq.experimental.flow_matching.FlowMatchingWorkflow` and
`EndpointFlowMaloqConfig`. The workflow must support `esen`,
`maloq_nte_v2`, and `qhflow3`, and every backbone must consume the current
node state, directed-edge state, and per-graph time equivariantly. QHFlow3 must
resolve the feature-scoped `null` grid setting to latitude 10 and longitude 11;
the runner rejects any drift. Matrix metrics must be enabled every epoch and
must use joint node/edge Euler endpoints rather than the random-time validation
head output. This bundle must not be enqueued if that contract or the relevant
CUDA/checkpoint tests are not yet satisfied.
