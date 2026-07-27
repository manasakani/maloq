# NablaDFT E3/Muon/SHIFT FlowMatching

Status: all three corrected server-2 two-rank CUDA smokes passed with endpoint
matrix metrics enabled; GPU 4-7 remained reserved. The first full attempts were
intentionally stopped before this correction, and their logs/checkpoints remain
preserved while the jobs are prepared for committed-source retry.

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

The only intended objective change is FlowMatching. Node and directed-edge
coupled Hamiltonian blocks share one graph time, are both corrupted, and are
both integrated with three endpoint-parameterized Euler steps. The trainer
uses `10 * masked coupled Frobenius MSE`. Validation loss remains the random-time
endpoint-matching loss used by the scheduler. Every epoch, the existing matrix,
node/edge MAE, and corresponding MSE metrics are computed from joint Euler3
inference with a fixed rank-and-batch prior, then delegated to canonical dense-AO
accounting in physical SHIFT-restored units.

SHIFT subtracts the frozen elementwise mean from node `l=0` targets only. Edge
targets remain unshifted, matching the completed direct baselines. The artifact
is:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/scale-shift-statistics/nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt
SHA-256: 375167ad551fb0b60dbe9cd049a4995276b54ce075e09906639ef3daa4f79475
```

## Identity and outputs

W&B project: `MALOQ-nablaDFT-v2`

Display names deliberately include `FlowMatching` so they cannot be confused
with the completed direct V2 runs:

- `NablaDFT | MALOQ-E3 | Muon | SHIFT | FlowMatching | V2`
- `NablaDFT | NTEV2-E3 | Muon | SHIFT | FlowMatching | V2`
- `NablaDFT | QHFlow3-E3 | Muon | SHIFT | FlowMatching | V2`

Full outputs use collision-resistant directories below:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-flow-matching-<lane>-v2-2gpu-eb20-mb5-ga2-full-e20-<timestamp>-<pid>/
```

Each rank-shared run directory records `resolved_flow_matching_config.json`
with the selected lane, typed config, base-config hash, database metadata, and
SHIFT artifact hash. The resolved provenance also records the effective QHFlow3
grid shape as `[10, 11]`.

## Commands

Validate the immutable inputs and all three typed lane configs without
training:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/04_nabladft_flow_matching_e3_muon_shift_2gpu.sh validate all
```

Run one full-size, one-epoch, two-GPU smoke:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/04_nabladft_flow_matching_e3_muon_shift_2gpu.sh smoke qhflow3-e3-muon-shift 6,7
```

Run one 20-epoch lane directly only after its smoke passes:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/04_nabladft_flow_matching_e3_muon_shift_2gpu.sh full qhflow3-e3-muon-shift 6,7
```

The launcher rejects duplicate, invalid, unavailable, or materially busy GPUs.
`EXPECTED_HOST` may optionally pin an interactive invocation to a hostname.

## Queue manifest

The server-1 smoke manifest used for the three passed CUDA smokes is:

```text
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/queue_nabladft_flow_matching_e3_muon_shift_smoke_server1.yaml
```

The full-run manifest is:

```text
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-28/queue_nabladft_flow_matching_e3_muon_shift.yaml
```

It defines exactly three two-GPU jobs, permits only `server-1` so server-2 GPUs
4-7 remain reserved, contains exactly one `{gpus}` placeholder per job, and
pins the launcher, runner, base config, and SHIFT artifact as immutable inputs.
All three CUDA smokes passed on server-1 GPU 6,7 at 04:21-04:22 KST. Initial
full attempts were stopped on request so endpoint-inference metrics could be
added before the comparison continued. Retry the preserved job IDs only after
the corrected source is tested and committed.

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
