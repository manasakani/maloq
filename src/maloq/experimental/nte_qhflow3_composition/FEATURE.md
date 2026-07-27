# `nte_qhflow3_composition`

- Status: `abandoned` (deprecated selector implementation)
- Owner: SC26-seongsu
- Created: `2026-07-27`
- Intended promotion target: one fixed MALOQ-NTE architecture after matched validation

## Hypothesis

A controlled composition of NTE scheduling, QHFlow3 matrix conditioning and
pair operations can improve matrix prediction without changing equivariant
feature conventions.

## Baseline

- Git commit: `3c5f955e65f5a010ca0fc7fa621ce95396f5073f`
- Config: `_my_script/experiment/2026-07-21/maloq_qh9stable.yaml` and dated selector configs
- Dataset and exact split: QH9Stable official split or NablaDFT recorded split
- Seed(s): primarily `44`; individual dated configs are authoritative
- Effective batch and optimizer: recorded per config; MatrixMuon + auxiliary AdamW for the selected runs
- Reference output/W&B run: `outputs/experiment-queue/` request and result records

## Entry point

- Config: `from maloq.experimental.nte_qhflow3_composition.config import MaloqConfig`
- Workflow: `from maloq.experimental.nte_qhflow3_composition.workflow import TrainingWorkflow`
- Experimental config namespace: `experimental.nte_qhflow3_composition`
- Resolved component/profile ID: `core_experiments_v1`
- Optional dependencies: the same e3nn/fairchem stack as canonical MALOQ

The dated experiment runner imports this workflow only for its `maloq-nte`
variant. Canonical MALOQ does not import this package.

## Checkpoint contract

- Architecture/schema version: `core_experiments_v1`
- Compatible checkpoints: checkpoints created by the same feature profile and head/optimizer policy
- Historical selector checkpoints: unsupported; rerun the selected core experiments
- State-dict migration: intentionally not provided
- Deterministic initialization requirements: keep the selected profile, seed and construction order

Full-object Python pickles and arbitrary pre-migration selector combinations are
not covered. New run/checkpoint IDs include `nte_qhflow3_composition`.

## Verification

- [x] CPU import
- [x] Config validation
- [x] Shape and forward/backward
- [x] Dtype/device coverage
- [x] Equivariance, when applicable
- [x] AO/basis convention, when applicable
- [ ] CUDA train step after migration
- [ ] CUDA validation step after migration
- [x] Checkpoint save/reload
- [x] DDP smoke, when supported by the selected configuration
- [ ] Matched parameter/memory/throughput comparison
- [ ] Matched quality comparison

## Evidence

| Date | Config | Commit or source snapshot | Output/W&B | Result |
|---|---|---|---|---|
| 2026-07-27 | queued `nte64e2_*` configs | frozen queue fingerprints | `outputs/experiment-queue/` | completed ablation set |

## Known limitations

- The feature keeps flat selectors only for the small set of core reruns; other historical combinations are unsupported.
- Matrix conditioning does not support distributed graph partitioning.
- Cross-profile checkpoint migration is not supported.
- New experiments should prefer a fixed architecture rather than add selectors here.

## Promotion decision

The selector implementation is deprecated and abandoned. Its selected
Edge2/initial-envelope/Atom2-direct behavior was rewritten independently as
MALOQ-NTE-V2; rejected selectors must not return to canonical MALOQ. New
comparisons use `training_workflow_v2.py`, not this package.
