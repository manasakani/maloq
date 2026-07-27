# `op_projection`

- Status: `cuda_ddp_smoke_validated`
- Owner: SC26-seongsu research workspace
- Created: `2026-07-28`
- Intended promotion target: matrix-free Hamiltonian/density operator training

## Hypothesis

A MALOQ-NTE-V2 node trunk can encode each geometry once and a differentiable
callback can stream onsite and pair contributions into probe-vector products
without materializing persistent learned edge features or a dense AO matrix.

## Baseline

- Git commit: `fce46162cf88084f72e711af68ac50cb2e61e01f`
- Config: `_my_script/experiment/2026-07-27/maloq_nte_v2_nabladft.yaml`
- Dataset and exact split: NablaDFT `train_2k.db`, row 189 only
- Seed(s): `20260728`
- Effective batch and optimizer: one molecule, 2 resampled probes/step,
  AdamW(lr=1e-2), head-only, 120 steps
- Reference output: `outputs/op_projection_single_batch_row189/metrics.json`

## Entry point

- Import: `from maloq.experimental.op_projection import OpProjectionModel`
- Experimental config namespace: `experimental.op_projection`
- Resolved component/profile ID: `node_latent_callback_v1`
- Optional dependencies: none beyond canonical MALOQ

The dated experiment runner must import this entry point explicitly. Canonical
MALOQ must not import it.

## Checkpoint contract

- Architecture/schema version: `MALOQ-NTE-V2-OP-PROJECTION`, version 1
- Compatible checkpoints: shared node-trunk keys from MALOQ-NTE-V2
- State-dict migration: drop `edge_blocks.*`, `edge_norm.*`, and
  `edge_output_projection.*`; load the remaining state strictly into this trunk
- Deterministic initialization requirements: use the experiment seed or migrate
  the node output projection from a baseline checkpoint

`esen_osh_v2.py` and `esen_block_v2.py` remain the canonical NTE-V2 baseline.
This experiment is an independent node-latent branch and does not deprecate or
schedule either baseline file for removal.

## Verification

- [x] CPU import
- [x] Dated experiment entry point
- [x] Shape and forward/backward
- [x] Dtype/device coverage: CPU float32
- [ ] Equivariance
- [x] AO/basis convention
- [x] CPU train step
- [x] Real-data held-out-probe learning smoke
- [x] End-to-end backbone gradient
- [x] CUDA train step
- [x] CUDA validation step
- [x] Checkpoint save/reload
- [x] DDP smoke
- [ ] Matched parameter/memory/throughput comparison
- [ ] Matched quality comparison

## Evidence

| Date | Config | Commit or source snapshot | Output/W&B | Result |
|---|---|---|---|---|
| 2026-07-28 | focused CPU unit test | working tree at baseline commit above | none | node-only forward and callback backward smoke |
| 2026-07-28 | row 189, C=8, q_train=2, q_holdout=8, 120 steps | dirty research snapshot | `outputs/op_projection_single_batch_row189` | held-out relative action error 1.00379 -> 0.75469 (24.8% reduction); packed AO matvec 1.06e-14; backbone grad norm 4.16 |
| 2026-07-28 | `nabladft_op_projection.yaml`; 20/20 smoke; C128 trunk -> C64 pair head; chunk 2048; MB5/GA2; 2 GPUs | dirty source fingerprint pinned by queue | disposable smoke output | 2-rank train/validation passed; relative action error 0.97724 after one warmup update; backbone/head grad norms 0.13270/0.07098; checkpoint reload passed; peak 7.68 GB/GPU; 5.66 s measured training loop |
| 2026-07-28 | same 20/20 smoke plus streamed exact matrix metrics; identity chunk 64; fixed 1 train + 1 validation molecule/rank | dirty replacement snapshot | disposable smoke output | exact macro/micro MAE/MSE/RMSE and relative Frobenius emitted for both splits without dense matrix assembly; 8 unit tests and 2-rank checkpoint reload passed; peak 7.68 GB/GPU; 6.41 s measured loop |
| 2026-07-28 | V4 20/20 smoke; fixed 1 train molecule/rank; all 20 validation molecules; callback width strictly below AO dimension | dirty full-validation snapshot pinned by queue | disposable smoke output | 13 unit tests, launcher validation, and 2-rank checkpoint reload passed; validation coverage 20/20 with canonical MAE/MSE aliases equal to authoritative micro metrics; peak 7.68 GB/GPU; 10.49 s measured loop |

## Known limitations

- Pair geometry remains required. Only persistent learned edge features are
  removed.
- The head makes transient equivariant pair latents and atom-pair AO blocks in
  chunks. It never constructs a molecule-scale predicted matrix; autograd still
  retains the chunk computations needed for backward.
- The real-data smoke freezes the random node backbone while fitting the head;
  a separate real-batch backward verifies gradients reach 59 backbone tensors.
- The full trainer's predicted Hamiltonian and target operator action are
  matrix-free: coupled onsite/pair labels are streamed into probes. Loader
  preprocessing still reads dense reference H/S, retains dense overlap S, and
  the current QHFlow3-exact conditioner consumes S. This is not an end-to-end
  matrix-free input pipeline.
- Exact matrix monitoring is also streamed by applying identity-column chunks
  narrower than the AO matrix and retaining only scalar sums. Training uses a
  fixed 2-molecule global diagnostic, while validation covers the complete
  64-molecule split (32/rank). The canonical validation MAE/MSE aliases are raw
  directed cutoff-operator AO-entry micro errors; they are not symmetrized if
  the learned operator is asymmetric.
- Hermiticity, density idempotency, trace/electron count, and PSD constraints
  are not yet enforced by the internal representation.
- The feature-local gated e3nn projection is not checkpoint-compatible with the
  canonical Muon Fock head.
- A callback is valid for one autograd step and must not be cached across steps.
- The current matrix input conditioner remains canonical shared code.

## Promotion decision

Not promoted. The core learning path, AO convention, CUDA/DDP train/validation,
and checkpoint reload are validated. Promotion still requires an explicit
rotation-equivariance test, physical operator constraints, a completed full
dataset run, and matched memory-throughput-quality evidence against canonical
NTE-V2.
