# `matrix_composite_loss`

- Status: `draft`
- Owner: SC26-seongsu
- Created: `2026-07-28`
- Intended promotion target: none until the matched loss ablation is evaluated

## Hypothesis

Adding an entrywise MAE term to the existing padded RMSE+MSE objective may
improve the final unshifted matrix MAE of NTEV2. Multiplying the complete
objective by ten tests whether the optimization trajectory changes under the
matched Muon/AuxAdamW schedule and global gradient clipping.

## Baseline

- Git commit: `fce46162cf88084f72e711af68ac50cb2e61e01f` plus the recorded dirty source snapshot
- Config: `NablaDFT | NTEV2-E3 | Muon | SHIFT | V2`
- Dataset and exact split: NablaDFT `train_2k.db`, ordered train 12,081 / validation 64 / test 0
- Seed(s): 44
- Effective batch and optimizer: 20 (5 per rank, 2 ranks, accumulation 2), Muon+AuxAdamW
- Reference output/W&B run: `outputs/nabladft-v2-ofat-ntev2-e3-muon-shift-2gpu-eb20-mb5-ga2-full-e20-20260727-181411-3321867`

## Entry point

- Imports: `maloq.experimental.matrix_composite_loss.apply_matrix_composite_loss_profile`
  and `build_matrix_composite_loss_workflow`
- Experimental config namespace: `matrix_composite_loss_profile`
- Resolved component/profile IDs: `rmse_mse_mae`, `10x_rmse_mse_mae`
- Optional dependencies: none

Dated direct-regression and FlowMatching runners import this package
explicitly. The pure profile adapter copies an arbitrary workflow config and
replaces only its effective training callable plus provenance fields. Canonical
MALOQ and canonical FlowMatching defaults do not import or select these losses.

## Checkpoint contract

- Architecture/schema version: unchanged `TrainingWorkflowV2Fixed` model/checkpoint schema
- Compatible checkpoints: exact same profile and matched config only
- State-dict migration: none
- Deterministic initialization requirements: seed 44 and the ordered two-rank split

## Verification

- [x] CPU import
- [x] Config validation
- [x] Shape and forward/backward
- [x] Dtype/device coverage
- [x] CUDA train step
- [x] CUDA validation step
- [ ] Checkpoint save/reload
- [x] DDP smoke for direct NTEV2 regression
- [ ] DDP smoke for the composed NTEV2 FlowMatching route
- [ ] Matched parameter/memory/throughput comparison
- [ ] Matched quality comparison

## Evidence

| Date | Config | Commit or source snapshot | Output/W&B | Result |
|---|---|---|---|---|
| 2026-07-28 | NTEV2 scale-1 `RMSE+MSE+MAE` | dirty fingerprint rooted at `baa2474` | smoke complete; full W&B `xvx2sdyl` | Two-rank server-1 smoke passed. Full run reached epoch 2 with finite train/validation loss and matrix/outside-cutoff metrics. |

## Known limitations

- The scale-10 profile interacts with the existing global gradient clipping at 1.0; it is intentionally not algebraically redundant with the scale-1 profile.
- The composite training losses are calculated after the canonical node/edge
  padding masks are applied. Validation matrix MAE/MSE remain the common
  unscaled comparison metrics.
- MAE is componentwise in flattened coupled-irrep coordinates. For l>0 it is
  not invariant under a general SO(3) basis mixing, so this objective is a
  coordinate-dependent ablation even when the model forward remains equivariant.

## Promotion decision

Pending the matched 20-epoch results. This feature is an ablation and does not
change canonical loss names or defaults.
