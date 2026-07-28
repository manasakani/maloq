# `flow_matching`

- Status: `draft`
- Owner: SC26-seongsu research workflow
- Created: `2026-07-28`
- Intended promotion target: none until real CUDA/DDP/resume and matched-quality
  evidence exist

## Hypothesis

For NablaDFT Hamiltonians, jointly flowing node/diagonal and directed-edge
Hamiltonian blocks through the same feature-local conditioner can compare
MALOQ-E3, NTEV2-E3, and QHFlow3-E3 against matched direct regression without
breaking proper-rotation SO(3) covariance or real-Hermitian structure.

## Baseline

- Source snapshot: dirty development tree rooted at
  `fce46162cf88084f72e711af68ac50cb2e61e01f`
- Config:
  `_my_script/experiment/2026-07-28/flow_matching_e3_muon_shift_nabladft.yaml`
- Dataset and exact split: NablaDFT `train_2k.db`, train 12081, validation 64,
  test 0
- Seed(s): 44
- Structures: MALOQ-E3, NTEV2-E3, QHFlow3-E3
- Effective batch and optimizer: 2 ranks x micro-batch 5 x accumulation 2 = 20;
  MatrixMuon head with Muon+AuxAdamW
- Direct V2 reference W&B runs: MALOQ `dj650akx`, NTEV2 `mpb1vutt`, QHFlow3
  `lpw3svdy`
- Mechanism source snapshot: QHFlow2 commit
  `2b5193785c199dce57db43065142cc9a5759d556`, active config
  `src/config_qh9/config_flow_v2_simple.yaml`

## Entry point

- Import: `maloq.experimental.flow_matching.FlowMatchingWorkflow`
- Experimental config namespace: `experimental.flow_matching`
- Resolved component/profile ID: `full_matrix_endpoint_flow_v1`
- Optional dependencies: none beyond canonical MALOQ and its runtime

The dated experiment runner imports this entry point explicitly. Canonical
MALOQ does not import it.

## Checkpoint contract

- Architecture/schema version: 2
- Compatible checkpoints: architecture-v2 feature checkpoints with identical
  selected backbone, common conditioner, head, optimizer, and resume signatures
- State-dict migration: none. Architecture-v1 node-only flow checkpoints omit
  the edge-conditioning parameters and are not exact-resume compatible;
  canonical or v1 weights may be used only as explicit partial initialization
- Training corruption uses canonical global CPU/CUDA Torch RNG, so
  `TrainingWorkflowV2Fixed` captures and restores that state per rank.
  Validation endpoint metrics use a private device-local generator seeded by
  runtime seed, rank, and local validation-batch index; metric inference neither
  perturbs training RNG nor changes its fixed prior between epochs.

## Implemented contract

- Coupled node and directed-edge priors share one feature-owned factory for
  training corruption and validation sampling. The backward-compatible
  default is a masked full-rank coupled-irrep Gaussian with sigma 0.1.
- `prior_type: tensor_expansion` selects the pinned-QHFlow2 unit-path Tensor
  Expansion distribution. One latent per orbital degree is summed over shell
  copies and reused across every compatible shell-pair path before the
  canonical Wigner-3j decoder. The active Nabla 5s+4p+3d padded basis therefore
  has 32 input features but covariance rank 9 (l=0,1,2); l=3,4 coupled output
  sectors are zero. `tensor_expansion_normalization` is fixed to
  `qhflow2_unit_path_sum`, and canonical masks must select complete irreps.
- One graph-level time in `[0.01, 0.99]`, shared by every node and edge in that
  graph.
- Joint paths: `Hnode_t=(1-t)Hnode_0+tHnode_1` and
  `Hedge_t=(1-t)Hedge_0+tHedge_1`; targets are both clean endpoints.
- `CoupledAOCodec` performs the reversible conversion
  `coupled <-> shell-pair-packed AO <-> padded dense AO`. It does not directly
  reshape the shell-packed output of `e3TensorDecomp.get_H()`.
- The coupled node state is installed as `batch.node_flow_t`, its decoded dense
  form as square `batch.init_ham_t`, the directed-edge state as
  `batch.edge_flow_t`, and graph time as `batch.t`. In the active direct profile
  the common conditioner consumes `node_flow_t`, `edge_flow_t`, and `t`;
  `init_ham_t` is a decoded mirror and is not consumed by any native trunk.
- `FlowConditionedBackbone` wraps `esen`, `maloq_nte_v2`, or `qhflow3` without
  architecture inheritance. Separate SO(3)-equivariant irrep linears map node
  and edge states into native embeddings; incoming projected edge state is
  averaged into destination nodes, and time enters output `l=0` components.
- Varying flow time is consumed once by this common conditioner. A native base
  sees fixed endpoint time `t=1`; QHFlow3 edge sorting and time mutation are
  isolated, then canonical loader edge order/time are restored before the head.
- `HamiltonianSymmetryProjector` applies `0.5*(Hii+Hii^T)` to node blocks and
  `0.5*(Hij+Hji^T)` to reverse-directed edge pairs, whose reverse map must be an
  involution.
- Canonical padding masks filter components before the configured loss is
  evaluated. The matched structure suite uses `rmse_mse_padded_loss`, i.e.
  `sqrt(mean(error^2)) + mean(error^2)`, without training MAE, an extra factor
  of 10, or time scaling.
- The dated NTEV2 loss-axis route explicitly composes
  `matrix_composite_loss.rmse_mse_mae`: `RMSE + MSE + MAE` over the same masked
  coupled-irrep components. The immutable FlowMatching YAML and default route
  remain canonical. Resolved provenance records the effective callable,
  formula, scale, coordinate space, and coordinate-invariance limitation.
- Sampling derives `(Hhat_1-H_t)/(1-t)` for both state families and uses three
  fixed joint Euler steps with symmetry projection.
- `EndpointFlowTrainer.sample_batch` connects a target-shaped validation batch
  directly to joint prior sampling, AO decoding, edge-state conditioning,
  backbone/head evaluations, symmetry projection, and joint Euler3.
- Canonical validation matrix metrics call the joint Euler sampler from a fixed
  prior, then delegate SHIFT restoration, dense-AO reconstruction, denominators,
  and DDP aggregation to `SplitTrainer.compute_validation_matrix_error_sums`.
  Existing `validation/matrix_*` names remain the configured ODE result.
  The same values are explicitly aliased under
  `validation/flow_matching_configured_ode/*`; an ODE1 one-head-evaluation result
  from the exact same batch and prior sample is recorded under
  `validation/flow_matching_one_shot/*`. Random-time endpoint-matching
  validation loss remains the scheduler signal.
- Workflow inherits `TrainingWorkflowV2Fixed`; trainer inherits `SplitTrainer`
  and delegates to `super().train` through corrupted loader wrappers.
- Parity is a proper-rotation contract: canonical all-even target labels are
  relabeled to degree parity for node/edge projections. SO(3) rotation
  matrices are unchanged; O(3)/reflection equivariance is not claimed.

## Verification

- [x] CPU import for codec, projector, and edge-conditioning wrapper
- [x] Architecture-v2 typed config and runner validation
- [x] CPU shape and real-backbone forward
- [x] Dtype/device contract coverage for the common conditioner
- [x] Codec, symmetry-projection, and edge-injection equivariance
- [x] CPU loader, real QHFlow3 edge feedback, and joint-Euler equivariance
- [x] AO/basis convention audit for full `def2-svp-nabla` shell layout
- [x] CUDA train step
- [x] CUDA validation step
- [x] Checkpoint save during bounded smoke
- [x] Three-lane two-rank CUDA smoke on server-1
- [x] Deterministic Euler-endpoint matrix-metric adapter and RNG isolation
- [x] Same-prior configured-ODE/one-shot validation metric variants and DDP sum
- [x] QHFlow2 unit-path Tensor Expansion prior, Nabla basis generalization, RNG,
  complete-irrep mask, and SO(3) regressions
- [x] Separate node-H_t, edge-H_t, and time gradients plus real-wrapper
  sensitivity for MALOQ, NTEV2, and QHFlow3
- [x] FlowMatching QHFlow3 default resolves to the matched 10x11 eSEN grid
- [ ] Matched parameter/memory/throughput comparison
- [x] Corrected three-lane two-rank CUDA metric smoke on server-2 GPU0-3
- [ ] NTEV2 FlowMatching `RMSE+MSE+MAE` guarded two-rank CUDA smoke
- [ ] Matched quality comparison

## Evidence

| Date | Config | Commit or source snapshot | Output/W&B | Result |
|---|---|---|---|---|
| 2026-07-28 | `qhflow2_endpoint_flow_nabladft.yaml` | dirty development tree rooted at `fce4616`; QHFlow2 audit `2b51937` | none | 30 focused CPU tests plus 4 import-boundary/trainer-factory tests passed. They cover the joint loader/sampler, real QHFlow3 edge feedback, codec/projector covariance, and per-step symmetry projection. The full NablaDFT shell-unpack audit reduced rotation error from 2.49 to 3.1e-7. CUDA/DDP execution remains pending. |
| 2026-07-28 | `flow_matching_e3_muon_shift_nabladft.yaml` | current dirty source rooted at `fce4616` | failed diagnostic evidence under `outputs/nabladft-flow-matching-maloq-...-040533-*` and `...-040705-*`; successful temporary smokes removed by launcher | 42 CPU/integration tests and all three typed configs pass. Diagnostics exposed and fixed a PATH-dependent torchrun wrapper plus ASE int32 edge indices. At 04:21-04:22 KST, MALOQ-E3, NTEV2-E3, and QHFlow3-E3 each passed a server-1 GPU6,7 two-rank train/validation/checkpoint smoke. |
| 2026-07-28 | `flow_matching_e3_muon_shift_nabladft.yaml` | pre-commit metric-correction tree | preserved first-attempt outputs; MALOQ W&B `0skkk54x` | The initial MALOQ full attempt completed epoch 1 and checkpointed with finite train/validation loss, then MALOQ and a newly claimed NTEV2 attempt were stopped on request before adding Euler-endpoint matrix metrics. QHFlow3 was cancelled before start. All target PIDs, claims, and GPU locks were released while outputs remained intact. |
| 2026-07-28 | corrected `flow_matching_e3_muon_shift_nabladft.yaml` | pre-commit corrected tree | temporary server-2 smokes cleaned by launcher | 43 focused CPU/canonical-metric tests passed. MALOQ, NTEV2, and QHFlow3 each completed two-rank train/validation/checkpoint smoke on server-2 GPU0-3 while 4-7 remained unselected. MALOQ endpoint matrix MAE/MSE was 0.272778/0.221840; QHFlow3 10x11 was 0.242831/0.198371. Both logged node/edge matrix metrics; the NTEV2 launcher completed the same enabled metric path with exit 0. |
| 2026-07-28 | canonical-MALOQ-loss `flow_matching_e3_muon_shift_nabladft.yaml` | current source fingerprint | temporary server-1 smokes cleaned by launcher | 66 FlowMatching/canonical-metric CPU tests passed. MALOQ, NTEV2, and QHFlow3 each completed a fresh two-rank CUDA train/validation/checkpoint smoke with configured `rmse_mse_padded_loss`; `validation/matrix_mae` remained enabled. |

## Known limitations

- The working profile predicts full `H` directly (`delta_learning=false`). The
  audited QHFlow2 QH9 run predicts residual `H-Hinit`; that parameterization is
  documented but not ported.
- Flowing directed edges is an intentional architecture-v2 extension beyond
  the active QHFlow2 QH9 executable, which integrates only node blocks.
- The active QHFlow2 executable loss is `10*(MAE+MSE)`. The matched structure
  suite instead uses MALOQ's coupled-coordinate `RMSE+MSE`: both terms depend
  only on the squared norm and remain invariant under orthogonal SO(3) irrep
  actions. The optional NTEV2 `RMSE+MSE+MAE` route uses componentwise MAE in
  flattened coupled-irrep coordinates, not QHFlow2 AO-space MAE. For `l>0`
  that term is coordinate dependent and not generally SO(3)-invariant, even
  though the model forward remains equivariant. Physical-space
  `validation/matrix_mae` remains a common reported endpoint metric.
- Node/edge state is injected at the final native embedding level, not
  recurrently into every internal message-passing block. In particular,
  QHFlow3 uses `default_hamiltonian_input="zero"` in this direct profile, so its
  native trunk does not consume decoded dense `init_ham_t`; enabling that path
  would create a second, deeper conditioning route and is a separate ablation.
- The Tensor Expansion prior is intentionally singular and is an explicit
  prior ablation, not the default. It follows the pinned QHFlow2 global sigma
  and all-one path-sum convention; it does not add geometry, species, initial-H,
  or empirical per-block scale conditioning.
- The parity bridge is SO(3)-only; reflections and full O(3) are outside this
  profile's contract.
- The matched FlowMatching suite deliberately sets QHFlow3's lmax=4 grid to the
  eSEN default latitude 10 by longitude 11. The maintained QHFlow3 adapter uses
  48x48 globally because 10x11 showed visible pair-GridAtomwise SO(3) aliasing at
  a strict float32 `1e-4` covariance tolerance. This suite choice is for matched
  structural comparison and is not a stronger equivariance guarantee; the
  strict high-resolution equivariance regression remains at 48x48.
- Validation quality metrics use one reproducible prior per rank/batch, not an
  average over multiple stochastic priors.
- Focused projection tests use a synthetic multi-shell basis. A heterogeneous
  NablaDFT reverse-edge pair with direction-dependent transpose masks remains
  part of the pending end-to-end validation.
- Checkpoint reload, throughput, memory, and matched quality evidence remain
  pending; bounded CUDA forward/backward, validation, and save are covered.
- The dated runner exposes validate, bounded smoke, full, and durable queue modes.

## Promotion decision

No promotion decision. Keep the feature isolated until the unchecked
verification items are complete and a matched direct-regression comparison
shows a reproducible scientific benefit.
