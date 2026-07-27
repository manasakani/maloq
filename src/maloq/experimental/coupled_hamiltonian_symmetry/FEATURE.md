# `coupled_hamiltonian_symmetry`

- Status: `smoke`
- Owner: SC26-seongsu
- Created: `2026-07-28`
- Intended promotion target: matrix-head output contract after matched validation

## Hypothesis

A symmetry-adapted output parameterization improves sample efficiency and
validation matrix error by learning only independent coupled-irrep channels.
This is the same contract as `reduce_node_intra=True`: diagonal orbital
self-interaction blocks omit antisymmetric odd-`L` channels instead of
predicting them and projecting them away later.

Node outputs learn the unique orbital upper triangle. Diagonal blocks contain
only even-`L` irreps, and the full onsite matrix is reconstructed by the
Clebsch-Gordan exchange sign. Edge outputs pair reverse directions before the
head: `alpha = E_ij + E_ji` and `beta = E_ij - E_ji`. Exchange-even diagonal
channels are learned in alpha, exchange-odd diagonal channels in beta, and
off-diagonal orbital-pair channels are remixed to reconstruct both directions.

The full-output projector remains only as an independent correctness oracle in
tests. It is not part of the training forward path.

## Baseline

- Git commit: working tree rooted at `SC26-seongsu`
- Config:
  `_my_script/experiment/2026-07-27/nabladft_v2_ofat_common.yaml`
- Dataset and exact split: NablaDFT ordered 12,081 train / 64 validation / 0 test
- Seed(s): 44
- Effective batch and optimizer: 20, Muon plus auxiliary AdamW
- Reference output/W&B run: E3 Muon+SHIFT V2 structure axis

## Entry point

- Import:
  `maloq.experimental.coupled_hamiltonian_symmetry.CoupledHamiltonianSymmetryWorkflow`
- Experimental config namespace: none; importing the explicit workflow is the
  feature switch
- Resolved component/profile ID: `node_intra_edge_pair_irrep_reduction_v1`
- Optional dependencies: none beyond canonical MALOQ

The dated runner leaves the legacy `reduce_*` config flags false. The
experimental head owns a fixed `reduce_node=True`, `reduce_node_intra=True`,
`reduce_edge=True` contract, so historical workflow branches are not selected.
Canonical MALOQ does not import this package.

## Checkpoint contract

- Architecture/schema version: `node_intra_edge_pair_irrep_reduction_v1`
- Compatible checkpoints: new checkpoints only
- State-dict migration: no automatic migration from the full-output Muon head
- Deterministic initialization: standard e3nn initialization is transferred
  into three Muon-visible matrices for node, edge-alpha, and edge-beta

## Verification

- [x] CPU import
- [x] Shape and forward/backward
- [x] Float32 coverage
- [x] SO(3) equivariance
- [x] Dense AO node and reverse-edge symmetry
- [x] Diagonal odd-`L` omission
- [x] Checkpoint save/reload
- [x] Config validation
- [x] CUDA train step
- [x] CUDA validation step
- [x] DDP smoke
- [ ] Matched parameter/memory/throughput comparison
- [ ] Matched quality comparison

## Evidence

| Date | Config | Source snapshot | Output/W&B | Result |
|---|---|---|---|---|
| 2026-07-28 | focused `s+p` basis CPU tests | dirty working tree | local pytest | 30 related tests passed: AO symmetry, equivariance, gradient, checkpoint, Muon/V2 regressions |
| 2026-07-28 | NablaDFT 12-shell, 64-channel head | dirty working tree | CPU forward/backward | full 1024; node 528; edge 528+496; zero residual |
| 2026-07-28 | MALOQ/NTEV2/QHFlow3 E3 Muon+SHIFT | server 2 GPUs 0-1 | 2-rank smoke | all train+validation smokes passed |

## Known limitations

- Every directed edge must have one unique reverse-directed partner.
- The coupled coefficient order must be the canonical row-major shell-pair
  order used by `Fock_Targets`.
- Initial validation targets closed-shell NablaDFT; open-shell tensors are
  structurally supported but do not yet have a full CUDA run.

## Promotion decision

Pending matched validation against the E3 Muon+SHIFT structure runs. Promotion
requires quality, throughput, memory, checkpoint, and multi-seed evidence.
