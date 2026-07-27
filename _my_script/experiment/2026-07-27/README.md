# Experiments — 2026-07-27

## OMol_CSH HELM paper-contract experiment

This experiment makes the published OMol_CSH train H5 directly usable by the
current MALOQ/HELM pipeline while preserving the paper-era input contract:

- one restricted Fock target;
- actual def2-TZVPD shell order after undoing the released l-sorted transform;
- atomic number and geometry as the model inputs;
- H5 charge/spin attributes retained only as audit metadata;
- model conditioning fixed to closed-shell charge `0`, multiplicity `1`;
- no charge/spin embedding or scalar mixer in the eSEN backbone;
- no overlap, initial-density, or initial-Hamiltonian conditioning.

Overlap-dependent eigenvalue and total-energy metrics are disabled, not
approximated with an identity overlap.

The H5 reader is sample-streaming. It does not materialize the 276.5 GB train
file, or a rank's whole dense-matrix slice, in host memory. The immutable e3
basis transform and CuPy orbital-template pointer table are initialized once
per rank and reused; molecule-specific graph and matrix labels remain
on-demand so memory stays bounded.

### Files and paths

- Dataset: `/dataset/seongsu/shared-home/datasets/omol_csh/omol_csh_58k_train.h5`
- Config: `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/omol_csh_helm_paper_contract.yaml`
- Launcher: `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_omol_csh_helm_paper_contract_2gpu.sh`
- Environment: `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26`
- Outputs: `/dataset/seongsu/shared-home/workspace/project/outputs/omol-csh-helm-paper-contract-<timestamp>/`

### Commands

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_omol_csh_helm_paper_contract_2gpu.sh prepare
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_omol_csh_helm_paper_contract_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_omol_csh_helm_paper_contract_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_omol_csh_helm_paper_contract_2gpu.sh full 0,1
```

`prepare` verifies all three downloaded H5 files and writes their
`*.h5.keys.json` lazy-index sidecars. `validate` does not initialize
distributed training. `smoke` uses one train and one validation molecule per
GPU for one epoch and deletes its output only after success.

### Parity boundary

The architecture, Fock convention, cutoff, uncoupled loss, normalization
artifact, optimizer, and plateau scheduler follow the paper-era HELM setup.
The operational two-GPU config uses 20 epochs. The historical run used 3,000
epochs over 64 ranks with approximately 18,000-edge dynamic bins; reproducing
that schedule on two GPUs would require 86,241,000 optimizer steps. This is a
paper data/model-contract reproduction, not a bitwise reproduction of the
historical distributed run.

Even the bounded 20-epoch `full` run is large: 28,747 optimizer steps per rank
per epoch and 574,940 per rank in total. The launcher is ready, but capacity
and wall-time should be planned before starting it.

`scale_and_shift` intentionally reuses
`src/maloq/fock_utils/element_scale_shifts_omol.pt`. That legacy artifact has
the expected 58-element schema but no embedded provenance record. Automatic
H5 recomputation is deliberately rejected because the legacy statistics path
is neither streaming- nor DDP-safe. A separately audited, training-only
artifact is required before calling a final full run provenance-closed.

Status: implementation, unit tests, a real-sample contract check, and a
two-GPU one-epoch smoke run passed on 2026-07-27. The smoke included 62-atom
and 150-atom training samples, completed uncoupled-loss forward/backward,
globally stepped the plateau scheduler once, and reported physical-unit
validation matrix MAE `0.3771897422`. Peak allocated memory was about
10.14 GB. No full training was launched.

## NablaDFT EdgeBlock-1 atomwise-output control

This final-MAE control starts from completed SplitOutNorm run `n7z3h6o0` and
changes only `model.direct_atomwise_layers: [1]`. It retrains from seed 44;
it does not warm-start the completed checkpoint.

### Exact layer-operation delta

EdgePre normalizes the final node state before EdgeBlock 1, while
`direct_edgewise_layers: [1]` already prevents the initial edge state from
being added back at the first edgewise boundary. Let:

```text
m1 = EdgeWise1(Norm1(node), E0, geometry)
a1 = AtomWise1(Norm2(m1))
```

The completed SplitOutNorm base and this candidate differ only at the
EdgeBlock-1 return:

```text
SplitOutNorm base: E1 = m1 + AtomLayerScale1(a1)
Edge1AtomDirect:   E1 = a1
```

Thus the candidate removes the first atomwise residual and its bounded
degree-wise LayerScale from the E1 return, matching QHFlow3 `xy2`'s direct
atomwise return at that boundary. EdgeBlock 2 remains recurrent and
`residual_scaled`; `direct_edgewise_layers: [1]`, EdgePre, QHF conditioning,
QHFlow3 irrep projection, separate node/edge output norms, and the MALOQ head
remain unchanged.

### Base and selection thresholds

The primary selection metric is epoch-20 validation matrix MAE. Epoch 16–20
means are stability diagnostics.

- SplitOutNorm `n7z3h6o0`
  - final matrix/node/edge:
    `5.3684045842383014e-5 / 1.9448897111227616e-4 /
    6.472260160257549e-5`
  - epoch 16–20 mean matrix/node/edge:
    `7.998210265337332e-5 / 3.9827224251590593e-4 /
    9.206929769077577e-5`
- QHFlow3 comparator `aeorq52s`
  - final matrix MAE: `5.216202903956794e-5`

SplitOutNorm improved final matrix MAE by 3.5199% relative to `tq5e9a5p` and
remains 2.9179% above QHFlow3. This candidate must finish below
`5.3684045842383014e-5` to become the best NTE lane. Matching QHFlow3 requires
`<= 5.216202903956794e-5`, an additional 2.8351% reduction from the
SplitOutNorm base.

### Fixed training contract

- dataset:
  `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
- rows: train 12,081; validation 64; test 0
- RAW Fock-matrix target; no scale/shift
- 20 epochs; seed 44
- two data-parallel GPUs
- micro-batch 5 per rank; gradient accumulation 2; effective batch 20
- `distribute_graphs: false`
- MatrixMuon plus auxiliary AdamW
- W&B: `kaist-korea/maloq-nablaDFT`, online, every 10 optimizer steps
- display name:
  `NablaDFT | NTE-64/2 | MatrixMuon+AuxAdamW | RAW | QHFcond |
  EdgePre+Edge1+QHFProj+SplitOutNorm+Edge1AtomDirect | V1`

### Commands

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_nabladft_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct_2gpu.sh prepare
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_nabladft_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_nabladft_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/01_nabladft_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct_2gpu.sh full 0,1
```

Successful smoke artifacts are removed. Failed smoke evidence and full outputs
remain under:

`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-nte64-edgepre-edge1-qhfproj-splitoutnorm-edge1atom-v1-2gpu-eb20-mb5-ga2-<scope>-<timestamp>-<pid>/`

The durable queue manifest is
`queue_nte64_edgepre_edge1_qhflow3_projection_split_output_norm_edge1_atom_direct.yaml`.
It hashes the launcher and model config, requests two GPUs on one host, and
permits either SC26 server. The launcher passed `bash -n`, `prepare`, and
runner validation; the queue manifest schema, single-`{gpus}` contract,
config-only semantic delta, and `git diff --check` passed. A two-GPU
full-model 20-train/20-validation smoke also passed and removed its temporary
artifacts. No full job has been enqueued.

## QH9Stable V2 reset

The 24 historical W&B runs in `kaist-korea/maloq-qh9` are retained for
provenance but now have the display-name prefix
`[DEPRECATED 2026-07-27]`. Their local outputs and all converted databases are
unchanged. See `QH9_DEPRECATION.md` and:

`/dataset/seongsu/shared-home/workspace/project/outputs/wandb-naming/qh9-pre-deprecation-20260727.json`

The replacement suite compares corrected-Muon-head MALOQ, MALOQ-NTE-64/2,
and QHFlow3 on both QH9Stable Hamiltonian delta and density delta targets.
Every invocation is one independent single-GPU experiment. The default
micro-batch is 16 with two-step accumulation, giving effective batch 32.

### QHFlow3 edge and performance audit

QHFlow3 and MALOQ receive the same loader-created directed pair graph:
`rcut_orbitals=7.5`, `reduce_edge=false`, and no QHFlow3-specific top-k edge
removal. In the retained full run, the first micro-batch had 106 atoms and
696 directed edges for all three models.

The historical Hamiltonian-delta averages were:

- MALOQ: 1,024.594 seconds per train epoch;
- MALOQ-NTE: 958.535 seconds;
- QHFlow3: 3,799.771 seconds, or 3.71 times MALOQ.

Average recorded forward time was 0.04173 seconds per batch for MALOQ and
0.21200 seconds for QHFlow3. The main cost is therefore QHFlow3's Grid48
SO(3) grid transforms and GridAtomwise MLP in both node and pair blocks, not a
different edge count. The old `qhflow3_grid_ffn_chunk_size=512` added a
secondary cost by checkpointing and recomputing ordinary roughly 700-edge
batches.

A same-data 512-sample benchmark measured 12.407 seconds at chunk 512,
12.122 seconds at chunk 1024, and 10.741 seconds unchunked. Unchunked peak
memory reached about 57.7 GB even in the short benchmark, so it is not safe
for the full edge distribution. V2 uses chunk 1024: ordinary batches avoid
checkpointing while large batches remain bounded. Grid48 is retained because
lower native grids do not meet the established general-rotation equivariance
tolerance.

### Files

- MALOQ config:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/qh9stable_v2_maloq_muon.yaml`
- MALOQ-NTE config:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/qh9stable_v2_maloq_nte_muon.yaml`
- QHFlow3 config:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/qh9stable_v2_qhflow3_muon_grid48_chunk1024.yaml`
- Single-run launcher:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh`

### Commands

Validation does not train or write W&B runs:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh maloq hamiltonian validate 0
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh maloq-nte hamiltonian validate 1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh qhflow3 hamiltonian validate 2
```

Example full six-lane placement when GPUs 0 through 5 are free:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh maloq hamiltonian full 0
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh maloq-nte hamiltonian full 1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh qhflow3 hamiltonian full 2
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh maloq density full 3
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh maloq-nte density full 4
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/02_qh9stable_v2_single_run.sh qhflow3 density full 5
```

Use `smoke` in place of `full` before launching each lane. Successful smoke
artifacts are deleted; failed evidence and full outputs remain below the
absolute project `outputs/` directory. Full runs log to
`kaist-korea/maloq-qh9` every 10 optimizer steps with group
`qh9stable-v2-reset-20260727`. No full V2 run was launched while preparing
this suite.

## NTE EdgeBlock-2 post-atomwise residual control

This full NablaDFT control starts from the completed SplitOutNorm NTE-64/2
base. It changes only the second recurrent edge block:

```text
base:      E2 = (E1 + Se2 * F2) + Sa2 * A2(Norm2(E1 + Se2 * F2))
candidate: E2 = E1 + (Se2 * F2 + Sa2 * A2(Norm2(Se2 * F2)))
```

The candidate therefore preserves the incoming recurrent `E1` residual but
moves it outside the complete local EdgeBlock-2 branch. This matches the
QHFlow3 layer boundary more closely without changing conditioning, EdgePre,
EdgeBlock-1 direct edgewise behavior, QHFlow3 irrep output projection,
separate output norms, head, optimizer, or training schedule.

- immutable runtime:
  `/dataset/seongsu/shared-home/workspace/project/outputs/experiment-source-trees/nabla-nte64-edge2postres-fee42435`
- runtime fingerprint:
  `e770d7b9dfecf3a18e988da21511ac2bed6f9e107d214d206cbfba9a7e46169c`
- runtime archive:
  `/dataset/seongsu/shared-home/workspace/project/outputs/experiment-source-archives/nabla-nte64-edge2postres-fee42435.tar.gz`
- archive SHA-256:
  `36a7f09f160b26fc538c537625e00ede92c081c06696e1605267fc2f12b7ba5c`
- queue manifest:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/queue_nte64_edge2_post_residual_frozen.yaml`
- queue ID:
  `nabla-nte64-edge2postres-frozen-v1-20260727b`
- data/split: NablaDFT `12081 / 64 / 0`
- training: 20 epochs, two GPUs, micro-batch 5/rank, accumulation 2,
  effective batch 20, seed 44
- W&B: `kaist-korea/maloq-nablaDFT`, every 10 optimizer steps
- primary selection: epoch-20 validation matrix MAE

The fixed runtime and canonical outer launcher passed checksum validation,
runner validation, and a two-GPU full-size smoke. A successful smoke left no
output directory or alias. The full queue run pins the canonical outer
launcher as an immutable input and stores the complete dirty canonical
worktree diff, untracked-file hashes/archive, queue request, and frozen
runtime provenance with the experiment output.

## NTE best-run server-1 backlog

Four independent full NablaDFT ablations keep the completed `n7z3h6o0`
contract fixed and are restricted to server 1:

- priority 18: `InitialEdgeZero+QHFProj+SplitOutNorm`;
- priority 17: `QHFPair+QHFProj+SplitOutNorm`;
- priority 16: `Best+GaussianWidth2`;
- priority 15: `Best+EdgePostRMS`.

All jobs use NablaDFT `12081 / 64 / 0`, RAW targets, seed 44, two GPUs,
micro-batch 5 per rank, accumulation 2, effective batch 20, MatrixMuon plus
auxiliary AdamW, and 20 complete epochs. W&B logs to
`kaist-korea/maloq-nablaDFT` every 10 optimizer steps under group
`nabla-nte64-best-n7-backlog-v1`. Epoch-20 validation matrix MAE is the
primary comparison and the epoch 16–20 mean is secondary.

- immutable runtime:
  `/dataset/seongsu/shared-home/workspace/project/outputs/experiment-source-trees/nabla-nte64-best-backlog-42e50501`
- runtime patch SHA-256:
  `42e505016b14bdd67fbe982924ad4bc44c3acbeaecf03be6314d94c5fdbd529d`
- runtime fingerprint:
  `bea057e0638f09710a57e5bcc5d1bd91fd836558617c206a88b7f93f02548564`
- queue manifest:
  `/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/queue_nte64_best_n7_backlog_server1_frozen.yaml`
- output prefixes:
  `/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-nte64-{initzero-qhfproj-splitoutnorm,qhfpair-qhfproj-splitoutnorm,best-gw2,best-edgepostrms}-v1-2gpu-eb20-mb5-ga2-full-e20-*`

Each of the four exact frozen config paths passed runner validation and a
two-GPU full-size smoke. Successful smoke artifacts were removed. The
manifest can be enqueued from any directory with:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python /dataset/seongsu/shared-home/workspace/project/_auto_script/experiment_queue/sc26_queue.py enqueue --allow-dirty /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/queue_nte64_best_n7_backlog_server1_frozen.yaml
```

The manifest pins the complete dirty canonical source fingerprint, each
canonical and frozen config hash, the frozen archive, and the queue-source
snapshot. Do not change the canonical worktree while any of these jobs remain
pending, because a worker will block a job whose source fingerprint changes.

## NTE next-layer controls (server-1 only)

The completed InitialEdgeZero run `67ji47fh` and Edge2PostResidual run
`6h06e4e9` are the bases for six controlled follow-ups:

- priority 30: `InitialEdgeZero+NodeSO3Proj+EdgeQHFProj`;
- priority 29: `InitialEdgeZero+Edge2PostResidual`;
- priority 28: `Edge2PostResidual+NodeSO3Proj+EdgeQHFProj`;
- priority 27: `Edge2PostResidual+InitialEdgeEnvelope`;
- priority 20: `Edge2PostResidual+GaussianWidth2`;
- priority 19: `Edge2PostResidual+Edge2PostRMS`.

The data, split, optimizer, head, batch, accumulation, seed, W&B cadence, and
20-epoch contract remain unchanged. Epoch-20 validation matrix MAE is primary.
The immutable runtime is
`outputs/experiment-source-trees/nabla-nte64-next-controls-79397692`
with source fingerprint
`79397692425321f754ecd4d7a4fbe68605ce92d067956a18fcf9011510767eec`.

The canonical launcher is only a short dispatcher. Queue runs immediately
`exec` the hash-pinned frozen implementation, which exports the relocated
OpenMPI prefixes, pins the absolute `prterun` helper for absolute `mpirun`,
and appends to `train.log`.
This prevents a mutable long-running launcher or a second invocation from
truncating completed-run evidence.

The server-1-only manifest is:

`/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-27/queue_nte64_next_controls_server1_frozen_v2.yaml`

Each job is enqueued only after its exact frozen config passes runner
validation and a two-GPU full-size smoke. Successful smoke outputs are not
retained.

After the first three v2 controls completed without improving
InitialEdgeZero, one additional single-change control was prepared:
`InitialEdgeZero+InitialEdgeEnvelope` (priority 31). Its server-1-only
manifest is
`queue_nte64_initial_edge_zero_initial_envelope_server1_frozen.yaml`, and its
frozen runtime is
`outputs/experiment-source-trees/nabla-nte64-initzero-envelope-aea6e61f`.
