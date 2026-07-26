# NablaDFT semantic-global-Muon baselines

These runs retain the exact RAW V1 model and training recipes for MALOQ,
NTE-64/2, and NTE-128/3 while making the optimizer contract explicit.
Ordinary trainable tensors with `ndim >= 2` use Muon, scalar/vector parameters
use AdamW, and the MALOQ node/edge path contractions are materialized as one
global matrix per branch and placed in the named
`semantic_global_head_muon` group.

The V2 label denotes explicit optimizer provenance and new output/W&B
identifiers; it does not change the backbone, target, seed, batch, scheduler,
or mathematical head forward from the corresponding V1 run.

Dataset and split:

- `/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db`
- train rows: 12,081
- validation rows: 64
- RAW Fock-matrix target

Training:

- environment:
  `/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26`
- 20 epochs, seed 44
- two data-parallel GPUs per model
- micro-batch 5 per rank, gradient accumulation 2, effective batch 20
- Muon LR 0.02, auxiliary AdamW LR 0.0005
- W&B: `kaist-korea/maloq-nablaDFT`, logging every 10 optimizer steps
- W&B group: `nabla-semantic-global-muon-raw-v2`

W&B display names:

- `NablaDFT | MALOQ | MatrixMuon+AuxAdamW+SGHead | RAW | V2`
- `NablaDFT | NTE-64/2 | MatrixMuon+AuxAdamW+SGHead | RAW | V2`
- `NablaDFT | NTE-128/3 | MatrixMuon+AuxAdamW+SGHead | RAW | V2`

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/01_nabladft_semantic_global_muon_baselines_3x2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/01_nabladft_semantic_global_muon_baselines_3x2gpu.sh smoke 0,1 2,3 4,5
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/01_nabladft_semantic_global_muon_baselines_3x2gpu.sh full 0,1 2,3 4,5
```

Successful smoke artifacts are removed. Failed smoke evidence and all full
outputs remain below
`/dataset/seongsu/shared-home/workspace/project/outputs/`.

## Launch status

The three full runs started on 2026-07-26 10:00 KST on `scp-gpu-2`
(`usr310-gpumngc-02`) in tmux session `sc26-semglobal-muon-v2`.

| Lane | GPUs | Semantic-global parameters | W&B |
|---|---:|---:|---|
| MALOQ | 0,1 | 50,688 | [8x17hoxs](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/8x17hoxs) |
| NTE-64/2 | 2,3 | 25,344 | [a86zytgu](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/a86zytgu) |
| NTE-128/3 | 4,5 | 50,688 | [07blkhuu](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/07blkhuu) |

The retained full output group is:

`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-semglobal-muon-3x2gpu-eb20-mb5-ga2-full-e20-20260726-100048/`

Before the full launch, all three lanes passed config validation and a
full-model, two-GPU, 20-train/20-validation smoke. The smoke completed
forward, backward, semantic-global head Muon updates, and validation matrix
reconstruction; its temporary outputs were removed.

## NTE-64/2 layer-structure ablations

The trained-checkpoint analysis compares the exact cited runs on the same 64
NablaDFT validation molecules:

- QHFlow3: `80sa5m4j`, OV0, default 10x11 grid, V2
- NTE-64/2: `fao0946w`, QHFcond, V1

The generated report is
`outputs/nabladft-qhf-ov0-ntegrid-vs-nte-qhfcond-layer-analysis/report.md`.
It isolates three higher-priority causes than input conditioning:

1. QHFlow3 NodeBlock 2 grows the `l=4` edgewise update to 5.91 times its
   incoming state; NTE's effective update is only 0.46 times after its learned
   degree scale.
2. NTE EdgeBlock 2 grows `l=0` by 5.55 times but `l=4` by only 1.09 times.
   QHFlow3 instead sums two independent pair branches.
3. NTE's degree-batched 128-to-64 projection is seen by generic Muon as a
   `5 x 8192` matrix, while QHFlow3's flat e3nn projection is routed to AdamW.

The initial three single-factor configs completed all 20 training epochs and
synced their final metrics to W&B:

- `nte64e2_qcond_node2_nolayerscale_nabladft.yaml`
  - W&B `u2hb8tzd`: matrix MAE `8.5813e-5`; worse than the cited NTE baseline.
- `nte64e2_qcond_qhflow3_parallel_edge_nabladft.yaml`
  - W&B `sb9afgms`: matrix MAE `6.0942e-5`; the epoch 16-20 mean
    (`9.2118e-5`) is within 0.1% of QHFlow3 (`9.2024e-5`).
- `nte64e2_qcond_projection_adamw_nabladft.yaml`
  - W&B `bpbuztlq`: matrix MAE `7.7622e-5`; projection optimizer routing alone
    did not close the gap.

The next single-operation candidates are:

- `nte64e2_qcond_repeat_system_nabladft.yaml`
  - Keeps the recurrent NTE topology and bounded degree residuals fixed.
  - Reproduces QHFlow3's layer-local operation that adds its learned
    zero-charge/spin system scalar after each NodeBlock's first normalization.
- `nte64e2_qcond_nte_parallel_edge_nabladft.yaml`
  - Keeps the original NTE EdgeBlock normalization, bounded residual scales,
    and initial edge state.
  - Changes only the stack topology: both EdgeBlocks read the same initial edge
    state independently, and their outputs are summed.
  - Comparison with `sb9afgms` separates topology from QHFlow3 pair-block math.

Each preserves the cited NTE run's QHF conditioning, seed, 20 epochs, two-GPU
data parallelism, micro-batch 5, gradient accumulation 2, effective batch 20,
RAW target, and W&B logging every 10 optimizer steps.

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/02_nabladft_nte64_layer_ablation_2gpu.sh validate node2
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/02_nabladft_nte64_layer_ablation_2gpu.sh smoke qhfpair 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/02_nabladft_nte64_layer_ablation_2gpu.sh full projadamw 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/02_nabladft_nte64_layer_ablation_2gpu.sh validate repeatsys
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/02_nabladft_nte64_layer_ablation_2gpu.sh validate nteparallel
```

Successful smoke output is removed; failed smoke evidence and all full outputs
are retained below `outputs/`.

Queue manifest:

`queue_nte64_layer_ablations.yaml`

It contains three two-GPU jobs restricted to `server-1`. Priority order is
NodeBlock 2 LayerScale, QHFlow3-style pair topology, then projection AdamW.
The queue records the source fingerprint and hashes the launcher plus the
selected config for every job. The first three jobs trained and synced all 20
epochs, but their queue attempts ended as false failures after the shared
launcher was edited while those shell processes were still open. The training
outputs and W&B runs are complete; future queued jobs must use a launcher that
is not modified between claim and exit.

Follow-up queue manifest:

`queue_nte64_layer_followups.yaml`

It pins `RepeatSys` to `server-1` and `NTEParallel` to `server-2`, with one
two-GPU full job per host. Both use the already smoke-tested launcher and keep
the baseline seed, effective batch, scheduler, and W&B logging unchanged.

### Source-target-message ablation

`nte64e2_qcond_source_target_message_nabladft.yaml` keeps the cited
NTE-64/2 QHFcond baseline's recurrent edge stack, bounded degree residuals,
seed, scheduler, and MatrixMuon+AuxAdamW head fixed. Its only model change is
`message_type: source-target-message`, so the initial or previous edge state
is concatenated with source and target node features inside every first SO(2)
convolution.

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/04_nabladft_nte64_stmessage_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/04_nabladft_nte64_stmessage_2gpu.sh smoke 2,3
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/04_nabladft_nte64_stmessage_2gpu.sh full 2,3
```

The full run uses two data-parallel GPUs, micro-batch 5 per rank, gradient
accumulation 2, effective batch 20, 20 epochs, and W&B logging every 10
optimizer steps. `queue_nte64_stmessage.yaml` restricts its full job to
`server-1`. Successful smoke output is removed; failed smoke and full outputs
remain below `outputs/`.

## NTE-64/2 semantic gate-Muon comparison

This V3 lane changes one optimizer-routing factor relative to W&B baseline
[`a86zytgu`](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/a86zytgu):
the 64-to-320 scalar/gate projection is materialized as a semantic matrix and
routed through Muon. The backbone, semantic-global node/edge head, RAW target,
12,081/64 split, seed 44, 20 epochs, micro-batch 5 per rank, two ranks,
gradient accumulation 2, scheduler, and learning rates match the V2 baseline.

- Baseline: `NablaDFT | NTE-64/2 | MatMuon+SGHead | RAW | V2`
- Gate-Muon: `NablaDFT | NTE-64/2 | MatMuon+SGHead+GateMuon | RAW | V3`
- Shared W&B comparison group: `nabla-semantic-global-muon-raw-v2`
- Gate-Muon output base:
  `/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-nte64-sghead-gatemuon-raw-v3-2gpu-eb20-mb5-ga2-<scope>-<timestamp>/`

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/03_nabladft_nte64_gate_muon_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/03_nabladft_nte64_gate_muon_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/03_nabladft_nte64_gate_muon_2gpu.sh full 0,1
```

The GPU pair is explicit and may be changed. Successful smoke artifacts are
removed; failed smoke and all full-run artifacts are retained under
`/dataset/seongsu/shared-home/workspace/project/outputs/`.


## NTE recurrent edge-normalization ablations

These three single-factor runs separate the causes of EdgeBlock 2's observed
scalar bias while keeping the NTE-64/2 QHFcond recurrent topology, node stack,
bounded degree LayerScale, MatrixMuon+AuxAdamW head, RAW target, seed 44,
split, scheduler, and effective batch 20 fixed.

| Ablation | Only active change | Hypothesis |
|---|---|---|
| `EdgePostRMS` | Apply `rms_norm_sh` after each edge block's final atomwise residual sum | Tests whether the missing block-local post-residual norm permits magnitude accumulation |
| `EdgeSplitNorm` | Change only edge-block `norm_2` to `layer_norm_sh` | Tests whether scalar `l=0` suppresses the joint `l>0` tensor sector through a shared RMS denominator |
| `EdgeDegreeNorm` | Change only edge-block `norm_2` to per-degree `layer_norm` | Tests whether imbalance within `l>0`, especially `l=4`, remains after scalar/tensor separation |

The options are edge-only: node-block norms and the final trunk norm remain the
baseline `rms_norm_sh`. `EdgePostRMS` changes the stored recurrent edge state;
the other two normalize only the input to the edge atomwise FFN and preserve
the unnormalized residual identity path.

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/05_nabladft_nte64_edge_norm_ablation_2gpu.sh validate postrms
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/05_nabladft_nte64_edge_norm_ablation_2gpu.sh validate split
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/05_nabladft_nte64_edge_norm_ablation_2gpu.sh validate degree
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/05_nabladft_nte64_edge_norm_ablation_2gpu.sh smoke postrms 4,5
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/05_nabladft_nte64_edge_norm_ablation_2gpu.sh full postrms 4,5
```

Every full run uses 20 epochs, two data-parallel GPUs, micro-batch 5 per rank,
gradient accumulation 2, and W&B logging every 10 optimizer steps in
`kaist-korea/maloq-nablaDFT`, group
`nabla-nte64-edge-norm-ablation-v1`. Successful smoke output is removed; failed
smoke and full outputs remain below `outputs/`. The durable queue manifest is
`queue_nte64_edge_norm_ablations.yaml`.


## NTE direct edge-atomwise output ablation

`EdgeAtomDirect` isolates the final operation inside each recurrent NTE
EdgeBlock. The baseline returns
`bounded_degree_scale(atomwise(x)) + residual`; this ablation returns
`atomwise(x)` directly, matching the original QHFlow3 `xy2` block at that
stage. It intentionally keeps NTE's recurrent edge topology, raw-node
edgewise input, first edgewise residual, normalization types, node stack,
conditioning, optimizer, and data order unchanged.

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/06_nabladft_nte64_edge_atom_direct_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/06_nabladft_nte64_edge_atom_direct_2gpu.sh smoke 4,5
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/06_nabladft_nte64_edge_atom_direct_2gpu.sh full 4,5
```

The full run uses 20 epochs, two data-parallel GPUs, micro-batch 5 per rank,
gradient accumulation 2, effective batch 20, RAW Fock targets, seed 44, no
graph distribution, and W&B logging every 10 optimizer steps in
`kaist-korea/maloq-nablaDFT`. The durable queue manifest is
`queue_nte64_edge_atom_direct.yaml`.

### Completed structural results

Every row below completed 20 epochs with the same NablaDFT rows, seed,
optimizer, effective batch, and validation cadence. `Tail-5` is the mean matrix
MAE over epochs 16–20.

| Variant | W&B | Final matrix MAE | Tail-5 matrix MAE |
|---|---|---:|---:|
| QHFlow3 reference | `80sa5m4j` | 5.5018e-5 | 9.2024e-5 |
| NTE-64/2 QHFcond reference | `fao0946w` | 7.0757e-5 | 1.0854e-4 |
| QHFPair | `sb9afgms` | 6.0942e-5 | 9.2118e-5 |
| RepeatSys | `cjea73l0` | 7.2947e-5 | 1.0225e-4 |
| NTEParallel | `ixli76xw` | 7.3554e-5 | 1.0923e-4 |
| STMessage | `bcwrm9wc` | 7.4219e-5 | 1.1011e-4 |
| EdgePostRMS | `uh1mdkgv` | 6.8514e-5 | 1.0064e-4 |
| EdgeSplitNorm | `gu0eanf4` | 7.3127e-5 | 1.0286e-4 |
| EdgeDegreeNorm | `1xt6c4xb` | 7.3935e-5 | 1.0601e-4 |
| EdgeAtomDirect | `6w6yjvzc` | 7.8412e-5 | 1.1432e-4 |
| EdgePreNodeNorm | `3uq7prdf` | 6.1804e-5 | 9.0054e-5 |

Independent branch topology alone (`NTEParallel`) and carrying the old edge
state in the message (`STMessage`) do not explain the gap. Post-residual RMS
normalization gives a real but partial improvement. The complete QHFlow3 pair
operation (`QHFPair`) ties QHFlow3 on tail matrix MAE. `EdgeAtomDirect` shows
that removing the final atomwise residual and bounded update scale alone is
harmful. `EdgePreNodeNorm` is the strongest recurrent single-operation result:
its epoch 16-20 matrix mean is 2.14% lower than QHFlow3 and 17.03% lower than
the cited NTE baseline. Its corresponding node and edge means are also 9.37%
and 0.86% lower than QHFlow3. The final-epoch matrix MAE remains 12.33% above
QHFlow3, so the robust conclusion is parity over the common late-training
window rather than superiority from one endpoint. Together with `QHFPair`,
this isolates QHFlow3's `node norm -> edgewise` boundary as the leading tested
pair-layer difference.


## NTE edge pre-node normalization ablation

`EdgePreNodeNorm` moves each recurrent EdgeBlock's first RMS normalization from
after edgewise message construction to the final node state before edgewise.
This matches the original QHFlow3 pair-block ordering at exactly one operation
boundary. It keeps NTE's recurrent edge state, first and second residuals,
bounded degree LayerScale, atomwise output rule, normalization types, node
stack, data order, and optimizer unchanged.

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/07_nabladft_nte64_edge_pre_node_norm_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/07_nabladft_nte64_edge_pre_node_norm_2gpu.sh smoke 4,5
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/07_nabladft_nte64_edge_pre_node_norm_2gpu.sh full 4,5
```

The durable queue manifest is `queue_nte64_edge_pre_node_norm.yaml`; the full
run uses the same 20-epoch, two-GPU, effective-batch-20, seed-44, RAW-target,
W&B-every-10-steps contract as the preceding pair-operation ablations.

The full queue job
`nabla-nte64-qcond-edge-pre-node-norm-v1-20260726a` completed all 20 epochs
from clean source commit `8d92906` on server 1 GPUs 0 and 1. Its retained output
is:

`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-nte64-edge-pre-node-norm-2gpu-eb20-mb5-ga2-full-e20-20260726-162158/`

W&B [`3uq7prdf`](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/3uq7prdf)
finished with final matrix/node/edge MAE of `6.1804e-5`, `2.3096e-4`, and
`7.4229e-5`. Its epoch 16-20 means are `9.0054e-5`, `4.2162e-4`, and
`1.0474e-4`, respectively. Queue doctor reported no remaining claim, GPU lock,
or invalid state after completion.


## NTE EdgeBlock-1 direct edgewise output ablation

`Edge1Direct` isolates the initial residual boundary in the recurrent edge
stack. Only EdgeBlock 1 changes from
`bounded_degree_scale(edgewise(x)) + initial_edge` to the normalized edgewise
message itself. EdgeBlock 2 remains `residual_scaled`, so it still adds
EdgeBlock 1's output and preserves a loss/gradient path through both blocks.
The baseline post-edgewise `norm_1` position, final atomwise residuals and
bounded scales, node stack, QHF conditioning, optimizer, split, seed, and data
order remain unchanged. This is the first-residual counterpart to the
completed `EdgeAtomDirect` final-residual test.

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/08_nabladft_nte64_edge1_direct_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/08_nabladft_nte64_edge1_direct_2gpu.sh smoke 0,1
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/08_nabladft_nte64_edge1_direct_2gpu.sh full 0,1
```

The full output pattern is
`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-nte64-edge1-direct-2gpu-eb20-mb5-ga2-full-e20-<timestamp>/`.
It uses the project environment, 20 epochs, two data-parallel GPUs,
micro-batch 5 per rank, gradient accumulation 2, effective batch 20, RAW Fock
targets, seed 44, no graph distribution, and W&B logging every 10 optimizer
steps. The durable queue manifest is `queue_nte64_edge1_direct.yaml`.
The config validation, 46 QH9 tests, three fixed-resume tests, and a full-model
two-GPU 20-train/20-validation smoke passed; successful smoke artifacts were
removed.


## QHFlow3 output-projection Muon ablation

`QHF3-ProjMuon` starts from QHFlow3 reference `80sa5m4j` and changes only the
two final `128 -> 64` node/edge projections (`81,920` parameters). The original
e3nn projection stores each weight as a flat vector and therefore sends it to
AdamW under shape-based Muon routing. This ablation stores the same paths as
`[degree, output, input]`, making them Muon-visible.

The new projection still invokes the original external-weight e3nn operation.
With the same seed, its mapped initial weights, forward output, input gradient,
and parameter gradient are bitwise identical to the reference before optimizer
step 1. The shared `MuonFockIrrepsHead`, all preceding QHFlow3 layers, OV0
conditioning, NTE 10x11 grid, data rows/order, and loss remain unchanged.

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/09_nabladft_qhflow3_projection_muon_2gpu.sh validate
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/09_nabladft_qhflow3_projection_muon_2gpu.sh smoke 6,7
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/09_nabladft_qhflow3_projection_muon_2gpu.sh full 6,7
```

The durable queue manifest is `queue_qhflow3_projection_muon.yaml`. The full
run uses 20 epochs, two data-parallel GPUs, micro-batch 5 per rank, gradient
accumulation 2, effective batch 20, RAW Fock targets, seed 44, no graph
distribution, W&B logging every 10 optimizer steps in
`kaist-korea/maloq-nablaDFT`, and the compact display name
`NablaDFT | QHFlow3 | MatrixMuon+ProjMuon+AuxAdamW | RAW | OV0 |
NTEGrid10x11 | V3`.


## Literal QHFlow3 layer-transplant factorial

The earlier `QHFPair` run (`sb9afgms`) reproduced QHFlow3's pair-stack order
with NTE implementations of SO(2) convolution, gate activation, normalization,
and grid FFN. It did not instantiate QHFlow3's original layer classes. These
three V2 runs instead import and instantiate `eSCNMD_Block` and
`eSCNMD_Block_xy2` directly from `src/maloq/helm/qhflow3_clean.py`.

Together with NTE reference `fao0946w`, they form a node-by-pair 2x2 factorial:

| Variant | Node stack | Pair stack |
|---|---|---|
| NTE reference | NTE NodeBlock x3 | recurrent NTE EdgeBlock x2 |
| `QHFNodeExact` | literal QHFlow3 NodeBlock x3 | recurrent NTE EdgeBlock x2 |
| `QHFPairExact` | NTE NodeBlock x3 | literal QHFlow3 xy2 x2, independent sum |
| `QHFBlocksExact` | literal QHFlow3 NodeBlock x3 | literal QHFlow3 xy2 x2, independent sum |

The exact blocks receive QHFlow3's original layer input contract: m-major
Wigner mapping, Gaussian width 2.0, `distance/source/target` radial-feature
order, sigmoid gate, default 10x11 SO(3) grid, and a 512-row grid-FFN chunk.
The NTE QHF conditioner, one shared NTE graph, 128-to-64 SO(3) projections,
Muon Fock head, target, data rows/order, and optimizer schedule remain fixed.
`QHFNodeExact` and `QHFBlocksExact` also repeat the learned system scalar in
each node block, as the original QHFlow3 block does; completed `RepeatSys`
provides the controlled single-operation comparison for that constituent.

Pre-launch parameter routing confirms that the exact node replacement is
nearly capacity-neutral: it removes only 30 scalar degree-LayerScale
parameters and leaves shape-routed Muon parameters unchanged (`28,180,864`).
The exact pair replacement has `28,463,646` backbone parameters versus
`28,231,730` for NTE, with `28,410,880` versus `28,180,864` routed to Muon.
Almost all of that increase is QHFlow3's original second scalar-modulation
branch (`fc2`), which is functionally dead because its second SO(2)
convolution uses internal weights; it is retained deliberately so the imported
pair layer is literal rather than a cleaned reimplementation.

Configs:

- `nte64e2_qcond_qhflow3_exact_node_nabladft.yaml`
- `nte64e2_qcond_qhflow3_exact_pair_nabladft.yaml`
- `nte64e2_qcond_qhflow3_exact_blocks_nabladft.yaml`

Commands:

```bash
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/10_nabladft_nte64_qhflow3_layer_transplant_2gpu.sh validate node
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/10_nabladft_nte64_qhflow3_layer_transplant_2gpu.sh smoke pair 4,5
/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/10_nabladft_nte64_qhflow3_layer_transplant_2gpu.sh full blocks 6,7
```

Every full lane uses 20 epochs, two data-parallel GPUs, micro-batch 5 per rank,
gradient accumulation 2, effective batch 20, RAW Fock targets, seed 44, no
distributed graph, and W&B logging every 10 optimizer steps in
`kaist-korea/maloq-nablaDFT`, group
`nabla-nte64-qhflow3-layer-transplant-v1`. Full outputs use:

`/dataset/seongsu/shared-home/workspace/project/outputs/nabladft-nte64-qhf*exact-v2-2gpu-eb20-mb5-ga2-full-e20-<timestamp>/`

The durable manifest is `queue_nte64_qhflow3_layer_transplants.yaml`. Status:
all three configs passed runner validation; 63 related regression tests passed;
and `QHFNodeExact`, `QHFPairExact`, and `QHFBlocksExact` each passed a
full-model two-GPU 20-train/20-validation smoke. Successful smoke artifacts
were removed. A separate general-rotation check measured maximum covariance
errors of `3.22e-6` for node features and `7.73e-7` for pair features. Full
queue launch is pending the clean source commit.
