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
