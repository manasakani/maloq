# NablaDFT MALOQ-NTE Muon scaling law

This experiment measures parameter scaling at a fixed data and optimizer
budget. It extends the existing 33.75M MALOQ-NTE Muon-head run without
changing the target, split, optimizer, or effective batch.

## Model ladder

| Label | Width | Node depth | Edge depth | Parameters | Micro-batch / GPU | Grad accumulation | Effective batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| p16m-w88-d3 | 88 | 3 | 3 | 16,125,037 | 5 | 2 | 20 |
| p33m-w128-d3 | 128 | 3 | 3 | 33,750,157 | 5 | 2 | 20 |
| p125m-w192-d5 | 192 | 5 | 5 | 125,004,341 | 2 | 5 | 20 |
| p500m-w384-d5 | 384 | 5 | 5 | 496,331,189 | 1 | 10 | 20 |

`l_embedding_dim`, `hidden_dim`, and `output_l_embedding_dim` are equal at
every point. Node and edge depths are also equal. This avoids adding a
readout bottleneck or changing only one side of the NTE schedule.

## Fixed controls

- dataset: NablaDFT `train_2k.db`
- ordered split: 12,081 train / 64 validation / 0 test
- objective: absolute Fock matrix, no label scale/shift
- head: `maloq_muon`
- schedule: node then edge
- distance basis: 512
- orbital / Gaussian cutoffs: 8 / 16 Bohr
- optimizer: Muon, LR 0.02, momentum 0.95, Nesterov, 5 NS steps
- auxiliary AdamW: LR 5e-4, betas (0.9, 0.95), epsilon 1e-10
- weight decay: 1e-4
- scheduler: 1,000-step warmup followed by polynomial decay
- epochs: 20
- dtype: float32
- seed: 44
- primary scaling metric: validation matrix MAE
- W&B group: `nabla-nte-muon-scaling`
- experiment version: `V1`

This is a fixed-data model scaling experiment, not a compute-optimal scaling
law. All points receive the same 20-epoch data budget.

## GPU schedule

The default schedule uses three two-GPU data-parallel lanes:

- GPUs 2,3: 500M
- GPUs 4,5: 125M
- GPUs 6,7: 33M, then 16M

The 500M point is smoke-tested first with micro-batch 1. Full runs preserve
effective batch 20 by changing gradient accumulation.

## Commands

```bash
LAUNCHER=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-24/02_nabladft_maloq_nte_scaling_law_6gpu.sh

"${LAUNCHER}" prepare
"${LAUNCHER}" validate 2,3 4,5 6,7
EXPECTED_HOST=usr310-gpumngc-01 "${LAUNCHER}" smoke 2,3 4,5 6,7
EXPECTED_HOST=usr310-gpumngc-01 "${LAUNCHER}" full 2,3 4,5 6,7
```

`validate` is single-process and therefore previews a smaller effective batch.
The actual two-rank smoke and full jobs have effective batch 20.

## Outputs

Full output groups follow:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/nabla-nte-muon-scaling-4point-v1-eb20-full-e20-seed44-<timestamp>/
```

Each group contains a launch manifest, source revision, coordinator log,
per-model logs and statuses, and one run directory per model. The full
launcher refuses existing output paths and refuses any GPU using more than
1,024 MiB before launch.

## Status

- all four configurations passed validation
- the 496,331,189-parameter full-width smoke passed on GPUs 2,3
- smoke effective batch: 20; forward, backward, and Muon step verified
- historical V1 full group launched on `usr310-gpumngc-01`; its original
  output path is retained for provenance:
  `ndft-nte-muon-scaling-4point-eb20-full-e20-seed44-20260724-165100`
- 500M, 125M, and 33M lanes reached their first training batch
- 16M is queued on GPUs 6,7 and starts automatically after 33M
- initial W&B run IDs: 500M `z9gt0ohd`, 125M `wosfc6ww`,
  33M `w9m2o09g`
