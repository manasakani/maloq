# NablaDFT 2-GPU effective-batch-20 checks

This experiment validates two distinct 2-GPU execution modes with NablaDFT,
Muon, seed 44, one train step, and one validation step:

- `data-parallel`: each rank owns 10 molecules; averaged gradients represent an
  effective global batch of 20.
- `distributed-graph`: each rank contributes 10 molecules to one 20-molecule
  global supergraph, which is partitioned across both ranks with
  `linear-edgewise`. The PyG DataLoader batch is 1 because that item is the
  global supergraph.

Both modes use Open MPI because distributed-graph internals combine MPI and
`torch.distributed`. `torchrun` is intentionally rejected for distributed
graphs when its ranks do not match `MPI.COMM_WORLD`.

Run MALOQ on GPUs 6 and 7:

```bash
cd /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-22
GPUS=6,7 MASTER_PORT=29570 \
  ./04_nabladft_2gpu_effective_batch20.sh data-parallel maloq smoke
```

Run the distributed-graph check:

```bash
cd /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-22
GPUS=6,7 MASTER_PORT=29571 \
  ./04_nabladft_2gpu_effective_batch20.sh distributed-graph maloq smoke
```

Replace `maloq` with `maloq-nte` to test the NTE model. For the data-parallel
mode, `qhflow3` and `all` are also accepted; `all` runs MALOQ, MALOQ-NTE, and
QHFlow3 sequentially. Outputs and local W&B files go below
`outputs/nabladft-*-2gpu-eb20-*/`; online metrics go to
`kaist-korea/maloq-nablaDFT`.

The canonical matched comparison command is:

```bash
cd /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-22
GPUS=6,7 MASTER_PORT=29570 \
  ./04_nabladft_2gpu_effective_batch20.sh data-parallel all full
```

It creates one `nabladft-data-parallel-2gpu-eb20-three-model-comparison-...`
root with `maloq/`, `maloq-nte/`, and `qhflow3/` model directories.

## Three dedicated 2-GPU pairs with safer micro-batches

Use `05_nabladft_three_models_3x2gpu_mb5_ga2.sh` to run the models concurrently
on fixed, non-overlapping GPU pairs:

| Model | GPUs | Per-rank micro-batch | Accumulation | Effective batch |
| --- | --- | ---: | ---: | ---: |
| MALOQ | 0,1 | 5 | 2 | 20 |
| MALOQ-NTE | 2,3 | 5 | 2 | 20 |
| QHFlow3 | 6,7 | 5 | 2 | 20 |

Run a one-epoch production-size-model smoke first:

```bash
cd /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-22
./05_nabladft_three_models_3x2gpu_mb5_ga2.sh all smoke
```

Then launch the matched 20-epoch runs:

```bash
./05_nabladft_three_models_3x2gpu_mb5_ga2.sh all full
```

To launch lanes separately, replace `all` with `maloq`, `maloq-nte`, or
`qhflow3`. The `all` form launches the three processes concurrently and waits
for all of them. Each model writes its own log and output directory under one
timestamped `outputs/nabladft-three-model-parallel-3x2gpu-eb20-mb5-ga2-.../`
group. After the completed lanes exit, their result rows are merged into the
group-level `comparison.csv`.

The `smoke` scope keeps the production channel dimensions (`--full-size-smoke`)
so it exercises the relevant memory path, while using only 20 train and 20
validation structures. It disables W&B and deletes the temporary run group
after all selected lanes pass. If any lane fails, the group and logs are kept
for diagnosis.

For the full split, each rank owns 6,040 training structures. Micro-batch 5
therefore gives 1,208 micro-batches and 604 optimizer updates per epoch after
accumulation, matching the previous batch-10 schedule while halving peak
activation memory per forward/backward pass.

QHFlow3 uses NablaDFT's `def2-svp-nabla` basis (32-AO padded atom blocks,
including S/Cl/Br) and a fully connected molecular pair graph. Its smoke path
is integrated into the existing 2-GPU script:

```bash
cd /dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-22
GPUS=6,7 MASTER_PORT=29572 \
  ./04_nabladft_2gpu_effective_batch20.sh data-parallel qhflow3 smoke
```

Replace the last argument with `full` for the 20-epoch, 12,081/64 run. Use
variant `all` to train MALOQ, MALOQ-NTE, and QHFlow3 sequentially. The same full
commands are also kept as copyable blocks `(c)` and `(d)` in
`03_nabladft_muon_20epoch_script.sh`.

QHFlow3 does not support the separate distributed-graph mode yet. Its two GPUs
therefore shard molecules data-parallel with batch size 10 per rank, for an
effective batch of 20.

The grid48 backbone passes the same general SO(3) rotation regression for both
the 14-AO QH9 basis and the 32-AO NablaDFT basis, including finite backward
gradients.

W&B receives rank-averaged `train_step/*` losses every 10 optimizer steps.
Validation metrics and complete train/validation summaries remain epoch-level;
their W&B step is the epoch's final optimizer step.

The 20-epoch command sheet uses two GPUs per experiment and `batch_size=10` per
rank. It keeps the official 12,081/64 train/validation boundary; molecule-level
sharding assigns 6,040 train molecules to each rank and ignores the single odd
training remainder. All 604 optimizer steps therefore have 10 local molecules
and global effective batch 20.

## Verified status (2026-07-22)

All four reduced-model smoke runs completed one optimizer step and one
validation step on GPUs 6 and 7 with Muon and effective batch 20:

| Model | Mode | Result | W&B run |
| --- | --- | --- | --- |
| MALOQ | data parallel | pass | [dnbiv03m](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/dnbiv03m) |
| MALOQ | distributed graph | pass | [mzdnxle0](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/mzdnxle0) |
| MALOQ-NTE | data parallel | pass | [yy5s73nv](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/yy5s73nv) |
| MALOQ-NTE | distributed graph | pass | [ij15ubpj](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/ij15ubpj) |
| QHFlow3 | data parallel | pass | [as7amzph](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/as7amzph) |

The 10-step cadence was separately exercised for 11 optimizer steps; W&B has a
rank-averaged train-step point at step 10 and the epoch summary at step 11 in
[run 374a3i1y](https://wandb.ai/kaist-korea/maloq-nablaDFT/runs/374a3i1y).

For distributed graphs, the 20-molecule training supergraph had 782 nodes and
23,648 edges. `linear-edgewise` assigned 11,844 and 11,804 edges to the two
ranks. Validation matrix-reconstruction metrics remain disabled in this mode;
distributed train/validation node, edge, and total losses are logged normally.
