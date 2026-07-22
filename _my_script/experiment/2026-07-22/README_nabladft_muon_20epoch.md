# NablaDFT full 20-epoch Muon training

## Scope

- Dataset: native NablaDFT `train_2k.db` (12,145 rows)
- Official ordered split boundary: 12,081 train, 64 validation, 0 test
- Models: MALOQ, MALOQ-NTE, and QHFlow3
- Optimizer: Muon for matrix parameters plus AdamW for remaining parameters
- Muon settings: learning rate 0.02, momentum 0.95, Nesterov enabled,
  Newton–Schulz steps 5
- AdamW fallback learning rate: 0.0005
- Epochs: 20
- GPUs per run: 2
- Per-rank micro-batch size: 5
- Gradient accumulation steps: 2
- Effective global batch size: 20
- The molecule-level sharder uses 6,040 training rows per rank and ignores the
  one odd remainder row. This produces 1,208 micro-batches and 604 optimizer
  updates per epoch. Validation still begins at the official row 12,081.
- Seed: 44
- W&B: enabled in online mode
- W&B entity/project: `kaist-korea/maloq-nablaDFT`
- Outputs: timestamped directories below the repository `outputs/` tree

QHFlow3 uses the NablaDFT `def2-svp-nabla` bridge with 32-AO padding and a
fully connected pair graph. It supports the same 2-GPU molecule data-parallel
setup; its separate distributed-graph mode remains unsupported.

## Command

Inspect the command sheet, then copy exactly one complete block:

```bash
cd /dataset/seongsu/shared-home/workspace/project
sed -n '1,260p' \
  _my_script/experiment/2026-07-22/03_nabladft_muon_20epoch_script.sh
```

- `(a)`: MALOQ only
- `(b)`: MALOQ-NTE only
- `(c)`: QHFlow3 only
- `(d)`: matched MALOQ, MALOQ-NTE, then QHFlow3 comparison on one 2-GPU pair

Change only `GPUS` and `MASTER_PORT` if those resources are already occupied.
The command sheet is protected by a shell here-document, so executing the file
itself does not accidentally start the long production jobs.

## W&B outputs

Remote runs are recorded under:

`https://wandb.ai/kaist-korea/maloq-nablaDFT`

Each run name includes the timestamped output-root name plus its model variant,
so repeated experiments do not all appear with the same name.
The local W&B cache is stored below each model output directory as `wandb/`.
Training losses are additionally recorded every 10 optimizer steps; validation
and full epoch summaries remain epoch-level.
The matched `(d)` run uses one output root named
`nabladft-three-model-comparison-...`, with `maloq/`, `maloq-nte/`, and
`qhflow3/` subdirectories. On completion it writes `comparison.json` and
`comparison.csv` for direct metric comparison.

The machine must already be logged into a W&B account with write access to the
`kaist-korea` entity.

For the concurrent three-pair launch, use
`05_nabladft_three_models_3x2gpu_mb5_ga2.sh`. It uses the same micro-batch 5 and
gradient accumulation 2 settings as this command sheet.

## Verification and status

MALOQ and MALOQ-NTE passed 2-GPU train/validation smoke runs with effective
batch 20 in data-parallel and distributed-graph modes. QHFlow3 passed the same
2-GPU data-parallel smoke path.

Older production jobs launched with `--num-train 12080` are marked
`INVALID_SPLIT_DO_NOT_COMPARE.txt` and must not be used. The corrected command
sheet uses `--num-train 12081` and preserves the official validation boundary.
Interrupted configuration-only runs are marked `INCOMPLETE_DO_NOT_COMPARE.txt`.
No full 20-epoch production job was running when the three-model naming cleanup
was completed.

The canonical naming and result table were verified with a one-GPU reduced
smoke run at
`outputs/nabladft-three-model-comparison-naming-smoke-20260722-055212/`.
