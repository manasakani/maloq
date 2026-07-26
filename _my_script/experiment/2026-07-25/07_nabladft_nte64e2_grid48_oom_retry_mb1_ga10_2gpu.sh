#!/usr/bin/env bash
set -euo pipefail

# Worst-case-safe retry of NTE-64/2 Grid48. Keep effective batch 20:
# micro-batch 1 x world size 2 x gradient accumulation 10.
DEFAULT_GPUS=4,5
DEFAULT_MASTER_PORT=29637

EXPERIMENT_NAME=nabla-nte64e2-muon-ss0-g48-mb1ga10-r2
MODEL_CONFIG=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/nte64e2_grid48_nabladft.yaml
MODEL_VARIANT=maloq-nte
MODEL_HEAD_TYPE=maloq_muon
SCALE_SHIFT_ENABLED=0
COMPACT_NAMES=1
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=10
WANDB_RUN_NAME_OVERRIDE="NablaDFT | NTE-64/2 | Muon | RAW | Grid48 | MB1 GA10 | R2"
WANDB_TAGS_CSV="retry:oom-r2,batch-size:1,grad-accum:10,effective-batch:20,supersedes:7gupb927"

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export MODEL_VARIANT MODEL_HEAD_TYPE SCALE_SHIFT_ENABLED COMPACT_NAMES
export MICRO_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export WANDB_RUN_NAME_OVERRIDE WANDB_TAGS_CSV
exec /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh "$@"
