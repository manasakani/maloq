#!/usr/bin/env bash
set -euo pipefail

# OOM-safe retry of NTE-64/2 Grid48. Keep effective batch 20:
# micro-batch 2 x world size 2 x gradient accumulation 5.
DEFAULT_GPUS=4,5
DEFAULT_MASTER_PORT=29635

EXPERIMENT_NAME=nabla-nte64e2-muon-ss0-g48-mb2ga5-r1
MODEL_CONFIG=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/nte64e2_grid48_nabladft.yaml
MODEL_VARIANT=maloq-nte
MODEL_HEAD_TYPE=maloq_muon
SCALE_SHIFT_ENABLED=0
COMPACT_NAMES=1
MICRO_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=5
WANDB_RUN_NAME_OVERRIDE="NablaDFT | NTE-64/2 | Muon | RAW | Grid48 | MB2 GA5 | R1"
WANDB_TAGS_CSV="retry:oom-r1,batch-size:2,grad-accum:5,effective-batch:20,supersedes:c9yy08ci"

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export MODEL_VARIANT MODEL_HEAD_TYPE SCALE_SHIFT_ENABLED COMPACT_NAMES
export MICRO_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export WANDB_RUN_NAME_OVERRIDE WANDB_TAGS_CSV
exec /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh "$@"
