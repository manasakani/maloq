#!/usr/bin/env bash
set -euo pipefail

# QHFlow3 OV0 with NTE's lmax=4 default 10x11 SO(3) grid.
DEFAULT_GPUS=6,7
DEFAULT_MASTER_PORT=29636

EXPERIMENT_NAME=nabla-qhf3-muon-ss0-ov0-ntegrid-v2
MODEL_CONFIG=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/qhflow3_v2_no_overlap_nte_grid_nabladft.yaml
MODEL_VARIANT=qhflow3
MODEL_HEAD_TYPE=maloq_muon
SCALE_SHIFT_ENABLED=0
COMPACT_NAMES=1
WANDB_RUN_NAME_OVERRIDE="NablaDFT | QHFlow3 | Muon | RAW | OV0 | NTEGrid10x11 | V2"
WANDB_TAGS_CSV="grid:10x11,grid-policy:nte-default,ablation:qhflow3-grid,baseline:g0l50g72"

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export MODEL_VARIANT MODEL_HEAD_TYPE SCALE_SHIFT_ENABLED COMPACT_NAMES
export WANDB_RUN_NAME_OVERRIDE WANDB_TAGS_CSV
exec /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh "$@"
