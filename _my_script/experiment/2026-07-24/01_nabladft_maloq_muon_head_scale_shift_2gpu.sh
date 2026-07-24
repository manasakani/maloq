#!/usr/bin/env bash
set -euo pipefail

# MALOQ interleaved backbone + corrected Muon head with train-only l=0
# label scale/shift. Pass a GPU pair as the second argument when needed.
DEFAULT_GPUS=0,1
DEFAULT_MASTER_PORT=29616

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
EXPERIMENT_NAME=nabla-maloq-muon-ss1-v1
MODEL_CONFIG=${PROJECT_ROOT}/_my_script/experiment/2026-07-24/maloq_muon_head_scale_shift_nabladft.yaml
MODEL_VARIANT=maloq
MODEL_HEAD_TYPE=maloq_muon
SCALE_SHIFT_ENABLED=1
COMPACT_NAMES=1

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export MODEL_VARIANT MODEL_HEAD_TYPE SCALE_SHIFT_ENABLED COMPACT_NAMES
exec "${PROJECT_ROOT}/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh" "$@"
