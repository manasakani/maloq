#!/usr/bin/env bash
set -euo pipefail

# QHFlow3-clean + corrected Muon head, with train-only l=0 label scale-shift.
# Edit this default or pass a pair as the second argument, e.g. "full 2,3".
DEFAULT_GPUS=0,1
DEFAULT_MASTER_PORT=29615

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
EXPERIMENT_NAME=nabla-qhf3-muon-ss1-v1
MODEL_CONFIG=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/qhflow3_local_muon_head_scale_shift_nabladft.yaml
MODEL_VARIANT=qhflow3
MODEL_HEAD_TYPE=maloq_muon
SCALE_SHIFT_ENABLED=1
COMPACT_NAMES=1

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export MODEL_VARIANT MODEL_HEAD_TYPE SCALE_SHIFT_ENABLED COMPACT_NAMES
exec /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh "$@"
