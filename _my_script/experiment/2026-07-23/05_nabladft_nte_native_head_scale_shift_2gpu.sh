#!/usr/bin/env bash
set -euo pipefail

# Edit this default or pass a pair as the second argument, e.g. "full 2,3".
DEFAULT_GPUS=6,7
DEFAULT_MASTER_PORT=29611

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
EXPERIMENT_NAME=nabla-nte128e3-native-ss1-v1
MODEL_CONFIG=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-23/maloq_nte_do128_le3_native_head_scale_shift_nabladft.yaml
SCALE_SHIFT_ENABLED=1
COMPACT_NAMES=1

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export SCALE_SHIFT_ENABLED COMPACT_NAMES
exec /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh "$@"
