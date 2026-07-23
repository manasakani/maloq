#!/usr/bin/env bash
set -euo pipefail

# Edit this default or pass a pair as the second argument, e.g. "full 2,3".
DEFAULT_GPUS=6,7
DEFAULT_MASTER_PORT=29611

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${SC26_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}
EXPERIMENT_NAME=nabladft-nte-do128-le3-native-head-scale-shift
MODEL_CONFIG=${SCRIPT_DIR}/maloq_nte_do128_le3_native_head_scale_shift_nabladft.yaml
SCALE_SHIFT_ENABLED=1

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export SCALE_SHIFT_ENABLED
exec "${PROJECT_ROOT}/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh" "$@"
