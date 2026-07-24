#!/usr/bin/env bash
set -euo pipefail

# QHFlow3 V2 overlap ablation: every baseline setting is preserved while
# qhflow3_use_overlap=false prevents overlap extraction and conditioning.
DEFAULT_GPUS=0,1
DEFAULT_MASTER_PORT=29631

EXPERIMENT_NAME=nabla-qhf3-muon-ss0-ov0-v2
MODEL_CONFIG=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/qhflow3_v2_no_overlap_nabladft.yaml
MODEL_VARIANT=qhflow3
MODEL_HEAD_TYPE=maloq_muon
SCALE_SHIFT_ENABLED=0
COMPACT_NAMES=1

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export MODEL_VARIANT MODEL_HEAD_TYPE SCALE_SHIFT_ENABLED COMPACT_NAMES
exec /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh "$@"
