#!/usr/bin/env bash
set -euo pipefail

# NTE-64/2 overlap-only input-conditioning ablation.
DEFAULT_GPUS=0,1
DEFAULT_MASTER_PORT=29633

EXPERIMENT_NAME=nabla-nte64e2-muon-ss0-scond-v1
MODEL_CONFIG=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-25/nte64e2_scond_nabladft.yaml
MODEL_VARIANT=maloq-nte
MODEL_HEAD_TYPE=maloq_muon
SCALE_SHIFT_ENABLED=0
COMPACT_NAMES=1

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export MODEL_VARIANT MODEL_HEAD_TYPE SCALE_SHIFT_ENABLED COMPACT_NAMES
exec /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh "$@"
