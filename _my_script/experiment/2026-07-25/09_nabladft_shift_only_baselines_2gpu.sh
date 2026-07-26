#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
COMMON_LAUNCHER=${PROJECT_ROOT}/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh
EXPERIMENT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-25

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {prepare|validate|smoke|full} {qhflow3|nte128|nte64|maloq} [GPU0,GPU1]" >&2
  exit 2
fi

SCOPE=$1
MODEL=$2
GPU_OVERRIDE=${3:-}

case "${MODEL}" in
  qhflow3)
    EXPERIMENT_NAME=nabla-qhf3-muon-shift-only-v2
    MODEL_CONFIG=${EXPERIMENT_ROOT}/qhflow3_muon_shift_only_nabladft.yaml
    MODEL_VARIANT=qhflow3
    MODEL_HEAD_TYPE=maloq_muon
    DEFAULT_GPUS=2,3
    DEFAULT_MASTER_PORT=29720
    WANDB_RUN_NAME_OVERRIDE="NablaDFT | QHFlow3 | Muon | SHIFT | V2"
    PREVIOUS_RAW_RUN=zqs1eohc
    ;;
  nte128)
    EXPERIMENT_NAME=nabla-nte128e3-muon-shift-only-v1
    MODEL_CONFIG=${EXPERIMENT_ROOT}/nte128e3_muon_shift_only_nabladft.yaml
    MODEL_VARIANT=maloq-nte
    MODEL_HEAD_TYPE=maloq_muon
    DEFAULT_GPUS=4,5
    DEFAULT_MASTER_PORT=29721
    WANDB_RUN_NAME_OVERRIDE="NablaDFT | NTE-128/3 | Muon | SHIFT | V1"
    PREVIOUS_RAW_RUN=119izc66
    ;;
  nte64)
    EXPERIMENT_NAME=nabla-nte64e2-muon-shift-only-v1
    MODEL_CONFIG=${EXPERIMENT_ROOT}/nte64e2_muon_shift_only_nabladft.yaml
    MODEL_VARIANT=maloq-nte
    MODEL_HEAD_TYPE=maloq_muon
    DEFAULT_GPUS=6,7
    DEFAULT_MASTER_PORT=29722
    WANDB_RUN_NAME_OVERRIDE="NablaDFT | NTE-64/2 | Muon | SHIFT | V1"
    PREVIOUS_RAW_RUN=loaiifgp
    ;;
  maloq)
    EXPERIMENT_NAME=nabla-maloq-muon-shift-only-v1
    MODEL_CONFIG=${EXPERIMENT_ROOT}/maloq_muon_shift_only_nabladft.yaml
    MODEL_VARIANT=maloq
    MODEL_HEAD_TYPE=maloq_muon
    DEFAULT_GPUS=6,7
    DEFAULT_MASTER_PORT=29723
    WANDB_RUN_NAME_OVERRIDE="NablaDFT | MALOQ | Muon | SHIFT | V1"
    PREVIOUS_RAW_RUN=9ldrunh9
    ;;
  *)
    echo "Model must be qhflow3, nte128, nte64, or maloq." >&2
    exit 2
    ;;
esac

SCALE_SHIFT_ENABLED=1
COMPACT_NAMES=1
WANDB_TAGS_CSV="normalization:l0-shift-only,target:mean-centered,baseline:original-maloq-shift,previous-raw-run:${PREVIOUS_RAW_RUN}"

export EXPERIMENT_NAME MODEL_CONFIG MODEL_VARIANT MODEL_HEAD_TYPE
export DEFAULT_GPUS DEFAULT_MASTER_PORT SCALE_SHIFT_ENABLED COMPACT_NAMES
export WANDB_RUN_NAME_OVERRIDE WANDB_TAGS_CSV

if [[ -n "${GPU_OVERRIDE}" ]]; then
  exec "${COMMON_LAUNCHER}" "${SCOPE}" "${GPU_OVERRIDE}"
fi
exec "${COMMON_LAUNCHER}" "${SCOPE}"
