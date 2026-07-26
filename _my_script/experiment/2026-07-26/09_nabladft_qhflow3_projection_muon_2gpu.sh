#!/usr/bin/env bash
set -euo pipefail

# QHFlow3 80sa5m4j parity ablation: only the final 128->64 node/edge
# projections move from flat e3nn/AdamW to degree-batched Muon parameters.
DEFAULT_GPUS=6,7
DEFAULT_MASTER_PORT=29659

EXPERIMENT_NAME=nabla-qhf3-projmuon-ov0-ntegrid-v3
MODEL_CONFIG=/dataset/seongsu/shared-home/workspace/project/_my_script/experiment/2026-07-26/qhflow3_ov0_ntegrid_projection_muon_nabladft.yaml
MODEL_VARIANT=qhflow3
MODEL_HEAD_TYPE=maloq_muon
SCALE_SHIFT_ENABLED=0
COMPACT_NAMES=1
WANDB_RUN_NAME_OVERRIDE="NablaDFT | QHFlow3 | MatrixMuon+ProjMuon+AuxAdamW | RAW | OV0 | NTEGrid10x11 | V3"
WANDB_TAGS_CSV="output-projection-optimizer:muon,ablation:output-projection-routing,reference:80sa5m4j"

export DEFAULT_GPUS DEFAULT_MASTER_PORT EXPERIMENT_NAME MODEL_CONFIG
export MODEL_VARIANT MODEL_HEAD_TYPE SCALE_SHIFT_ENABLED COMPACT_NAMES
export WANDB_RUN_NAME_OVERRIDE WANDB_TAGS_CSV
exec /dataset/seongsu/shared-home/workspace/project/_auto_script/nabladft_scale_shift/run_single_nte_2gpu.sh "$@"
