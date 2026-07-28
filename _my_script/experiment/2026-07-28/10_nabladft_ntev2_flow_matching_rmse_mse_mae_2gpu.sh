#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z "${1}" ]]; then
  echo "Usage: $0 GPU0,GPU1" >&2
  exit 2
fi

GPUS=$1
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project/MALOQ
EXPERIMENT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-28
FLOW_LAUNCHER=${EXPERIMENT_ROOT}/04_nabladft_flow_matching_e3_muon_shift_2gpu.sh
LANE=ntev2-e3-muon-shift
LOSS_PROFILE=rmse-mse-mae

if [[ ! -x "${FLOW_LAUNCHER}" ]]; then
  echo "FlowMatching launcher is missing or not executable: ${FLOW_LAUNCHER}" >&2
  exit 1
fi

"${FLOW_LAUNCHER}" smoke "${LANE}" "${GPUS}" "${LOSS_PROFILE}"
exec "${FLOW_LAUNCHER}" full "${LANE}" "${GPUS}" "${LOSS_PROFILE}"
