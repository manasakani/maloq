#!/usr/bin/env bash
set -euo pipefail

SCOPE=${1:-validate}
EXPECTED_HOST=${EXPECTED_HOST:-}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project/MALOQ
SCRIPT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-28
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
TORCHRUN=${ENV_ROOT}/bin/torchrun
RUNNER=${SCRIPT_ROOT}/run_qhflow2_endpoint_flow_nabladft.py
CONFIG=${SCRIPT_ROOT}/qhflow2_endpoint_flow_nabladft.yaml
RUN_PREFIX=full-matrix-endpoint-flow-qhflow3-e2-muon-raw-v2

usage() {
  echo "Usage: $0 {prepare|validate|smoke}" >&2
}

case "${SCOPE}" in
  prepare|validate|smoke) ;;
  *)
    usage
    exit 2
    ;;
esac

for required in "${PY}" "${TORCHRUN}" "${RUNNER}" "${CONFIG}"; do
  if [[ ! -r "${required}" ]]; then
    echo "Required endpoint-flow input is missing: ${required}" >&2
    exit 1
  fi
done
if [[ ! -x "${PY}" || ! -x "${TORCHRUN}" || ! -x "${RUNNER}" ]]; then
  echo "Python, torchrun, and the endpoint-flow runner must be executable." >&2
  exit 1
fi
if [[ -n "${EXPECTED_HOST}" && "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "Expected host ${EXPECTED_HOST}, got $(hostname)." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ "${SCOPE}" == "prepare" || "${SCOPE}" == "validate" ]]; then
  exec env \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${PY}" "${RUNNER}" --config "${CONFIG}" --scope validate
fi

if [[ "${CUDA_DEVICES}" != *,* || "${CUDA_DEVICES}" == *,*,* ]]; then
  echo "Smoke requires exactly two comma-separated CUDA devices; got ${CUDA_DEVICES}." >&2
  exit 1
fi

RUN_ID=$(date +%Y%m%d-%H%M%S)-$$
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/${RUN_PREFIX}-smoke-${RUN_ID}
LOG_PATH=${OUTPUT_ROOT}.log
if [[ -e "${OUTPUT_ROOT}" || -e "${LOG_PATH}" ]]; then
  echo "Smoke output or log already exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi

env \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  PYTHONPATH="${PROJECT_ROOT}/src" \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${TORCHRUN}" --standalone --nproc-per-node=2 \
    "${RUNNER}" \
      --config "${CONFIG}" \
      --scope smoke \
      --output-root "${OUTPUT_ROOT}" \
    >"${LOG_PATH}" 2>&1

echo "Endpoint-flow smoke completed: ${OUTPUT_ROOT}"
echo "Log: ${LOG_PATH}"
