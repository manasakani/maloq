#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {validate|smoke|full} [zero-channel-mean-layerscale-s64|degreewise-l34-gate|all]" >&2
  exit 2
fi

SCOPE=$1
SELECTION=${2:-all}
case "${SCOPE}" in validate|smoke|full) ;; *) echo "Invalid scope: ${SCOPE}" >&2; exit 2 ;; esac
case "${SELECTION}" in
  zero-channel-mean-layerscale-s64|degreewise-l34-gate) PRESETS=("${SELECTION}") ;;
  all) PRESETS=(zero-channel-mean-layerscale-s64 degreewise-l34-gate) ;;
  *) echo "Invalid preset: ${SELECTION}" >&2; exit 2 ;;
esac

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nte_reference_tricks_qh9stable.py
ZERO_GPU=${ZERO_GPU:-0}
GATE_GPU=${GATE_GPU:-1}

gpu_for() {
  case "$1" in
    zero-channel-mean-layerscale-s64) printf '%s' "${ZERO_GPU}" ;;
    degreewise-l34-gate) printf '%s' "${GATE_GPU}" ;;
  esac
}

port_for() {
  case "$1" in
    zero-channel-mean-layerscale-s64) printf '29621' ;;
    degreewise-l34-gate) printf '29622' ;;
  esac
}

cd "${PROJECT_ROOT}"
pids=()
for preset in "${PRESETS[@]}"; do
  gpu=$(gpu_for "${preset}")
  port=$(port_for "${preset}")
  command=(
    "${PY}" "${RUNNER}"
    --preset "${preset}"
    --scope "${SCOPE}"
    --gpu "${gpu}"
    --master-port "${port}"
  )
  if [[ "${SCOPE}" == "validate" ]]; then
    PYTHONPATH=src "${command[@]}"
  else
    env \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      PYTHONPATH=src \
      "${command[@]}" &
    pids+=("$!")
    echo "Started ${preset} on GPU ${gpu} (PID ${pids[-1]})."
  fi
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
