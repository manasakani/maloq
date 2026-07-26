#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
LAUNCHER=${PROJECT_ROOT}/_my_script/experiment/2026-07-25/09_nabladft_shift_only_baselines_2gpu.sh
GPU_PAIR=${1:-}
shift || true

if [[ -z "${GPU_PAIR}" || $# -eq 0 ]]; then
  echo "Usage: $0 GPU0,GPU1 MODEL [MODEL ...]" >&2
  exit 2
fi

IFS=',' read -r GPU0 GPU1 EXTRA <<< "${GPU_PAIR}"
if [[ -z "${GPU0}" || -z "${GPU1}" || -n "${EXTRA:-}" ]] ||
  [[ ! "${GPU0}" =~ ^[0-9]+$ || ! "${GPU1}" =~ ^[0-9]+$ ]] ||
  [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU pair must contain two distinct integer indices." >&2
  exit 2
fi

HOST=$(hostname)
QUEUE_NAME="nabladft-shift-only-${HOST}-g${GPU0}-${GPU1}"
QUEUE_ROOT=${PROJECT_ROOT}/outputs/${QUEUE_NAME}
if [[ -e "${QUEUE_ROOT}" ]]; then
  echo "Queue output already exists: ${QUEUE_ROOT}" >&2
  exit 1
fi
mkdir -p "${QUEUE_ROOT}"

printf 'model\tstatus\texit_code\tupdated_at\n' > "${QUEUE_ROOT}/status.tsv"
for model in "$@"; do
  printf '%s\tqueued\t\t%s\n' "${model}" "$(date --iso-8601=seconds)" \
    >> "${QUEUE_ROOT}/status.tsv"
done

overall_status=0
for model in "$@"; do
  printf '%s\tstarting\t\t%s\n' "${model}" "$(date --iso-8601=seconds)" \
    >> "${QUEUE_ROOT}/status.tsv"
  set +e
  EXPECTED_HOST="${HOST}" "${LAUNCHER}" full "${model}" "${GPU_PAIR}" \
    2>&1 | tee "${QUEUE_ROOT}/${model}.log"
  model_status=${PIPESTATUS[0]}
  set -e
  if (( model_status == 0 )); then
    state=complete
  else
    state=failed
    overall_status=1
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "${model}" "${state}" "${model_status}" "$(date --iso-8601=seconds)" \
    >> "${QUEUE_ROOT}/status.tsv"
done

exit "${overall_status}"
