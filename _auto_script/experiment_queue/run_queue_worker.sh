#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
QUEUE=${PROJECT_ROOT}/_auto_script/experiment_queue/sc26_queue.py
QUEUE_ROOT=${PROJECT_ROOT}/outputs/experiment-queue
HOST_LABEL=${SC26_QUEUE_HOST_LABEL:-$(hostname)}
WORKER_SLOT=${SC26_QUEUE_WORKER_SLOT:-0}
SESSION="sc26-experiment-queue-${HOST_LABEL}-s${WORKER_SLOT}"
WORKER_ROOT=${QUEUE_ROOT}/workers/${HOST_LABEL}/slot-${WORKER_SLOT}
LOG_FILE=${WORKER_ROOT}/worker.log

if [[ ! "${HOST_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "SC26_QUEUE_HOST_LABEL contains unsupported characters." >&2
  exit 2
fi
if [[ ! "${WORKER_SLOT}" =~ ^[0-3]$ ]]; then
  echo "SC26_QUEUE_WORKER_SLOT must be between 0 and 3." >&2
  exit 2
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {start|status|logs|run-once|doctor}" >&2
  exit 2
fi

mkdir -p "${WORKER_ROOT}"

case "$1" in
  start)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "Queue worker is already running: ${SESSION}"
      exit 0
    fi
    tmux new-session -d -s "${SESSION}" \
      "${PY} ${QUEUE} --queue-root ${QUEUE_ROOT} worker --host-label ${HOST_LABEL} >> ${LOG_FILE} 2>&1"
    sleep 1
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "Queue worker failed to stay running; inspect ${LOG_FILE}" >&2
      exit 1
    fi
    echo "Queue worker started: ${SESSION}"
    echo "Log: ${LOG_FILE}"
    ;;
  status)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "worker slot ${WORKER_SLOT}: running (${SESSION})"
    else
      echo "worker slot ${WORKER_SLOT}: stopped (${SESSION})"
    fi
    "${PY}" "${QUEUE}" --queue-root "${QUEUE_ROOT}" list
    ;;
  logs)
    if [[ ! -f "${LOG_FILE}" ]]; then
      echo "No worker log yet: ${LOG_FILE}"
      exit 0
    fi
    tail -n 100 "${LOG_FILE}"
    ;;
  run-once)
    exec "${PY}" "${QUEUE}" --queue-root "${QUEUE_ROOT}" \
      run-once --host-label "${HOST_LABEL}"
    ;;
  doctor)
    exec "${PY}" "${QUEUE}" --queue-root "${QUEUE_ROOT}" doctor
    ;;
  *)
    echo "Usage: $0 {start|status|logs|run-once|doctor}" >&2
    exit 2
    ;;
esac
