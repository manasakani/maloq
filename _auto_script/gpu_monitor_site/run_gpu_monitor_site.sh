#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-status}
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PYTHON=/usr/bin/python3
TMUX=/usr/bin/tmux
CURL=/usr/bin/curl
SERVER=/dataset/seongsu/shared-home/workspace/project/_auto_script/gpu_monitor_site/server.py
OUTPUT_DIR=/dataset/seongsu/shared-home/workspace/project/outputs/gpu-monitor-site
LOG_FILE=/dataset/seongsu/shared-home/workspace/project/outputs/gpu-monitor-site/server.log
HISTORY_DB=/dataset/seongsu/shared-home/workspace/project/outputs/gpu-monitor-site/history.sqlite3
SESSION=sc26-gpu-monitor-site
BIND=127.0.0.1
PORT=8787
URL=http://${BIND}:${PORT}

start_monitor() {
  mkdir -p "${OUTPUT_DIR}"
  if "${TMUX}" has-session -t "${SESSION}" 2>/dev/null; then
    echo "SC26 GPU monitor is already running."
    status_monitor
    return
  fi

  command_line="${PYTHON} ${SERVER} --bind ${BIND} --port ${PORT} --refresh-seconds 5 --history-seconds 60 --history-db ${HISTORY_DB} --log-file ${LOG_FILE}"
  "${TMUX}" new-session -d -s "${SESSION}" "${command_line}"

  for _ in $(seq 1 20); do
    if "${CURL}" --fail --silent --max-time 2 "${URL}/healthz" >/dev/null; then
      echo "SC26 GPU monitor started."
      status_monitor
      return
    fi
    sleep 0.25
  done

  echo "Monitor did not become healthy. Recent log:" >&2
  tail -n 30 "${LOG_FILE}" >&2 || true
  exit 1
}

stop_monitor() {
  if ! "${TMUX}" has-session -t "${SESSION}" 2>/dev/null; then
    echo "SC26 GPU monitor is not running."
    return
  fi
  "${TMUX}" kill-session -t "${SESSION}"
  echo "SC26 GPU monitor stopped. Logs remain at ${LOG_FILE}"
}

status_monitor() {
  if "${TMUX}" has-session -t "${SESSION}" 2>/dev/null; then
    echo "status: running"
    echo "tmux:   ${SESSION}"
    echo "local:  ${URL}"
    echo "tunnel: ssh -N -L ${PORT}:${BIND}:${PORT} scp-gpu-1"
    echo "open:   http://127.0.0.1:${PORT}"
    echo "log:    ${LOG_FILE}"
    echo "history: ${HISTORY_DB}"
    "${CURL}" --fail --silent --max-time 2 "${URL}/healthz" || true
    echo
  else
    echo "status: stopped"
    echo "log:    ${LOG_FILE}"
    return 1
  fi
}

case "${ACTION}" in
  start)
    start_monitor
    ;;
  stop)
    stop_monitor
    ;;
  restart)
    stop_monitor
    start_monitor
    ;;
  status)
    status_monitor
    ;;
  logs)
    if [[ -f "${LOG_FILE}" ]]; then
      tail -n 80 "${LOG_FILE}"
    else
      echo "No log file exists yet: ${LOG_FILE}"
    fi
    ;;
  foreground)
    mkdir -p "${OUTPUT_DIR}"
    exec "${PYTHON}" "${SERVER}" \
      --bind "${BIND}" \
      --port "${PORT}" \
      --refresh-seconds 5 \
      --history-seconds 60 \
      --history-db "${HISTORY_DB}" \
      --log-file "${LOG_FILE}"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|foreground}" >&2
    exit 2
    ;;
esac
