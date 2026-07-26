#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
STATE_ROOT=${PROJECT_ROOT}/outputs/omol-transfer-monitor
INTERVAL_SECONDS=${OMOL_MONITOR_INTERVAL_SECONDS:-300}
MAX_RESTARTS=${OMOL_MONITOR_MAX_RESTARTS:-20}

OPEN_SESSION=sc26-omol-open-shell-sync
OPEN_COMMAND=${PROJECT_ROOT}/_auto_script/omol_open_shell_transfer/sync_processed_open_shell.sh
OPEN_LOG=${PROJECT_ROOT}/outputs/omol-open-shell-sync/sync.log
OPEN_ROOT=/dataset/seongsu/shared-home/datasets/omol25_open_shell_maloq_ase

CSH_SESSION=sc26-omol-csh-download
CSH_COMMAND=${PROJECT_ROOT}/_auto_script/omol_csh_download/download_omol_csh.sh
CSH_LOG=${PROJECT_ROOT}/outputs/omol-csh-download/download.log
CSH_ROOT=/dataset/seongsu/shared-home/datasets/omol_csh

ELECTROLYTE_SESSION=sc26-omol-electrolyte-sync
ELECTROLYTE_COMMAND=${PROJECT_ROOT}/_auto_script/omol_electrolyte_transfer/sync_completed_electrolyte.sh
ELECTROLYTE_LOG=${PROJECT_ROOT}/outputs/omol-electrolyte-sync/sync.log
ELECTROLYTE_ROOT=/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb

mkdir -p "${STATE_ROOT}"

usage() {
  echo "Usage: $0 {monitor|status}" >&2
  exit 2
}

session_state() {
  local session=$1
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo running
  else
    echo stopped
  fi
}

json_verified() {
  local path=$1
  [[ -f "${path}" ]] || return 1
  /dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python - \
    "${path}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
raise SystemExit(0 if data.get("status") == "verified" else 1)
PY
}

count_open_shell_shards() {
  local split=$1
  find "${OPEN_ROOT}/${split}" -maxdepth 1 -type f -name '*.db' 2>/dev/null \
    | wc -l
}

open_shell_complete() {
  if json_verified "${STATE_ROOT}/open-shell-verification.json"; then
    return 0
  fi
  [[ "$(count_open_shell_shards train)" == 945 ]] || return 1
  [[ "$(count_open_shell_shards val)" == 9 ]] || return 1
  [[ "$(count_open_shell_shards test)" == 11 ]] || return 1
  if "${OPEN_COMMAND}" verify >"${STATE_ROOT}/open-shell-verify.log" 2>&1; then
    /dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python - \
      "${STATE_ROOT}/open-shell-verify.log" \
      "${STATE_ROOT}/open-shell-verification.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

text = Path(sys.argv[1]).read_text()
payload = {
    "status": "verified",
    "verified_at": datetime.now(timezone.utc).isoformat(),
    "verification_output": text,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    return 0
  fi
  return 1
}

csh_complete() {
  json_verified "${CSH_ROOT}/verification.json"
}

electrolyte_complete() {
  json_verified "${ELECTROLYTE_ROOT}/verification.json"
}

restart_job() {
  local label=$1
  local session=$2
  local command=$3
  local action=$4
  local log=$5
  local count_file=${STATE_ROOT}/${label}.restart-count
  local count=0

  if [[ -f "${count_file}" ]]; then
    read -r count <"${count_file}"
  fi
  if (( count >= MAX_RESTARTS )); then
    echo "$(date --iso-8601=seconds) ${label}: restart limit reached" \
      >>"${STATE_ROOT}/monitor.log"
    return 1
  fi
  count=$((count + 1))
  echo "${count}" >"${count_file}"
  tmux new-session -d -s "${session}" \
    "bash -lc '${command} ${action} 2>&1 | tee -a ${log}'"
  echo "$(date --iso-8601=seconds) ${label}: restarted attempt ${count}" \
    >>"${STATE_ROOT}/monitor.log"
}

write_status() {
  local open_done=$1
  local csh_done=$2
  local electrolyte_done=$3
  local temporary=${STATE_ROOT}/latest.txt.tmp
  {
    printf 'checked_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'open_shell_complete=%s session=%s shards=%s/%s/%s\n' \
      "${open_done}" "$(session_state "${OPEN_SESSION}")" \
      "$(count_open_shell_shards train)" \
      "$(count_open_shell_shards val)" \
      "$(count_open_shell_shards test)"
    printf 'csh_complete=%s session=%s\n' \
      "${csh_done}" "$(session_state "${CSH_SESSION}")"
    "${CSH_COMMAND}" status
    printf 'electrolyte_complete=%s session=%s shards=%s/%s/%s\n' \
      "${electrolyte_done}" "$(session_state "${ELECTROLYTE_SESSION}")" \
      "$(find "${ELECTROLYTE_ROOT}/train" -mindepth 2 -maxdepth 2 -type f -name data.mdb 2>/dev/null | wc -l)" \
      "$(find "${ELECTROLYTE_ROOT}/val" -mindepth 2 -maxdepth 2 -type f -name data.mdb 2>/dev/null | wc -l)" \
      "$(find "${ELECTROLYTE_ROOT}/test" -mindepth 2 -maxdepth 2 -type f -name data.mdb 2>/dev/null | wc -l)"
    df -P -B1 /dataset | tail -1
  } >"${temporary}"
  mv "${temporary}" "${STATE_ROOT}/latest.txt"
  cat "${STATE_ROOT}/latest.txt" >>"${STATE_ROOT}/history.log"
}

monitor() {
  exec 9>"${STATE_ROOT}/monitor.lock"
  if ! flock -n 9; then
    echo "OMol transfer monitor is already running" >&2
    exit 1
  fi

  while true; do
    local open_done=false
    local csh_done=false
    local electrolyte_done=false

    if open_shell_complete; then open_done=true; fi
    if csh_complete; then csh_done=true; fi
    if electrolyte_complete; then electrolyte_done=true; fi

    write_status "${open_done}" "${csh_done}" "${electrolyte_done}"
    if [[ "${open_done}" == true && "${csh_done}" == true && \
          "${electrolyte_done}" == true ]]; then
      cp "${STATE_ROOT}/latest.txt" "${STATE_ROOT}/COMPLETE"
      echo "$(date --iso-8601=seconds) all OMol transfers verified" \
        >>"${STATE_ROOT}/monitor.log"
      return
    fi

    if [[ "${open_done}" == false ]] && \
       ! tmux has-session -t "${OPEN_SESSION}" 2>/dev/null; then
      restart_job open-shell "${OPEN_SESSION}" "${OPEN_COMMAND}" sync "${OPEN_LOG}" \
        || true
    fi
    if [[ "${csh_done}" == false ]] && \
       ! tmux has-session -t "${CSH_SESSION}" 2>/dev/null; then
      restart_job csh "${CSH_SESSION}" "${CSH_COMMAND}" download "${CSH_LOG}" \
        || true
    fi
    if [[ "${electrolyte_done}" == false ]] && \
       ! tmux has-session -t "${ELECTROLYTE_SESSION}" 2>/dev/null; then
      restart_job electrolyte "${ELECTROLYTE_SESSION}" \
        "${ELECTROLYTE_COMMAND}" sync "${ELECTROLYTE_LOG}" || true
    fi
    sleep "${INTERVAL_SECONDS}"
  done
}

[[ $# -eq 1 ]] || usage
case "$1" in
  monitor) monitor ;;
  status)
    if [[ -f "${STATE_ROOT}/latest.txt" ]]; then
      cat "${STATE_ROOT}/latest.txt"
    else
      echo "No monitor snapshot exists yet." >&2
      exit 1
    fi
    ;;
  *) usage ;;
esac
