#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")

PROJECT=/dataset/seongsu/shared-home/workspace/project
TOOL_ROOT=${PROJECT}/_auto_script/omol_electrolyte_process
PROCESSOR=${TOOL_ROOT}/processor/process_omol_electrolyte_shards.py
VERIFY_ROOT=${TOOL_ROOT}/verification
PYTHON=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
V1_ROOT=/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb
MANIFEST=${V1_ROOT}/manifests/unsolvated_electrolytes_all_supported_elements_85_5_10_v1
RAW_ROOT=/dataset/seongsu/shared-home/datasets/omol25_electrolyte_raw_selected_v2
OVERLAY=/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_v2_overlay
VIEW=/dataset/seongsu/shared-home/datasets/omol25_electrolyte_maloq_lmdb_corrected_full_v2
GLOBAL_STATE=/dataset/seongsu/shared-home/workspace/_global_auto_script_output/omol_electrolyte_preprocess
WORKER_ROOT=${GLOBAL_STATE}/v2_selection/worker_lists

ACTIVE_KIND=
ACTIVE_NAME=
ACTIVE_STATE=
ACTIVE_ATTEMPT=
ACTIVE_ATTEMPT_STATE=
ACTIVE_STAGE=

usage() {
  echo "Usage: $0 {status|start-be|start-missing|start-finalize|_run-phase PHASE|_finalize}" >&2
  exit 2
}

phase_config() {
  local phase=$1
  case "${phase}" in
    be16)
      PHASE_WORKERS=16
      PHASE_LIST_DIR=${WORKER_ROOT}/be16
      PHASE_TRANSFER_MARKER=${GLOBAL_STATE}/raw_sync/be.complete
      PHASE_SESSION=sc26-omol-electrolyte-v2-be
      ;;
    missing24)
      PHASE_WORKERS=24
      PHASE_LIST_DIR=${WORKER_ROOT}/missing24
      PHASE_TRANSFER_MARKER=${GLOBAL_STATE}/raw_sync/missing.complete
      PHASE_SESSION=sc26-omol-electrolyte-v2-missing
      ;;
    *)
      echo "Unknown phase: ${phase}" >&2
      exit 2
      ;;
  esac
  PHASE_STATE=${OVERLAY}/_pipeline/${phase}
}

require_file() {
  local path=$1
  if [[ ! -f "${path}" ]]; then
    echo "Required file is missing: ${path}" >&2
    exit 1
  fi
}

atomic_write_marker() {
  local path=$1
  shift
  local temporary="${path}.tmp-p$$-${RANDOM}"
  mkdir -p "$(dirname "${path}")"
  printf '%s\n' "$@" >"${temporary}"
  mv -f -- "${temporary}" "${path}"
}

activate_attempt() {
  ACTIVE_KIND=$1
  ACTIVE_NAME=$2
  ACTIVE_STATE=$3
  ACTIVE_ATTEMPT=$4
  ACTIVE_ATTEMPT_STATE=$5
  ACTIVE_STAGE=preflight

  trap state_exit_trap EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP

  mkdir -p "${ACTIVE_ATTEMPT_STATE}"
  rm -f "${ACTIVE_STATE}/FAILED"
  atomic_write_marker "${ACTIVE_STATE}/CURRENT_ATTEMPT" "${ACTIVE_ATTEMPT}"
  atomic_write_marker \
    "${ACTIVE_STATE}/RUNNING" \
    "kind=${ACTIVE_KIND}" \
    "name=${ACTIVE_NAME}" \
    "attempt=${ACTIVE_ATTEMPT}" \
    "pid=$$" \
    "host=$(hostname)" \
    "started_at=$(date -Is)"
  atomic_write_marker \
    "${ACTIVE_ATTEMPT_STATE}/RUNNING" \
    "kind=${ACTIVE_KIND}" \
    "name=${ACTIVE_NAME}" \
    "attempt=${ACTIVE_ATTEMPT}" \
    "pid=$$" \
    "host=$(hostname)" \
    "started_at=$(date -Is)"

}

state_exit_trap() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  if [[ -n "${ACTIVE_STATE}" ]]; then
    if ((rc == 0)); then
      rc=1
    fi
    rm -f \
      "${ACTIVE_STATE}/RUNNING" \
      "${ACTIVE_ATTEMPT_STATE}/RUNNING"
    atomic_write_marker \
      "${ACTIVE_STATE}/FAILED" \
      "kind=${ACTIVE_KIND}" \
      "name=${ACTIVE_NAME}" \
      "attempt=${ACTIVE_ATTEMPT}" \
      "stage=${ACTIVE_STAGE}" \
      "rc=${rc}" \
      "pid=$$" \
      "host=$(hostname)" \
      "failed_at=$(date -Is)"
    atomic_write_marker \
      "${ACTIVE_ATTEMPT_STATE}/FAILED" \
      "kind=${ACTIVE_KIND}" \
      "name=${ACTIVE_NAME}" \
      "attempt=${ACTIVE_ATTEMPT}" \
      "stage=${ACTIVE_STAGE}" \
      "rc=${rc}" \
      "pid=$$" \
      "host=$(hostname)" \
      "failed_at=$(date -Is)"
  fi
  exit "${rc}"
}

complete_attempt() {
  local completed_at
  completed_at=$(date -Is)
  atomic_write_marker \
    "${ACTIVE_ATTEMPT_STATE}/COMPLETE" \
    "kind=${ACTIVE_KIND}" \
    "name=${ACTIVE_NAME}" \
    "attempt=${ACTIVE_ATTEMPT}" \
    "completed_at=${completed_at}"
  atomic_write_marker \
    "${ACTIVE_STATE}/COMPLETE" \
    "kind=${ACTIVE_KIND}" \
    "name=${ACTIVE_NAME}" \
    "attempt=${ACTIVE_ATTEMPT}" \
    "completed_at=${completed_at}"
  rm -f \
    "${ACTIVE_STATE}/FAILED" \
    "${ACTIVE_STATE}/RUNNING" \
    "${ACTIVE_ATTEMPT_STATE}/FAILED" \
    "${ACTIVE_ATTEMPT_STATE}/RUNNING"
  ACTIVE_KIND=
  ACTIVE_NAME=
  ACTIVE_STATE=
  ACTIVE_ATTEMPT=
  ACTIVE_ATTEMPT_STATE=
  ACTIVE_STAGE=
  trap - EXIT INT TERM HUP
}

tmux_start_command() {
  local session=$1
  shift
  local command argument quoted
  printf -v command 'exec /bin/bash %q' "${SCRIPT_PATH}"
  for argument in "$@"; do
    printf -v quoted '%q' "${argument}"
    command+=" ${quoted}"
  done
  tmux new-session -d -s "${session}" -c "${PROJECT}" "${command}"
}

phase_list_path() {
  local worker=$1
  printf '%s/worker_%02d_of_%02d.txt\n' \
    "${PHASE_LIST_DIR}" "${worker}" "${PHASE_WORKERS}"
}

preflight_phase_files() {
  require_file "${PROCESSOR}"
  require_file "${PHASE_TRANSFER_MARKER}"
  local worker
  for ((worker = 0; worker < PHASE_WORKERS; worker++)); do
    require_file "$(phase_list_path "${worker}")"
  done
}

validate_expected_index() {
  local index_root=$1
  "${PYTHON}" -B - \
    "${index_root}" "${MANIFEST}" "${V1_ROOT}" "${OVERLAY}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"invalid resumable full-v2 index: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).absolute()
manifest = Path(sys.argv[2]).resolve()
v1_root = Path(sys.argv[3]).resolve()
v2_root = Path(sys.argv[4]).absolute()
summary_path = root / "summary.json"
if root.is_symlink() or not root.is_dir() or not summary_path.is_file():
    fail(f"missing atomically published index root/summary: {root}")
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception as exc:
    fail(f"unreadable summary: {type(exc).__name__}: {exc}")
expected = {
    "schema": "omol_electrolyte_full_v2_index_v1",
    "status": "complete",
    "manifest_dir": str(manifest),
    "v1_root": str(v1_root),
    "v2_root": str(v2_root),
    "missing_count": 0,
}
for key, value in expected.items():
    if summary.get(key) != value:
        fail(f"{key}={summary.get(key)!r}, expected {value!r}")
if int(summary.get("indexed_total", -1)) != int(summary.get("expected_total", -2)):
    fail("indexed_total does not equal expected_total")

manifest_hashes = {}
for name in ("train.jsonl", "val.jsonl", "test.jsonl", "summary.json"):
    candidate = manifest / name
    if candidate.is_file():
        manifest_hashes[name] = sha256(candidate)
if summary.get("manifest_sha256") != manifest_hashes:
    fail("manifest hashes no longer match")

index_hashes = summary.get("index_sha256")
if not isinstance(index_hashes, dict):
    fail("index_sha256 is not a dictionary")
for split in ("train", "val", "test"):
    name = f"{split}.index.jsonl"
    candidate = root / name
    if not candidate.is_file():
        fail(f"missing {name}")
    if index_hashes.get(name) != sha256(candidate):
        fail(f"hash mismatch for {name}")
PY
}

validate_expected_view() {
  local index_root=$1
  local view_root=$2
  "${PYTHON}" -B - \
    "${index_root}" "${view_root}" "${MANIFEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"invalid resumable full-v2 view: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


index_root = Path(sys.argv[1]).resolve()
view_root = Path(sys.argv[2]).absolute()
manifest = Path(sys.argv[3]).resolve()
summary_path = view_root / "summary.json"
view_index_summary_path = view_root / "_index" / "summary.json"
if view_root.is_symlink() or not view_root.is_dir():
    fail(f"view root is absent, not a directory, or a symlink: {view_root}")
if not summary_path.is_file() or not view_index_summary_path.is_file():
    fail("view summary or view index summary is missing")
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    view_index_summary = json.loads(
        view_index_summary_path.read_text(encoding="utf-8")
    )
except Exception as exc:
    fail(f"unreadable summary: {type(exc).__name__}: {exc}")
expected = {
    "schema": "omol_electrolyte_full_v2_symlink_view_v1",
    "status": "view_built_not_fully_verified",
    "index_root": str(index_root),
    "manifest_dir": str(manifest),
    "out_view_root": str(view_root),
    "symlink_only": True,
    "source_data_modified": False,
}
for key, value in expected.items():
    if summary.get(key) != value:
        fail(f"{key}={summary.get(key)!r}, expected {value!r}")
source_index_summary = index_root / "summary.json"
if summary.get("index_summary_sha256") != sha256(source_index_summary):
    fail("source index summary hash mismatch")
if view_index_summary.get("schema") != "omol_electrolyte_full_v2_view_index_v1":
    fail("unexpected view index summary schema")
if view_index_summary.get("source_index_root") != str(index_root):
    fail("view index source root mismatch")
if view_index_summary.get("view_root") != str(view_root):
    fail("view index root mismatch")

view_hashes = summary.get("view_index_sha256")
if not isinstance(view_hashes, dict):
    fail("view_index_sha256 is not a dictionary")
for split in ("train", "val", "test"):
    name = f"{split}.index.jsonl"
    candidate = view_root / "_index" / name
    top_link = view_root / name
    if not candidate.is_file() or not top_link.is_symlink():
        fail(f"missing view index or top-level symlink for {split}")
    if view_hashes.get(name) != sha256(candidate):
        fail(f"view index hash mismatch for {name}")

complete_path = view_root / "COMPLETE"
if complete_path.is_file():
    verification_path = view_root / "verification.json"
    if not verification_path.is_file():
        fail("COMPLETE exists without verification.json")
    try:
        verification = json.loads(
            verification_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        fail(f"unreadable verification.json: {type(exc).__name__}: {exc}")
    expected_verification = {
        "schema": "omol_electrolyte_full_v2_verification_v1",
        "status": "verified",
        "mode": "full",
        "manifest_dir": str(manifest),
        "index_root": str((view_root / "_index").resolve()),
        "view_root": str(view_root),
        "all_shards_checked": True,
        "all_records_checked": True,
    }
    for key, value in expected_verification.items():
        if verification.get(key) != value:
            fail(
                f"verification {key}={verification.get(key)!r}, "
                f"expected {value!r}"
            )
    if verification.get("manifest_sha256") != {
        name: sha256(manifest / name)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "summary.json")
        if (manifest / name).is_file()
    }:
        fail("verification manifest hashes no longer match")
    if verification.get("index_sha256") != {
        f"{split}.index.jsonl": sha256(
            view_root / "_index" / f"{split}.index.jsonl"
        )
        for split in ("train", "val", "test")
    }:
        fail("verification index hashes no longer match")
    limits = verification.get("limits", {})
    if float(limits.get("density_trace_error", -1.0)) != 0.05:
        fail("verification density trace limit is not 0.05")
    if float(limits.get("initial_density_trace_error", -1.0)) != 0.001:
        fail("verification initial-density trace limit is not 0.001")
PY
}

print_state() {
  local name=$1
  local state=$2
  local complete=$3
  if [[ -f "${complete}" ]]; then
    printf 'complete\t%s\t%s\n' "${name}" "$(head -1 "${complete}")"
  elif [[ -f "${state}/FAILED" ]]; then
    printf 'failed\t%s\t%s\n' "${name}" "$(head -1 "${state}/FAILED")"
  elif [[ -f "${state}/RUNNING" ]]; then
    printf 'running\t%s\t%s\n' "${name}" "$(head -1 "${state}/RUNNING")"
  else
    printf 'pending\t%s\n' "${name}"
  fi
}

start_phase() {
  local phase=$1
  phase_config "${phase}"
  if tmux has-session -t "${PHASE_SESSION}" 2>/dev/null; then
    echo "Session already exists: ${PHASE_SESSION}" >&2
    exit 1
  fi
  if [[ -f "${PHASE_STATE}/COMPLETE" ]]; then
    echo "Phase is already complete: ${phase}" >&2
    exit 1
  fi
  preflight_phase_files
  mkdir -p "${PHASE_STATE}"
  tmux_start_command "${PHASE_SESSION}" _run-phase "${phase}"
  echo "Started ${phase} in tmux session ${PHASE_SESSION}"
}

run_phase() {
  local phase=$1
  phase_config "${phase}"
  mkdir -p "${PHASE_STATE}"

  local phase_lock_fd
  exec {phase_lock_fd}>"${PHASE_STATE}/LOCK"
  if ! flock -n "${phase_lock_fd}"; then
    echo "Phase is locked by another orchestrator: ${phase}" >&2
    exit 75
  fi
  if [[ -f "${PHASE_STATE}/COMPLETE" ]]; then
    echo "Phase is already complete: ${phase}" >&2
    exit 1
  fi

  local attempt attempt_state
  attempt=$(date -u +%Y%m%dT%H%M%SZ)-p$$
  attempt_state="${PHASE_STATE}/attempts/${attempt}"
  mkdir -p "${attempt_state}/logs" "${attempt_state}/exit"
  activate_attempt phase "${phase}" "${PHASE_STATE}" "${attempt}" "${attempt_state}"

  ACTIVE_STAGE=preflight
  preflight_phase_files

  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1

  local -a pids=()
  local -a labels=()
  local worker list label
  ACTIVE_STAGE=launch-workers
  for ((worker = 0; worker < PHASE_WORKERS; worker++)); do
    label=$(printf '%02d' "${worker}")
    list=$(phase_list_path "${worker}")
    "${PYTHON}" "${PROCESSOR}" run \
      --manifest-dir "${MANIFEST}" \
      --out "${OVERLAY}" \
      --density-root "${RAW_ROOT}/data/omol25/electronic" \
      --parquet-root "${RAW_ROOT}/datasets/omol25_train_4M" \
      --shard-list "${list}" \
      --require-density-dtype float32 \
      --max-initial-trace-error 0.001 \
      --lmdb-map-size-gb 8 \
      --replace-invalid-existing \
      --run-id "${phase}-${attempt}-worker-${label}" \
      >"${attempt_state}/logs/worker-${label}.log" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
  done

  local failed=0
  local index rc
  ACTIVE_STAGE=wait-workers
  for index in "${!pids[@]}"; do
    rc=0
    wait "${pids[index]}" || rc=$?
    atomic_write_marker \
      "${attempt_state}/exit/worker-${labels[index]}.exit" "${rc}"
    if ((rc != 0)); then
      failed=1
    fi
  done

  if ((failed)); then
    echo "${phase} failed; inspect ${attempt_state}/logs" >&2
    exit 1
  fi
  ACTIVE_STAGE=complete
  complete_attempt
  echo "${phase} complete"
}

start_finalize() {
  local session=sc26-omol-electrolyte-v2-finalize
  local state=${OVERLAY}/_pipeline/finalize
  require_file "${OVERLAY}/_pipeline/be16/COMPLETE"
  require_file "${OVERLAY}/_pipeline/missing24/COMPLETE"
  require_file "${VERIFY_ROOT}/build_full_v2_index.py"
  require_file "${VERIFY_ROOT}/build_full_v2_view.py"
  require_file "${VERIFY_ROOT}/verify_full_v2.py"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2
    exit 1
  fi
  if [[ -f "${state}/COMPLETE" ]]; then
    echo "Finalize is already complete: ${state}" >&2
    exit 1
  fi
  mkdir -p "${state}"
  tmux_start_command "${session}" _finalize
  echo "Started resumable final index/view/full verification in ${session}"
}

finalize() {
  local state=${OVERLAY}/_pipeline/finalize
  local index_root=${OVERLAY}/_full_index
  mkdir -p "${state}"

  local finalize_lock_fd
  exec {finalize_lock_fd}>"${state}/LOCK"
  if ! flock -n "${finalize_lock_fd}"; then
    echo "Finalize is locked by another orchestrator" >&2
    exit 75
  fi
  if [[ -f "${state}/COMPLETE" ]]; then
    validate_expected_index "${index_root}"
    validate_expected_view "${index_root}" "${VIEW}"
    require_file "${VIEW}/COMPLETE"
    echo "Finalize is already complete"
    return 0
  fi

  local attempt attempt_state
  attempt=$(date -u +%Y%m%dT%H%M%SZ)-p$$
  attempt_state="${state}/attempts/${attempt}"
  mkdir -p \
    "${attempt_state}/logs" \
    "${attempt_state}/reports" \
    "${attempt_state}/steps"
  activate_attempt finalize full-v2 "${state}" "${attempt}" "${attempt_state}"

  ACTIVE_STAGE=preflight
  require_file "${OVERLAY}/_pipeline/be16/COMPLETE"
  require_file "${OVERLAY}/_pipeline/missing24/COMPLETE"
  require_file "${VERIFY_ROOT}/build_full_v2_index.py"
  require_file "${VERIFY_ROOT}/build_full_v2_view.py"
  require_file "${VERIFY_ROOT}/verify_full_v2.py"
  export PYTHONDONTWRITEBYTECODE=1

  if [[ -e "${index_root}" || -L "${index_root}" ]]; then
    ACTIVE_STAGE=validate-existing-index
    validate_expected_index "${index_root}" \
      >"${attempt_state}/logs/index-validation.log" 2>&1
    atomic_write_marker \
      "${attempt_state}/steps/index.COMPLETE" \
      "action=reused" \
      "completed_at=$(date -Is)"
  else
    ACTIVE_STAGE=build-index
    "${PYTHON}" "${VERIFY_ROOT}/build_full_v2_index.py" \
      --manifest-dir "${MANIFEST}" \
      --v1-root "${V1_ROOT}" \
      --v2-root "${OVERLAY}" \
      --out-index-root "${index_root}" \
      >"${attempt_state}/logs/index.log" 2>&1
    ACTIVE_STAGE=validate-new-index
    validate_expected_index "${index_root}" \
      >"${attempt_state}/logs/index-validation.log" 2>&1
    atomic_write_marker \
      "${attempt_state}/steps/index.COMPLETE" \
      "action=built" \
      "completed_at=$(date -Is)"
  fi

  if [[ -e "${VIEW}" || -L "${VIEW}" ]]; then
    ACTIVE_STAGE=validate-existing-view
    validate_expected_view "${index_root}" "${VIEW}" \
      >"${attempt_state}/logs/view-validation.log" 2>&1
    atomic_write_marker \
      "${attempt_state}/steps/view.COMPLETE" \
      "action=reused" \
      "completed_at=$(date -Is)"
  else
    ACTIVE_STAGE=build-view
    "${PYTHON}" "${VERIFY_ROOT}/build_full_v2_view.py" \
      --index-root "${index_root}" \
      --manifest-dir "${MANIFEST}" \
      --out-view-root "${VIEW}" \
      >"${attempt_state}/logs/view.log" 2>&1
    ACTIVE_STAGE=validate-new-view
    validate_expected_view "${index_root}" "${VIEW}" \
      >"${attempt_state}/logs/view-validation.log" 2>&1
    atomic_write_marker \
      "${attempt_state}/steps/view.COMPLETE" \
      "action=built" \
      "completed_at=$(date -Is)"
  fi

  if [[ -f "${VIEW}/COMPLETE" ]]; then
    ACTIVE_STAGE=repair-finalize-state
    atomic_write_marker \
      "${attempt_state}/steps/full-verification.COMPLETE" \
      "action=reused-existing-view-complete" \
      "completed_at=$(date -Is)"
    complete_attempt
    echo "Recovered finalize COMPLETE from verified view"
    return 0
  fi

  ACTIVE_STAGE=metadata-verification
  "${PYTHON}" "${VERIFY_ROOT}/verify_full_v2.py" \
    --manifest-dir "${MANIFEST}" \
    --index-root "${VIEW}/_index" \
    --v2-root "${OVERLAY}" \
    --view-root "${VIEW}" \
    --mode metadata \
    --report-path "${attempt_state}/reports/metadata-verification.json" \
    >"${attempt_state}/logs/metadata.log" 2>&1
  atomic_write_marker \
    "${attempt_state}/steps/metadata-verification.COMPLETE" \
    "completed_at=$(date -Is)"

  ACTIVE_STAGE=sampled-verification
  "${PYTHON}" "${VERIFY_ROOT}/verify_full_v2.py" \
    --manifest-dir "${MANIFEST}" \
    --index-root "${VIEW}/_index" \
    --v2-root "${OVERLAY}" \
    --view-root "${VIEW}" \
    --mode sampled \
    --records-per-shard 1 \
    --report-path "${attempt_state}/reports/sampled-verification.json" \
    >"${attempt_state}/logs/sampled.log" 2>&1
  atomic_write_marker \
    "${attempt_state}/steps/sampled-verification.COMPLETE" \
    "completed_at=$(date -Is)"

  ACTIVE_STAGE=full-verification
  "${PYTHON}" "${VERIFY_ROOT}/verify_full_v2.py" \
    --manifest-dir "${MANIFEST}" \
    --index-root "${VIEW}/_index" \
    --v2-root "${OVERLAY}" \
    --view-root "${VIEW}" \
    --mode full \
    --max-density-trace-error 0.05 \
    --max-initial-trace-error 0.001 \
    --mark-complete "${VIEW}" \
    >"${attempt_state}/logs/full.log" 2>&1
  require_file "${VIEW}/COMPLETE"
  atomic_write_marker \
    "${attempt_state}/steps/full-verification.COMPLETE" \
    "action=verified" \
    "completed_at=$(date -Is)"

  ACTIVE_STAGE=complete
  complete_attempt
  echo "Finalize complete"
}

status() {
  local marker
  for marker in \
    "${GLOBAL_STATE}/raw_sync/be.complete" \
    "${GLOBAL_STATE}/raw_sync/missing.complete"; do
    if [[ -f "${marker}" ]]; then
      printf 'complete\t%s\t%s\n' "${marker}" "$(head -1 "${marker}")"
    else
      printf 'pending\t%s\n' "${marker}"
    fi
  done

  print_state \
    be16 "${OVERLAY}/_pipeline/be16" \
    "${OVERLAY}/_pipeline/be16/COMPLETE"
  print_state \
    missing24 "${OVERLAY}/_pipeline/missing24" \
    "${OVERLAY}/_pipeline/missing24/COMPLETE"
  print_state \
    finalize "${OVERLAY}/_pipeline/finalize" \
    "${OVERLAY}/_pipeline/finalize/COMPLETE"

  marker=${VIEW}/COMPLETE
  if [[ -f "${marker}" ]]; then
    printf 'complete\tview\t%s\n' "$(head -1 "${marker}")"
  else
    printf 'pending\tview\n'
  fi

  printf 'overlay shards\t'
  if [[ -d "${OVERLAY}" ]]; then
    find "${OVERLAY}" -mindepth 3 -maxdepth 3 -type f -name data.mdb | wc -l
  else
    echo 0
  fi
  tmux list-sessions 2>/dev/null | rg 'sc26-omol-electrolyte-v2-' || true
}

[[ $# -ge 1 ]] || usage
case "$1" in
  status)
    [[ $# -eq 1 ]] || usage
    status
    ;;
  start-be)
    [[ $# -eq 1 ]] || usage
    start_phase be16
    ;;
  start-missing)
    [[ $# -eq 1 ]] || usage
    start_phase missing24
    ;;
  start-finalize)
    [[ $# -eq 1 ]] || usage
    start_finalize
    ;;
  _run-phase)
    [[ $# -eq 2 ]] || usage
    run_phase "$2"
    ;;
  _finalize)
    [[ $# -eq 1 ]] || usage
    finalize
    ;;
  *)
    usage
    ;;
esac
