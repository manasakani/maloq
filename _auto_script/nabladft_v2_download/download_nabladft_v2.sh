#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
DATASET_ROOT=/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/nabladft-v2-download
PYTHON=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
VERIFY=${PROJECT_ROOT}/_auto_script/nabladft_v2_download/verify_nabladft_v2.py
BASE_URL=https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/hamiltonian_databases
MAX_ATTEMPTS=${NABLADFT_MAX_ATTEMPTS:-100}

names=(
  train_10k.db
  test_2k_conformers.db
)
bytes=(
  68388278272
  3099738112
)
etags=(
  41f03a745d88afa8689f7a41e0afb54f-8153
  0b9a02f0e3d1dee44bb4d40353845a1c-370
)
artifacts=(
  train_10k
  test_2k_conformers
)

usage() {
  echo "Usage: $0 {status|download|verify}" >&2
  exit 2
}

file_size() {
  local path=$1
  if [[ -f "${path}" ]]; then
    stat -c '%s' "${path}"
  else
    echo 0
  fi
}

status() {
  local index final partial actual expected percent state sidecar
  for index in "${!names[@]}"; do
    final=${DATASET_ROOT}/${names[$index]}
    partial=${final}.part
    sidecar=${final}.verification.json
    expected=${bytes[$index]}
    if [[ -f "${final}" ]]; then
      actual=$(file_size "${final}")
      state=complete
      if [[ -f "${sidecar}" ]]; then
        state=verified
      fi
    else
      actual=$(file_size "${partial}")
      state=partial
    fi
    percent=$(awk -v actual="${actual}" -v expected="${expected}" \
      'BEGIN { printf "%.3f", 100 * actual / expected }')
    printf '%s state=%s bytes=%s/%s percent=%s%%\n' \
      "${names[$index]}" "${state}" "${actual}" "${expected}" "${percent}"
  done
}

verify_remote_metadata() {
  local name=$1
  local expected_bytes=$2
  local expected_etag=$3
  local log=$4
  local headers remote_bytes remote_etag

  headers=$(curl \
    --fail \
    --silent \
    --show-error \
    --location \
    --head \
    --connect-timeout 30 \
    --retry 10 \
    --retry-delay 10 \
    --retry-all-errors \
    "${BASE_URL}/${name}")
  remote_bytes=$(printf '%s\n' "${headers}" | tr -d '\r' \
    | awk 'tolower($1) == "content-length:" {value=$2} END {print value}')
  remote_etag=$(printf '%s\n' "${headers}" | tr -d '\r' \
    | awk 'tolower($1) == "etag:" {gsub(/"/, "", $2); value=$2} END {print value}')
  if [[ "${remote_bytes}" != "${expected_bytes}" ]]; then
    echo "${name}: remote Content-Length ${remote_bytes:-missing}, expected ${expected_bytes}" \
      | tee -a "${log}" >&2
    return 1
  fi
  if [[ "${remote_etag}" != "${expected_etag}" ]]; then
    echo "${name}: remote ETag ${remote_etag:-missing}, expected ${expected_etag}" \
      | tee -a "${log}" >&2
    return 1
  fi
  echo "${name}: remote metadata matches size and ETag" | tee -a "${log}"
}

download_one() {
  local name=$1
  local expected=$2
  local expected_etag=$3
  local final=${DATASET_ROOT}/${name}
  local partial=${final}.part
  local log=${OUTPUT_ROOT}/${name}.log
  local attempt=1
  local actual

  verify_remote_metadata "${name}" "${expected}" "${expected_etag}" "${log}"

  while (( attempt <= MAX_ATTEMPTS )); do
    if [[ -f "${final}" ]]; then
      actual=$(file_size "${final}")
      if [[ "${actual}" == "${expected}" ]]; then
        echo "${name}: already complete (${actual} bytes)" | tee -a "${log}"
        return
      fi
      echo "${name}: completed filename has unexpected size ${actual}" >&2
      return 1
    fi

    actual=$(file_size "${partial}")
    if [[ "${actual}" == "${expected}" ]]; then
      mv "${partial}" "${final}"
      echo "${name}: promoted complete partial file" | tee -a "${log}"
      return
    fi
    if (( actual > expected )); then
      echo "${name}: partial file is larger than expected (${actual} > ${expected})" >&2
      return 1
    fi

    echo "${name}: attempt ${attempt}, resuming at ${actual}/${expected}" \
      | tee -a "${log}"
    if curl \
      --fail \
      --location \
      --continue-at - \
      --connect-timeout 30 \
      --speed-limit 1024 \
      --speed-time 300 \
      --retry 20 \
      --retry-delay 15 \
      --retry-all-errors \
      --remote-time \
      --output "${partial}" \
      "${BASE_URL}/${name}" >>"${log}" 2>&1; then
      actual=$(file_size "${partial}")
      if [[ "${actual}" == "${expected}" ]]; then
        mv "${partial}" "${final}"
        echo "${name}: download complete (${actual} bytes)" | tee -a "${log}"
        return
      fi
      echo "${name}: curl succeeded with unexpected size ${actual}" \
        | tee -a "${log}"
    fi
    attempt=$((attempt + 1))
    sleep 30
  done
  echo "${name}: failed after ${MAX_ATTEMPTS} attempts" >&2
  return 1
}

verify_all() {
  local args=()
  local artifact
  for artifact in "${artifacts[@]}"; do
    args+=(--artifact "${artifact}")
  done
  "${PYTHON}" "${VERIFY}" \
    --root "${DATASET_ROOT}" \
    --hash \
    --write-report \
    "${args[@]}"
}

download_all() {
  exec 9>"${OUTPUT_ROOT}/download.lock"
  if ! flock -n 9; then
    echo "Another NablaDFT downloader owns ${OUTPUT_ROOT}/download.lock" >&2
    exit 1
  fi

  local pids=()
  local index
  for index in "${!names[@]}"; do
    download_one "${names[$index]}" "${bytes[$index]}" "${etags[$index]}" &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  (( failed == 0 )) || return 1
  verify_all
}

mkdir -p "${DATASET_ROOT}" "${OUTPUT_ROOT}"

[[ $# -eq 1 ]] || usage
case "$1" in
  status) status ;;
  download) download_all ;;
  verify) verify_all ;;
  *) usage ;;
esac
