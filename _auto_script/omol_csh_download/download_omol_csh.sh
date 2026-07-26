#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
DATASET_ROOT=/dataset/seongsu/shared-home/datasets/omol_csh
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/omol-csh-download
PYTHON=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
VERIFY=${PROJECT_ROOT}/_auto_script/omol_csh_download/verify_omol_csh.py
MAX_ATTEMPTS=${OMOL_CSH_MAX_ATTEMPTS:-100}

mkdir -p "${DATASET_ROOT}" "${OUTPUT_ROOT}"

names=(
  omol_csh_58k_train.h5
  omol_csh_5k_test_all.h5
  omol_csh_1k_test_common.h5
)
urls=(
  https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/omol_csh/omol_csh_58k_train.h5
  https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/omol_csh/omol_csh_5k_test_all.h5
  https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/omol_csh/omol_csh_1k_test_common.h5
)
bytes=(
  276516996009
  33350917699
  8203349480
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
  local index final partial actual expected percent state
  for index in "${!names[@]}"; do
    final=${DATASET_ROOT}/${names[$index]}
    partial=${final}.part
    expected=${bytes[$index]}
    if [[ -f "${final}" ]]; then
      actual=$(file_size "${final}")
      state=complete
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

download_one() {
  local name=$1
  local url=$2
  local expected=$3
  local final=${DATASET_ROOT}/${name}
  local partial=${final}.part
  local log=${OUTPUT_ROOT}/${name}.log
  local attempt=1
  local actual

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
      "${url}" >>"${log}" 2>&1; then
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

download_all() {
  exec 9>"${OUTPUT_ROOT}/download.lock"
  if ! flock -n 9; then
    echo "Another OMol_CSH downloader owns ${OUTPUT_ROOT}/download.lock" >&2
    exit 1
  fi

  local pids=()
  local index
  for index in "${!names[@]}"; do
    download_one "${names[$index]}" "${urls[$index]}" "${bytes[$index]}" &
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
  "${PYTHON}" "${VERIFY}" --root "${DATASET_ROOT}"
}

[[ $# -eq 1 ]] || usage
case "$1" in
  status) status ;;
  download) download_all ;;
  verify) "${PYTHON}" "${VERIFY}" --root "${DATASET_ROOT}" ;;
  *) usage ;;
esac
