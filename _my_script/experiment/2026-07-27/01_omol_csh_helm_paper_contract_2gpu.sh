#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PYTHON=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
CONFIG=${PROJECT_ROOT}/_my_script/experiment/2026-07-27/omol_csh_helm_paper_contract.yaml
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-27/run_omol_csh_helm_paper_contract.py
PREPARE=${PROJECT_ROOT}/_auto_script/omol_csh_download/download_omol_csh.sh
OUTPUT_ROOT=${PROJECT_ROOT}/outputs
DEFAULT_GPUS=0,1
DEFAULT_MASTER_PORT=29727

cd "${PROJECT_ROOT}"

usage() {
  echo "Usage: $0 {prepare|validate|smoke|full} [GPU_LIST]" >&2
  exit 2
}

validate_gpus() {
  local gpu_list=$1
  local gpu pid_list
  [[ "${gpu_list}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "Invalid GPU list: ${gpu_list}" >&2
    exit 2
  }
  IFS=',' read -r -a requested_gpus <<<"${gpu_list}"
  for gpu in "${requested_gpus[@]}"; do
    nvidia-smi -i "${gpu}" --query-gpu=index --format=csv,noheader,nounits \
      >/dev/null
    pid_list=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    if [[ -n "${pid_list}" && "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
      echo "GPU ${gpu} is materially busy (PIDs: ${pid_list//$'\n'/,})." >&2
      echo "Choose idle GPUs or set ALLOW_BUSY_GPUS=1 explicitly." >&2
      exit 1
    fi
  done
  NUM_PROCESSES=${#requested_gpus[@]}
}

[[ $# -ge 1 && $# -le 2 ]] || usage
scope=$1
gpus=${2:-${DEFAULT_GPUS}}

case "${scope}" in
  prepare)
    bash "${PREPARE}" verify
    ;;
  validate)
    env PYTHONPATH=${PROJECT_ROOT}/src \
      "${PYTHON}" "${RUNNER}" --config "${CONFIG}" --scope validate
    ;;
  smoke|full)
    validate_gpus "${gpus}"
    timestamp=$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S)
    if [[ "${scope}" == smoke ]]; then
      output_folder=${OUTPUT_ROOT}/omol-csh-helm-paper-contract-smoke-${timestamp}-$$
    else
      output_folder=${OUTPUT_ROOT}/omol-csh-helm-paper-contract-${timestamp}
    fi
    mkdir -p "${output_folder}"
    export CUDA_VISIBLE_DEVICES=${gpus}
    export PYTHONPATH=${PROJECT_ROOT}/src
    export OMOL_CSH_OUTPUT_FOLDER=${output_folder}
    "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node="${NUM_PROCESSES}" \
      --master_addr=127.0.0.1 \
      --master_port="${OMOL_CSH_MASTER_PORT:-${DEFAULT_MASTER_PORT}}" \
      "${RUNNER}" --config "${CONFIG}" --scope "${scope}"
    if [[ "${scope}" == smoke ]]; then
      rm -rf -- "${output_folder}"
      echo "OMol_CSH smoke passed; removed ${output_folder}"
    else
      echo "OMol_CSH full output: ${output_folder}"
    fi
    ;;
  *)
    usage
    ;;
esac
