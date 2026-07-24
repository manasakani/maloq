#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 {prepare|validate|smoke|full} [GPU0,GPU1] [GPU2,GPU3] [GPU4,GPU5]" >&2
  exit 2
fi

SCOPE=$1
PAIR0=${2:-2,3}
PAIR1=${3:-4,5}
PAIR2=${4:-6,7}
EXPECTED_HOST=${EXPECTED_HOST:-}

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py
NABLA_DB=/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db
EXPERIMENT_DIR=${PROJECT_ROOT}/_my_script/experiment/2026-07-24

LABELS=(
  p16m-w88-d3
  p33m-w128-d3
  p125m-w192-d5
  p500m-w384-d5
)
PARAMETERS=(16125037 33750157 125004341 496331189)
DISPLAY_SIZES=(16M 33M 125M 500M)
WIDTHS=(88 128 192 384)
DEPTHS=(3 3 5 5)
MICRO_BATCHES=(5 5 2 1)
ACCUMULATIONS=(2 2 5 10)
CONFIGS=(
  "${EXPERIMENT_DIR}/maloq_nte_scaling_p16m_w88_d3_nabladft.yaml"
  "${EXPERIMENT_DIR}/maloq_nte_scaling_p33m_w128_d3_nabladft.yaml"
  "${EXPERIMENT_DIR}/maloq_nte_scaling_p125m_w192_d5_nabladft.yaml"
  "${EXPERIMENT_DIR}/maloq_nte_scaling_p500m_w384_d5_nabladft.yaml"
)

case "${SCOPE}" in
  prepare)
    echo "No preprocessing is required: all four scaling runs use scale_and_shift=false."
    exit 0
    ;;
  validate | smoke | full) ;;
  *)
    echo "Scope must be prepare, validate, smoke, or full." >&2
    exit 2
    ;;
esac

if [[ ! -x "${PY}" || ! -x "${MPIRUN}" || ! -f "${RUNNER}" ]]; then
  echo "SC26 environment, mpirun, or experiment runner is missing." >&2
  exit 1
fi
for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Model config is missing: ${config}" >&2
    exit 1
  fi
done

ACTUAL_HOST=$(hostname)
if [[ -n "${EXPECTED_HOST}" && "${ACTUAL_HOST}" != "${EXPECTED_HOST}" ]]; then
  echo "Expected host ${EXPECTED_HOST}; current host is ${ACTUAL_HOST}." >&2
  exit 1
fi

validate_pair() {
  local pair=$1
  local -a indices
  IFS=',' read -r -a indices <<< "${pair}"
  if [[ ${#indices[@]} -ne 2 ]]; then
    echo "GPU pair must contain exactly two comma-separated indices: ${pair}" >&2
    return 2
  fi
  if [[ "${indices[0]}" == "${indices[1]}" ]]; then
    echo "A GPU pair cannot repeat an index: ${pair}" >&2
    return 2
  fi
  local gpu
  for gpu in "${indices[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      echo "Invalid GPU index in pair ${pair}: ${gpu}" >&2
      return 2
    fi
  done
}

for pair in "${PAIR0}" "${PAIR1}" "${PAIR2}"; do
  validate_pair "${pair}"
done

declare -A SEEN_GPUS=()
for pair in "${PAIR0}" "${PAIR1}" "${PAIR2}"; do
  IFS=',' read -r -a indices <<< "${pair}"
  for gpu in "${indices[@]}"; do
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
      echo "GPU ${gpu} occurs in more than one pair." >&2
      exit 2
    fi
    SEEN_GPUS[${gpu}]=1
  done
done

check_pair_free() {
  local pair=$1
  local -a memory_used indices
  mapfile -t memory_used < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  IFS=',' read -r -a indices <<< "${pair}"
  local gpu
  for gpu in "${indices[@]}"; do
    if [[ -z "${memory_used[${gpu}]:-}" ]]; then
      echo "GPU ${gpu} does not exist on ${ACTUAL_HOST}." >&2
      return 2
    fi
    if (( memory_used[gpu] > 1024 )); then
      echo "GPU ${gpu} uses ${memory_used[gpu]} MiB; refusing overlap." >&2
      return 1
    fi
  done
}

if [[ "${SCOPE}" == "validate" ]]; then
  for idx in "${!LABELS[@]}"; do
    echo "Validating ${LABELS[idx]} (${PARAMETERS[idx]} parameters expected)"
    "${PY}" "${RUNNER}" \
      --dataset nabladft \
      --variant maloq-nte \
      --dbpath "${NABLA_DB}" \
      --model-config "${CONFIGS[idx]}" \
      --head-type maloq_muon \
      --optimizer-type muon \
      --batch-size "${MICRO_BATCHES[idx]}" \
      --gradient-accumulation-steps "${ACCUMULATIONS[idx]}" \
      --no-distribute-graphs \
      --gpu "${PAIR0}" \
      --master-port "$((29620 + idx))" \
      --run-name "nabla-nte-scale-${LABELS[idx]}-v1" \
      --wandb-run-name "NablaDFT | NTE Scaling | ${DISPLAY_SIZES[idx]} | W${WIDTHS[idx]} D${DEPTHS[idx]} | V1" \
      --wandb-group nabla-nte-muon-scaling \
      --wandb-job-type validate \
      --wandb-tag scaling-law \
      --wandb-tag "params:${PARAMETERS[idx]}" \
      --wandb-tag "width:${WIDTHS[idx]}" \
      --wandb-tag "depth:${DEPTHS[idx]}" \
      --wandb-tag version:v1 \
      --validate-only \
      --no-use-wandb \
      --flat-output
  done
  exit 0
fi

if [[ "${SCOPE}" == "smoke" ]]; then
  check_pair_free "${PAIR0}"
else
  check_pair_free "${PAIR0}"
  check_pair_free "${PAIR1}"
  check_pair_free "${PAIR2}"
fi

RUN_ID=${SCALING_RUN_ID:-$(date +%Y%m%d-%H%M%S)}
if [[ "${SCOPE}" == "smoke" ]]; then
  GROUP_NAME=nabla-nte-muon-scaling-p500m-smoke-v1-seed44-${RUN_ID}
else
  GROUP_NAME=nabla-nte-muon-scaling-4point-v1-eb20-full-e20-seed44-${RUN_ID}
fi
GROUP_ROOT=${PROJECT_ROOT}/outputs/${GROUP_NAME}
if [[ -e "${GROUP_ROOT}" ]]; then
  echo "Output group already exists: ${GROUP_ROOT}" >&2
  exit 1
fi
mkdir -p "${GROUP_ROOT}/logs" "${GROUP_ROOT}/runs" "${GROUP_ROOT}/status"
exec > >(tee -a "${GROUP_ROOT}/coordinator.log") 2>&1

printf 'label\tparameters\twidth\tdepth\tmicro_batch\tworld_size\taccumulation\teffective_batch\tconfig\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"
for idx in "${!LABELS[@]}"; do
  printf '%s\t%s\t%s\t%s\t%s\t2\t%s\t20\t%s\n' \
    "${LABELS[idx]}" "${PARAMETERS[idx]}" "${WIDTHS[idx]}" "${DEPTHS[idx]}" \
    "${MICRO_BATCHES[idx]}" "${ACCUMULATIONS[idx]}" "${CONFIGS[idx]}" \
    >> "${GROUP_ROOT}/launch_manifest.tsv"
done
printf 'source_commit\t%s\n' "$(git rev-parse HEAD)" > "${GROUP_ROOT}/source_revision.tsv"
printf 'host\t%s\nscope\t%s\nrun_id\t%s\n' \
  "${ACTUAL_HOST}" "${SCOPE}" "${RUN_ID}" > "${GROUP_ROOT}/launch_context.tsv"
echo "Output group: ${GROUP_ROOT}"

run_one() {
  local idx=$1
  local pair=$2
  local port=$3
  local label=${LABELS[idx]}
  local output_dir=${GROUP_ROOT}/runs/${label}
  local log_file=${GROUP_ROOT}/logs/${label}.log
  local status_file=${GROUP_ROOT}/status/${label}.tsv
  local -a scope_args

  check_pair_free "${pair}"
  if [[ "${SCOPE}" == "smoke" ]]; then
    scope_args=(
      --smoke
      --full-size-smoke
      --keep-smoke-output
      --num-epochs 1
      --num-train 20
      --num-val 20
      --num-test 0
      --no-use-wandb
      --wandb-job-type smoke
    )
  else
    scope_args=(
      --num-epochs 20
      --num-train 12081
      --num-val 64
      --num-test 0
      --use-wandb
      --wandb-project maloq-nablaDFT
      --wandb-entity kaist-korea
      --wandb-mode online
      --wandb-log-every-n-steps 10
      --wandb-job-type full
    )
  fi

  printf 'status\texit_code\tgpu_pair\tmaster_port\nrunning\t\t%s\t%s\n' \
    "${pair}" "${port}" > "${status_file}"
  echo "Starting ${label} on GPUs ${pair}, port ${port}"
  set +e
  env \
    CUDA_VISIBLE_DEVICES="${pair}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${port}" \
    OPAL_PREFIX="${ENV_ROOT}" \
    PRTE_PREFIX="${ENV_ROOT}" \
    PMIX_PREFIX="${ENV_ROOT}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "${MPIRUN}" -np 2 --bind-to none \
    "${PY}" "${RUNNER}" \
      --dataset nabladft \
      --variant maloq-nte \
      --dbpath "${NABLA_DB}" \
      --model-config "${CONFIGS[idx]}" \
      --head-type maloq_muon \
      --optimizer-type muon \
      --batch-size "${MICRO_BATCHES[idx]}" \
      --gradient-accumulation-steps "${ACCUMULATIONS[idx]}" \
      --no-distribute-graphs \
      --gpu "${pair}" \
      --master-port "${port}" \
      --run-name "nabla-nte-scale-${label}-v1" \
      --wandb-run-name "NablaDFT | NTE Scaling | ${DISPLAY_SIZES[idx]} | W${WIDTHS[idx]} D${DEPTHS[idx]} | V1" \
      --wandb-group nabla-nte-muon-scaling \
      --wandb-tag scaling-law \
      --wandb-tag "params:${PARAMETERS[idx]}" \
      --wandb-tag "width:${WIDTHS[idx]}" \
      --wandb-tag "depth:${DEPTHS[idx]}" \
      --wandb-tag scale-shift:off \
      --wandb-tag seed:44 \
      --wandb-tag version:v1 \
      --flat-output \
      "${scope_args[@]}" \
      --output-root "${output_dir}" \
      2>&1 | tee "${log_file}"
  local run_status=${PIPESTATUS[0]}
  set -e

  if [[ ${run_status} -ne 0 ]]; then
    printf 'status\texit_code\tgpu_pair\tmaster_port\nfailed\t%s\t%s\t%s\n' \
      "${run_status}" "${pair}" "${port}" > "${status_file}"
    echo "${label} failed with exit code ${run_status}" >&2
    return "${run_status}"
  fi
  printf 'status\texit_code\tgpu_pair\tmaster_port\ncomplete\t0\t%s\t%s\n' \
    "${pair}" "${port}" > "${status_file}"
  echo "Completed ${label}"
}

if [[ "${SCOPE}" == "smoke" ]]; then
  run_one 3 "${PAIR0}" 29623
  MODEL_SUMMARY=${GROUP_ROOT}/runs/${LABELS[3]}/model_summary.json
  "${PY}" - "${MODEL_SUMMARY}" "${PARAMETERS[3]}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected = int(sys.argv[2])
actual = int(json.loads(summary_path.read_text())["total_parameters"])
if actual != expected:
    raise SystemExit(f"Parameter mismatch: expected {expected}, found {actual}")
print(f"Verified largest model parameter count: {actual:,}")
PY
  SMOKE_OUTPUT=${GROUP_ROOT}/runs/${LABELS[3]}
  case "${SMOKE_OUTPUT}" in
    "${GROUP_ROOT}"/runs/p500m-w384-d5) rm -rf -- "${SMOKE_OUTPUT}" ;;
    *) echo "Refusing to remove unexpected smoke path: ${SMOKE_OUTPUT}" >&2; exit 1 ;;
  esac
  printf 'status\texit_code\ncomplete\t0\n' > "${GROUP_ROOT}/coordinator_status.tsv"
  echo "Largest-model smoke passed; temporary training artifacts removed."
  exit 0
fi

queue_small_models() {
  local queue_status=0
  run_one 1 "${PAIR2}" 29622 || queue_status=$?
  run_one 0 "${PAIR2}" 29624 || queue_status=$?
  return "${queue_status}"
}

run_one 3 "${PAIR0}" 29620 &
PID_500M=$!
run_one 2 "${PAIR1}" 29621 &
PID_125M=$!
queue_small_models &
PID_SMALL_QUEUE=$!
printf 'lane\tpid\tgpu_pair\n500m\t%s\t%s\n125m\t%s\t%s\n33m_then_16m\t%s\t%s\n' \
  "${PID_500M}" "${PAIR0}" "${PID_125M}" "${PAIR1}" \
  "${PID_SMALL_QUEUE}" "${PAIR2}" > "${GROUP_ROOT}/coordinator_pids.tsv"

set +e
wait "${PID_500M}"
STATUS_500M=$?
wait "${PID_125M}"
STATUS_125M=$?
wait "${PID_SMALL_QUEUE}"
STATUS_SMALL_QUEUE=$?
set -e

if (( STATUS_500M != 0 || STATUS_125M != 0 || STATUS_SMALL_QUEUE != 0 )); then
  printf 'status\texit_code_500m\texit_code_125m\texit_code_small_queue\nfailed\t%s\t%s\t%s\n' \
    "${STATUS_500M}" "${STATUS_125M}" "${STATUS_SMALL_QUEUE}" \
    > "${GROUP_ROOT}/coordinator_status.tsv"
  exit 1
fi
printf 'status\texit_code_500m\texit_code_125m\texit_code_small_queue\ncomplete\t0\t0\t0\n' \
  > "${GROUP_ROOT}/coordinator_status.tsv"
echo "All four scaling runs completed: ${GROUP_ROOT}"
