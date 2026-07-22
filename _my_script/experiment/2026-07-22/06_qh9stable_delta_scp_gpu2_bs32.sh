#!/usr/bin/env bash
set -euo pipefail

# QH9Stable delta comparison on scp-gpu-2.
# Six single-GPU lanes keep the effective batch equal to the requested 32:
# H: MALOQ=GPU0, MALOQ-NTE=GPU1, QHFlow3=GPU2
# D: MALOQ=GPU3, MALOQ-NTE=GPU4, QHFlow3=GPU5

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {hamiltonian|density|both} {validate|smoke|full} [maloq|maloq-nte|qhflow3|all]" >&2
  exit 2
fi

TARGET_SELECTION=$1
SCOPE=$2
MODEL_SELECTION=${3:-all}

case "${TARGET_SELECTION}" in
  hamiltonian) TARGETS=(hamiltonian) ;;
  density) TARGETS=(density) ;;
  both) TARGETS=(hamiltonian density) ;;
  *) echo "Target must be hamiltonian, density, or both." >&2; exit 2 ;;
esac
case "${MODEL_SELECTION}" in
  maloq|maloq-nte|qhflow3) MODELS=("${MODEL_SELECTION}") ;;
  all) MODELS=(maloq maloq-nte qhflow3) ;;
  *) echo "Model must be maloq, maloq-nte, qhflow3, or all." >&2; exit 2 ;;
esac

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py
EXPECTED_HOST=${EXPECTED_HOST:-usr310-gpumngc-02}
ACTUAL_HOST=$(hostname)
NUM_EPOCHS=${NUM_EPOCHS:-80}
RUN_ID=$(date +%Y%m%d-%H%M%S)
GROUP_ROOT=${PROJECT_ROOT}/outputs/qh9stable-delta-${TARGET_SELECTION}-six-lane-bs32-${SCOPE}-seed44-${RUN_ID}

if [[ ! -x "${PY}" || ! -f "${RUNNER}" ]]; then
  echo "SC26 environment or runner is missing." >&2
  exit 1
fi
if [[ "${SCOPE}" != "validate" && "${ACTUAL_HOST}" != "${EXPECTED_HOST}" ]]; then
  echo "Run ${SCOPE} on scp-gpu-2 (${EXPECTED_HOST}); current host is ${ACTUAL_HOST}." >&2
  exit 1
fi
if [[ ! "${NUM_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_EPOCHS must be a positive integer." >&2
  exit 2
fi

case "${SCOPE}" in
  validate)
    RUN_ARGS=(--validate-only --num-train 64 --num-val 32 --num-test 0)
    TRACKING_ARGS=(--no-use-wandb)
    ;;
  smoke)
    RUN_ARGS=(--smoke --full-size-smoke --num-epochs 1 --num-train 64 --num-val 32 --num-test 0)
    TRACKING_ARGS=(--no-use-wandb)
    ;;
  full)
    RUN_ARGS=(--num-epochs "${NUM_EPOCHS}" --num-train 104664 --num-val 13083 --num-test 13084)
    TRACKING_ARGS=(
      --use-wandb
      --wandb-project maloq-qh9
      --wandb-entity kaist-korea
      --wandb-mode online
      --wandb-log-every-n-steps 10
    )
    ;;
  *) echo "Scope must be validate, smoke, or full." >&2; exit 2 ;;
esac

declare -A LANE_GPUS=(
  [hamiltonian:maloq]=0
  [hamiltonian:maloq-nte]=1
  [hamiltonian:qhflow3]=2
  [density:maloq]=3
  [density:maloq-nte]=4
  [density:qhflow3]=5
)
declare -A LANE_PORTS=(
  [hamiltonian:maloq]=29601
  [hamiltonian:maloq-nte]=29602
  [hamiltonian:qhflow3]=29603
  [density:maloq]=29604
  [density:maloq-nte]=29605
  [density:qhflow3]=29606
)
declare -A LANE_PIDS=()

if [[ "${SCOPE}" != "validate" ]]; then
  mapfile -t GPU_MEMORY_USED < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  for target in "${TARGETS[@]}"; do
    for model in "${MODELS[@]}"; do
      key=${target}:${model}
      gpu=${LANE_GPUS[${key}]}
      used_mib=${GPU_MEMORY_USED[${gpu}]:-999999}
      if (( used_mib > 1024 )); then
        echo "GPU ${gpu} is already using ${used_mib} MiB; refusing to overlap ${key}." >&2
        exit 1
      fi
    done
  done
fi

mkdir -p "${GROUP_ROOT}/logs"
printf 'target\tmodel\tgpu\tbatch_size\teffective_batch\toutput\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"

launch_lane() {
  local target=$1 model=$2 key=${1}:${2}
  local gpu=${LANE_GPUS[${key}]} port=${LANE_PORTS[${key}]}
  local output_dir=${GROUP_ROOT}/${target}-${model}
  local log_file=${GROUP_ROOT}/logs/${target}-${model}.log

  printf '%s\t%s\t%s\t32\t32\t%s\n' \
    "${target}" "${model}" "${gpu}" "${output_dir}" \
    >> "${GROUP_ROOT}/launch_manifest.tsv"

  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${port}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "${PY}" "${RUNNER}" \
      --dataset "qh9-${target}" \
      --variant "${model}" \
      --optimizer-type muon \
      --batch-size 32 \
      --gradient-accumulation-steps 1 \
      --delta-learning \
      --no-distribute-graphs \
      "${TRACKING_ARGS[@]}" \
      --gpu "${gpu}" \
      --master-port "${port}" \
      --flat-output \
      "${RUN_ARGS[@]}" \
      --output-root "${output_dir}" \
      > "${log_file}" 2>&1 &

  LANE_PIDS[${key}]=$!
  echo "Started ${target}/${model} on ${ACTUAL_HOST} GPU ${gpu}: PID ${LANE_PIDS[${key}]}"
  echo "  log: ${log_file}"
}

stop_children() {
  local key
  for key in "${!LANE_PIDS[@]}"; do
    kill "${LANE_PIDS[${key}]}" 2>/dev/null || true
  done
}
trap stop_children INT TERM

cd "${PROJECT_ROOT}"
for target in "${TARGETS[@]}"; do
  for model in "${MODELS[@]}"; do
    launch_lane "${target}" "${model}"
  done
done

printf 'target\tmodel\tstatus\texit_code\n' > "${GROUP_ROOT}/status.tsv"
overall_status=0
for target in "${TARGETS[@]}"; do
  for model in "${MODELS[@]}"; do
    key=${target}:${model}
    if wait "${LANE_PIDS[${key}]}"; then
      exit_code=0 status=complete
    else
      exit_code=$? status=failed overall_status=1
    fi
    printf '%s\t%s\t%s\t%s\n' \
      "${target}" "${model}" "${status}" "${exit_code}" \
      >> "${GROUP_ROOT}/status.tsv"
    echo "${target}/${model}: ${status} (exit ${exit_code})"
  done
done
trap - INT TERM

comparison_files=()
for target in "${TARGETS[@]}"; do
  for model in "${MODELS[@]}"; do
    comparison_file=${GROUP_ROOT}/${target}-${model}/comparison.csv
    [[ -f "${comparison_file}" ]] && comparison_files+=("${comparison_file}")
  done
done
if [[ ${#comparison_files[@]} -gt 0 ]]; then
  awk 'FNR == 1 && NR != 1 { next } { print }' \
    "${comparison_files[@]}" > "${GROUP_ROOT}/comparison.csv"
fi

if [[ "${SCOPE}" == "smoke" && ${overall_status} -eq 0 ]]; then
  case "${GROUP_ROOT}" in
    "${PROJECT_ROOT}"/outputs/qh9stable-delta-*-six-lane-bs32-smoke-seed44-*) ;;
    *) echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2; exit 1 ;;
  esac
  rm -rf -- "${GROUP_ROOT}"
  echo "Smoke passed; temporary outputs removed: ${GROUP_ROOT}"
else
  echo "Run group retained: ${GROUP_ROOT}"
fi

exit "${overall_status}"
