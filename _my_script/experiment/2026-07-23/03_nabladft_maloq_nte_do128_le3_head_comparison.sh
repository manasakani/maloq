#!/usr/bin/env bash
set -euo pipefail

# Controlled MALOQ-NTE head comparison with MALOQ-matched do=128 and Le=3.
# Native head runs on GPUs 0,1 while corrected Muon head runs on GPUs 2,3.
# Effective batch: 5 molecules/rank * 2 ranks * 2 accumulation steps = 20.

SCOPE=${1:-validate}
NATIVE_GPUS=${NATIVE_GPUS:-0,1}
MUON_GPUS=${MUON_GPUS:-2,3}
NATIVE_MASTER_PORT=${NATIVE_MASTER_PORT:-29594}
MUON_MASTER_PORT=${MUON_MASTER_PORT:-29595}

case "${SCOPE}" in
  validate)
    RUN_ARGS=(--validate-only --no-use-wandb)
    ;;
  smoke)
    RUN_ARGS=(
      --smoke
      --full-size-smoke
      --keep-smoke-output
      --num-epochs 1
      --num-train 20
      --num-val 20
      --num-test 0
      --no-use-wandb
    )
    SCOPE_SLUG=full-size-smoke-e1
    ;;
  full)
    RUN_ARGS=(
      --num-epochs 20
      --num-train 12081
      --num-val 64
      --num-test 0
      --use-wandb
      --wandb-project maloq-nablaDFT
      --wandb-entity kaist-korea
      --wandb-mode online
      --wandb-log-every-n-steps 10
    )
    SCOPE_SLUG=full-e20
    ;;
  *)
    echo "Usage: $0 {validate|smoke|full}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py
SCRIPT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-23
LANES=(native-head muon-head)

declare -A CONFIGS=(
  [native-head]=${SCRIPT_ROOT}/maloq_nte_do128_le3_native_head_nabladft.yaml
  [muon-head]=${SCRIPT_ROOT}/maloq_nte_do128_le3_muon_head_nabladft.yaml
)
declare -A HEAD_TYPES=(
  [native-head]=maloq
  [muon-head]=maloq_muon
)
declare -A LANE_GPUS=(
  [native-head]=${NATIVE_GPUS}
  [muon-head]=${MUON_GPUS}
)
declare -A LANE_PORTS=(
  [native-head]=${NATIVE_MASTER_PORT}
  [muon-head]=${MUON_MASTER_PORT}
)

common_args() {
  local config=$1 gpus=$2 port=$3
  printf '%s\0' \
    --dataset nabladft \
    --variant maloq-nte \
    --model-config "${config}" \
    --optimizer-type muon \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --no-distribute-graphs \
    --gpu "${gpus}" \
    --master-port "${port}" \
    --flat-output
}

cd "${PROJECT_ROOT}"
if [[ "${SCOPE}" == "validate" ]]; then
  for lane in "${LANES[@]}"; do
    mapfile -d '' -t COMMON_ARGS < <(
      common_args \
        "${CONFIGS[${lane}]}" \
        "${LANE_GPUS[${lane}]}" \
        "${LANE_PORTS[${lane}]}"
    )
    echo "Validating ${lane} (${HEAD_TYPES[${lane}]})"
    "${PY}" "${RUNNER}" "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}"
  done
  exit 0
fi

for lane in "${LANES[@]}"; do
  port=${LANE_PORTS[${lane}]}
  if [[ ! "${port}" =~ ^[0-9]+$ ]] ||
    (( 10#${port} < 1 || 10#${port} > 65535 )); then
    echo "Invalid master port for ${lane}: ${port}" >&2
    exit 2
  fi
done
if [[ "${NATIVE_MASTER_PORT}" == "${MUON_MASTER_PORT}" ]]; then
  echo "The two concurrent lanes must use different master ports." >&2
  exit 2
fi

mapfile -t GPU_MEMORY_USED < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
declare -A SEEN_GPUS=()
for lane in "${LANES[@]}"; do
  IFS=',' read -r -a GPU_INDICES <<< "${LANE_GPUS[${lane}]}"
  if [[ ${#GPU_INDICES[@]} -ne 2 ]]; then
    echo "${lane} must have exactly two comma-separated GPU indices." >&2
    exit 2
  fi
  for gpu in "${GPU_INDICES[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ || -z "${GPU_MEMORY_USED[${gpu}]:-}" ]]; then
      echo "Invalid GPU index for ${lane}: ${gpu}" >&2
      exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
      echo "GPU ${gpu} is assigned to more than one lane." >&2
      exit 2
    fi
    SEEN_GPUS[${gpu}]=${lane}
    if (( GPU_MEMORY_USED[gpu] > 1024 )); then
      echo "GPU ${gpu} already uses ${GPU_MEMORY_USED[gpu]} MiB; refusing overlap." >&2
      exit 1
    fi
  done
done

RUN_ID=$(date +%Y%m%d-%H%M%S)
GROUP_ROOT=${PROJECT_ROOT}/outputs/nabladft-maloq-nte-do128-le3-head-comparison-parallel-2x2gpu-eb20-mb5-ga2-${SCOPE_SLUG}-seed44-${RUN_ID}
if [[ -e "${GROUP_ROOT}" ]]; then
  echo "Output group already exists: ${GROUP_ROOT}" >&2
  exit 1
fi
mkdir -p "${GROUP_ROOT}/logs"
printf 'lane\thead_type\tdt\tdo\tLn\tLe\tgpus\tmicro_batch\tworld_size\taccumulation\teffective_batch\tconfig\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"
printf 'lane\tstatus\texit_code\n' > "${GROUP_ROOT}/status.tsv"

declare -A LANE_PIDS=()

launch_lane() {
  local lane=$1
  local config=${CONFIGS[${lane}]}
  local head_type=${HEAD_TYPES[${lane}]}
  local gpus=${LANE_GPUS[${lane}]}
  local port=${LANE_PORTS[${lane}]}
  local output_dir=${GROUP_ROOT}/${lane}
  local log_file=${GROUP_ROOT}/logs/${lane}.log
  local common=()

  printf '%s\t%s\t128\t128\t3\t3\t%s\t5\t2\t2\t20\t%s\n' \
    "${lane}" "${head_type}" "${gpus}" "${config}" \
    >> "${GROUP_ROOT}/launch_manifest.tsv"
  mapfile -d '' -t common < <(common_args "${config}" "${gpus}" "${port}")

  echo "Starting ${lane} (${head_type}) on GPUs ${gpus}."
  env \
    CUDA_VISIBLE_DEVICES="${gpus}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${port}" \
    OPAL_PREFIX="${ENV_ROOT}" \
    PRTE_PREFIX="${ENV_ROOT}" \
    PMIX_PREFIX="${ENV_ROOT}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "${MPIRUN}" -np 2 --bind-to none \
    "${PY}" "${RUNNER}" \
      "${common[@]}" \
      "${RUN_ARGS[@]}" \
      --output-root "${output_dir}" \
      > "${log_file}" 2>&1 &
  LANE_PIDS[${lane}]=$!
  echo "  PID ${LANE_PIDS[${lane}]}; log: ${log_file}"
}

stop_children() {
  local lane
  for lane in "${LANES[@]}"; do
    if [[ -n "${LANE_PIDS[${lane}]:-}" ]]; then
      kill "${LANE_PIDS[${lane}]}" 2>/dev/null || true
    fi
  done
}
trap stop_children INT TERM

for lane in "${LANES[@]}"; do
  launch_lane "${lane}"
done

overall_status=0
for lane in "${LANES[@]}"; do
  if wait "${LANE_PIDS[${lane}]}"; then
    exit_code=0
    status=complete
  else
    exit_code=$?
    status=failed
    overall_status=1
  fi
  printf '%s\t%s\t%s\n' "${lane}" "${status}" "${exit_code}" \
    >> "${GROUP_ROOT}/status.tsv"
  echo "${lane}: ${status} (exit ${exit_code})"
done
trap - INT TERM

comparison_files=()
for lane in "${LANES[@]}"; do
  comparison_file=${GROUP_ROOT}/${lane}/comparison.csv
  [[ -f "${comparison_file}" ]] && comparison_files+=("${comparison_file}")
done
if [[ ${#comparison_files[@]} -gt 0 ]]; then
  awk 'FNR == 1 && NR != 1 { next } { print }' \
    "${comparison_files[@]}" > "${GROUP_ROOT}/comparison.csv"
fi

if [[ "${SCOPE}" == "smoke" && ${overall_status} -eq 0 ]]; then
  case "${GROUP_ROOT}" in
    "${PROJECT_ROOT}"/outputs/nabladft-maloq-nte-do128-le3-head-comparison-parallel-2x2gpu-eb20-mb5-ga2-full-size-smoke-e1-seed44-*) ;;
    *) echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2; exit 1 ;;
  esac
  rm -rf -- "${GROUP_ROOT}"
  echo "Both smoke lanes passed; temporary output removed: ${GROUP_ROOT}"
else
  echo "Run group retained: ${GROUP_ROOT}"
fi

exit "${overall_status}"
