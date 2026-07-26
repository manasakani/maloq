#!/usr/bin/env bash
set -euo pipefail

SCOPE=${1:-validate}
MALOQ_GPUS=${2:-0,1}
NTE64_GPUS=${3:-2,3}
NTE128_GPUS=${4:-4,5}
EXPECTED_HOST=${EXPECTED_HOST:-}

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
SCRIPT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-26
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-25/run_training_workflow_fixed.py

LANES=(maloq nte64 nte128)
declare -A CONFIGS=(
  [maloq]=${SCRIPT_ROOT}/maloq_semantic_global_muon_raw_nabladft.yaml
  [nte64]=${SCRIPT_ROOT}/nte64e2_semantic_global_muon_raw_nabladft.yaml
  [nte128]=${SCRIPT_ROOT}/nte128e3_semantic_global_muon_raw_nabladft.yaml
)
declare -A VARIANTS=(
  [maloq]=maloq
  [nte64]=maloq-nte
  [nte128]=maloq-nte
)
declare -A LANE_GPUS=(
  [maloq]=${MALOQ_GPUS}
  [nte64]=${NTE64_GPUS}
  [nte128]=${NTE128_GPUS}
)
declare -A PORTS=(
  [maloq]=29631
  [nte64]=29632
  [nte128]=29633
)

case "${SCOPE}" in
  validate)
    RUN_ARGS=(--validate-only --no-use-wandb)
    ;;
  smoke)
    RUN_ARGS=(
      --smoke
      --full-size-smoke
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
      --wandb-group nabla-semantic-global-muon-raw-v2
    )
    SCOPE_SLUG=full-e20
    ;;
  *)
    echo "Usage: $0 {validate|smoke|full} [MALOQ_GPUS] [NTE64_GPUS] [NTE128_GPUS]" >&2
    exit 2
    ;;
esac

if [[ -n "${EXPECTED_HOST}" && "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "Expected host ${EXPECTED_HOST}, got $(hostname)." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ "${SCOPE}" == "validate" ]]; then
  for lane in "${LANES[@]}"; do
    "${PY}" "${RUNNER}" \
      --dataset nabladft \
      --variant "${VARIANTS[${lane}]}" \
      --model-config "${CONFIGS[${lane}]}" \
      --optimizer-type muon \
      --head-type maloq_semantic_global_muon \
      --batch-size 5 \
      --gradient-accumulation-steps 2 \
      --no-distribute-graphs \
      --gpu 0 \
      --master-port "${PORTS[${lane}]}" \
      "${RUN_ARGS[@]}"
  done
  exit 0
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
      echo "GPU ${gpu} is assigned to both ${SEEN_GPUS[${gpu}]} and ${lane}." >&2
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
GROUP_ROOT=${PROJECT_ROOT}/outputs/nabladft-semglobal-muon-3x2gpu-eb20-mb5-ga2-${SCOPE_SLUG}-${RUN_ID}
if [[ -e "${GROUP_ROOT}" ]]; then
  echo "Output group already exists: ${GROUP_ROOT}" >&2
  exit 1
fi
mkdir -p "${GROUP_ROOT}/logs"
printf 'lane\tgpus\toptimizer_groups\thead_routing\tmicro_batch\tworld_size\tgrad_acc\teffective_batch\tconfig\toutput\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"

declare -A LANE_PIDS=()
launch_lane() {
  local lane=$1
  local gpus=${LANE_GPUS[${lane}]}
  local port=${PORTS[${lane}]}
  local output_dir=${GROUP_ROOT}/${lane}
  local log_file=${GROUP_ROOT}/logs/${lane}.log

  printf '%s\t%s\tshape_matrix_muon+semantic_global_head_muon+auxiliary_adamw\tsemantic_global_node_edge\t5\t2\t2\t20\t%s\t%s\n' \
    "${lane}" "${gpus}" "${CONFIGS[${lane}]}" "${output_dir}" \
    >> "${GROUP_ROOT}/launch_manifest.tsv"

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
      --dataset nabladft \
      --variant "${VARIANTS[${lane}]}" \
      --model-config "${CONFIGS[${lane}]}" \
      --optimizer-type muon \
      --head-type maloq_semantic_global_muon \
      --batch-size 5 \
      --gradient-accumulation-steps 2 \
      --no-distribute-graphs \
      --gpu "${gpus}" \
      --master-port "${port}" \
      --flat-output \
      --output-root "${output_dir}" \
      "${RUN_ARGS[@]}" \
      > "${log_file}" 2>&1 &
  LANE_PIDS[${lane}]=$!
  echo "Started ${lane} on GPUs ${gpus}: PID ${LANE_PIDS[${lane}]}"
  echo "  log: ${log_file}"
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

printf 'lane\tstatus\texit_code\n' > "${GROUP_ROOT}/status.tsv"
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

if [[ "${SCOPE}" == "smoke" && ${overall_status} -eq 0 ]]; then
  case "${GROUP_ROOT}" in
    "${PROJECT_ROOT}"/outputs/nabladft-semglobal-muon-3x2gpu-eb20-mb5-ga2-full-size-smoke-e1-*) ;;
    *)
      echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2
      exit 1
      ;;
  esac
  rm -rf -- "${GROUP_ROOT}"
  echo "Smoke passed; temporary artifacts removed."
else
  echo "Run group retained: ${GROUP_ROOT}"
fi

exit "${overall_status}"
