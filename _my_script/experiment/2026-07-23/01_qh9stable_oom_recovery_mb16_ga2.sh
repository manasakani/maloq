#!/usr/bin/env bash
set -euo pipefail

# Restart only the four QH9Stable lanes that failed with batch-32 CUDA OOM.
# Use micro-batch 16 and two-step gradient accumulation (effective batch 32).

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {validate|smoke|full}" >&2
  exit 2
fi

SCOPE=$1
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py
EXPECTED_HOST=${EXPECTED_HOST:-usr310-gpumngc-02}
ACTUAL_HOST=$(hostname)
NUM_EPOCHS=${NUM_EPOCHS:-80}
MICRO_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2
EFFECTIVE_BATCH_SIZE=32
RUN_ID=$(date +%Y%m%d-%H%M%S)
GROUP_ROOT=${PROJECT_ROOT}/outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-${SCOPE}-seed44-${RUN_ID}

if [[ ! -x "${PY}" || ! -f "${RUNNER}" ]]; then
  echo "SC26 environment or QH9 runner is missing." >&2
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
  *)
    echo "Scope must be validate, smoke, or full." >&2
    exit 2
    ;;
esac

LANES=(
  hamiltonian:maloq
  hamiltonian:maloq-nte
  hamiltonian:qhflow3
  density:maloq-nte
)
declare -A LANE_GPUS=(
  [hamiltonian:maloq]=0
  [hamiltonian:maloq-nte]=1
  [hamiltonian:qhflow3]=2
  [density:maloq-nte]=4
)
declare -A LANE_PORTS=(
  [hamiltonian:maloq]=29701
  [hamiltonian:maloq-nte]=29702
  [hamiltonian:qhflow3]=29703
  [density:maloq-nte]=29704
)
declare -A LANE_PIDS=()

if [[ "${SCOPE}" != "validate" ]]; then
  mapfile -t GPU_MEMORY_USED < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  for key in "${LANES[@]}"; do
    gpu=${LANE_GPUS[${key}]}
    used_mib=${GPU_MEMORY_USED[${gpu}]:-999999}
    if (( used_mib > 1024 )); then
      echo "GPU ${gpu} uses ${used_mib} MiB; refusing to overlap ${key}." >&2
      exit 1
    fi
  done
fi

mkdir -p "${GROUP_ROOT}/logs"
printf 'target\tmodel\tgpu\tmicro_batch\tgradient_accumulation\teffective_batch\toutput\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"
printf 'source_commit\t%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" \
  > "${GROUP_ROOT}/source_revision.tsv"

launch_lane() {
  local key=$1
  local target=${key%%:*}
  local model=${key#*:}
  local gpu=${LANE_GPUS[${key}]}
  local port=${LANE_PORTS[${key}]}
  local output_dir=${GROUP_ROOT}/${target}-${model}
  local log_file=${GROUP_ROOT}/logs/${target}-${model}.log

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${target}" "${model}" "${gpu}" "${MICRO_BATCH_SIZE}" \
    "${GRADIENT_ACCUMULATION_STEPS}" "${EFFECTIVE_BATCH_SIZE}" \
    "${output_dir}" >> "${GROUP_ROOT}/launch_manifest.tsv"

  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${port}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "${PY}" "${RUNNER}" \
      --dataset "qh9-${target}" \
      --variant "${model}" \
      --optimizer-type muon \
      --batch-size "${MICRO_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
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
  echo "Started ${target}/${model} on GPU ${gpu}: PID ${LANE_PIDS[${key}]}"
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
for key in "${LANES[@]}"; do
  launch_lane "${key}"
done

printf 'target\tmodel\tstatus\texit_code\n' > "${GROUP_ROOT}/status.tsv"
overall_status=0
for key in "${LANES[@]}"; do
  target=${key%%:*}
  model=${key#*:}
  if wait "${LANE_PIDS[${key}]}"; then
    exit_code=0
    status=complete
  else
    exit_code=$?
    status=failed
    overall_status=1
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "${target}" "${model}" "${status}" "${exit_code}" \
    >> "${GROUP_ROOT}/status.tsv"
  echo "${target}/${model}: ${status} (exit ${exit_code})"
done
trap - INT TERM

comparison_files=()
for key in "${LANES[@]}"; do
  target=${key%%:*}
  model=${key#*:}
  comparison_file=${GROUP_ROOT}/${target}-${model}/comparison.csv
  [[ -f "${comparison_file}" ]] && comparison_files+=("${comparison_file}")
done
if [[ ${#comparison_files[@]} -gt 0 ]]; then
  awk 'FNR == 1 && NR != 1 { next } { print }' \
    "${comparison_files[@]}" > "${GROUP_ROOT}/comparison.csv"
fi

if [[ "${SCOPE}" == "smoke" && ${overall_status} -eq 0 ]]; then
  case "${GROUP_ROOT}" in
    "${PROJECT_ROOT}"/outputs/qh9stable-oom-recovery-four-lane-mb16-ga2-eb32-smoke-seed44-*) ;;
    *) echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2; exit 1 ;;
  esac
  rm -rf -- "${GROUP_ROOT}"
  echo "Smoke passed; temporary outputs removed: ${GROUP_ROOT}"
else
  echo "Run group retained: ${GROUP_ROOT}"
fi

exit "${overall_status}"
