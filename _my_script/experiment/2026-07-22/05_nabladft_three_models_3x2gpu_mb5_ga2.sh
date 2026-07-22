#!/usr/bin/env bash
set -euo pipefail

# Run one model or all three matched NablaDFT lanes.
#
# GPU mapping:
#   MALOQ     -> 0,1
#   MALOQ-NTE -> 2,3
#   QHFlow3   -> 6,7
#
# Each rank uses micro-batch 5 and accumulates two micro-batches:
#   5 molecules/rank * 2 ranks * 2 accumulation steps = effective batch 20.

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {maloq|maloq-nte|qhflow3|all} {smoke|full}" >&2
  exit 2
fi

SELECTION=$1
SCOPE=$2

case "${SELECTION}" in
  maloq|maloq-nte|qhflow3)
    MODELS=("${SELECTION}")
    ;;
  all)
    MODELS=(maloq maloq-nte qhflow3)
    ;;
  *)
    echo "Model must be maloq, maloq-nte, qhflow3, or all." >&2
    exit 2
    ;;
esac

case "${SCOPE}" in
  smoke)
    RUN_ARGS=(
      --smoke
      --full-size-smoke
      --keep-smoke-output
      --num-epochs 1
      --num-train 20
      --num-val 20
      --num-test 0
    )
    TRACKING_ARGS=(--no-use-wandb)
    SCOPE_SLUG=full-size-smoke-e1
    ;;
  full)
    RUN_ARGS=(
      --num-epochs 20
      --num-train 12081
      --num-val 64
      --num-test 0
    )
    TRACKING_ARGS=(
      --use-wandb
      --wandb-project maloq-nablaDFT
      --wandb-entity kaist-korea
      --wandb-mode online
      --wandb-log-every-n-steps 10
    )
    SCOPE_SLUG=full-e20
    ;;
  *)
    echo "Scope must be smoke or full." >&2
    exit 2
    ;;
esac

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py
RUN_ID=$(date +%Y%m%d-%H%M%S)
GROUP_ROOT=${PROJECT_ROOT}/outputs/nabladft-three-model-parallel-3x2gpu-eb20-mb5-ga2-${SCOPE_SLUG}-seed44-${RUN_ID}

declare -A MODEL_GPUS=(
  [maloq]=0,1
  [maloq-nte]=2,3
  [qhflow3]=6,7
)
declare -A MODEL_PORTS=(
  [maloq]=29561
  [maloq-nte]=29562
  [qhflow3]=29563
)
declare -A MODEL_PIDS=()

if [[ -e "${GROUP_ROOT}" ]]; then
  echo "Output group already exists: ${GROUP_ROOT}" >&2
  exit 1
fi
mkdir -p "${GROUP_ROOT}/logs"

printf 'model\tgpus\tmaster_port\tmicro_batch\taccumulation\teffective_batch\toutput\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"

launch_model() {
  local model=$1
  local gpus=${MODEL_GPUS[${model}]}
  local port=${MODEL_PORTS[${model}]}
  local output_dir=${GROUP_ROOT}/${model}
  local log_file=${GROUP_ROOT}/logs/${model}.log

  printf '%s\t%s\t%s\t5\t2\t20\t%s\n' \
    "${model}" "${gpus}" "${port}" "${output_dir}" \
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
      --variant "${model}" \
      --optimizer-type muon \
      --batch-size 5 \
      --gradient-accumulation-steps 2 \
      --no-distribute-graphs \
      "${TRACKING_ARGS[@]}" \
      --gpu "${gpus}" \
      --master-port "${port}" \
      --flat-output \
      "${RUN_ARGS[@]}" \
      --output-root "${output_dir}" \
      > "${log_file}" 2>&1 &

  MODEL_PIDS[${model}]=$!
  echo "Started ${model} on GPUs ${gpus}: PID ${MODEL_PIDS[${model}]}"
  echo "  log: ${log_file}"
}

stop_children() {
  local model
  for model in "${MODELS[@]}"; do
    if [[ -n "${MODEL_PIDS[${model}]:-}" ]]; then
      kill "${MODEL_PIDS[${model}]}" 2>/dev/null || true
    fi
  done
}
trap stop_children INT TERM

cd "${PROJECT_ROOT}"
for model in "${MODELS[@]}"; do
  launch_model "${model}"
done

printf 'model\tstatus\texit_code\n' > "${GROUP_ROOT}/status.tsv"
overall_status=0
for model in "${MODELS[@]}"; do
  if wait "${MODEL_PIDS[${model}]}"; then
    exit_code=0
    status=complete
  else
    exit_code=$?
    status=failed
    overall_status=1
  fi
  printf '%s\t%s\t%s\n' "${model}" "${status}" "${exit_code}" \
    >> "${GROUP_ROOT}/status.tsv"
  echo "${model}: ${status} (exit ${exit_code})"
done
trap - INT TERM

comparison_files=()
for model in "${MODELS[@]}"; do
  comparison_file=${GROUP_ROOT}/${model}/comparison.csv
  if [[ -f "${comparison_file}" ]]; then
    comparison_files+=("${comparison_file}")
  fi
done

if [[ ${#comparison_files[@]} -gt 0 ]]; then
  awk 'FNR == 1 && NR != 1 { next } { print }' \
    "${comparison_files[@]}" > "${GROUP_ROOT}/comparison.csv"
  echo "Comparison table: ${GROUP_ROOT}/comparison.csv"
fi

if [[ "${SCOPE}" == "smoke" && ${overall_status} -eq 0 ]]; then
  case "${GROUP_ROOT}" in
    "${PROJECT_ROOT}"/outputs/nabladft-three-model-parallel-3x2gpu-eb20-mb5-ga2-full-size-smoke-e1-seed44-*) ;;
    *)
      echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2
      exit 1
      ;;
  esac
  rm -rf -- "${GROUP_ROOT}"
  echo "Smoke passed; temporary outputs removed: ${GROUP_ROOT}"
else
  echo "Run group retained: ${GROUP_ROOT}"
fi

exit "${overall_status}"
