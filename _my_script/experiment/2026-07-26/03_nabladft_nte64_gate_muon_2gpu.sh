#!/usr/bin/env bash
set -euo pipefail

SCOPE=${1:-validate}
GPUS=${2:-0,1}
EXPECTED_HOST=${EXPECTED_HOST:-}
MASTER_PORT=${MASTER_PORT:-29651}

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
SCRIPT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-26
CONFIG=${SCRIPT_ROOT}/nte64e2_semantic_global_gate_muon_raw_nabladft.yaml
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-25/run_training_workflow_fixed.py

case "${SCOPE}" in
  validate)
    RUN_ARGS=(--validate-only --no-use-wandb)
    SCOPE_SLUG=validate
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
      --wandb-tag baseline:a86zytgu
      --wandb-tag comparison:gate-muon
    )
    SCOPE_SLUG=full-e20
    ;;
  *)
    echo "Usage: $0 {validate|smoke|full} GPU0,GPU1" >&2
    exit 2
    ;;
esac

if [[ -n "${EXPECTED_HOST}" && "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "Expected host ${EXPECTED_HOST}, got $(hostname)." >&2
  exit 1
fi
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]] ||
  (( 10#${MASTER_PORT} < 1 || 10#${MASTER_PORT} > 65535 )); then
  echo "MASTER_PORT must be between 1 and 65535." >&2
  exit 2
fi
if [[ ! -x "${PY}" || ! -x "${MPIRUN}" || ! -f "${RUNNER}" ]]; then
  echo "SC26 interpreter, mpirun, or fixed runner is missing." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ "${SCOPE}" == "validate" ]]; then
  exec "${PY}" "${RUNNER}" \
    --dataset nabladft \
    --variant maloq-nte \
    --model-config "${CONFIG}" \
    --optimizer-type muon \
    --head-type maloq_semantic_global_gate_muon \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --no-distribute-graphs \
    --gpu 0 \
    --master-port "${MASTER_PORT}" \
    "${RUN_ARGS[@]}"
fi

IFS=',' read -r -a GPU_INDICES <<< "${GPUS}"
if [[ ${#GPU_INDICES[@]} -ne 2 ||
  "${GPU_INDICES[0]}" == "${GPU_INDICES[1]}" ]]; then
  echo "Exactly two different comma-separated GPU indices are required." >&2
  exit 2
fi
mapfile -t GPU_MEMORY_USED < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
declare -A SEEN_GPUS=()
for gpu in "${GPU_INDICES[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ || -z "${GPU_MEMORY_USED[${gpu}]:-}" ]]; then
    echo "Invalid GPU index: ${gpu}" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "Duplicate GPU index: ${gpu}" >&2
    exit 2
  fi
  SEEN_GPUS[${gpu}]=1
  if (( GPU_MEMORY_USED[gpu] > 1024 )); then
    echo "GPU ${gpu} already uses ${GPU_MEMORY_USED[gpu]} MiB; refusing overlap." >&2
    exit 1
  fi
done

RUN_ID=$(date +%Y%m%d-%H%M%S)
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/nabladft-nte64-sghead-gatemuon-raw-v3-2gpu-eb20-mb5-ga2-${SCOPE_SLUG}-${RUN_ID}
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output already exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
printf 'model\tbaseline_wandb\tgpus\tmicro_batch\tworld_size\tgrad_acc\teffective_batch\thead_routing\tgate_routing\tconfig\n' \
  > "${OUTPUT_ROOT}/launch_manifest.tsv"
printf 'nte64e2\ta86zytgu\t%s\t5\t2\t2\t20\tsemantic_global_node_edge\tsemantic_scalar_gate\t%s\n' \
  "${GPUS}" "${CONFIG}" >> "${OUTPUT_ROOT}/launch_manifest.tsv"

set +e
env \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${MASTER_PORT}" \
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
    --model-config "${CONFIG}" \
    --optimizer-type muon \
    --head-type maloq_semantic_global_gate_muon \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --no-distribute-graphs \
    --gpu "${GPUS}" \
    --master-port "${MASTER_PORT}" \
    --flat-output \
    --output-root "${OUTPUT_ROOT}/run" \
    "${RUN_ARGS[@]}" \
    > "${OUTPUT_ROOT}/train.log" 2>&1
EXIT_CODE=$?
set -e

if [[ "${SCOPE}" == "smoke" && ${EXIT_CODE} -eq 0 ]]; then
  case "${OUTPUT_ROOT}" in
    "${PROJECT_ROOT}"/outputs/nabladft-nte64-sghead-gatemuon-raw-v3-2gpu-eb20-mb5-ga2-full-size-smoke-e1-*) ;;
    *)
      echo "Refusing to remove unexpected smoke path: ${OUTPUT_ROOT}" >&2
      exit 1
      ;;
  esac
  rm -rf -- "${OUTPUT_ROOT}"
  echo "Smoke passed; temporary artifacts removed."
elif [[ ${EXIT_CODE} -ne 0 ]]; then
  echo "Run failed; evidence retained at ${OUTPUT_ROOT}" >&2
else
  echo "Run complete: ${OUTPUT_ROOT}"
fi

exit "${EXIT_CODE}"
