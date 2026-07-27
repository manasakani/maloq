#!/usr/bin/env bash
set -euo pipefail

SCOPE=${1:-validate}
VARIANT=${2:-maloq-nte}
GPUS=${3:-0,1}
EXPECTED_HOST=${EXPECTED_HOST:-}

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-25/run_training_workflow_fixed.py
DOWNLOAD=${PROJECT_ROOT}/_auto_script/nabladft_v2_download/download_nabladft_v2.sh
VERIFY=${PROJECT_ROOT}/_auto_script/nabladft_v2_download/verify_nabladft_v2.py
DATASET_ROOT=/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases
TRAIN_DB=${DATASET_ROOT}/train_10k.db
TEST_DB=${DATASET_ROOT}/test_2k_conformers.db
VAL_ROWS=${NABLADFT_VAL_ROWS:-64}

case "${VARIANT}" in
  maloq)
    CONFIG=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/maloq_muon_head_nabladft.yaml
    DISPLAY_MODEL=MALOQ
    ;;
  maloq-nte)
    CONFIG=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/maloq_nte_muon_head_nabladft.yaml
    DISPLAY_MODEL=MALOQ-NTE
    ;;
  qhflow3)
    CONFIG=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/qhflow3_clean_muon_head_nabladft.yaml
    DISPLAY_MODEL=QHFlow3
    ;;
  *)
    echo "Variant must be one of: maloq, maloq-nte, qhflow3" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PY}" || ! -x "${MPIRUN}" || ! -f "${RUNNER}" ]]; then
  echo "SC26 Python, mpirun, or fixed training runner is missing." >&2
  exit 1
fi
if [[ ! -r "${CONFIG}" || ! -x "${DOWNLOAD}" || ! -r "${VERIFY}" ]]; then
  echo "Model config or NablaDFT helper is missing." >&2
  exit 1
fi

case "${SCOPE}" in
  prepare)
    exec "${DOWNLOAD}" download
    ;;
  validate|smoke|full) ;;
  *)
    echo "Usage: $0 {prepare|validate|smoke|full} {maloq|maloq-nte|qhflow3} GPU0,GPU1" >&2
    exit 2
    ;;
esac

if [[ ! -r "${TRAIN_DB}" ]]; then
  echo "NablaDFT 10k training DB is missing: ${TRAIN_DB}" >&2
  echo "Run: ${DOWNLOAD} download" >&2
  exit 1
fi

TRAIN_ROWS=$(
  "${PY}" "${VERIFY}" \
    --root "${DATASET_ROOT}" \
    --artifact train_10k \
    --print-rows
)
if [[ ! "${TRAIN_ROWS}" =~ ^[0-9]+$ ]]; then
  echo "Could not read a numeric row count from ${TRAIN_DB}: ${TRAIN_ROWS}" >&2
  exit 1
fi
if [[ ! "${VAL_ROWS}" =~ ^[1-9][0-9]*$ ]] || (( VAL_ROWS >= TRAIN_ROWS )); then
  echo "NABLADFT_VAL_ROWS must be positive and smaller than ${TRAIN_ROWS}." >&2
  exit 2
fi
NUM_TRAIN=$((TRAIN_ROWS - VAL_ROWS))
WANDB_DISPLAY_NAME="NablaDFT-10k | ${DISPLAY_MODEL} | Muon-head | RAW | V1"

if [[ -n "${EXPECTED_HOST}" && "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "Expected host ${EXPECTED_HOST}, got $(hostname)." >&2
  exit 1
fi

if [[ -n "${MASTER_PORT:-}" ]]; then
  PORT=${MASTER_PORT}
else
  PORT=$(
    "${PY}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
  )
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "MASTER_PORT must be an integer in [1024, 65535], got ${PORT}." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

if [[ "${SCOPE}" == "validate" ]]; then
  if [[ -r "${TEST_DB}" ]]; then
    "${PY}" "${VERIFY}" \
      --root "${DATASET_ROOT}" \
      --artifact test_2k_conformers >/dev/null
  else
    echo "Tiny-conformer test DB is still missing: ${TEST_DB}" >&2
    exit 1
  fi
  exec env PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" "${RUNNER}" \
    --dataset nabladft \
    --variant "${VARIANT}" \
    --model-config "${CONFIG}" \
    --dbpath "${TRAIN_DB}" \
    --num-train "${NUM_TRAIN}" \
    --num-val "${VAL_ROWS}" \
    --num-test 0 \
    --optimizer-type muon \
    --head-type maloq_muon \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --no-distribute-graphs \
    --gpu 0 \
    --master-port "${PORT}" \
    --validate-only \
    --no-use-wandb \
    --wandb-run-name "${WANDB_DISPLAY_NAME}"
fi

IFS=',' read -r -a GPU_INDICES <<< "${GPUS}"
if [[ ${#GPU_INDICES[@]} -ne 2 ]]; then
  echo "Exactly two comma-separated GPU indices are required." >&2
  exit 2
fi
mapfile -t GPU_MEMORY_USED < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
declare -A SEEN_GPUS=()
for gpu in "${GPU_INDICES[@]}"; do
  if [[ ! "${gpu}" =~ ^(0|[1-9][0-9]*)$ || -z "${GPU_MEMORY_USED[${gpu}]:-}" ]]; then
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

if [[ "${SCOPE}" == "smoke" ]]; then
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
else
  RUN_ARGS=(
    --num-epochs 20
    --num-train "${NUM_TRAIN}"
    --num-val "${VAL_ROWS}"
    --num-test 0
    --use-wandb
    --wandb-project maloq-nablaDFT
    --wandb-entity kaist-korea
    --wandb-mode online
    --wandb-log-every-n-steps 10
    --wandb-group nabladft-10k-muon-head-v1
  )
  SCOPE_SLUG=full-e20
fi

RUN_ID=$(date +%Y%m%d-%H%M%S)-$$
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/nabladft-10k-${VARIANT}-muon-head-2gpu-eb20-mb5-ga2-${SCOPE_SLUG}-${RUN_ID}
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output already exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

set +e
env \
  PYTHONPATH="${PROJECT_ROOT}/src" \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${PORT}" \
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
    --variant "${VARIANT}" \
    --model-config "${CONFIG}" \
    --dbpath "${TRAIN_DB}" \
    --optimizer-type muon \
    --head-type maloq_muon \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --no-distribute-graphs \
    --gpu "${GPUS}" \
    --master-port "${PORT}" \
    --flat-output \
    --output-root "${OUTPUT_ROOT}/run" \
    --wandb-run-name "${WANDB_DISPLAY_NAME}" \
    --wandb-tag split:train10k \
    --wandb-tag test:test2k-conformers \
    "${RUN_ARGS[@]}" \
    >"${OUTPUT_ROOT}/train.log" 2>&1
EXIT_CODE=$?
set -e

if [[ "${SCOPE}" == "smoke" && ${EXIT_CODE} -eq 0 ]]; then
  case "${OUTPUT_ROOT}" in
    "${PROJECT_ROOT}"/outputs/nabladft-10k-*-muon-head-2gpu-eb20-mb5-ga2-full-size-smoke-e1-*) ;;
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
