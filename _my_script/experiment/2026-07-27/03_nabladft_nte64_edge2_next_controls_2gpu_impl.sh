#!/usr/bin/env bash
set -euo pipefail

SCOPE=${1:-validate}
GPUS=${2:-0,1}
EXPECTED_HOST=${EXPECTED_HOST:-}

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
SOURCE_ROOT=${SC26_SOURCE_ROOT:-${PROJECT_ROOT}}
SOURCE_MANIFEST=${SC26_SOURCE_MANIFEST:-}
SOURCE_ARCHIVE=${SC26_SOURCE_ARCHIVE:-}
SOURCE_FINGERPRINT=${SC26_SOURCE_FINGERPRINT:-}
QUEUE_SOURCE_SNAPSHOT=${SC26_QUEUE_SOURCE_SNAPSHOT:-}
SCRIPT_ROOT=${SOURCE_ROOT}/_my_script/experiment/2026-07-27
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
PRTERUN=${ENV_ROOT}/bin/prterun
RUNNER=${SOURCE_ROOT}/_my_script/experiment/2026-07-25/run_training_workflow_fixed.py
NABLA_DB=/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db
CONFIG=${SC26_MODEL_CONFIG:-${SCRIPT_ROOT}/nte64e2_edge2postres_gaussian_width2_nabladft.yaml}
WANDB_DISPLAY_NAME=${SC26_WANDB_DISPLAY_NAME:-"NablaDFT | NTE-64/2 | MatrixMuon+AuxAdamW | RAW | QHFcond | Edge2PostResidual+GaussianWidth2 | V1"}
WANDB_GROUP=${SC26_WANDB_GROUP:-nabla-nte64-edge2-next-layer-controls-v1}
RUN_PREFIX=${SC26_RUN_PREFIX:-nabladft-nte64-edge2-gw2-v1-2gpu-eb20-mb5-ga2}

if [[ ! -x "${PY}" || ! -x "${MPIRUN}" || ! -x "${PRTERUN}" || ! -f "${RUNNER}" ]]; then
  echo "SC26 Python, mpirun/prterun, or fixed training runner is missing." >&2
  exit 1
fi
if [[ ! -r "${NABLA_DB}" ]]; then
  echo "NablaDFT database is missing or unreadable: ${NABLA_DB}" >&2
  exit 1
fi
if [[ ! -r "${CONFIG}" ]]; then
  echo "Model config is missing or unreadable: ${CONFIG}" >&2
  exit 1
fi
if [[ "${SOURCE_ROOT}" != "${PROJECT_ROOT}" ]]; then
  if [[ ! -d "${SOURCE_ROOT}/src/maloq" || ! -r "${SOURCE_MANIFEST}" ]]; then
    echo "Frozen source tree or checksum manifest is missing." >&2
    exit 1
  fi
  (
    cd "${SOURCE_ROOT}"
    sha256sum --check --quiet "${SOURCE_MANIFEST}"
  )
  if [[ -n "${SOURCE_ARCHIVE}" && ! -r "${SOURCE_ARCHIVE}" ]]; then
    echo "Frozen source archive is missing: ${SOURCE_ARCHIVE}" >&2
    exit 1
  fi
  if [[ -n "${SOURCE_FINGERPRINT}" ]]; then
    ACTUAL_SOURCE_FINGERPRINT=$(
      "${PY}" - "${SOURCE_ROOT}/PROVENANCE/source-snapshot/source-final.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["fingerprint"])
PY
    )
    if [[ "${ACTUAL_SOURCE_FINGERPRINT}" != "${SOURCE_FINGERPRINT}" ]]; then
      echo "Frozen source fingerprint mismatch." >&2
      exit 1
    fi
  fi
fi
if [[ -n "${QUEUE_SOURCE_SNAPSHOT}" ]]; then
  if [[ ! -d "${QUEUE_SOURCE_SNAPSHOT}" || ! -r "${QUEUE_SOURCE_SNAPSHOT}/package.sha256" ]]; then
    echo "Queue source snapshot or checksum package is missing." >&2
    exit 1
  fi
  (
    cd "${QUEUE_SOURCE_SNAPSHOT}"
    sha256sum --check --quiet package.sha256
  )
fi

export OPAL_PREFIX="${ENV_ROOT}"
export PRTE_PREFIX="${ENV_ROOT}"
export PMIX_PREFIX="${ENV_ROOT}"
export OMPI_PRTERUN="${PRTERUN}"
if ! "${MPIRUN}" --version >/dev/null 2>&1; then
  echo "Relocated mpirun preflight failed." >&2
  exit 1
fi

case "${SCOPE}" in
  prepare)
    echo "No preprocessing is required; NablaDFT and the fixed experiment inputs are ready."
    exit 0
    ;;
  validate)
    RUN_ARGS=(
      --validate-only
      --no-use-wandb
      --wandb-run-name "${WANDB_DISPLAY_NAME}"
    )
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
      --wandb-run-name "${WANDB_DISPLAY_NAME}"
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
      --wandb-group "${WANDB_GROUP}"
      --wandb-run-name "${WANDB_DISPLAY_NAME}"
    )
    SCOPE_SLUG=full-e20
    ;;
  *)
    echo "Usage: $0 {prepare|validate|smoke|full} GPU0,GPU1" >&2
    exit 2
    ;;
esac

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
  exec env \
    PYTHONPATH="${SOURCE_ROOT}/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${PY}" "${RUNNER}" \
    --dataset nabladft \
    --variant maloq-nte \
    --model-config "${CONFIG}" \
    --optimizer-type muon \
    --head-type maloq_muon \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --no-distribute-graphs \
    --gpu 0 \
    --master-port "${PORT}" \
    "${RUN_ARGS[@]}"
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

RUN_ID=$(date +%Y%m%d-%H%M%S)-$$
RUN_BASENAME=${RUN_PREFIX}-${SCOPE_SLUG}-${RUN_ID}
OUTPUT_ALIAS=
if [[ "${SOURCE_ROOT}" == "${PROJECT_ROOT}" ]]; then
  OUTPUT_BASE=${PROJECT_ROOT}/outputs
else
  OUTPUT_BASE=${SOURCE_ROOT}/outputs
  OUTPUT_ALIAS=${PROJECT_ROOT}/outputs/${RUN_BASENAME}
fi
OUTPUT_ROOT=${OUTPUT_BASE}/${RUN_BASENAME}
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output already exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
if [[ -n "${OUTPUT_ALIAS}" ]]; then
  if [[ -e "${OUTPUT_ALIAS}" || -L "${OUTPUT_ALIAS}" ]]; then
    echo "Output alias already exists: ${OUTPUT_ALIAS}" >&2
    exit 1
  fi
  ln -s "${OUTPUT_ROOT}" "${OUTPUT_ALIAS}"
fi
if [[ "${SOURCE_ROOT}" != "${PROJECT_ROOT}" ]]; then
  mkdir -p "${OUTPUT_ROOT}/source-snapshot"
  cp -a -- \
    "${SOURCE_ROOT}/PROVENANCE/source-snapshot/." \
    "${OUTPUT_ROOT}/source-snapshot/"
  cp -- "${SOURCE_MANIFEST}" "${OUTPUT_ROOT}/SOURCE_TREE.sha256"
  if [[ -n "${SOURCE_ARCHIVE}" ]]; then
    cp -- "${SOURCE_ARCHIVE}" "${OUTPUT_ROOT}/"
  fi
  printf '%s\n' \
    "source_root=${SOURCE_ROOT}" \
    "source_manifest=${SOURCE_MANIFEST}" \
    "source_archive=${SOURCE_ARCHIVE}" \
    "source_fingerprint=${SOURCE_FINGERPRINT}" \
    > "${OUTPUT_ROOT}/runtime-source.txt"
  if [[ -n "${SC26_QUEUE_JOB_ID:-}" ]]; then
    QUEUE_REQUEST=${PROJECT_ROOT}/outputs/experiment-queue/jobs/${SC26_QUEUE_JOB_ID}/request.json
    if [[ -r "${QUEUE_REQUEST}" ]]; then
      cp -- "${QUEUE_REQUEST}" "${OUTPUT_ROOT}/queue-request.json"
    fi
  fi
fi
if [[ -n "${QUEUE_SOURCE_SNAPSHOT}" ]]; then
  mkdir -p "${OUTPUT_ROOT}/queue-source-snapshot"
  cp -a -- "${QUEUE_SOURCE_SNAPSHOT}/." "${OUTPUT_ROOT}/queue-source-snapshot/"
fi
: > "${OUTPUT_ROOT}/train.log"

set +e
env \
  PYTHONPATH="${SOURCE_ROOT}/src" \
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${PORT}" \
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
    --head-type maloq_muon \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --no-distribute-graphs \
    --gpu "${GPUS}" \
    --master-port "${PORT}" \
    --flat-output \
    --output-root "${OUTPUT_ROOT}/run" \
    "${RUN_ARGS[@]}" \
    >> "${OUTPUT_ROOT}/train.log" 2>&1
EXIT_CODE=$?
set -e

if [[ "${SCOPE}" == "smoke" && ${EXIT_CODE} -eq 0 ]]; then
  case "${OUTPUT_ROOT}" in
    "${OUTPUT_BASE}"/"${RUN_PREFIX}"-full-size-smoke-e1-*) ;;
    *)
      echo "Refusing to remove unexpected smoke path: ${OUTPUT_ROOT}" >&2
      exit 1
      ;;
  esac
  rm -rf -- "${OUTPUT_ROOT}"
  if [[ -n "${OUTPUT_ALIAS}" && -L "${OUTPUT_ALIAS}" ]]; then
    unlink -- "${OUTPUT_ALIAS}"
  fi
  echo "Smoke passed; temporary artifacts removed."
elif [[ ${EXIT_CODE} -ne 0 ]]; then
  echo "Run failed; evidence retained at ${OUTPUT_ROOT}" >&2
else
  echo "Run complete: ${OUTPUT_ALIAS:-${OUTPUT_ROOT}}"
fi

exit "${EXIT_CODE}"
