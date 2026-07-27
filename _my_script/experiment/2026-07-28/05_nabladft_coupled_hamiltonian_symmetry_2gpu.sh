#!/usr/bin/env bash
set -euo pipefail

SCOPE=${1:-validate}
LANE=${2:-all}
GPUS=${3:-}
EXPECTED_HOST=${EXPECTED_HOST:-}

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
SCRIPT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-28
BASE_SCRIPT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-27
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${SCRIPT_ROOT}/run_nabladft_coupled_hamiltonian_symmetry.py
BASE_RUNNER=${BASE_SCRIPT_ROOT}/run_nabladft_v2_ofat.py
BASE_CONFIG=${BASE_SCRIPT_ROOT}/nabladft_v2_ofat_common.yaml
NABLA_DB=/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db
SCALE_SHIFT=${PROJECT_ROOT}/outputs/scale-shift-statistics/nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt
SCALE_SHIFT_SHA256=375167ad551fb0b60dbe9cd049a4995276b54ce075e09906639ef3daa4f79475
RUN_PREFIX=nabladft-v2-ofat
FEATURE_SLUG=coupled-hamiltonian-symmetry

LANES=(
  maloq-e3-muon-shift
  ntev2-e3-muon-shift
  qhflow3-e3-muon-shift
)

usage() {
  echo "Usage: $0 {prepare|validate|smoke|full} {all|LANE} [GPU0,GPU1]" >&2
}

is_lane() {
  local candidate=$1
  local known
  for known in "${LANES[@]}"; do
    if [[ "${candidate}" == "${known}" ]]; then
      return 0
    fi
  done
  return 1
}

for required in \
  "${PY}" \
  "${MPIRUN}" \
  "${RUNNER}" \
  "${BASE_RUNNER}" \
  "${BASE_CONFIG}" \
  "${NABLA_DB}" \
  "${SCALE_SHIFT}"; do
  if [[ ! -r "${required}" ]]; then
    echo "Required experiment input is missing or unreadable: ${required}" >&2
    exit 1
  fi
done
if [[ ! -x "${PY}" || ! -x "${MPIRUN}" || ! -x "${RUNNER}" ]]; then
  echo "Python, mpirun, and the feature runner must be executable." >&2
  exit 1
fi

ACTUAL_SCALE_SHIFT_SHA256=$(sha256sum "${SCALE_SHIFT}" | awk '{print $1}')
if [[ "${ACTUAL_SCALE_SHIFT_SHA256}" != "${SCALE_SHIFT_SHA256}" ]]; then
  echo "SHIFT artifact SHA-256 mismatch." >&2
  echo "expected=${SCALE_SHIFT_SHA256}" >&2
  echo "actual=${ACTUAL_SCALE_SHIFT_SHA256}" >&2
  exit 1
fi

case "${SCOPE}" in
  prepare)
    if [[ "${LANE}" != "all" ]]; then
      echo "prepare validates the complete suite; use lane 'all'." >&2
      exit 2
    fi
    ;;
  validate)
    if [[ "${LANE}" != "all" ]] && ! is_lane "${LANE}"; then
      echo "Unknown lane: ${LANE}" >&2
      usage
      exit 2
    fi
    ;;
  smoke|full)
    if ! is_lane "${LANE}"; then
      echo "smoke/full requires one concrete lane, got: ${LANE}" >&2
      usage
      exit 2
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ -n "${EXPECTED_HOST}" && "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "Expected host ${EXPECTED_HOST}, got $(hostname)." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ "${SCOPE}" == "prepare" || "${SCOPE}" == "validate" ]]; then
  exec env \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${PY}" "${RUNNER}" \
      --base-config "${BASE_CONFIG}" \
      --lane "${LANE}" \
      --scope validate
fi

if [[ -z "${GPUS}" ]]; then
  echo "smoke/full requires exactly two comma-separated GPU indices." >&2
  usage
  exit 2
fi
IFS=',' read -r -a GPU_INDICES <<< "${GPUS}"
if [[ ${#GPU_INDICES[@]} -ne 2 ]]; then
  echo "Exactly two comma-separated GPU indices are required: ${GPUS}" >&2
  exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPU_INDICES[@]}"; do
  if [[ ! "${gpu}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "Invalid GPU index: ${gpu}" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "Duplicate GPU index: ${gpu}" >&2
    exit 2
  fi
  SEEN_GPUS[${gpu}]=1

  GPU_STATS=$(nvidia-smi -i "${gpu}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null) || {
      echo "GPU index ${gpu} is not available." >&2
      exit 2
    }
  IFS=',' read -r MEMORY_USED GPU_UTILIZATION <<< "${GPU_STATS}"
  MEMORY_USED=${MEMORY_USED//[[:space:]]/}
  GPU_UTILIZATION=${GPU_UTILIZATION//[[:space:]]/}
  if [[ ! "${MEMORY_USED}" =~ ^[0-9]+$ || ! "${GPU_UTILIZATION}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse live status for GPU ${gpu}: ${GPU_STATS}" >&2
    exit 1
  fi

  ACTIVE_PIDS=$(nvidia-smi -i "${gpu}" \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {print $1}')
  if [[ -n "${ACTIVE_PIDS}" ]]; then
    echo "GPU ${gpu} has active compute PIDs: ${ACTIVE_PIDS//$'\n'/,}" >&2
    exit 1
  fi
  if (( MEMORY_USED > 1024 || GPU_UTILIZATION > 10 )); then
    echo "GPU ${gpu} is materially busy: ${MEMORY_USED} MiB, ${GPU_UTILIZATION}%." >&2
    exit 1
  fi
done

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

if [[ "${SCOPE}" == "smoke" ]]; then
  SCOPE_SLUG=smoke-e1
else
  SCOPE_SLUG=full-e20
fi
RUN_FEATURE_SLUG=${FEATURE_SLUG}
if [[ "${LANE}" == "qhflow3-e3-muon-shift" ]]; then
  RUN_FEATURE_SLUG=${FEATURE_SLUG}-grid10x11
fi
RUN_ID=$(date +%Y%m%d-%H%M%S)-$$
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/${RUN_PREFIX}-${LANE}-${RUN_FEATURE_SLUG}-2gpu-eb20-mb5-ga2-${SCOPE_SLUG}-${RUN_ID}
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output already exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

unset MALOQ_FIXED_RESUME_FROM
unset MALOQ_FIXED_ALLOW_CONFIG_MISMATCH
unset MALOQ_FIXED_STOP_AFTER_EPOCH

set +e
env \
  PYTHONPATH="${PROJECT_ROOT}/src" \
  PYTHONDONTWRITEBYTECODE=1 \
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
    --base-config "${BASE_CONFIG}" \
    --lane "${LANE}" \
    --scope "${SCOPE}" \
    --output-root "${OUTPUT_ROOT}/run" \
    >"${OUTPUT_ROOT}/train.log" 2>&1
EXIT_CODE=$?
set -e

if [[ "${SCOPE}" == "smoke" && ${EXIT_CODE} -eq 0 ]]; then
  case "${OUTPUT_ROOT}" in
    "${PROJECT_ROOT}"/outputs/"${RUN_PREFIX}"-"${LANE}"-"${RUN_FEATURE_SLUG}"-2gpu-eb20-mb5-ga2-smoke-e1-*) ;;
    *)
      echo "Refusing to remove unexpected smoke output: ${OUTPUT_ROOT}" >&2
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
