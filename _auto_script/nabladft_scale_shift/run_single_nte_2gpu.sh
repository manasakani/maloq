#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT_NAME:?Wrapper must set EXPERIMENT_NAME}"
: "${MODEL_CONFIG:?Wrapper must set MODEL_CONFIG}"
: "${DEFAULT_GPUS:?Wrapper must set DEFAULT_GPUS}"
: "${DEFAULT_MASTER_PORT:?Wrapper must set DEFAULT_MASTER_PORT}"
: "${SCALE_SHIFT_ENABLED:?Wrapper must set SCALE_SHIFT_ENABLED}"

MODEL_VARIANT=${MODEL_VARIANT:-maloq-nte}
MODEL_HEAD_TYPE=${MODEL_HEAD_TYPE:-}
COMPACT_NAMES=${COMPACT_NAMES:-0}

case "${MODEL_VARIANT}" in
  maloq | maloq-nte | qhflow3) ;;
  *)
    echo "MODEL_VARIANT must be maloq, maloq-nte, or qhflow3." >&2
    exit 2
    ;;
esac
case "${MODEL_HEAD_TYPE}" in
  "" | maloq | maloq_muon | static_te) ;;
  *)
    echo "MODEL_HEAD_TYPE must be empty, maloq, maloq_muon, or static_te." >&2
    exit 2
    ;;
esac
case "${COMPACT_NAMES}" in
  0 | 1) ;;
  *)
    echo "COMPACT_NAMES must be 0 or 1." >&2
    exit 2
    ;;
esac

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {prepare|validate|smoke|full} [GPU0,GPU1]" >&2
  exit 2
fi

SCOPE=$1
GPU_PAIR=${2:-${GPUS:-${DEFAULT_GPUS}}}
MASTER_PORT=${MASTER_PORT:-${DEFAULT_MASTER_PORT}}
EXPECTED_HOST=${EXPECTED_HOST:-}

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py
PREPROCESSOR=${PROJECT_ROOT}/_auto_script/nabladft_scale_shift/process_nabladft_scale_shift.py
NABLA_DB=/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db
SCALE_SHIFT_STATS=${PROJECT_ROOT}/outputs/scale-shift-statistics/nabladft-train12081-fock-l0-mean-std-rcut8-float32.pt

IFS=',' read -r -a GPU_INDICES <<< "${GPU_PAIR}"
if [[ ${#GPU_INDICES[@]} -ne 2 ]]; then
  echo "GPU pair must contain exactly two comma-separated indices." >&2
  exit 2
fi
if [[ "${GPU_INDICES[0]}" == "${GPU_INDICES[1]}" ]]; then
  echo "The two data-parallel ranks require different GPUs." >&2
  exit 2
fi
for gpu in "${GPU_INDICES[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index: ${gpu}" >&2
    exit 2
  fi
done
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]] ||
  (( 10#${MASTER_PORT} < 1 || 10#${MASTER_PORT} > 65535 )); then
  echo "MASTER_PORT must be between 1 and 65535." >&2
  exit 2
fi

if [[ ! -x "${PY}" || ! -x "${MPIRUN}" || ! -f "${RUNNER}" ]]; then
  echo "SC26 environment, mpirun, or experiment runner is missing." >&2
  exit 1
fi
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model config is missing: ${MODEL_CONFIG}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
if [[ "${SCOPE}" == "prepare" ]]; then
  if [[ "${SCALE_SHIFT_ENABLED}" != "1" ]]; then
    echo "This no-scale-shift experiment does not require preparation."
    exit 0
  fi
  mapfile -t GPU_MEMORY_USED < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  prepare_gpu=${GPU_INDICES[0]}
  if [[ -z "${GPU_MEMORY_USED[${prepare_gpu}]:-}" ]]; then
    echo "GPU ${prepare_gpu} does not exist on $(hostname)." >&2
    exit 2
  fi
  if (( GPU_MEMORY_USED[prepare_gpu] > 1024 )); then
    echo "GPU ${prepare_gpu} uses ${GPU_MEMORY_USED[prepare_gpu]} MiB; refusing overlap." >&2
    exit 1
  fi
  env \
    CUDA_VISIBLE_DEVICES="${prepare_gpu}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "${PY}" "${PREPROCESSOR}" \
      --dbpath "${NABLA_DB}" \
      --output "${SCALE_SHIFT_STATS}" \
      --num-train 12081 \
      --rcut 8.0 \
      --batch-size 64 \
      --dtype float32
  exit 0
fi

if [[ "${SCALE_SHIFT_ENABLED}" == "1" && "${SCOPE}" != "validate" &&
  ! -f "${SCALE_SHIFT_STATS}" ]]; then
  echo "Train-only scale-shift statistics are missing: ${SCALE_SHIFT_STATS}" >&2
  echo "Run this script with 'prepare ${GPU_PAIR}' first." >&2
  exit 1
fi

SCALE_SHIFT_ARGS=()
if [[ "${SCALE_SHIFT_ENABLED}" == "1" && -f "${SCALE_SHIFT_STATS}" ]]; then
  SCALE_SHIFT_ARGS=(--scale-shift-path "${SCALE_SHIFT_STATS}")
fi
HEAD_TYPE_ARGS=()
if [[ -n "${MODEL_HEAD_TYPE}" ]]; then
  HEAD_TYPE_ARGS=(--head-type "${MODEL_HEAD_TYPE}")
fi
RUN_NAME_ARGS=()
if [[ "${COMPACT_NAMES}" == "1" ]]; then
  RUN_NAME_ARGS=(--run-name "${EXPERIMENT_NAME}")
fi

case "${SCOPE}" in
  validate)
    "${PY}" "${RUNNER}" \
      --dataset nabladft \
      --variant "${MODEL_VARIANT}" \
      --dbpath "${NABLA_DB}" \
      --model-config "${MODEL_CONFIG}" \
      "${SCALE_SHIFT_ARGS[@]}" \
      "${HEAD_TYPE_ARGS[@]}" \
      "${RUN_NAME_ARGS[@]}" \
      --optimizer-type muon \
      --batch-size 5 \
      --gradient-accumulation-steps 2 \
      --no-distribute-graphs \
      --gpu "${GPU_PAIR}" \
      --master-port "${MASTER_PORT}" \
      --validate-only \
      --no-use-wandb \
      --flat-output
    exit 0
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
    echo "Scope must be prepare, validate, smoke, or full." >&2
    exit 2
    ;;
esac

ACTUAL_HOST=$(hostname)
if [[ -n "${EXPECTED_HOST}" && "${ACTUAL_HOST}" != "${EXPECTED_HOST}" ]]; then
  echo "Expected host ${EXPECTED_HOST}; current host is ${ACTUAL_HOST}." >&2
  exit 1
fi

mapfile -t GPU_MEMORY_USED < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
for gpu in "${GPU_INDICES[@]}"; do
  if [[ -z "${GPU_MEMORY_USED[${gpu}]:-}" ]]; then
    echo "GPU ${gpu} does not exist on ${ACTUAL_HOST}." >&2
    exit 2
  fi
  if (( GPU_MEMORY_USED[gpu] > 1024 )); then
    echo "GPU ${gpu} uses ${GPU_MEMORY_USED[gpu]} MiB; refusing overlap." >&2
    exit 1
  fi
done

if [[ "${COMPACT_NAMES}" == "1" ]]; then
  if [[ "${SCOPE}" == "smoke" ]]; then
    GROUP_ROOT=${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}-smoke
  else
    GROUP_ROOT=${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}
  fi
else
  RUN_ID=$(date +%Y%m%d-%H%M%S)
  GROUP_ROOT=${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}-2gpu-eb20-mb5-ga2-${SCOPE_SLUG}-seed44-${RUN_ID}
fi
OUTPUT_DIR=${GROUP_ROOT}/run
LOG_FILE=${GROUP_ROOT}/run.log
if [[ -e "${GROUP_ROOT}" ]]; then
  echo "Output group already exists: ${GROUP_ROOT}" >&2
  exit 1
fi
mkdir -p "${GROUP_ROOT}"
MANIFEST_STATS=none
if [[ "${SCALE_SHIFT_ENABLED}" == "1" ]]; then
  MANIFEST_STATS=${SCALE_SHIFT_STATS}
fi
printf 'experiment\tvariant\thead_type\tgpus\tmicro_batch\tworld_size\taccumulation\teffective_batch\tconfig\tscale_shift_stats\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"
printf '%s\t%s\t%s\t%s\t5\t2\t2\t20\t%s\t%s\n' \
  "${EXPERIMENT_NAME}" "${MODEL_VARIANT}" "${MODEL_HEAD_TYPE:-config}" \
  "${GPU_PAIR}" "${MODEL_CONFIG}" \
  "${MANIFEST_STATS}" >> "${GROUP_ROOT}/launch_manifest.tsv"
printf 'source_commit\t%s\n' "$(git rev-parse HEAD)" \
  > "${GROUP_ROOT}/source_revision.tsv"

set +e
env \
  CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
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
    --variant "${MODEL_VARIANT}" \
    --dbpath "${NABLA_DB}" \
    --model-config "${MODEL_CONFIG}" \
    "${SCALE_SHIFT_ARGS[@]}" \
    "${HEAD_TYPE_ARGS[@]}" \
    "${RUN_NAME_ARGS[@]}" \
    --optimizer-type muon \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --no-distribute-graphs \
    --gpu "${GPU_PAIR}" \
    --master-port "${MASTER_PORT}" \
    --flat-output \
    "${RUN_ARGS[@]}" \
    --output-root "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [[ ${RUN_STATUS} -ne 0 ]]; then
  printf 'status\texit_code\nfailed\t%s\n' "${RUN_STATUS}" \
    > "${GROUP_ROOT}/status.tsv"
  echo "Run failed; artifacts retained: ${GROUP_ROOT}" >&2
  exit "${RUN_STATUS}"
fi
printf 'status\texit_code\ncomplete\t0\n' > "${GROUP_ROOT}/status.tsv"

if [[ "${SCOPE}" == "smoke" ]]; then
  if [[ "${COMPACT_NAMES}" == "1" ]]; then
    [[ "${GROUP_ROOT}" == "${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}-smoke" ]] ||
      { echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2; exit 1; }
  else
    case "${GROUP_ROOT}" in
      "${PROJECT_ROOT}"/outputs/"${EXPERIMENT_NAME}"-2gpu-eb20-mb5-ga2-full-size-smoke-e1-seed44-*) ;;
      *) echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2; exit 1 ;;
    esac
  fi
  rm -rf -- "${GROUP_ROOT}"
  echo "Smoke passed; temporary output removed: ${GROUP_ROOT}"
else
  echo "Full output retained: ${GROUP_ROOT}"
fi
