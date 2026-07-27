#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
EXPERIMENT_ROOT=${PROJECT_ROOT}/_my_script/experiment/2026-07-27
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py
HAMILTONIAN_DB=/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db
DENSITY_DB=/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9StableMatrices_random.db

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 {maloq|maloq-nte|qhflow3} {hamiltonian|density} {prepare|validate|smoke|full} [GPU]" >&2
  exit 2
fi

MODEL=$1
TARGET=$2
SCOPE=$3
GPU=${4:-0}

case "${MODEL}" in
  maloq)
    MODEL_CONFIG=${EXPERIMENT_ROOT}/qh9stable_v2_maloq_muon.yaml
    MODEL_SLUG=maloq
    MODEL_LABEL=MALOQ
    MODEL_TAG=model:maloq
    PORT_MODEL_OFFSET=0
    ;;
  maloq-nte)
    MODEL_CONFIG=${EXPERIMENT_ROOT}/qh9stable_v2_maloq_nte_muon.yaml
    MODEL_SLUG=nte64e2
    MODEL_LABEL=NTE-64/2
    MODEL_TAG=model:nte64e2
    PORT_MODEL_OFFSET=20
    ;;
  qhflow3)
    MODEL_CONFIG=${EXPERIMENT_ROOT}/qh9stable_v2_qhflow3_muon_grid48_chunk1024.yaml
    MODEL_SLUG=qhf3-grid48-chunk1024
    MODEL_LABEL=QHFlow3
    MODEL_TAG=model:qhf3
    PORT_MODEL_OFFSET=40
    ;;
  *)
    echo "Model must be maloq, maloq-nte, or qhflow3." >&2
    exit 2
    ;;
esac

case "${TARGET}" in
  hamiltonian)
    DATASET=qh9-hamiltonian
    DBPATH=${HAMILTONIAN_DB}
    TARGET_SLUG=hdelta
    TARGET_LABEL=HΔ
    TARGET_TAG=target:hamiltonian-delta
    PORT_TARGET_OFFSET=0
    ;;
  density)
    DATASET=qh9-density
    DBPATH=${DENSITY_DB}
    TARGET_SLUG=ddelta
    TARGET_LABEL=DΔ
    TARGET_TAG=target:density-delta
    PORT_TARGET_OFFSET=10
    ;;
  *)
    echo "Target must be hamiltonian or density." >&2
    exit 2
    ;;
esac

case "${SCOPE}" in
  prepare|validate|smoke|full) ;;
  *)
    echo "Scope must be prepare, validate, smoke, or full." >&2
    exit 2
    ;;
esac

if [[ ! "${GPU}" =~ ^[0-7]$ ]]; then
  echo "GPU must be one physical index from 0 through 7." >&2
  exit 2
fi
if [[ ! -x "${PY}" || ! -f "${RUNNER}" || ! -f "${MODEL_CONFIG}" ]]; then
  echo "SC26 environment, runner, or model config is missing." >&2
  exit 1
fi
if [[ ! -r "${DBPATH}" ]]; then
  echo "Converted QH9Stable database is missing or unreadable: ${DBPATH}" >&2
  exit 1
fi

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
NUM_EPOCHS=${NUM_EPOCHS:-80}
EXPECTED_HOST=${EXPECTED_HOST:-}

for numeric_value in \
  "${MICRO_BATCH_SIZE}" \
  "${GRADIENT_ACCUMULATION_STEPS}" \
  "${NUM_EPOCHS}"; do
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch, accumulation, and epoch values must be positive integers." >&2
    exit 2
  fi
done

EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( EFFECTIVE_BATCH_SIZE != 32 )); then
  echo "QH9Stable V2 comparisons require effective batch 32; got ${EFFECTIVE_BATCH_SIZE}." >&2
  exit 2
fi

ACTUAL_HOST=$(hostname)
if [[ -n "${EXPECTED_HOST}" && "${ACTUAL_HOST}" != "${EXPECTED_HOST}" ]]; then
  echo "EXPECTED_HOST=${EXPECTED_HOST}, but current host is ${ACTUAL_HOST}." >&2
  exit 1
fi

MASTER_PORT=${MASTER_PORT:-$((29830 + PORT_MODEL_OFFSET + PORT_TARGET_OFFSET + GPU))}
if (( MASTER_PORT < 1024 || MASTER_PORT > 65535 )); then
  echo "MASTER_PORT must be between 1024 and 65535." >&2
  exit 2
fi

COMMON_ARGS=(
  --dataset "${DATASET}"
  --variant "${MODEL}"
  --model-config "${MODEL_CONFIG}"
  --optimizer-type muon
  --head-type maloq_muon
  --batch-size "${MICRO_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --delta-learning
  --no-distribute-graphs
  --gpu "${GPU}"
  --master-port "${MASTER_PORT}"
  --flat-output
  --dbpath "${DBPATH}"
)

if [[ "${SCOPE}" == "prepare" ]]; then
  printf 'project=%s\nconfig=%s\ndatabase=%s\nrunner=%s\nenvironment=%s\n' \
    "${PROJECT_ROOT}" "${MODEL_CONFIG}" "${DBPATH}" "${RUNNER}" "${ENV_ROOT}"
  printf 'model=%s\ntarget=%s\nmicro_batch=%s\ngradient_accumulation=%s\neffective_batch=%s\n' \
    "${MODEL_LABEL}" "${TARGET_LABEL}" "${MICRO_BATCH_SIZE}" \
    "${GRADIENT_ACCUMULATION_STEPS}" "${EFFECTIVE_BATCH_SIZE}"
  exit 0
fi

if [[ "${SCOPE}" == "validate" ]]; then
  exec "${PY}" "${RUNNER}" \
    "${COMMON_ARGS[@]}" \
    --validate-only \
    --num-train 64 \
    --num-val 32 \
    --num-test 0 \
    --no-use-wandb
fi

GPU_MEMORY_USED=$(
  nvidia-smi \
    --id="${GPU}" \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits
)
if [[ ! "${GPU_MEMORY_USED}" =~ ^[0-9]+$ ]]; then
  echo "Could not read GPU ${GPU} memory usage." >&2
  exit 1
fi
if (( GPU_MEMORY_USED > 1024 )); then
  echo "GPU ${GPU} already uses ${GPU_MEMORY_USED} MiB; refusing to overlap." >&2
  exit 1
fi

TIMESTAMP=$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S)
RUN_BASENAME=qh9stable-v2-${TARGET_SLUG}-${MODEL_SLUG}-muon-eb32-mb${MICRO_BATCH_SIZE}-ga${GRADIENT_ACCUMULATION_STEPS}-${SCOPE}-seed44-${TIMESTAMP}
GROUP_ROOT=${PROJECT_ROOT}/outputs/${RUN_BASENAME}
OUTPUT_DIR=${GROUP_ROOT}/run
LOG_FILE=${GROUP_ROOT}/${RUN_BASENAME}.log

if [[ -e "${GROUP_ROOT}" ]]; then
  echo "Output already exists: ${GROUP_ROOT}" >&2
  exit 1
fi
mkdir -p "${GROUP_ROOT}"

if [[ "${SCOPE}" == "smoke" ]]; then
  RUN_ARGS=(
    --smoke
    --full-size-smoke
    --num-epochs 1
    --num-train 64
    --num-val 32
    --num-test 0
    --no-use-wandb
  )
else
  DISPLAY_SUFFIX=
  EXTRA_TAGS=()
  if [[ "${MODEL}" == "qhflow3" ]]; then
    DISPLAY_SUFFIX=" | Grid48+Chunk1024"
    EXTRA_TAGS=(
      --wandb-tag grid:48x48
      --wandb-tag grid-ffn-chunk:1024
      --wandb-tag output-projection:muon
    )
  fi
  RUN_ARGS=(
    --num-epochs "${NUM_EPOCHS}"
    --num-train 104664
    --num-val 13083
    --num-test 13084
    --use-wandb
    --wandb-project maloq-qh9
    --wandb-entity kaist-korea
    --wandb-mode online
    --wandb-log-every-n-steps 10
    --run-name "${RUN_BASENAME}"
    --wandb-run-name "QH9Stable | ${TARGET_LABEL} | ${MODEL_LABEL} | Muon${DISPLAY_SUFFIX} | V2"
    --wandb-group qh9stable-v2-reset-20260727
    --wandb-job-type full
    --wandb-tag dataset:qh9stable
    --wandb-tag "${MODEL_TAG}"
    --wandb-tag head:muon
    --wandb-tag "${TARGET_TAG}"
    --wandb-tag delta-learning:on
    --wandb-tag effective-batch:32
    --wandb-tag seed:44
    --wandb-tag version:v2
    --wandb-tag reset:20260727
    --wandb-tag sc26-seongsu
    "${EXTRA_TAGS[@]}"
  )
fi

printf 'host=%s\ngpu=%s\nmodel=%s\ntarget=%s\nscope=%s\noutput=%s\nlog=%s\n' \
  "${ACTUAL_HOST}" "${GPU}" "${MODEL_LABEL}" "${TARGET_LABEL}" "${SCOPE}" \
  "${OUTPUT_DIR}" "${LOG_FILE}"

set +e
env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${MASTER_PORT}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${PY}" "${RUNNER}" \
    "${COMMON_ARGS[@]}" \
    "${RUN_ARGS[@]}" \
    --output-root "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [[ "${SCOPE}" == "smoke" && ${RUN_STATUS} -eq 0 ]]; then
  case "${GROUP_ROOT}" in
    "${PROJECT_ROOT}"/outputs/qh9stable-v2-*-smoke-seed44-*) ;;
    *)
      echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2
      exit 1
      ;;
  esac
  rm -rf -- "${GROUP_ROOT}"
  echo "Smoke passed; temporary artifacts removed: ${GROUP_ROOT}"
elif (( RUN_STATUS != 0 )); then
  echo "Run failed with status ${RUN_STATUS}; evidence retained: ${GROUP_ROOT}" >&2
else
  echo "Run completed: ${GROUP_ROOT}"
fi

exit "${RUN_STATUS}"
