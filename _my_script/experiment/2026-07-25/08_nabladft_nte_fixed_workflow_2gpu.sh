#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-25/run_training_workflow_fixed.py
FULL_CONFIG=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/maloq_nte_muon_head_nabladft.yaml
SMOKE_CONFIG=${PROJECT_ROOT}/_my_script/experiment/2026-07-25/training_workflow_fixed_nte_smoke.yaml
NABLA_DB=/dataset/seongsu/shared-home/datasets/nablaDFT/hamiltonian_databases/train_2k.db

DEFAULT_GPUS=0,1
DEFAULT_MASTER_PORT=29641
EXPECTED_HOST=${EXPECTED_HOST:-}
MASTER_PORT=${MASTER_PORT:-${DEFAULT_MASTER_PORT}}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage:" >&2
  echo "  $0 validate [GPU0,GPU1]" >&2
  echo "  $0 smoke [GPU0,GPU1]" >&2
  echo "  $0 full [GPU0,GPU1]" >&2
  echo "  $0 resume CHECKPOINT_OR_DIRECTORY [GPU0,GPU1]" >&2
  exit 2
fi

SCOPE=$1
RESUME_FROM=
if [[ "${SCOPE}" == "resume" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "resume requires a checkpoint file or directory." >&2
    exit 2
  fi
  RESUME_FROM=$2
  GPU_PAIR=${3:-${GPUS:-${DEFAULT_GPUS}}}
  if [[ ! -e "${RESUME_FROM}" ]]; then
    echo "Resume source does not exist: ${RESUME_FROM}" >&2
    exit 1
  fi
else
  if [[ $# -gt 2 ]]; then
    echo "${SCOPE} accepts at most one GPU-pair argument." >&2
    exit 2
  fi
  GPU_PAIR=${2:-${GPUS:-${DEFAULT_GPUS}}}
fi

case "${SCOPE}" in
  validate | smoke | full | resume) ;;
  *)
    echo "Scope must be validate, smoke, full, or resume." >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPU_INDICES <<< "${GPU_PAIR}"
if [[ ${#GPU_INDICES[@]} -ne 2 ||
  "${GPU_INDICES[0]}" == "${GPU_INDICES[1]}" ]]; then
  echo "GPU pair must contain two different comma-separated indices." >&2
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
  echo "SC26 interpreter, mpirun, or fixed runner is missing." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
if [[ "${SCOPE}" == "validate" ]]; then
  env \
    CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${MASTER_PORT}" \
    OPAL_PREFIX="${ENV_ROOT}" \
    PRTE_PREFIX="${ENV_ROOT}" \
    PMIX_PREFIX="${ENV_ROOT}" \
    "${MPIRUN}" -np 2 --bind-to none \
    "${PY}" "${RUNNER}" \
      --dataset nabladft \
      --variant maloq-nte \
      --dbpath "${NABLA_DB}" \
      --model-config "${FULL_CONFIG}" \
      --head-type maloq_muon \
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
fi

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

run_distributed() {
  local log_file=$1
  shift
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
    "${PY}" "${RUNNER}" "$@" 2>&1 | tee "${log_file}"
  return "${PIPESTATUS[0]}"
}

RUN_ID=$(date +%Y%m%d-%H%M%S)
if [[ "${SCOPE}" == "smoke" ]]; then
  GROUP_ROOT=${PROJECT_ROOT}/outputs/training-workflow-fixed-smoke-${RUN_ID}
  mkdir -p "${GROUP_ROOT}/stage1" "${GROUP_ROOT}/stage2"
  COMMON_SMOKE_ARGS=(
    --dataset nabladft
    --variant maloq-nte
    --dbpath "${NABLA_DB}"
    --model-config "${SMOKE_CONFIG}"
    --head-type maloq_muon
    --optimizer-type muon
    --batch-size 1
    --gradient-accumulation-steps 1
    --num-train 4
    --num-val 2
    --num-test 0
    --no-distribute-graphs
    --gpu "${GPU_PAIR}"
    --master-port "${MASTER_PORT}"
    --no-use-wandb
    --flat-output
  )
  if ! run_distributed "${GROUP_ROOT}/stage1/run.log" \
    "${COMMON_SMOKE_ARGS[@]}" \
    --num-epochs 2 \
    --fixed-stop-after-epoch 1 \
    --output-root "${GROUP_ROOT}/stage1/run"; then
    echo "Fixed-workflow smoke stage 1 failed; retained: ${GROUP_ROOT}" >&2
    exit 1
  fi
  if ! run_distributed "${GROUP_ROOT}/stage2/run.log" \
    --resume-from "${GROUP_ROOT}/stage1/run" \
    "${COMMON_SMOKE_ARGS[@]}" \
    --num-epochs 2 \
    --output-root "${GROUP_ROOT}/stage2/run"; then
    echo "Fixed-workflow resume stage failed; retained: ${GROUP_ROOT}" >&2
    exit 1
  fi
  "${PY}" -c \
    'import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
stage1 = json.loads((root / "stage1/run/training_state.meta.json").read_text())
stage2 = json.loads((root / "stage2/run/training_state.meta.json").read_text())
assert stage1["completed_epoch"] == 0, stage1
assert stage2["completed_epoch"] == 1, stage2
assert stage2["optimizer_step"] > stage1["optimizer_step"], (stage1, stage2)
assert stage2["resume_parent"], stage2
print("Fixed-workflow distributed resume smoke passed.")' \
    "${GROUP_ROOT}"
  case "${GROUP_ROOT}" in
    "${PROJECT_ROOT}"/outputs/training-workflow-fixed-smoke-*) ;;
    *)
      echo "Refusing to remove unexpected smoke path: ${GROUP_ROOT}" >&2
      exit 1
      ;;
  esac
  rm -rf -- "${GROUP_ROOT}"
  echo "Smoke artifacts removed: ${GROUP_ROOT}"
  exit 0
fi

if [[ "${SCOPE}" == "full" ]]; then
  GROUP_ROOT=${PROJECT_ROOT}/outputs/nabla-nte64e2-fixed-v1-${RUN_ID}
  RESUME_ARGS=()
else
  GROUP_ROOT=${PROJECT_ROOT}/outputs/nabla-nte64e2-fixed-v1-resume-${RUN_ID}
  RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi
OUTPUT_DIR=${GROUP_ROOT}/run
LOG_FILE=${GROUP_ROOT}/run.log
mkdir -p "${GROUP_ROOT}"
printf 'workflow\tmodel\tgpus\tmicro_batch\tworld_size\taccumulation\teffective_batch\tresume_from\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"
printf 'fixed\tmaloq-nte\t%s\t5\t2\t2\t20\t%s\n' \
  "${GPU_PAIR}" "${RESUME_FROM:-none}" >> "${GROUP_ROOT}/launch_manifest.tsv"

if run_distributed "${LOG_FILE}" \
  "${RESUME_ARGS[@]}" \
  --dataset nabladft \
  --variant maloq-nte \
  --dbpath "${NABLA_DB}" \
  --model-config "${FULL_CONFIG}" \
  --head-type maloq_muon \
  --optimizer-type muon \
  --batch-size 5 \
  --gradient-accumulation-steps 2 \
  --num-train 12081 \
  --num-val 64 \
  --num-test 0 \
  --num-epochs 20 \
  --no-distribute-graphs \
  --gpu "${GPU_PAIR}" \
  --master-port "${MASTER_PORT}" \
  --use-wandb \
  --wandb-project maloq-nablaDFT \
  --wandb-entity kaist-korea \
  --wandb-mode online \
  --wandb-log-every-n-steps 10 \
  --run-name nabla-nte64e2-fixed-v1 \
  --wandb-run-name "NablaDFT | NTE-64/2 | Muon | RAW | FixedResume | V1" \
  --wandb-group nabla-nte64e2-fixed-resume \
  --wandb-job-type full \
  --wandb-tag workflow:fixed-resume \
  --flat-output \
  --output-root "${OUTPUT_DIR}"; then
  printf 'status\texit_code\ncomplete\t0\n' > "${GROUP_ROOT}/status.tsv"
  echo "Fixed workflow completed: ${GROUP_ROOT}"
else
  exit_code=$?
  printf 'status\texit_code\nfailed\t%s\n' "${exit_code}" \
    > "${GROUP_ROOT}/status.tsv"
  echo "Fixed workflow failed; retained: ${GROUP_ROOT}" >&2
  exit "${exit_code}"
fi
