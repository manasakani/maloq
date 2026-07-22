#!/usr/bin/env bash
set -euo pipefail

# MALOQ-NTE backbone + corrected MUON-compatible MALOQ head.
# Effective batch: 5 molecules/rank * 2 ranks * 2 accumulation steps = 20.

SCOPE=${1:-validate}
GPUS=${GPUS:-2,3}
MASTER_PORT=${MASTER_PORT:-29582}

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
    )
    SCOPE_SLUG=full-e20
    ;;
  *)
    echo "Usage: $0 {validate|smoke|full}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUNNER=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py
CONFIG=${PROJECT_ROOT}/_my_script/experiment/2026-07-22/maloq_nte_muon_head_nabladft.yaml

COMMON_ARGS=(
  --dataset nabladft
  --variant maloq-nte
  --model-config "${CONFIG}"
  --head-type maloq_muon
  --optimizer-type muon
  --batch-size 5
  --gradient-accumulation-steps 2
  --no-distribute-graphs
  --gpu "${GPUS}"
  --master-port "${MASTER_PORT}"
  --flat-output
)

cd "${PROJECT_ROOT}"
if [[ "${SCOPE}" == "validate" ]]; then
  "${PY}" "${RUNNER}" "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}"
  exit 0
fi

RUN_ID=$(date +%Y%m%d-%H%M%S)
GROUP_ROOT=${PROJECT_ROOT}/outputs/nabladft-maloq-nte-muon-head-2gpu-eb20-mb5-ga2-${SCOPE_SLUG}-seed44-${RUN_ID}
MODEL_OUTPUT=${GROUP_ROOT}/maloq-nte
LOG_FILE=${GROUP_ROOT}/logs/maloq-nte.log

if [[ -e "${GROUP_ROOT}" ]]; then
  echo "Output group already exists: ${GROUP_ROOT}" >&2
  exit 1
fi
mkdir -p "${GROUP_ROOT}/logs"
printf 'model\tbackbone\thead_type\tgpus\tmicro_batch\tworld_size\taccumulation\teffective_batch\tconfig\n' \
  > "${GROUP_ROOT}/launch_manifest.tsv"
printf 'maloq-nte\tesen-node-then-edge\tmaloq_muon\t%s\t5\t2\t2\t20\t%s\n' \
  "${GPUS}" "${CONFIG}" >> "${GROUP_ROOT}/launch_manifest.tsv"

set +e
env \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${MASTER_PORT}" \
  OPAL_PREFIX="${ENV_ROOT}" \
  PRTE_PREFIX="${ENV_ROOT}" \
  PMIX_PREFIX="${ENV_ROOT}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${MPIRUN}" -np 2 --bind-to none \
  "${PY}" "${RUNNER}" \
    "${COMMON_ARGS[@]}" \
    "${RUN_ARGS[@]}" \
    --output-root "${MODEL_OUTPUT}" \
    > "${LOG_FILE}" 2>&1
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  status=complete
  if [[ -f "${MODEL_OUTPUT}/comparison.csv" ]]; then
    cp "${MODEL_OUTPUT}/comparison.csv" "${GROUP_ROOT}/comparison.csv"
  fi
else
  status=failed
fi
printf 'model\tstatus\texit_code\nmaloq-nte\t%s\t%s\n' "${status}" "${exit_code}" \
  > "${GROUP_ROOT}/status.tsv"
echo "MALOQ-NTE ${SCOPE}: ${status} (exit ${exit_code})"
echo "Output: ${GROUP_ROOT}"
echo "Log: ${LOG_FILE}"
exit "${exit_code}"
