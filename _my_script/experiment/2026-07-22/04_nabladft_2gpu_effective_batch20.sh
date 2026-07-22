#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPUS=6,7 ./04_nabladft_2gpu_effective_batch20.sh data-parallel maloq smoke
#   GPUS=6,7 ./04_nabladft_2gpu_effective_batch20.sh data-parallel qhflow3 full
#   GPUS=6,7 ./04_nabladft_2gpu_effective_batch20.sh distributed-graph maloq smoke
# The second positional argument may be maloq, maloq-nte, qhflow3, or all.
# `all` runs MALOQ, MALOQ-NTE, and QHFlow3 sequentially.
# The third positional argument may be smoke or full and defaults to smoke.

MODE=${1:-data-parallel}
VARIANT=${2:-all}
SCOPE=${3:-smoke}
GPUS=${GPUS:-6,7}
MASTER_PORT=${MASTER_PORT:-29570}
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_ROOT=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
PY=${ENV_ROOT}/bin/python
MPIRUN=${ENV_ROOT}/bin/mpirun
RUN_ID=$(date +%Y%m%d-%H%M%S)

case "${MODE}" in
  data-parallel)
    MODE_ARGS=(--no-distribute-graphs)
    MODE_SLUG=data-parallel
    ;;
  distributed-graph)
    MODE_ARGS=(--distribute-graphs --partition-type linear-edgewise)
    MODE_SLUG=distributed-graph
    ;;
  *)
    echo "MODE must be data-parallel or distributed-graph" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in
  maloq|maloq-nte|qhflow3|all) ;;
  *)
    echo "VARIANT must be maloq, maloq-nte, qhflow3, or all" >&2
    exit 2
    ;;
esac

if [[ "${VARIANT}" == "all" ]]; then
  VARIANT_SLUG=three-model-comparison
else
  VARIANT_SLUG=${VARIANT}
fi

if [[ "${MODE}" == "distributed-graph" && ( "${VARIANT}" == "qhflow3" || "${VARIANT}" == "all" ) ]]; then
  echo "QHFlow3 supports 2-GPU data parallelism, not distributed graphs." >&2
  exit 2
fi

case "${SCOPE}" in
  smoke)
    RUN_ARGS=(
      --smoke
      --num-epochs 1
      --num-train 20
      --num-val 20
      --num-test 0
    )
    TRACKING_ARGS=(--no-use-wandb)
    SCOPE_SLUG=smoke
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
    echo "SCOPE must be smoke or full" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
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
  "${PY}" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py" \
    --dataset nabladft \
    --variant "${VARIANT}" \
    --optimizer-type muon \
    --batch-size 10 \
    "${TRACKING_ARGS[@]}" \
    --gpu "${GPUS}" \
    --master-port "${MASTER_PORT}" \
    "${MODE_ARGS[@]}" \
    "${RUN_ARGS[@]}" \
    --output-root \
    "${PROJECT_ROOT}/outputs/nabladft-${MODE_SLUG}-2gpu-eb20-${VARIANT_SLUG}-${SCOPE_SLUG}-${RUN_ID}"
