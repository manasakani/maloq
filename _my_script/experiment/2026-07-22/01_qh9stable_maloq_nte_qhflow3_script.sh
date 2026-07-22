#!/usr/bin/env bash
# Command sheet. Copy/run one complete block at a time.

cd /dataset/seongsu/shared-home/workspace/project

: <<'COMMANDS'

# (a) maloq_qh9stable: production run.
# Train the native interleaved 3+3 MALOQ baseline on the official QH9Stable split.
GPU=0
MASTER_PORT=29541
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
DB=/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db
RUN_ID=$(date +%Y%m%d-%H%M%S)
env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${ENV_PREFIX}/bin/python" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-21/compare_maloq_qh9.py" \
    --variant maloq \
    --dbpath "${DB}" \
    --gpu "${GPU}" \
    --master-port "${MASTER_PORT}" \
    --output-root "${PROJECT_ROOT}/outputs/qh9stable-maloq-full-seed44-${RUN_ID}"

# (b) maloq_nte_qh9stable: production run.
# Train MALOQ-NTE with 3 node then 2 edge blocks and the matched native MALOQ head.
GPU=0
MASTER_PORT=29542
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
DB=/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db
RUN_ID=$(date +%Y%m%d-%H%M%S)
env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${ENV_PREFIX}/bin/python" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-21/compare_maloq_qh9.py" \
    --variant maloq-nte \
    --dbpath "${DB}" \
    --gpu "${GPU}" \
    --master-port "${MASTER_PORT}" \
    --output-root "${PROJECT_ROOT}/outputs/qh9stable-maloq-nte-full-seed44-${RUN_ID}"

# (c) qhflow3_qh9stable: production run.
# Train the equivariant grid48 QHFlow3 trunk with zero-H/overlap inputs and the native MALOQ head.
GPU=0
MASTER_PORT=29543
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
DB=/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db
RUN_ID=$(date +%Y%m%d-%H%M%S)
env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${ENV_PREFIX}/bin/python" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-21/compare_maloq_qh9.py" \
    --variant qhflow3 \
    --dbpath "${DB}" \
    --gpu "${GPU}" \
    --master-port "${MASTER_PORT}" \
    --output-root "${PROJECT_ROOT}/outputs/qh9stable-qhflow3-grid48-full-seed44-${RUN_ID}"

# (d) qh9stable_three_models: matched production comparison.
# Train MALOQ, MALOQ-NTE, and QHFlow3 sequentially with the same split, loss, optimizer, and seed.
GPU=0
MASTER_PORT=29544
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
ENV_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26
DB=/dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9Stable_random.db
RUN_ID=$(date +%Y%m%d-%H%M%S)
env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${ENV_PREFIX}/bin/python" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-21/compare_maloq_qh9.py" \
    --variant all \
    --dbpath "${DB}" \
    --gpu "${GPU}" \
    --master-port "${MASTER_PORT}" \
    --output-root "${PROJECT_ROOT}/outputs/qh9stable-three-models-full-seed44-${RUN_ID}"

COMMANDS
