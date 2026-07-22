#!/usr/bin/env bash
# Command sheet: copy one complete block from (a), (b), (c), or (d).
# Running this file directly does not launch training.

cd /dataset/seongsu/shared-home/workspace/project

: <<'COMMANDS'

# (a) MALOQ: 2 GPUs, micro-batch 5, accumulation 2, effective batch 20.
GPUS=0,1
MASTER_PORT=29561
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
MPIRUN=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/mpirun
RUN_ID=$(date +%Y%m%d-%H%M%S)
env \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${MASTER_PORT}" \
  OPAL_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  PRTE_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  PMIX_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${MPIRUN}" -np 2 --bind-to none \
  "${PY}" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py" \
    --dataset nabladft \
    --variant maloq \
    --optimizer-type muon \
    --use-wandb \
    --wandb-project maloq-nablaDFT \
    --wandb-entity kaist-korea \
    --wandb-mode online \
    --wandb-log-every-n-steps 10 \
    --num-epochs 20 \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --num-train 12081 \
    --num-val 64 \
    --num-test 0 \
    --gpu "${GPUS}" \
    --master-port "${MASTER_PORT}" \
    --output-root "${PROJECT_ROOT}/outputs/nabladft-maloq-muon-e20-full-seed44-${RUN_ID}"

# (b) MALOQ-NTE: 2 GPUs, micro-batch 5, accumulation 2, effective batch 20.
GPUS=2,3
MASTER_PORT=29562
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
MPIRUN=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/mpirun
RUN_ID=$(date +%Y%m%d-%H%M%S)
env \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${MASTER_PORT}" \
  OPAL_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  PRTE_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  PMIX_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${MPIRUN}" -np 2 --bind-to none \
  "${PY}" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py" \
    --dataset nabladft \
    --variant maloq-nte \
    --optimizer-type muon \
    --use-wandb \
    --wandb-project maloq-nablaDFT \
    --wandb-entity kaist-korea \
    --wandb-mode online \
    --wandb-log-every-n-steps 10 \
    --num-epochs 20 \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --num-train 12081 \
    --num-val 64 \
    --num-test 0 \
    --gpu "${GPUS}" \
    --master-port "${MASTER_PORT}" \
    --output-root "${PROJECT_ROOT}/outputs/nabladft-maloq-nte-muon-e20-full-seed44-${RUN_ID}"

# (c) QHFlow3: 2 GPUs, micro-batch 5, accumulation 2, effective batch 20.
GPUS=6,7
MASTER_PORT=29563
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
MPIRUN=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/mpirun
RUN_ID=$(date +%Y%m%d-%H%M%S)
env \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${MASTER_PORT}" \
  OPAL_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  PRTE_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  PMIX_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${MPIRUN}" -np 2 --bind-to none \
  "${PY}" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py" \
    --dataset nabladft \
    --variant qhflow3 \
    --optimizer-type muon \
    --use-wandb \
    --wandb-project maloq-nablaDFT \
    --wandb-entity kaist-korea \
    --wandb-mode online \
    --wandb-log-every-n-steps 10 \
    --num-epochs 20 \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --num-train 12081 \
    --num-val 64 \
    --num-test 0 \
    --gpu "${GPUS}" \
    --master-port "${MASTER_PORT}" \
    --output-root "${PROJECT_ROOT}/outputs/nabladft-qhflow3-muon-e20-full-seed44-${RUN_ID}"

# (d) Matched comparison: train MALOQ, MALOQ-NTE, and QHFlow3 sequentially.
GPUS=6,7
MASTER_PORT=29564
PROJECT_ROOT=/dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
MPIRUN=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/mpirun
RUN_ID=$(date +%Y%m%d-%H%M%S)
env \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${MASTER_PORT}" \
  OPAL_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  PRTE_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  PMIX_PREFIX=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "${MPIRUN}" -np 2 --bind-to none \
  "${PY}" \
  "${PROJECT_ROOT}/_my_script/experiment/2026-07-22/run_nabladft_qh9_density.py" \
    --dataset nabladft \
    --variant all \
    --optimizer-type muon \
    --use-wandb \
    --wandb-project maloq-nablaDFT \
    --wandb-entity kaist-korea \
    --wandb-mode online \
    --wandb-log-every-n-steps 10 \
    --num-epochs 20 \
    --batch-size 5 \
    --gradient-accumulation-steps 2 \
    --num-train 12081 \
    --num-val 64 \
    --num-test 0 \
    --gpu "${GPUS}" \
    --master-port "${MASTER_PORT}" \
    --output-root "${PROJECT_ROOT}/outputs/nabladft-three-model-comparison-muon-e20-full-seed44-${RUN_ID}"

COMMANDS
