#!/usr/bin/env bash
# Copy one complete block into the shell. The file is a command sheet, not a
# wrapper that launches every experiment automatically.
# The QH9 blocks below are serial/local examples. For the current target-
# separated batch-32 W&B run on scp-gpu-2, use
# 06_qh9stable_delta_scp_gpu2_bs32.sh.

# Common project and interpreter:
cd /dataset/seongsu/shared-home/workspace/project
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python

# (a) Validate native NablaDFT without launching training.
"${PY}" _my_script/experiment/2026-07-22/run_nabladft_qh9_density.py \
  --dataset nabladft \
  --variant all \
  --validate-only

# (b) NablaDFT CUDA smoke: MALOQ + MALOQ-NTE + QHFlow3, one epoch.
GPU=0
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" \
  _my_script/experiment/2026-07-22/run_nabladft_qh9_density.py \
  --dataset nabladft \
  --variant all \
  --smoke \
  --gpu "${GPU}" \
  --master-port 29545

# (c) NablaDFT production comparison: MALOQ + MALOQ-NTE + QHFlow3.
GPU=0
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" \
  _my_script/experiment/2026-07-22/run_nabladft_qh9_density.py \
  --dataset nabladft \
  --variant all \
  --gpu "${GPU}" \
  --master-port 29546

# (d) Convert one density-target 2/1/1 QH9 delta-learning sample.
mkdir -p /dataset_tmp/qh9_matrix_maloq_ase
"${PY}" _auto_script/qh9_matrix_lmdb_to_maloq/process_qh9_matrix_lmdb_to_maloq_ase.py \
  --input-lmdb /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/data.lmdb \
  --split-file /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/QH9Stable_random_split.json \
  --output-db /dataset_tmp/qh9_matrix_maloq_ase/QH9StableMatrices_random_2_1_1.db \
  --subset-limit train=2 \
  --subset-limit val=1 \
  --subset-limit test=1

# (e1) QH9 Hamiltonian delta-learning CUDA smoke: all three models.
GPU=0
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" \
  _my_script/experiment/2026-07-22/run_nabladft_qh9_density.py \
  --dataset qh9-hamiltonian \
  --variant all \
  --delta-learning \
  --smoke \
  --gpu "${GPU}" \
  --master-port 29547

# (e2) QH9 density delta-learning CUDA smoke: all three models.
GPU=0
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" \
  _my_script/experiment/2026-07-22/run_nabladft_qh9_density.py \
  --dataset qh9-density \
  --variant all \
  --delta-learning \
  --smoke \
  --gpu "${GPU}" \
  --master-port 29548

# (f) Full density-target QH9 matrix conversion. Run once.
"${PY}" _auto_script/qh9_matrix_lmdb_to_maloq/process_qh9_matrix_lmdb_to_maloq_ase.py \
  --input-lmdb /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/data.lmdb \
  --split-file /dataset/seongsu/shared-home/datasets/qh9_b3lyp5_with_density/QH9Stable_random_split.json \
  --output-db /dataset/seongsu/shared-home/data/QH9_maloq_ase/QH9StableMatrices_random.db \
  --matrix-dtype float64 \
  --progress-every 100

# (g1) QH9 Hamiltonian delta-learning production comparison: all three models.
GPU=0
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" \
  _my_script/experiment/2026-07-22/run_nabladft_qh9_density.py \
  --dataset qh9-hamiltonian \
  --variant all \
  --delta-learning \
  --gpu "${GPU}" \
  --master-port 29549

# (g2) QH9 density delta-learning production comparison: all three models.
GPU=0
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" \
  _my_script/experiment/2026-07-22/run_nabladft_qh9_density.py \
  --dataset qh9-density \
  --variant all \
  --delta-learning \
  --gpu "${GPU}" \
  --master-port 29550
