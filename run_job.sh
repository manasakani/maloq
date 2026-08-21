#!/bin/bash -l
#SBATCH --job-name=train_nabla_flash
#SBATCH --account=c33
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --output=slurm_output/output_%j.txt
#SBATCH --error=slurm_output/error_file_%j.err 

# NOTE: 
# Distributed MALOQ training on ALPS (CSCS), 8 nodes x 4 GPUs, self-contained
# `maloq` conda env (its own CUDA/NCCL/MPICH + clean-main flash_so2).
# The config file to run is the FIRST ARGUMENT (defaults to the FP32 example).
# Submit different precisions concurrently, e.g.:
#   sbatch --job-name=train_nabla_tf32 run_job.sh examples/run_nablaDFT_tf32.py
#   sbatch --job-name=train_nabla_bf16 run_job.sh examples/run_nablaDFT_bf16.py

# START ENVIRONMENT SETUP - CSCS Alps
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# NCCL runs on its built-in socket transport; pin it to the Slingshot (hsn*) interfaces.
export NCCL_SOCKET_IFNAME=hsn

export TRITON_CACHE_DIR="/capstor/scratch/cscs/dlu/triton_cache"
source /capstor/scratch/cscs/dlu/miniforge3/etc/profile.d/conda.sh
conda activate maloq
export PYTHONNOUSERSITE=1
# END ENVIRONMENT SETUP

# Config file to run (arg 1; default = FP32 example). Exported so the launcher reads it.
export MALOQ_CONFIG="${1:-/capstor/scratch/cscs/dlu/iclr/maloq/examples/run_nablaDFT.py}"

# Raise the NCCL collective timeout (env-level launcher; no maloq source edits).
# The heaviest ranks need >10 min to build Fock-matrix labels at startup, which
# overruns the stock 600s collective timeout; a longer one covers the one-time skew.
# Launcher path is per-job (SLURM_JOB_ID) so concurrent jobs never clash.
LAUNCHER=/capstor/scratch/cscs/dlu/iclr/_nccl_timeout_launcher_${SLURM_JOB_ID}.py
cat > "$LAUNCHER" <<PYEOF
import datetime
import runpy
import torch.distributed as dist

_orig_init = dist.init_process_group


def _init_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", datetime.timedelta(minutes=60))
    return _orig_init(*args, **kwargs)


dist.init_process_group = _init_with_timeout
runpy.run_path("$MALOQ_CONFIG", run_name="__main__")
PYEOF

srun --cpu-bind=socket bash -c "export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID; python $LAUNCHER"