#!/bin/bash --login 
#SBATCH --account=s1119
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --constraint=gpu
#SBATCH --time=03:00:00
#SBATCH --job-name=noDGLtest

module load daint-gpu
module load PyTorch

export OMP_NUM_THREADS=12
MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500
# echo "Master node address: $MASTER_ADDR"

# debug mode
# TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

source ./myvenv/bin/activate
srun python main-HfO2.py

# torch version: 1.10.1+cu111
# torch scatter version: [pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html]
# compatibility page: https://data.pyg.org/whl/
