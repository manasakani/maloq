#!/bin/bash --login 
#SBATCH --account=s1119
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1  
#SBATCH --constraint=gpu
#SBATCH --time=00:10:00
#SBATCH --job-name=5proc

module load daint-gpu
module load PyTorch

export OMP_NUM_THREADS=12
MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500
echo "Master node address: $MASTER_ADDR"

# export MASTER_PORT=12355
# export MASTER_ADDR=$(hostname -s)

# debug mode
# TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

# source ./pyenv_gnn/bin/activate
source ./myvenv/bin/activate
srun python main-HfO2.py

# torch version: 1.10.1+cu111
# torch scatter version: [pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html]
# compatibility page: https://data.pyg.org/whl/
