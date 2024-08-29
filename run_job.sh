#!/bin/bash --login 
#SBATCH --account=s1119
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1  
#SBATCH --constraint=gpu
#SBATCH --time=00:10:00
#SBATCH --job-name=9proc

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
TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

# source ./pyenv_gnn/bin/activate
source ./myvenv/bin/activate
srun python main-HfO2.py

# torch version: 1.10.1+cu111
# torch scatter version: [pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html]
# compatibility page: https://data.pyg.org/whl/

# depreciate Pytorch to V1.12, so that it works with the latest available cudatoolkit version on daint (V11.3)
# pip install torch==1.12.0+cu113 torchvision==0.8.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
