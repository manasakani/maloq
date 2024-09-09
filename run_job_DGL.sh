#!/bin/bash --login 
#SBATCH --account=s1119
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1  
#SBATCH --constraint=gpu
#SBATCH --time=1:00:00
#SBATCH --job-name=DGLHfO2

# Note for Future Manasa: Using [Torch version: 1.10.1+cu111], [DGL version:  0.9.1post1] [Torch scatter version:  2.0.9]
# More recent versions of [torch, torch_scatter] require DGL compatible with torch2.2.1, 
# and that requires GLIBC_2.27 which is not available on daint

module load daint-gpu
module load PyTorch

export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=12
MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

echo "Creating ip_config.txt..."
> ip_config.txt  # Clearing the file if it exists
for node in $(scontrol show hostname $SLURM_NODELIST); do
    echo "$node:$MASTER_PORT" >> ip_config.txt
done

# debug mode option
# TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

# pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
# pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html
source ./myvenv/bin/activate

# srun python main-H2O_distributed.py  --ip_config ip_config.txt -f './'
srun python main-HfO2_distributed.py  --ip_config ip_config.txt -f './'


# later version: PyTorch-2.2.1, CUDA-11.8
# module load PyTorch/2.2.1-CrayGNU-21.09
# source ./distenv/bin/activate
# pip install e3nn ase torch_geometric dscribe typing
# required for virtual environment (PyTorch-2.2.1, CUDA-11.8)
# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.1+cu118.html
# pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2.1/cu118/repo.html
# the above is from the compatibility page: https://www.dgl.ai/pages/start.html