import os
import torch
import torch.distributed as dist
import numpy as np
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

def setup_env(rank, world_size):

    # copying over elastic launch for testing - nvm might have figured it out
    # launch_config = LaunchConfig(
    #             min_nodes=1,
    #             max_nodes=1,
    #             nproc_per_node=scheduler_cfg.ranks_per_node,
    #             rdzv_backend="c10d",
    #             max_restarts=0,
    #         )
    # elastic_launch(launch_config, _runner_wrapper)(cfg)

    # dist.init_process_group(
    #         backend=config["distributed_backend"],
    #         rank=int(os.environ.get("RANK")),
    #         world_size=config["world_size"],
    #         timeout=timeout,
    #     )
    
    # gpu_id = os.environ.get("RANK")
    # torch.cuda.set_device(gpu_id) 
    # device = torch.device('cuda:'+ str(gpu_id))
    # # gpu_id = os.environ['SLURM_PROCID']

    print("Initializing distributed process group... ")     
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # !! make sure visibility is restricted to "gpu 0" in .sh file !!
    gpu_id = 0
    torch.cuda.set_device(gpu_id) 
    device = torch.device('cuda:'+ str(gpu_id))
    
    return device

def split_indices(rank, world_size, total_num_idx):
    """
    Split data indices 
    """

    assert dist.is_initialized()

    if rank == 0:
        print(f"Processing {total_num_idx} structures between {world_size} GPUs")

    local_num_idx = total_num_idx//world_size
    counts = np.array([local_num_idx]*world_size, dtype=np.int32)
    for i in range(total_num_idx % world_size):
        counts[i] += 1

    displacements = np.zeros_like(counts)
    for i in range(1, len(counts)):
        displacements[i] = displacements[i-1] + counts[i-1]

    start_idx = displacements[rank]
    end_idx = displacements[rank] + counts[rank]
    local_num_idx = counts[rank]
    
    print(f"Rank {rank} does indices {start_idx} to {end_idx}")
    dist.barrier()

    return start_idx, end_idx, local_num_idx

