import os
import torch
import torch.distributed as dist
import numpy as np
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

def setup_env(rank, world_size):

    # dist.init_process_group(
    #         backend=config["distributed_backend"],
    #         rank=int(os.environ.get("RANK")),
    #         world_size=config["world_size"],
    #         timeout=timeout,
    #     )

    print("Initializing distributed process group... ", flush=True)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    # dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    print("Initialized distributed process group", flush=True)

    # barrier:
    dist.barrier()

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

    print(f"IMPORTANT NOTE: Ignoring {total_num_idx % world_size} indices to make the distribution even!")
    # for i in range(total_num_idx % world_size):
    #     counts[i] += 1

    displacements = np.zeros_like(counts)
    for i in range(1, len(counts)):
        displacements[i] = displacements[i-1] + counts[i-1]

    start_idx = displacements[rank]
    end_idx = displacements[rank] + counts[rank]
    local_num_idx = counts[rank]

    print(f"Rank {rank} does indices {start_idx} to {end_idx}", flush=True)
    dist.barrier()

    return start_idx, end_idx, local_num_idx
