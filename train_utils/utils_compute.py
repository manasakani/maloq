import os
import torch
import torch.distributed as dist
import numpy as np

def setup_env(rank, world_size):

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # device = torch.device('cuda')  
    # print(f"rank {rank} sees {os.environ['CUDA_VISIBLE_DEVICES']}")

    print("Initializing distributed process group... ")     
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # visibility is restricted to 0 in .sh file
    if len(os.environ['CUDA_VISIBLE_DEVICES']) == 1:
        gpu_id = 0
        torch.cuda.set_device(gpu_id) 
        device = torch.device('cuda:'+ str(gpu_id))
    else:
        print("len(os.environ['CUDA_VISIBLE_DEVICES']) ~= 1", flush=True)
        torch.cuda.set_device(rank) 
        device = torch.device(f'cuda:{rank}')

    print("Finished setting up compute environment.")
    return device

def split_indices(rank, world_size, total_num_idx):
    """
    Distributes data indices between ranks
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

