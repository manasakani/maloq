import os
import torch
import torch.distributed as dist
import numpy as np
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

def setup_env(rank, world_size, backend='nccl'):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # !! make sure visibility is restricted to "gpu 0" in .sh file !!
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    # Single-process local runs should not require MASTER_ADDR/MASTER_PORT.
    if world_size <= 1:
        if not dist.is_initialized():
            init_kwargs = {
                'backend': backend,
                'rank': 0,
                'world_size': 1,
                'init_method': 'tcp://127.0.0.1:29500',
            }
            if use_cuda:
                init_kwargs['device_id'] = device
            dist.init_process_group(**init_kwargs)
        print("Initialized local single-process distributed group", flush=True)
    else:
        init_kwargs = {
            'backend': backend,
            'rank': rank,
            'world_size': world_size,
        }
        if use_cuda:
            init_kwargs['device_id'] = device
        dist.init_process_group(**init_kwargs)  # Uses env:// when rank/world size > 1.
        print("Initialized distributed process group", flush=True)

    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    if dist.is_initialized():
        dist.barrier()

    return device

def print_cuda_env():
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = "N/A (Not Initialized)"

    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "No GPU"

    print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES: {cuda_devices} | Local Device Index: {current_device}")

def split_indices(rank, world_size, total_num_idx, distribute_graphs):
    """
    Split data indices
    """

    assert dist.is_initialized()

    dist.barrier()
    if rank == 0:
        print(f"Processing {total_num_idx} structures between {world_size} GPUs")
    dist.barrier()

    local_num_idx = total_num_idx//world_size
    counts = np.array([local_num_idx]*world_size, dtype=np.int32)

    if distribute_graphs:
        # Distribute the remainder (total_num_idx % world_size)
        for i in range(total_num_idx % world_size):
            counts[i] += 1
    else:
        print(f"IMPORTANT NOTE: Ignoring {total_num_idx % world_size} indices to make the distribution even!")

    displacements = np.zeros_like(counts)
    for i in range(1, len(counts)):
        displacements[i] = displacements[i-1] + counts[i-1]

    start_idx = displacements[rank]
    end_idx = displacements[rank] + counts[rank]
    local_num_idx = counts[rank]

    dist.barrier()
    print(f"Rank {rank} does indices {start_idx} to {end_idx}", flush=True)
    dist.barrier()

    return start_idx, end_idx, local_num_idx
