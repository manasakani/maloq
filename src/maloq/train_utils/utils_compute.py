# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

import os
import atexit
import torch
import torch.distributed as dist
import numpy as np
from torch.distributed.launcher.api import LaunchConfig, elastic_launch


_DIST_CLEANUP_REGISTERED = False


def cleanup_process_group(sync_barrier=True):
    """Safely tears down torch.distributed process group if initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return

    # Barrier before teardown helps avoid backend warnings on clean shutdown.
    if sync_barrier:
        try:
            dist.barrier()
        except Exception:
            # Do not block teardown if one of the ranks has already failed.
            pass

    try:
        dist.destroy_process_group()
    except Exception:
        # We intentionally swallow teardown errors so cleanup never masks root errors.
        pass


def register_dist_cleanup():
    """Registers a best-effort process-group cleanup hook for interpreter exit."""
    global _DIST_CLEANUP_REGISTERED
    if _DIST_CLEANUP_REGISTERED:
        return

    atexit.register(cleanup_process_group, False)
    _DIST_CLEANUP_REGISTERED = True

def distributed_context(env=None):
    """Return global rank, world size, and local rank for common launchers."""
    env = os.environ if env is None else env
    if 'RANK' in env or 'WORLD_SIZE' in env:
        if 'RANK' not in env or 'WORLD_SIZE' not in env:
            raise ValueError('RANK and WORLD_SIZE must be set together.')
        rank = int(env['RANK'])
        world_size = int(env['WORLD_SIZE'])
        local_rank = int(env.get('LOCAL_RANK', 0))
    elif 'OMPI_COMM_WORLD_RANK' in env:
        rank = int(env['OMPI_COMM_WORLD_RANK'])
        world_size = int(env['OMPI_COMM_WORLD_SIZE'])
        local_rank = int(env.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    else:
        rank = int(env.get('SLURM_PROCID', 0))
        world_size = int(env.get('SLURM_NTASKS', 1))
        local_rank = int(env.get('SLURM_LOCALID', 0))

    if world_size < 1:
        raise ValueError(f'WORLD_SIZE must be positive; got {world_size}.')
    if not 0 <= rank < world_size:
        raise ValueError(
            f'Rank must be in [0, {world_size}); got rank={rank}.'
        )
    if local_rank < 0:
        raise ValueError(f'LOCAL_RANK must be non-negative; got {local_rank}.')
    return rank, world_size, local_rank


def setup_env(rank, world_size, backend='nccl', local_rank=0):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        visible_device_count = torch.cuda.device_count()
        # A scheduler may expose one GPU per process while reporting a nonzero
        # node-local rank. In that case the only valid process-local index is 0.
        gpu_id = 0 if visible_device_count == 1 else int(local_rank)
        if not 0 <= gpu_id < visible_device_count:
            raise ValueError(
                f'Local rank {local_rank} cannot select one of '
                f'{visible_device_count} visible CUDA devices.'
            )
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    # Single-process local runs should not require MASTER_ADDR/MASTER_PORT.
    if world_size <= 1:
        if not dist.is_initialized():
            master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
            master_port = int(os.environ.get('MASTER_PORT', '29500'))
            if not 1 <= master_port <= 65535:
                raise ValueError(
                    f"MASTER_PORT must be between 1 and 65535; got {master_port}"
                )
            init_kwargs = {
                'backend': backend,
                'rank': 0,
                'world_size': 1,
                'init_method': f'tcp://{master_addr}:{master_port}',
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

    register_dist_cleanup()

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
