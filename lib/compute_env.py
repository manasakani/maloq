import os
import torch.distributed as dist
from mpi4py import MPI
import torch
import functools

# Initialize the compute environment for distributed training
def initialize_compute_env():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'SLURM_PROCID' in os.environ:  
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        backend = 'gloo'  # Use NCCL for Piz Daint (edit: RDMA may be broken, switching to gloo)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print("Initialized process group in: SLURM", flush=True)

    else:  
        comm = MPI.COMM_WORLD
        local_rank = comm.Get_rank()
        rank = local_rank
        world_size = comm.Get_size()

        # get the required env variables on rank 0 and broadcast them to all other ranks
        if local_rank == 0:
            os.environ["MASTER_ADDR"] = "127.0.0.1"  
            os.environ["MASTER_PORT"] = "29500"      
            master_addr = "127.0.0.1"
            master_port = "29500"

            comm.bcast(master_addr, root=0)
            comm.bcast(master_port, root=0)
        else:
            os.environ["MASTER_ADDR"] = comm.bcast(None, root=0)
            os.environ["MASTER_PORT"] = comm.bcast(None, root=0)

        # Set environment variables for torch.distributed
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        backend = 'gloo'  # Use Gloo for attelas (single GPU)
        dist.init_process_group(backend='gloo', rank=local_rank, world_size=world_size)
        rank_zero_print("Initialized process group in: local", flush=True)

    rank_zero_print(f"RANK: {rank}", flush=True)
    rank_zero_print(f"WORLD_SIZE: {world_size}", flush=True)
    rank_zero_print(f"LOCAL_RANK: {local_rank}", flush=True)
    dist.barrier()

    return device, world_size

# check this function
def remove_module_prefix(state_dict):
    prefix = 'module.'
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

# to run a function only on rank 0 in distributed training
def only_rank_zero(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

# Utility function that only prints if rank is 0
def rank_zero_print(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

def dist_restart(restart_file, model, optimizer):
    if restart_file is not None:
        print("Restarting training from a saved model and optimizer state...", flush=True)
        checkpoint = torch.load(restart_file)
        state_dict = checkpoint['model_state_dict']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if dist.is_available() and dist.is_initialized():
            # If the model was saved with DDP, remove the 'module' prefix that it might have (just in case)
            if 'module.' in next(iter(checkpoint['model_state_dict'].keys())):
                prefix = 'module.'
                state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            # with the current training setup, the module prefix is already removed
            model.load_state_dict(state_dict)
        else:
            state_dict = remove_module_prefix(checkpoint['model_state_dict'])
            model.load_state_dict(state_dict)

    return model, optimizer