import os
import torch.distributed as dist
from mpi4py import MPI
import torch
import functools
import numpy as np

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

        gpu_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        print(f"Rank {rank} is using GPU {gpu_id}")
        print("Total number of GPUs found: ", torch.cuda.device_count())

        backend = 'gloo'  # Use Gloo for attelas 
        # backend = 'nccl'  # Use Gloo for burmy

        dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)
        rank_zero_print("Initialized process group in: local", flush=True)
        rank_zero_print(f"Backend: {backend}", flush=True)

    rank_zero_print(f"RANK: {rank}", flush=True)
    rank_zero_print(f"WORLD_SIZE: {world_size}", flush=True)
    rank_zero_print(f"LOCAL_RANK: {local_rank}", flush=True)
    dist.barrier()

    return device, world_size

# check this function
def remove_module_prefix(state_dict):
    prefix = 'module.'
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def only_rank_zero(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

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


def allgatherv_cpu_numpy_1D(local_data, device, comm):

    local_data_shape = local_data.shape
    local_data = local_data.numpy().reshape(-1)
    local_data_size = len(local_data)

    global_data_counts = comm.allgather(local_data_size)
    global_data_displs = np.cumsum([0] + global_data_counts[:-1])
    global_data = np.empty(sum(global_data_counts), dtype=local_data.dtype)

    comm.Allgatherv(local_data, [global_data, global_data_counts, global_data_displs, MPI.DOUBLE])

    return global_data


class Domain_Decomp():
    def __init__(self, structure, device):
        
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.comm = MPI.COMM_WORLD
        self.device = device

        total_num_nodes = len(structure.atomic_numbers) 
        total_num_edges = structure.edge_matrix.shape[1]
        
        local_num_nodes = total_num_nodes // self.size
        local_num_edges = total_num_edges // self.size

        start_node = self.rank * local_num_nodes
        end_node = start_node + local_num_nodes
        start_edge = self.rank * local_num_edges
        end_edge = start_edge + local_num_edges

        if self.rank == self.size - 1:
            local_num_nodes += total_num_nodes % self.size
            end_node += total_num_nodes % self.size
        if self.rank == self.size - 1:
            local_num_edges += total_num_edges % self.size
            end_edge += total_num_edges % self.size
        
        self.start_node = start_node
        self.end_node = end_node
        self.start_edge = start_edge
        self.end_edge = end_edge

        # the numbers correspond to the full set of nodes and edges in the structure
        self.local_node_index = np.arange(start_node, end_node)
        self.local_edge_index = structure.edge_matrix[:, start_edge:end_edge]
        global_edge_index = structure.edge_matrix
        self.global_edge_index = torch.tensor(global_edge_index, device=self.device)

        # assigned during data creation:
        self.global_edge_distance_vec = None
        self.global_atomic_numbers = torch.tensor(structure.atomic_numbers, device=self.device)

    def print_info(self):
        dist.barrier()
        for i in range(self.size):
            if self.rank == i:
                print("________________________________________________________")
                print(f"Rank {self.rank} has {self.end_node - self.start_node} nodes and {self.end_edge - self.start_edge} edges:")
                print(f"Rank {self.rank} has nodes from {self.start_node} to {self.end_node}")
                print(f"Rank {self.rank} has edges from {self.start_edge} to {self.end_edge}")
            self.comm.Barrier()

        


        