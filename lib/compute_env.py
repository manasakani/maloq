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
        
        # splitting the nodes and edges among the ranks
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
        self.global_atomic_numbers = torch.tensor(structure.atomic_numbers, device=self.device)

        # created and assigned during data creation:
        self.global_edge_distance_vec = None

        # _________________________________________________________________________________________
        # initialize communication patterns for message passing

        # message creation
        self.expand_edge_0 = self.init_comm_pattern_expand(self.local_edge_index[0, :])     # dst node
        self.expand_edge_1 = self.init_comm_pattern_expand(self.local_edge_index[1, :])     # src node

        # aggregation
        self.reduce_edge = self.init_comm_pattern_reduce(self.local_edge_index[0, :])
        # self.reduce_edge = self.init_comm_pattern_reduce(self.local_edge_index[1, :]) # WRONG EDGE FOR TESTING


    def print_info(self):
        dist.barrier()
        for i in range(self.size):
            if self.rank == i:
                print("________________________________________________________")
                print(f"Rank {self.rank} has {self.end_node - self.start_node} nodes and {self.end_edge - self.start_edge} edges:")
                print(f"Rank {self.rank} has nodes from {self.start_node} to {self.end_node}: {self.local_node_index}")
                print(f"Rank {self.rank} has edges from {self.start_edge} to {self.end_edge}: {self.local_edge_index}")
            self.comm.Barrier()

    def init_comm_pattern_expand(self, edge_index):

        # expand edge:
        local_num_nodes = len(self.local_node_index)
        total_num_nodes = self.comm.allreduce(local_num_nodes, op=MPI.SUM)
        num_nodes_local = total_num_nodes // self.size
        local_node_nums = torch.arange(self.start_node, self.end_node)
        
        # start and end nodes on every rank:
        start_nodes = self.comm.allgather(self.start_node)
        end_nodes = self.comm.allgather(self.end_node)

        #  get 'remote' nodes in this rank to be recieved from remote ranks
        remote_node_ranks = []
        remote_nodes = []
        for node in edge_index:
            if node < self.start_node or node >= self.end_node:
                for i, (start, end) in enumerate(zip(start_nodes, end_nodes)):
                    if node >= start and node < end:
                        if node not in remote_nodes:
                            remote_node_ranks.append(i)
                            remote_nodes.append(node)
                        break                

        # Nodes to recieve on this rank
        nodes_to_recv = {}
        for i, remote_rank in enumerate(remote_node_ranks):
            if remote_rank not in nodes_to_recv:
                nodes_to_recv[remote_rank] = []
            if remote_nodes[i].item() not in nodes_to_recv[remote_rank]:
                nodes_to_recv[remote_rank].append(remote_nodes[i].item())

        # allgatherv the edge_indices on each rank:
        length_local_edge_idx = len(edge_index)
        edge_index_np = edge_index
        counts = self.comm.allgather(length_local_edge_idx)
        displacements = [0] + [sum(counts[:i]) for i in range(1, self.size)]

        total_length_edge_idx = sum(counts)        
        all_edge_idx = torch.zeros(total_length_edge_idx, dtype=torch.int64)
        self.comm.Allgatherv(edge_index_np, [all_edge_idx, counts, displacements, MPI.LONG])

        # Nodes to send from this rank
        nodes_to_send = {}
        # iterate over all_edge_idx, if this rank has a node which that rank does not, add it to the nodes to send
        for i, (c, d) in enumerate(zip(counts, displacements)):
            # look at all the nodes in the edge index for rank i
            for node in all_edge_idx[d:d+c]:
                # if the node is not in the local nodes for rank 1, but is in the current local nodes:
                if node in local_node_nums:
                    if node < start_nodes[i] or node >= end_nodes[i]:
                        # add the note to the send list, i is the rank to send to
                        if i not in nodes_to_send:
                            nodes_to_send[i] = []

                        if node not in nodes_to_send[i]:
                            nodes_to_send[i].append(node)

        dist.barrier()
        print("rank ", self.rank, " Nodes to send (during message creation): ", nodes_to_send)
        print("rank ", self.rank, " Nodes to recv (during message creation): ", nodes_to_recv)
        dist.barrier()

        expand_edge_dict = {}
        expand_edge_dict['nodes_to_send'] = nodes_to_send
        expand_edge_dict['nodes_to_recv'] = nodes_to_recv
        expand_edge_dict['remote_nodes'] = remote_nodes
        expand_edge_dict['remote_node_ranks'] = remote_node_ranks
        expand_edge_dict['local_node_nums'] = local_node_nums
        expand_edge_dict['start_node'] = self.start_node
        expand_edge_dict['end_node'] = self.end_node

        return expand_edge_dict


    def init_comm_pattern_reduce(self, edge_index):

        rank = dist.get_rank()
        size = dist.get_world_size()
        comm = MPI.COMM_WORLD

        local_num_nodes = len(self.local_node_index)
        total_num_nodes = self.comm.allreduce(local_num_nodes, op=MPI.SUM)
        num_nodes_local = total_num_nodes // self.size

        # nodes owned by this rank
        local_node_nums = torch.arange(self.start_node, self.end_node)

        # start and end nodes on every rank:
        start_nodes = self.comm.allgather(self.start_node)
        end_nodes = self.comm.allgather(self.end_node)
        # print("Total number of nodes: ", total_num_nodes, "rank: ", rank, "start nodes: ", start_nodes, " end nodes: ", end_nodes)

        # allgather the edge_indices on each rank to make a global_edge_index and counts and displacements:
        length_local_edge_idx = len(edge_index)
        edge_index_np = edge_index
        counts = comm.allgather(length_local_edge_idx)
        displacements = [0] + [sum(counts[:i]) for i in range(1, size)]

        total_length_edge_idx = sum(counts)
        global_edge_index = torch.zeros(total_length_edge_idx, dtype=torch.int64)
        comm.Allgatherv(edge_index_np, [global_edge_index, counts, displacements, MPI.LONG])

        local_edge_index_torch = torch.tensor(self.local_edge_index, device=self.device)
        local_edge_idx = (self.global_edge_index.T.unsqueeze(1) == local_edge_index_torch.T.unsqueeze(0)).all(dim=2).nonzero(as_tuple=True)[0]
        
        # messages to send are in the form of {rank: [indices of own self.embedding to send to rank]}
        messages_to_send = {}
        for i, target_node in enumerate(edge_index):
            if target_node in local_node_nums:
                pass # this is where the self-edges are handled
            else:
                for j, (start, end) in enumerate(zip(start_nodes, end_nodes)):
                    if target_node >= start and target_node < end:
                        if j not in messages_to_send:
                            messages_to_send[j] = []
                        messages_to_send[j].append(i)
                        break

        # messages to send are in the form of {rank: [indices of rank's embedding to be recieved]}
        messages_to_recv = {}
        for i, target_node in enumerate(global_edge_index):
            if target_node in local_node_nums and i not in local_edge_idx: # check here
                for j, (c, d) in enumerate(zip(counts, displacements)):
                    if i >= d and i < d + c:
                        if j not in messages_to_recv:
                            messages_to_recv[j] = []
                        messages_to_recv[j].append(i-d)
                        break

        # the messages are the indices of the local embeddings on the source rank
        dist.barrier()
        print(f"Rank {rank}: messages_to_send (during message aggregation) = {messages_to_send}")
        print(f"Rank {rank}: messages_to_recv (during message aggregation) = {messages_to_recv}")
        dist.barrier()
        
        reduce_edge_dict = {}
        reduce_edge_dict['messages_to_send'] = messages_to_send
        reduce_edge_dict['messages_to_recv'] = messages_to_recv
        reduce_edge_dict['local_node_nums'] = local_node_nums
        reduce_edge_dict['global_edge_index'] = global_edge_index
        reduce_edge_dict['start_node'] = self.start_node    
        reduce_edge_dict['end_node'] = self.end_node
        reduce_edge_dict['start_nodes'] = start_nodes
        reduce_edge_dict['counts'] = counts
        reduce_edge_dict['displacements'] = displacements

        return reduce_edge_dict
        