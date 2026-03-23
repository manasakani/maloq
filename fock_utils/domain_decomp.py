import torch
import numpy as np
import cupy as cp
import torch.distributed as dist
from mpi4py import MPI

class MergedStructure:
    def __init__(self, z, pos, edges):
        self.atomic_numbers = z
        self.atomic_positions = pos
        self.edge_matrix = edges
        # self.counts = None # Domain_Decomp will calculate split

        print("Created MergedStructure with ", len(z), " atoms and edge list ", edges)

class Domain_Decomp():
    def __init__(self, structure, device, partition_type='linear'):
        
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.comm = MPI.COMM_WORLD
        self.device = device
        self.global_edge_index = structure.edge_matrix

        # Partition nodes
        self.local_node_indices = self.partition_graph_nodes(structure, partition_type) 
        self.all_local_node_indices = self.comm.allgather(self.local_node_indices)              # outer list is per rank
        self.local_num_nodes = len(self.local_node_indices)
        # note: local_nodes would be the same as local_node_indices since the nodes are defined by their index in the global node list

        # Partion edges (each rank gets all edges with dst node in its local node list, so no communication needed for aggregation)
        is_own_edge = np.isin(structure.edge_matrix[0], self.local_node_indices)                # local edge mask in global edge list
        self.local_edge_indices = np.where(is_own_edge)[0]                                      # indices of the local edges in the global edge list
        self.all_local_edge_indices = self.comm.allgather(self.local_edge_indices)              # outer list is per rank
        
        self.local_num_edges = len(self.local_edge_indices)
        self.local_edges = structure.edge_matrix[:, self.local_edge_indices]     # the numbers correspond to the full set of nodes and edges in the structure

        # _________________________________________________________________________________________
        # initialize communication patterns for message passing

        # reorder the edge list so that the local edges are at the start of the list:
        # local_node_nums = np.arange(self.start_node, self.end_node)
        is_local = np.isin(self.local_edges[1, :], self.local_node_indices)
        src_edge_nodes = np.concatenate([self.local_edges[0, :][is_local], self.local_edges[0, :][~is_local]])
        dst_edge_nodes = np.concatenate([self.local_edges[1, :][is_local], self.local_edges[1, :][~is_local]])
        self.local_edges = np.stack([src_edge_nodes, dst_edge_nodes], axis=0)
        self.truly_local_num_edges = np.sum(is_local)
        self.is_truly_local_edge = is_local # store to perform this reorder on the fock edges later

        # message creation
        self.expand_edge_0 = self.init_comm_pattern_expand(0)     # dst node   
        self.expand_edge_1 = self.init_comm_pattern_expand(1)     # src node

        # aggregation
        self.reduce_edge = self.init_comm_pattern_reduce(self.local_edges[0, :])

        # --> Shuffle gpus for topology-optimized partition assignment
        # rank_topology_assignment = redistribute_partitions(self)
        # structure.shuffle_partitions(rank_topology_assignment) ?
        # call init on self?


    def print_info(self):
        self.comm.Barrier()
        for i in range(self.size):
            if self.rank == i:
                lines = [
                    "________________________________________________________",
                    f"Rank {self.rank} has {len(self.local_node_indices)} nodes and {len(self.local_edges[0])} edges:",
                    f"Rank {self.rank} has nodes: {self.local_node_indices}",
                    f"Rank {self.rank} has edges: {self.local_edges}",
                    f"Rank {self.rank} expand edge 0 (dst) nodes_to_send: {self.expand_edge_0['nodes_to_send']}",
                    f"Rank {self.rank} expand edge 1 (src) nodes_to_send: {self.expand_edge_1['nodes_to_send']}"
                ]
                print("\n".join(lines), flush=True)
            self.comm.Barrier()

    def partition_graph_nodes(self, structure, partition_type):
        """Partition the graph and return the local node indices for this rank."""

        # --> Split nodes between ranks
        if partition_type == 'linear':
            total_num_nodes = len(structure.atomic_numbers) 
            local_num_nodes = total_num_nodes // self.size
            counts = np.array([local_num_nodes] * self.size, dtype=np.int32)
            for i in range(total_num_nodes % self.size):
                counts[i] += 1

            displacements = np.zeros_like(counts)
            for i in range(1, len(counts)):
                displacements[i] = displacements[i-1] + counts[i-1]

            # --> Naive partition assignment (rank 0 gets the first partition, etc)
            start_node = displacements[self.rank]
            end_node = displacements[self.rank] + counts[self.rank]
            local_num_nodes = counts[self.rank]
            local_node_indices = np.arange(start_node, end_node) # indices of the local nodes in the global list
        
        else:
            raise ValueError("Graph node partition type not recognized!")
        
        return local_node_indices


    def init_comm_pattern_expand(self, dst0_or_src1):

        with torch.no_grad():

            edge_index = self.local_edges[dst0_or_src1, :]

            if torch.is_tensor(edge_index):
                edge_index_np = edge_index.detach().cpu().contiguous().numpy().astype(np.int32)
            else:
                edge_index_np = np.ascontiguousarray(edge_index, dtype=np.int32)

            # expand edge:
            local_num_nodes = len(self.local_node_indices)
            total_num_nodes = self.comm.allreduce(local_num_nodes, op=MPI.SUM)
            num_nodes_local = total_num_nodes // self.size

            #  get 'remote' nodes in this rank to be recieved from remote ranks
            remote_node_ranks = []
            remote_nodes = []
            for node in edge_index:
                # if the node is not local, it is remote
                if node not in self.local_node_indices:
                    for i in range(self.size):
                        if i == self.rank:
                            continue
                        # check if the node is in the local nodes for rank i
                        if node in self.all_local_node_indices[i]: 
                            if node not in remote_nodes:
                                remote_node_ranks.append(i)
                                remote_nodes.append(node)
                            break 
            
            # print(f"Rank {self.rank} remote nodes: {remote_nodes} from ranks {remote_node_ranks}", flush=True)

            # Nodes to recieve on this rank 
            nodes_to_recv = {}
            for i, remote_rank in enumerate(remote_node_ranks):
                if remote_rank not in nodes_to_recv:
                    nodes_to_recv[remote_rank] = []
                if remote_nodes[i].item() not in nodes_to_recv[remote_rank]:
                    nodes_to_recv[remote_rank].append(remote_nodes[i].item())

            # print(f"Rank {self.rank} nodes to receive: {nodes_to_recv}", flush=True)

            # the nodes in the global edge list corresponding to the dst or src of the local edges
            global_edge_nodes = self.global_edge_index[dst0_or_src1, :] 

            # Nodes to send from this rank
            nodes_to_send = {}
            # if 'this' rank has a node which 'that' rank does not, add it to the nodes to send
            for i in range(self.size):
                # look at all the nodes in the edge index for rank i ('that' rank)
                that_ranks_edge_nodes = global_edge_nodes[self.all_local_edge_indices[i]] 
                for node in that_ranks_edge_nodes:
                    # if the node is not in the local nodes for rank 1, but is in the current local nodes:
                    if node in self.local_node_indices and node not in self.all_local_node_indices[i]:
                        # add the note to the send list, i is the rank to send to
                        if i not in nodes_to_send:
                            nodes_to_send[i] = []

                        if node not in nodes_to_send[i]:
                            nodes_to_send[i].append(int(node))

            # print(f"Rank {self.rank} nodes to send: {nodes_to_send}", flush=True)

            indices_to_send = {}
            for target_rank, nodes in nodes_to_send.items():
                if nodes:
                    nodes_tensor = torch.tensor(nodes, dtype=torch.int64, requires_grad=False)
                    indices = torch.empty_like(nodes_tensor)
                    for j, node in enumerate(nodes_tensor):
                        idx = torch.where(self.local_node_indices == node)[0]
                        indices[j] = idx
                    indices_to_send[target_rank] = indices.to(self.device)
            
            # print(f"Rank {self.rank} indices to send: {indices_to_send}", flush=True)

            # indices of local embedding to slot into the new embedding
            # this should be the same as edge_index[is_local] - self.start_node, but works even if the local nodes are not a contiguous chunk of the global node list
            is_local = np.isin(edge_index, self.local_node_indices)
            node_to_local_pos = {int(node): i for i, node in enumerate(self.local_node_indices)}
            local_indices = [node_to_local_pos[int(node)] for node in edge_index[is_local]]
            # print(f"Rank {self.rank} local indices: {local_indices}", flush=True)

            # indices of recieved remote embeddings to slot into the new embedding
            is_remote = np.isin(edge_index, remote_nodes)
            remote_edge_nodes = torch.from_numpy(edge_index[is_remote]).to(self.device)
            remote_indices = torch.ones(len(remote_edge_nodes), dtype=torch.long, device=self.device) 

            node_track = 0
            for i, (source_rank, nodes) in enumerate(nodes_to_recv.items()):
                if nodes: 
                    for node in nodes:  # node is the identity of the recieved node, not the index
                        remote_indices[torch.where(remote_edge_nodes == node)[0]] = node_track # locations in the new embedding where this recieved embedding should go
                        node_track += 1 # track the number of nodes received

            # print(f"Rank {self.rank} remote indices: {remote_indices}", flush=True)

            expand_edge_dict = {}
            expand_edge_dict['local_indices'] = local_indices
            expand_edge_dict['remote_indices'] = remote_indices
            expand_edge_dict['is_local'] = is_local
            expand_edge_dict['is_remote'] = is_remote
            expand_edge_dict['nodes_to_send'] = nodes_to_send
            expand_edge_dict['indices_to_send'] = indices_to_send
            expand_edge_dict['nodes_to_recv'] = nodes_to_recv

            # Convert PyTorch tensor directly to CuPy array to use when indexing flattened embeddings
            indices_to_send_cp = {
                target_rank: cp.asarray(nodes.to(self.device))  
                for target_rank, nodes in indices_to_send.items()
            }
            expand_edge_dict['indices_to_send_cp'] = indices_to_send_cp

            # torch versions of some index arrays, to allow for skipping of memory copies
            local_indices_torch = torch.tensor(local_indices, dtype=torch.long, device=self.device)
            remote_indices_torch = torch.tensor(remote_indices, dtype=torch.long, device=self.device)
            expand_edge_dict['local_indices_torch'] = local_indices_torch
            expand_edge_dict['remote_indices_torch'] = remote_indices_torch

        return expand_edge_dict

    def init_comm_pattern_reduce(self, edge_index):

        # rank = dist.get_rank()
        # size = dist.get_world_size()
        # comm = MPI.COMM_WORLD

        # local_num_nodes = len(self.local_node_indices)
        # total_num_nodes = self.comm.allreduce(local_num_nodes, op=MPI.SUM)
        # num_nodes_local = total_num_nodes // self.size

        # # nodes owned by this rank
        # local_node_nums = torch.arange(self.start_node, self.end_node)

        # # start and end nodes on every rank:
        # start_nodes = self.comm.allgather(self.start_node)
        # end_nodes = self.comm.allgather(self.end_node)

        # # allgather the edge_indices on each rank to make a global_edge_index and counts and displacements:
        # length_local_edge_idx = len(edge_index)
        # edge_index_np = edge_index
        # counts = comm.allgather(length_local_edge_idx)
        # displacements = [0] + [sum(counts[:i]) for i in range(1, size)]

        # # total_length_edge_idx = sum(counts)
        # global_edge_index = self.global_edge_index[0, :] #torch.zeros(total_length_edge_idx, dtype=torch.int64)
        # # comm.Allgatherv(edge_index_np, [global_edge_index, counts, displacements, MPI.LONG]) # INCORRECT!!!
        
        # local_edge_idx = torch.arange(self.start_edge, self.end_edge)
        # local_edge_idx = local_edge_idx.to(self.device)
        # self.local_edge_idx = local_edge_idx

        # # messages to send are in the form of {rank: [indices of own self.embedding to send to rank]}
        # messages_to_send = {}
        # for i, target_node in enumerate(edge_index):
        #     if target_node in local_node_nums:
        #         pass # this is where the self-edges are handled
        #     else:
        #         for j, (start, end) in enumerate(zip(start_nodes, end_nodes)):
        #             if target_node >= start and target_node < end:
        #                 if j not in messages_to_send:
        #                     messages_to_send[j] = []
        #                 messages_to_send[j].append(i)
        #                 break

        # # messages to send are in the form of {rank: [indices of rank's embedding to be recieved]}
        # messages_to_recv = {}
        # for i, target_node in enumerate(global_edge_index):
        #     if target_node in local_node_nums and i not in local_edge_idx: # check here
        #         for j, (c, d) in enumerate(zip(counts, displacements)):
        #             if i >= d and i < d + c:
        #                 if j not in messages_to_recv:
        #                     messages_to_recv[j] = []
        #                 messages_to_recv[j].append(i-d)
        #                 break

        # for dest_rank, embedding_idxs in messages_to_send.items():
        #     if embedding_idxs:
        #         messages_to_send[dest_rank] = torch.tensor(embedding_idxs, dtype=torch.int64, device=self.device)

        # edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        # local_node_nums = torch.tensor(local_node_nums, dtype=torch.long, device=self.device)

        # is_local = (edge_index >= self.start_node) & (edge_index < self.end_node)
        # is_local = torch.isin(edge_index, self.local_node_indices)
        # local_indices = edge_index[is_local] - self.start_node
        # local_indices = self.local_node_indices.index(edge_index[is_local]) # this should be the same as edge_index[is_local] - self.start_node, but works even if the local nodes are not a contiguous chunk of the global node list
        is_local = np.isin(edge_index, self.local_node_indices)
        node_to_local_pos = {int(node): i for i, node in enumerate(self.local_node_indices)}
        local_indices = [node_to_local_pos[int(node)] for node in edge_index[is_local]]
        local_indices = torch.tensor(local_indices, dtype=torch.long, device=self.device)

        # # get remote indices to write into
        # recv_pointer = 0
        # slot_pointer = 0
        # num_msgs_to_recv = sum([len(msgs) for msgs in messages_to_recv.values()])
        # remote_indices = torch.zeros(num_msgs_to_recv, dtype=torch.long, device=self.device)       # already start collecting where the embeddings should go
        # for source_rank, embedding_idxs in messages_to_recv.items():

        #     if embedding_idxs:
        #         node_start = displacements[source_rank]
        #         for j, idx in enumerate(embedding_idxs):
        #             node_to_sum_into = global_edge_index[node_start + idx]
        #             remote_indices[slot_pointer] = torch.where(local_node_nums == node_to_sum_into)[0].item()
        #             slot_pointer += 1


        reduce_edge_dict = {}
        reduce_edge_dict['is_local'] = is_local
        reduce_edge_dict['local_indices'] = local_indices
        # reduce_edge_dict['remote_indices'] = remote_indices
        # reduce_edge_dict['messages_to_send'] = messages_to_send
        # reduce_edge_dict['messages_to_recv'] = messages_to_recv
        # reduce_edge_dict['local_node_nums'] = local_node_nums
        # reduce_edge_dict['global_edge_index'] = global_edge_index
        # reduce_edge_dict['start_node'] = self.start_node    
        # reduce_edge_dict['end_node'] = self.end_node
        # reduce_edge_dict['start_nodes'] = start_nodes
        # reduce_edge_dict['counts'] = counts
        # reduce_edge_dict['displacements'] = displacements

        return reduce_edge_dict