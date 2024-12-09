"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


TODO:
    1. Simplify the case when `num_resolutions` == 1.
    2. Remove indexing when the shape is the same.
    3. Move some functions outside classes and to separate files.
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np

try:
    from e3nn import o3
    from e3nn.o3 import FromS2Grid, ToS2Grid
except ImportError:
    pass

from wigner import wigner_D
from torch.nn import Linear
from mpi4py import MPI
import torch.distributed as dist

class CoefficientMappingModule(torch.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax,
        mmax,
    ):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax

        # Temporarily use `cpu` as device and this will be overwritten.
        self.device = 'cpu'
        
        # Compute the degree (l) and order (m) for each entry of the embedding
        l_harmonic = torch.tensor([], device=self.device).long()
        m_harmonic = torch.tensor([], device=self.device).long()
        m_complex  = torch.tensor([], device=self.device).long()

        res_size = torch.zeros([1], device=self.device).long() # 1 used to be `num_resolutions`

        offset = 0
        for l in range(0, self.lmax + 1):
            mmax = min(self.mmax, l)
            m = torch.arange(-mmax, mmax + 1, device=self.device).long()
            m_complex = torch.cat([m_complex, m], dim=0)
            m_harmonic = torch.cat(
                [m_harmonic, torch.abs(m).long()], dim=0
            )
            l_harmonic = torch.cat(
                [l_harmonic, m.fill_(l).long()], dim=0
            )
        res_size[0] = len(l_harmonic) - offset
        offset = len(l_harmonic)

        num_coefficients = len(l_harmonic)
        # `self.to_m` moves m components from different L to contiguous index
        to_m = torch.zeros([num_coefficients, num_coefficients], device=self.device)
        m_size = torch.zeros([self.mmax + 1], device=self.device).long()

        # The following is implemented poorly - very slow. It only gets called
        # a few times so haven't optimized.
        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)

            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)

            m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        # save tensors and they will be moved to GPU
        self.register_buffer('l_harmonic', l_harmonic)
        self.register_buffer('m_harmonic', m_harmonic)
        self.register_buffer('m_complex',  m_complex)
        self.register_buffer('res_size',   res_size)
        self.register_buffer('to_m',       to_m)
        self.register_buffer('m_size',     m_size)

        # for caching the output of `coefficient_idx`
        self.lmax_cache, self.mmax_cache = None, None
        self.mask_indices_cache = None
        self.rotate_inv_rescale_cache = None


    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        '''
            Add `m_complex` and `l_harmonic` to the input arguments 
            since we cannot use `self.m_complex`. 
        '''
        if lmax == -1:
            lmax = self.lmax

        indices = torch.arange(len(l_harmonic), device=self.device)
        # Real part
        mask_r = torch.bitwise_and(
            l_harmonic.le(lmax), m_complex.eq(m)
        )
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([], device=self.device).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(
                l_harmonic.le(lmax), m_complex.eq(-m)
            )
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i


    # Return mask containing coefficients less than or equal to degree (l) and order (m)
    def coefficient_idx(self, lmax, mmax):

        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.mask_indices_cache is not None:
                    return self.mask_indices_cache

        mask = torch.bitwise_and(
            self.l_harmonic.le(lmax), self.m_harmonic.le(mmax)
        )
        self.device = mask.device
        indices = torch.arange(len(mask), device=self.device)
        mask_indices = torch.masked_select(indices, mask)
        self.lmax_cache, self.mmax_cache = lmax, mmax
        self.mask_indices_cache = mask_indices
        return self.mask_indices_cache
    

    # Return the re-scaling for rotating back to original frame
    # this is required since we only use a subset of m components for SO(2) convolution
    def get_rotate_inv_rescale(self, lmax, mmax):

        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.rotate_inv_rescale_cache is not None:
                    return self.rotate_inv_rescale_cache
        
        if self.mask_indices_cache is None:
            self.coefficient_idx(lmax, mmax)
        
        rotate_inv_rescale = torch.ones((1, (lmax + 1)**2, (lmax + 1)**2), device=self.device)
        for l in range(lmax + 1):
            if l <= mmax:
                continue
            start_idx = l ** 2
            length = 2 * l + 1
            rescale_factor = math.sqrt(length / (2 * mmax + 1))
            rotate_inv_rescale[:, start_idx : (start_idx + length), start_idx : (start_idx + length)] = rescale_factor
        rotate_inv_rescale = rotate_inv_rescale[:, :, self.mask_indices_cache]        
        self.rotate_inv_rescale_cache = rotate_inv_rescale
        return self.rotate_inv_rescale_cache

    
    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, mmax={self.mmax})"


class SO3_Embedding():
    """
    Helper functions for performing operations on irreps embedding

    Args:
        length (int):           Batch size
        lmax   (int):           Maximum degree of the spherical harmonics
        num_channels (int):     Number of channels
        device:                 Device of the output
        dtype:                  type of the output tensors
    """

    def __init__(
        self,
        length,
        lmax,
        num_channels,
        device,
        dtype,
    ):
        super().__init__()

        self.lmax = lmax

        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype

        self.num_coefficients = 0
        self.num_coefficients = self.num_coefficients + int(
            (self.lmax + 1) ** 2
        )

        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(self.lmax, self.lmax)


    # Clone an embedding of irreps
    def clone(self):
        clone = SO3_Embedding(
            0,
            self.lmax,
            self.num_channels,
            self.device,
            self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone


    # Initialize an embedding of irreps
    def set_embedding(self, embedding):
        self.length = len(embedding)
        self.embedding = embedding


    # Set the maximum order to be the maximum degree
    def set_lmax_mmax(self, lmax, mmax):
        # if its a list: # TODO: check if this is correct
        if isinstance(lmax, list):
            self.lmax = lmax[0]
            self.mmax = mmax[0]
        else:
            self.lmax = lmax
            self.mmax = mmax

    # Flatten the input embedding for MPI4py
    def _flatten_embedding(self, embedding):
        """
        Flattens the embedding for communication via Allgatherv.
        """

        return embedding.view(-1).cpu().detach().numpy()

    # Flatten the self.embedding for MPI4py
    def flatten_embedding(self):
        """
        Flattens the embedding for communication via Allgatherv.
        """
        return self.embedding.view(-1).cpu().detach().numpy()

    # Restore embedding from flattened data
    def unflatten_embedding(self, flattened_data):
        """
        Reshapes the gathered flat data back into self.embedding's original shape.
        """
        restored_embedding = torch.from_numpy(flattened_data).to(self.device).to(self.dtype)
        restored_embedding = restored_embedding.view(self.length, self.num_coefficients, self.num_channels)
        self.set_embedding(restored_embedding)


    # Expand the node embeddings to the number of edges
    def _expand_edge(self, edge_index):
        """
        This function was originally quite simple on one node:
            edge_embeddings = self.embeddings[edge_index]
            self.set_embedding(edge_embeddings)
        
        It's now been extended to account for the distributed case. The node embeddings which are needed on the current rank (due to being in the
        edge_index) but only exist on other ranks are communicated using non-blocking p2p communication.
        """

        rank = dist.get_rank()
        size = dist.get_world_size()
        comm = MPI.COMM_WORLD

        print("____________________________________________________")
        print("start expand edge")
        print("____________________________________________________")

        print("rank ", rank, " edge_index ", edge_index)

        local_num_nodes = self.length
        total_num_nodes = comm.allreduce(local_num_nodes, op=MPI.SUM)
        num_nodes_local = total_num_nodes // size
        # num_edges_local = total_num_nodes // size

        start_node = rank * num_nodes_local
        end_node = start_node + num_nodes_local
        # start_edge = rank * num_edges_local
        # end_edge = start_edge + num_edges_local

        if rank == size - 1:
            end_node += total_num_nodes % size

        local_node_nums = torch.arange(start_node, end_node)
        print("Rank: ", rank, "Local nodes: ", local_node_nums)

        # start and end nodes on every rank:
        start_nodes = comm.allgather(start_node)
        end_nodes = comm.allgather(end_node)
        print("Total number of nodes: ", total_num_nodes, "rank: ", rank, "start nodes: ", start_nodes, " end nodes: ", end_nodes)

        #  get 'remote' nodes in this rank to be recieved from remote ranks
        remote_node_ranks = []
        remote_nodes = []
        for node in edge_index:
            if node < start_node or node >= end_node:
                if rank == 0:
                    print(node)
                for i, (start, end) in enumerate(zip(start_nodes, end_nodes)):
                    if node >= start and node < end:
                        remote_node_ranks.append(i)
                        remote_nodes.append(node)
                        break

        print("rank ", rank, " Remote node ranks: ", remote_node_ranks)
        print("rank ", rank, " Remote nodes: ", remote_nodes) 

        # Nodes to recieve on this rank
        nodes_to_recv = {}
        for i, remote_rank in enumerate(remote_node_ranks):
            if remote_rank not in nodes_to_recv:
                nodes_to_recv[remote_rank] = []
            if remote_nodes[i].item() not in nodes_to_recv[remote_rank]:
                nodes_to_recv[remote_rank].append(remote_nodes[i].item())

        # allgatherv the edge_indices on each rank:
        length_local_edge_idx = len(edge_index)
        edge_index_np = edge_index.cpu().numpy()
        counts = comm.allgather(length_local_edge_idx)
        displacements = [0] + [sum(counts[:i]) for i in range(1, size)]

        total_length_edge_idx = sum(counts)        
        all_edge_idx = torch.zeros(total_length_edge_idx, dtype=torch.int64)
        comm.Allgatherv(edge_index_np, [all_edge_idx, counts, displacements, MPI.LONG])

        # Nodes to send from this rank
        nodes_to_send = {}
        # iterate over all_edge_idx, if this rank has a node which that rank does not, add it to the nodes to send
        for i, (c, d) in enumerate(zip(counts, displacements)):
            # look at all the nodes in the edge index for rank i
            for node in all_edge_idx[d:d+c]:
                # if the node is not in the local nodes for rank 1, but is in the current local nodes:
                if node not in range(start_nodes[i], end_nodes[i]) and node in local_node_nums:
                    # add the note to the send list, i is the rank to send to
                    if i not in nodes_to_send:
                        nodes_to_send[i] = []

                    if node not in nodes_to_send[i]:
                        nodes_to_send[i].append(node)
                    
        dist.barrier()
        print("rank ", rank, " Nodes to send: ", nodes_to_send)
        print("rank ", rank, " Nodes to recv: ", nodes_to_recv)
        dist.barrier()

        num_nodes_to_recv = sum([len(nodes) for nodes in nodes_to_recv.values()])
                
        # Send/Receive embeddings
        send_requests = []
        recv_bufs = np.empty(num_nodes_to_recv * self.num_coefficients * self.num_channels, dtype=np.float64)  # Hardcoded datatype!!!
        recv_source = np.empty(num_nodes_to_recv)

        # Non-blocking sends
        for target_rank, nodes in nodes_to_send.items():

            if nodes:

                nodes_tensor = torch.tensor(nodes, dtype=torch.long)
                indices = []
                for node in nodes_tensor:
                    idx = torch.where(local_node_nums == node)[0]
                    if idx.numel() == 0:
                        raise ValueError(f"comm error, check ln 392 in SO3.py")
                    indices.append(idx.item())

                print("rank ", rank, "Sending nodes: ", nodes, " in indices ", indices, " to rank: ", target_rank)

                sendbuf = self._flatten_embedding(self.embedding[indices])
                req = comm.isend(sendbuf, dest=target_rank, tag=rank)
                send_requests.append(req)

        # Non-blocking recvs
        recv_pointer = 0
        # recv_nodes = []
        for i, (source_rank, nodes) in enumerate(nodes_to_recv.items()):

            if nodes: 
                print(f"Rank {rank}: getting nodes {nodes} from rank {source_rank}")
                req = comm.irecv(source=source_rank, tag=source_rank)

                start_idx = recv_pointer
                end_idx = start_idx + len(nodes) * self.num_coefficients * self.num_channels
                recv_bufs[start_idx:end_idx] = req.wait()
                recv_source[i] = source_rank
                # recv_nodes.append(nodes)
                recv_pointer = end_idx
        
        # flatten recv_nodes:
        # recv_nodes = [item for sublist in recv_nodes for item in sublist]

        MPI.Request.Waitall(send_requests)

        received_embeddings = recv_bufs.reshape(num_nodes_to_recv, self.num_coefficients, self.num_channels)
        print("rank ", rank, " received embeddings shape: ", received_embeddings.shape)

        for i in range(len(received_embeddings)):
            print(rank, " ", i, " sum embedding ", np.sum(received_embeddings[i]))
        dist.barrier()

        # Expand edge embeddings with received remote embeddings
        dist.barrier()
        print("rank ", rank, " expanding its edge embeddings:")
        dist.barrier()

        edge_embeddings = []
        for i, node in enumerate(edge_index):
            if node.cpu() in local_node_nums:
                local_idx = node - start_node
                edge_embeddings.append(self.embedding[local_idx])

            elif node in remote_nodes:
                owner_rank = remote_node_ranks[remote_nodes.index(node)]
                # print("rank ", rank, " node: ", node, " owner_rank: ", owner_rank)
                # num_recv = len(nodes_to_recv[owner_rank])

                nodes_in_this_rank = nodes_to_recv[owner_rank]
                # print("rank ", rank, " num_recv: ", num_recv, " from owner_rank: ", owner_rank, " which has nodes ", nodes_in_this_rank)

                offset = nodes_in_this_rank.index(node)

                embedding_idx = recv_source.tolist().index(owner_rank) + offset
                edge_embeddings.append(torch.tensor(received_embeddings[embedding_idx]).to(self.device))
        
        dist.barrier()
        
        edge_embeddings = torch.stack(edge_embeddings)
        self.set_embedding(edge_embeddings)
        

    # Initialize an embedding of irreps of a neighborhood
    def expand_edge(self, edge_index):
        x_expand = SO3_Embedding(
            0,
            self.lmax,
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand

    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge_old(self, edge_index, num_nodes):

        # make the new set of embeddings
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )

        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)


    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge(self, edge_index, local_edge_idx, remote_edge_idx, num_nodes):
        """
        This should in theory be done with 2x in-place index_add_ operations, once for the local messages and once for the remote messages.
        """

        # make the new set of embeddings
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )

        rank = dist.get_rank()
        size = dist.get_world_size()
        comm = MPI.COMM_WORLD

        local_num_nodes = num_nodes
        total_num_nodes = comm.allreduce(local_num_nodes, op=MPI.SUM)
        num_nodes_local = total_num_nodes // size

        start_node = rank * num_nodes_local
        end_node = start_node + num_nodes_local

        if rank == size - 1:
            end_node += total_num_nodes % size

        local_node_nums = torch.arange(start_node, end_node)
        start_nodes = comm.allgather(start_node)
        end_nodes = comm.allgather(end_node)
        print("Total number of nodes: ", total_num_nodes, "rank: ", rank, "start nodes: ", start_nodes, " end nodes: ", end_nodes)

        # allgather the edge_indices on each rank to make a global_edge_index and counts and displacements:
        length_local_edge_idx = len(edge_index)
        edge_index_np = edge_index.cpu().numpy()
        counts = comm.allgather(length_local_edge_idx)
        displacements = [0] + [sum(counts[:i]) for i in range(1, size)]

        total_length_edge_idx = sum(counts)
        global_edge_index = torch.zeros(total_length_edge_idx, dtype=torch.int64)
        comm.Allgatherv(edge_index_np, [global_edge_index, counts, displacements, MPI.LONG])

        print("rank ", rank, " edge_index ", edge_index)
        print("rank ", rank, " global_edge_index ", global_edge_index)
        

        # for i, target_node in enumerate(edge_index):
        #     print(rank, " ", i, " sum embedding ", torch.sum(self.embedding[i]))
        # messages to send are in the form of {rank: [indices of own self.embedding to send to rank]}
        messages_to_send = {}
        for i, target_node in enumerate(edge_index):
            if target_node.cpu() in local_node_nums:
                local_idx = target_node - start_node
                new_embedding[local_idx] += self.embedding[i]
                if rank == 0:
                    print("target node: ", target_node, " local_idx: ", local_idx, " i: ", i, " new_embedding.shape: ", new_embedding.shape)
                # print("new embedding shape: ", new_embedding[local_idx].shape)
            else:
                for j, (start, end) in enumerate(zip(start_nodes, end_nodes)):
                    if target_node >= start and target_node < end:
                        if j not in messages_to_send:
                            messages_to_send[j] = []
                        messages_to_send[j].append(i)
                        break

        print("within aggregate")
        for i in range(size):
            if rank == i:
                print("rank ", rank, " new_embedding embedding: ", new_embedding)
            dist.barrier()


        # messages to send are in the form of {rank: [indices of rank's embedding to be recieved]}
        messages_to_recv = {}
        for i, target_node in enumerate(global_edge_index):
            if target_node.cpu() in local_node_nums and i not in local_edge_idx:
                for j, (c, d) in enumerate(zip(counts, displacements)):
                    if i >= d and i < d + c:
                        if j not in messages_to_recv:
                            messages_to_recv[j] = []
                        messages_to_recv[j].append(i-d)
                        break

        # in each case, the messages are the indices of the local embeddings on the source rank
        dist.barrier()
        print(f"Rank {rank}: messages_to_send = {messages_to_send}")
        print(f"Rank {rank}: messages_to_recv = {messages_to_recv}")
        dist.barrier()
        
        num_msgs_to_recv = sum([len(msgs) for msgs in messages_to_recv.values()])
        print("rank ", rank, " num_msgs_to_recv: ", num_msgs_to_recv)
                
        # Send/Receive embeddings
        send_requests = []
        recv_bufs = np.empty(num_msgs_to_recv * self.num_coefficients * self.num_channels, dtype=np.float64)  # Hardcoded datatype!!!
        recv_target_nodes = np.empty(num_msgs_to_recv)
        print("rank ", rank, " recv_bufs shape: ", recv_bufs.shape)

        # Non-blocking sends
        for dest_rank, embedding_idxs in messages_to_send.items():

            if embedding_idxs:

                embedding_idxs_tensor = torch.tensor(embedding_idxs, dtype=torch.long)
                sendbuf = self._flatten_embedding(self.embedding[embedding_idxs_tensor])
                req = comm.isend(sendbuf, dest=dest_rank, tag=rank)
                send_requests.append(req)

                print("rank ", rank, "Sending embedding_idxs: ", embedding_idxs, " to rank: ", dest_rank)
                print("type of sendbuf: ", type(sendbuf))
                print("rank ", rank, " sendbuf shape: ", sendbuf.shape)
                print("rank ", rank, " recv_bufs shape: ", recv_bufs.shape)

        # Non-blocking recvs
        recv_pointer = 0
        for i, (source_rank, embedding_idxs) in enumerate(messages_to_recv.items()):

            if embedding_idxs:

                req = comm.irecv(source=source_rank, tag=source_rank)

                start_idx = recv_pointer
                end_idx = start_idx + len(embedding_idxs) * self.num_coefficients * self.num_channels
                    
                print("rank ", rank, " receiving message into buffer of size: ", len(recv_bufs[start_idx:end_idx]))
                recv_bufs[start_idx:end_idx] = req.wait()
                print("rank ", rank, "received message from rank ", source_rank)
                    
                recv_target_nodes[i] = source_rank # this is the rank that the message came from
                recv_pointer = end_idx

        dist.barrier()
        MPI.Request.Waitall(send_requests)
        print("rank ", rank, " recieved all messages")
        dist.barrier()

        received_embeddings = recv_bufs.reshape(num_msgs_to_recv, self.num_coefficients, self.num_channels)
        print("rank ", rank, " received embeddings shape: ", received_embeddings.shape)
        

        # sum recieved embeddings into the local target nodes:
        for i, target_node in enumerate(global_edge_index):

            if target_node.cpu() in local_node_nums and i not in local_edge_idx:
                # get the rank which owns this message
                for j, (c, d) in enumerate(zip(counts, displacements)):
                    if i >= d and i < d + c:

                        # the rank which owns this message is j
                        # the node is target_node

                        # CHECK THAT IT WORKS FOR MULTIPLE MESSAGES FROM THE SAME REMOTE RANK
                        indices_remote_rank = messages_to_recv[j] # indices of the embeddings on the remote rank
                        for k, idx in enumerate(indices_remote_rank):
                            print("rank ", rank, " target node: ", target_node, " indices_remote_rank: ", idx)
                            # rank 1 is recieving a message coming to its node 1 (target_node) from the position of 2 in rank 0's list of embeddings

                            local_idx = torch.where(local_node_nums == target_node)[0]
                            
                            print("size of new_embedding[local_idx]: ", new_embedding[local_idx].shape)
                            new_embedding[local_idx] += torch.tensor(received_embeddings[k]).to(self.device) # hardcode
                            break

        # *******
        self.set_embedding(new_embedding)
        # *******


    # Reshape the embedding l -> m
    def _m_primary(self, mapping):
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, mapping.to_m)


    # Reshape the embedding m -> l
    def _l_primary(self, mapping):
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, mapping.to_m)


    # Rotate the embedding
    def _rotate(self, SO3_rotation, lmax, mmax):
        
        embedding_rotate = SO3_rotation[0].rotate(self.embedding, lmax, mmax)

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax, mmax)


    # Rotate the embedding by the inverse of the rotation matrix
    def _rotate_inv(self, SO3_rotation, mappingReduced):

        embedding_rotate = SO3_rotation[0].rotate_inv(self.embedding, self.lmax, self.mmax)

        self.embedding = embedding_rotate

        # Assume mmax = lmax when rotating back
        self.mmax = int(self.lmax)
        self.set_lmax_mmax(self.lmax, self.mmax)


class SO3_Rotation(torch.nn.Module):
    """
    Helper functions for Wigner-D rotations

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
    """

    def __init__(
        self,
        lmax,
    ):
        super().__init__()
        self.lmax = lmax
        self.mapping = CoefficientMappingModule(self.lmax, self.lmax)


    def set_wigner(self, rot_mat3x3):
        self.device, self.dtype = rot_mat3x3.device, rot_mat3x3.dtype
        length = len(rot_mat3x3)
        self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, self.lmax)
        self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()
        self.wigner = self.wigner.detach()
        self.wigner_inv = self.wigner_inv.detach()


    # Rotate the embedding
    def rotate(self, embedding, out_lmax, out_mmax):
        out_mask = self.mapping.coefficient_idx(out_lmax, out_mmax)
        wigner = self.wigner[:, out_mask, :]
        return torch.bmm(wigner, embedding)


    # Rotate the embedding by the inverse of the rotation matrix
    def rotate_inv(self, embedding, in_lmax, in_mmax):
        in_mask = self.mapping.coefficient_idx(in_lmax, in_mmax)
        wigner_inv = self.wigner_inv[:, :, in_mask]
        wigner_inv_rescale = self.mapping.get_rotate_inv_rescale(in_lmax, in_mmax)
        wigner_inv = wigner_inv * wigner_inv_rescale
        return torch.bmm(wigner_inv, embedding)


    # Compute Wigner matrices from rotation matrix
    def RotationToWignerDMatrix(self, edge_rot_mat, start_lmax, end_lmax):
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(
                alpha, beta, torch.zeros_like(alpha)
            ).transpose(-1, -2)
            @ edge_rot_mat
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
        wigner = torch.zeros(len(alpha), size, size, device=self.device)
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner.detach()


class SO3_LinearV2(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, bias=True):
        '''
            1. Use `torch.einsum` to prevent slicing and concatenation
            2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
            3. Applies bias to scalar features only
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)
        

    def forward(self, input_embedding):

        weight = torch.index_select(self.weight, dim=0, index=self.expand_index) # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum('bmi, moi -> bmo', input_embedding.embedding, weight) # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias #add bias to scalar features

        out_embedding = SO3_Embedding(
            0, 
            input_embedding.lmax, 
            self.out_features, 
            device=input_embedding.device, 
            dtype=input_embedding.dtype
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(input_embedding.lmax, input_embedding.lmax)

        return out_embedding
        

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"