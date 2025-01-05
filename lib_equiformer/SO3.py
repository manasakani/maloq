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
import time
import cupy as cp
import utils

try:
    from e3nn import o3
    from e3nn.o3 import FromS2Grid, ToS2Grid
except ImportError:
    pass

from wigner import wigner_D
from torch.nn import Linear
from mpi4py import MPI
import torch.distributed as dist
from cupy import cuda
from cupy.cuda import nccl
from cupyx.distributed import NCCLBackend

class _global_nccl_comm():
    def __init__(
        self
    ):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        # uid = nccl.get_unique_id()
        # uid = comm.bcast(uid, root=0)
        # self.comm = nccl.NcclCommunicator(self.size, uid, self.rank)
        self.comm = NCCLBackend(self.size, self.rank, use_mpi=True)
        self.stream = cp.cuda.Stream(non_blocking=True)

comm_nccl = _global_nccl_comm()

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

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

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
        Flattens the embedding for communication 
        """

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.contiguous().view(-1)

        return cp.from_dlpack(embedding.detach()) # convert to cupy array
    
    # Expand the node embeddings to the number of edges
    def _expand_edge_old(self, edge_index, expand_edge_dict):
        
        edge_embeddings = self.embedding[edge_index]
        self.set_embedding(edge_embeddings)


    # Expand the node embeddings to the number of edges - Distributed version
    def _expand_edge(self, edge_index, expand_edge_dict):
        """
        It's now been extended to account for the distributed case. The node embeddings which are needed on the current rank (due to being in the
        edge_index) but only exist on other ranks are communicated using non-blocking p2p communication.
        """
        rank = self.rank
        size = self.size
        comm = comm_nccl.comm
        stream = comm_nccl.stream

        start_node = expand_edge_dict['start_node']
        end_node = expand_edge_dict['end_node']

        # Get precomputed communication plan
        local_indices = expand_edge_dict['local_indices']
        remote_indices = expand_edge_dict['remote_indices']
        is_local = expand_edge_dict['is_local'] 
        is_remote = expand_edge_dict['is_remote']
        nodes_to_send = expand_edge_dict['nodes_to_send']
        nodes_to_recv = expand_edge_dict['nodes_to_recv']
        remote_nodes = expand_edge_dict['remote_nodes']
        remote_node_ranks = expand_edge_dict['remote_node_ranks']
        local_node_nums = expand_edge_dict['local_node_nums']
        indices_to_send = expand_edge_dict['indices_to_send']
        displacements = expand_edge_dict['displacements']
        global_edge_idx = expand_edge_dict['global_edge_idx']

        # --> Send/Receive embeddings
        num_nodes_to_recv = len(remote_indices)

        cupy_buffer_dtype = utils.dtype_converter(self.dtype, input_library='torch', output_library='cupy')
        recv_bufs = cp.empty(
            num_nodes_to_recv * self.num_coefficients * self.num_channels,
            dtype=cupy_buffer_dtype
        )

        # Non-blocking sends
        sendbufs = {}
        for target_rank, nodes in nodes_to_send.items():
            if nodes:
                sendbufs[target_rank] = self._flatten_embedding(self.embedding[indices_to_send[target_rank]])

        cp.cuda.runtime.deviceSynchronize()
        nccl.groupStart()
        for target_rank, nodes in nodes_to_send.items():
            if nodes:
                comm.send(sendbufs[target_rank], target_rank, stream=stream)


        # Non-blocking recvs posts
        recv_pointer = 0
        for i, (source_rank, nodes) in enumerate(nodes_to_recv.items()):
            if nodes: 
                start_idx = recv_pointer
                end_idx = start_idx + len(nodes) * self.num_coefficients * self.num_channels
                comm.recv(recv_bufs[start_idx:end_idx], source_rank, stream=stream)
                recv_pointer = end_idx

        nccl.groupEnd()
        cp.cuda.runtime.deviceSynchronize()

        # --> Slot in the local embeddings while waiting for comm
        edge_embeddings = torch.empty((len(edge_index), self.embedding.shape[1], self.embedding.shape[2]), device=self.device)
        edge_embeddings[is_local] = self.embedding[local_indices]
        
        # --> Slot in the recieved remote embeddings
        if num_nodes_to_recv:
            received_embeddings = recv_bufs.reshape(num_nodes_to_recv, self.num_coefficients, self.num_channels)
            received_embeddings = torch.tensor(received_embeddings, device=self.device)
            edge_embeddings[is_remote] = received_embeddings[remote_indices]

        self.set_embedding(edge_embeddings)


    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge_old(self, edge_index, local_edge_idx, num_nodes, reduce_edge_dict):

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
    def _reduce_edge(self, edge_index, local_edge_idx, num_nodes, reduce_edge_dict):
        """
        Distributed aggregation is now done with 2x in-place index_add_ operations, 
        once for the local messages and once for the remote messages.
        """

        # make the new set of embeddings
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )

        # start_time = time.time()

        rank = self.rank
        size = self.size
        comm = comm_nccl.comm
        stream = comm_nccl.stream

        # Get precomputed communication plan
        is_local = reduce_edge_dict['is_local']
        local_indices = reduce_edge_dict['local_indices']
        remote_indices = reduce_edge_dict['remote_indices']
        local_node_nums = reduce_edge_dict['local_node_nums']               # numbers of the nodes onwed by the current rank
        start_node = reduce_edge_dict['start_node']                         # start node of the current rank
        end_node = reduce_edge_dict['end_node']                             # end node of the current rank
        messages_to_send = reduce_edge_dict['messages_to_send']             # dictionary of messages to send to other ranks in the form {rank: [indices to send from current rank]}
        messages_to_recv = reduce_edge_dict['messages_to_recv']             # dictionary of messages to receive from other ranks in the form {rank: [rank's indices to receive]}
        global_edge_index = reduce_edge_dict['global_edge_index']           # global edge index of the destination nodes
        displacements = reduce_edge_dict['displacements']                   # displacements of the nodes owned by each rank (used for allgatherv)
        
        num_msgs_to_recv = len(remote_indices)
        
        cupy_buffer_dtype = utils.dtype_converter(self.dtype, input_library='torch', output_library='cupy')
        recv_bufs = cp.empty(
            num_msgs_to_recv * self.num_coefficients * self.num_channels,
            dtype=cupy_buffer_dtype
        )

        sendbufs = {}
        for target_rank, embedding_idxs in messages_to_send.items():
            if len(embedding_idxs) > 0:
                sendbufs[target_rank] = self._flatten_embedding(self.embedding[embedding_idxs])

        cp.cuda.runtime.deviceSynchronize()
        nccl.groupStart()
        # Non-blocking sends
        for target_rank, embedding_idxs in messages_to_send.items():
            if len(embedding_idxs) > 0:
                comm.send(sendbufs[target_rank], target_rank, stream=stream)

        # Non-blocking recvs
        recv_pointer = 0
        for source_rank, embedding_idxs in messages_to_recv.items():
            if len(embedding_idxs) > 0:
                start_idx = recv_pointer
                end_idx = start_idx + len(embedding_idxs) * self.num_coefficients * self.num_channels
                comm.recv(recv_bufs[start_idx:end_idx], source_rank, stream=stream)
                recv_pointer = end_idx

        nccl.groupEnd()
        cp.cuda.runtime.deviceSynchronize()
        # NOTE: This is a blocking operation and such no overlap currently

        # --> aggregate the local embeddings while waiting for comm
        new_embedding.index_add_(0, local_indices, self.embedding[is_local])

        received_embeddings = recv_bufs.reshape(num_msgs_to_recv, self.num_coefficients, self.num_channels)
        received_embeddings = torch.tensor(received_embeddings, device=self.device)

        # --> aggregate the remote embeddings recieved
        new_embedding.index_add_(0, remote_indices, received_embeddings)

        self.set_embedding(new_embedding)


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