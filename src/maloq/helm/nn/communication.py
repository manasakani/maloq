# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

import torch
import torch.distributed as dist
from mpi4py import MPI
import cupy as cp
import numpy as np
from cupy import cuda
from cupy.cuda import nccl
from cupyx.distributed import NCCLBackend
from torch.utils.dlpack import to_dlpack, from_dlpack
import time

class _global_nccl_comm():
    def __init__(
        self
    ):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.comm = NCCLBackend(self.size, self.rank, use_mpi=True)
        self.stream = cp.cuda.Stream(non_blocking=True)

comm_nccl = _global_nccl_comm()

# def exchange_nodes(embedding, num_edges, communication_dict, comm):
#     """
#     Exchange node embeddings between different processes using NCCL backend, according to communication_dict.
#     """

#     # dist.barrier()
#     comm = comm_nccl.comm
#     stream = comm_nccl.stream

#     num_nodes = embedding.shape[0]
#     num_coefficients = embedding.shape[1]
#     num_channels = embedding.shape[2]
    
#     # Get precomputed communication plan
#     is_local = communication_dict['is_local']
#     is_remote = communication_dict['is_remote']
#     local_indices = communication_dict['local_indices']
#     local_indices_torch = communication_dict['local_indices_torch']
#     remote_indices = communication_dict['remote_indices']
#     remote_indices_torch = communication_dict['remote_indices_torch']
#     nodes_to_send = communication_dict['nodes_to_send']                
#     indices_to_send = communication_dict['indices_to_send']            
#     indices_to_send_cp = communication_dict['indices_to_send_cp']  
#     nodes_to_recv = communication_dict['nodes_to_recv']           
    

#     with torch.no_grad():

#         # --> Send/Receive embeddings
#         num_nodes_to_recv = len(remote_indices)

#         if num_nodes_to_recv:

#             # Prepare buffers for sends and recvs
#             cupy_buffer_dtype = dtype_converter(embedding.dtype, input_library='torch', output_library='cupy')
#             recv_bufs = cp.empty(
#                 num_nodes_to_recv * num_coefficients * num_channels,
#                 dtype=cupy_buffer_dtype
#             )

#             sendbufs = {}
#             for target_rank, nodes in nodes_to_send.items():
#                 if nodes:
#                     sendbufs[target_rank] = flatten_embedding(embedding[indices_to_send[target_rank]])

#             cp.cuda.runtime.deviceSynchronize()
#             nccl.groupStart()

#             # Sends
#             for target_rank, nodes in nodes_to_send.items():
#                 if nodes:
#                     comm.send(sendbufs[target_rank], target_rank, stream=stream)

#             # Recvs
#             recv_pointer = 0
#             for i, (source_rank, nodes) in enumerate(nodes_to_recv.items()):
#                 if nodes:
#                     start_idx = recv_pointer
#                     end_idx = start_idx + len(nodes) * num_coefficients * num_channels
#                     comm.recv(recv_bufs[start_idx:end_idx], source_rank, stream=stream)
#                     recv_pointer = end_idx

#             nccl.groupEnd()
#             cp.cuda.runtime.deviceSynchronize()
#             # NOTE: This is a blocking operation


#         # --> Slot in the local embeddings 
#         edge_embeddings = torch.empty((num_edges, embedding.shape[1], embedding.shape[2]), device=embedding.device, dtype=embedding.dtype)

#     edge_embeddings[is_local] = embedding[local_indices_torch]
#     cp.cuda.runtime.deviceSynchronize()

#     # --> Slot in the remote embeddings 
#     if num_nodes_to_recv:
#         with torch.no_grad():
#             received_embeddings = from_dlpack(recv_bufs.toDlpack()).reshape(num_nodes_to_recv, num_coefficients, num_channels)
        
#         edge_embeddings[is_remote] = received_embeddings[remote_indices_torch]    
    
#     return edge_embeddings

def exchange_nodes(embedding, num_edges, communication_dict, comm=None):
    """
    Exchange node embeddings using torch.distributed, mirrored after the original NCCL/CuPy logic.
    """
    # 1. Setup metadata
    device = embedding.device
    dtype = embedding.dtype
    num_coefficients = embedding.shape[1]
    num_channels = embedding.shape[2]

    # print(f"Rank {dist.get_rank()} - Active backend: {torch.distributed.get_backend()}", flush=True)
    
    is_local = communication_dict['is_local']
    is_remote = communication_dict['is_remote']
    local_indices_torch = communication_dict['local_indices_torch']
    remote_indices_torch = communication_dict['remote_indices_torch']
    nodes_to_send = communication_dict['nodes_to_send']                
    indices_to_send = communication_dict['indices_to_send']            
    nodes_to_recv = communication_dict['nodes_to_recv']           

    num_nodes_to_recv = len(remote_indices_torch)

    edge_embeddings = torch.empty(
        (num_edges, num_coefficients, num_channels), 
        device=device, dtype=dtype
    )

    # if there are no remote nodes to receive, we can skip the communication step and directly slot in the local embeddings
    if num_nodes_to_recv == 0:
        edge_embeddings[is_local] = embedding[local_indices_torch]

    # torch.cuda.synchronize(device)
    # dist.barrier()
    # comm_start_time = time.time()

    # 3. Communication Group
    if num_nodes_to_recv > 0:
        p2p_ops = []
        recv_buffers = {}

        # Prepare Receives (Mirrors recv_bufs allocation)
        for source_rank, nodes in nodes_to_recv.items():
            if nodes:
                # Allocate rank-specific buffer
                buf = torch.empty(
                    (len(nodes), num_coefficients, num_channels), 
                    device=device, dtype=dtype
                )
                recv_buffers[source_rank] = buf
                # Equivalent to comm.recv within a group
                p2p_ops.append(dist.P2POp(dist.irecv, buf, source_rank))

        # Prepare Sends (Mirrors sendbufs creation)
        for target_rank, nodes in nodes_to_send.items():
            if nodes:
                # NCCL requires contiguous memory for sends
                send_tensor = embedding[indices_to_send[target_rank]].contiguous()
                # Equivalent to comm.send within a group
                p2p_ops.append(dist.P2POp(dist.isend, send_tensor, target_rank))

        # 4. Execute Group Communication
        # This is the direct torch equivalent of nccl.groupStart() -> nccl.groupEnd()
        reqs = dist.batch_isend_irecv(p2p_ops)

        # Slot in the local embeddings while communication (overlap)
        edge_embeddings[is_local] = embedding[local_indices_torch]
        
        for req in reqs:
            req.wait()

        # 5. Slot in the remote embeddings
        # We concatenate the buffers in the order defined by nodes_to_recv 
        # to match the remote_indices_torch mapping.
        all_received = torch.cat([recv_buffers[rank] for rank in nodes_to_recv.keys()], dim=0)
        edge_embeddings[is_remote] = all_received[remote_indices_torch]
    
    # torch.cuda.synchronize(device)
    # comm_end_time = time.time()
    # print(f"Rank {dist.get_rank()} - Message exchange completed in {comm_end_time - comm_start_time:.4f} seconds", flush=True)

    return edge_embeddings

class ExchangeNodes(torch.autograd.Function):

    @staticmethod
    def forward(ctx, embedding, num_edges, communication_dict):

        # Store metadata for backward
        ctx.communication_dict = communication_dict
        ctx.num_nodes_input = embedding.shape[0]
        
        # 1. Setup metadata
        device = embedding.device
        dtype = embedding.dtype
        num_coefficients = embedding.shape[1]
        num_channels = embedding.shape[2]

        # print(f"Rank {dist.get_rank()} - Active backend: {torch.distributed.get_backend()}", flush=True)
        
        is_local = communication_dict['is_local']
        is_remote = communication_dict['is_remote']
        local_indices_torch = communication_dict['local_indices_torch']
        remote_indices_torch = communication_dict['remote_indices_torch']
        nodes_to_send = communication_dict['nodes_to_send']                
        indices_to_send = communication_dict['indices_to_send']            
        nodes_to_recv = communication_dict['nodes_to_recv']           

        num_nodes_to_recv = len(remote_indices_torch)


        edge_embeddings = torch.empty(
            (num_edges, num_coefficients, num_channels), 
            device=device, dtype=dtype
        )

        # if there are no remote nodes to receive, we can skip the communication step and directly slot in the local embeddings
        if num_nodes_to_recv == 0:
            edge_embeddings[is_local] = embedding[local_indices_torch]

        # torch.cuda.synchronize(device)
        # dist.barrier()
        # comm_start_time = time.time()

        # 3. Communication Group
        if num_nodes_to_recv > 0:
            p2p_ops = []
            recv_buffers = {}

            # Prepare Receives (Mirrors recv_bufs allocation)
            for source_rank, nodes in nodes_to_recv.items():
                if nodes:
                    # Allocate rank-specific buffer
                    buf = torch.empty(
                        (len(nodes), num_coefficients, num_channels), 
                        device=device, dtype=dtype
                    )
                    recv_buffers[source_rank] = buf
                    # Equivalent to comm.recv within a group
                    p2p_ops.append(dist.P2POp(dist.irecv, buf, source_rank))

            # Prepare Sends (Mirrors sendbufs creation)
            for target_rank, nodes in nodes_to_send.items():
                if nodes:
                    # NCCL requires contiguous memory for sends
                    send_tensor = embedding[indices_to_send[target_rank]].contiguous()
                    # Equivalent to comm.send within a group
                    p2p_ops.append(dist.P2POp(dist.isend, send_tensor, target_rank))

            # 4. Execute Group Communication
            # This is the direct torch equivalent of nccl.groupStart() -> nccl.groupEnd()
            reqs = dist.batch_isend_irecv(p2p_ops)

            # Slot in the local embeddings while communication (overlap)
            edge_embeddings[is_local] = embedding[local_indices_torch]
            
            for req in reqs:
                req.wait()

            # 5. Slot in the remote embeddings
            # We concatenate the buffers in the order defined by nodes_to_recv 
            # to match the remote_indices_torch mapping.
            all_received = torch.cat([recv_buffers[rank] for rank in nodes_to_recv.keys()], dim=0)
            edge_embeddings[is_remote] = all_received[remote_indices_torch]
        
        # torch.cuda.synchronize(device)
        # comm_end_time = time.time()
        # print(f"Rank {dist.get_rank()} - Message exchange completed in {comm_end_time - comm_start_time:.4f} seconds", flush=True)

        return edge_embeddings

    @staticmethod
    def backward(ctx, grad_output):
        comm_dict = ctx.communication_dict
        device = grad_output.device
        
        grad_embedding = torch.zeros(
            (ctx.num_nodes_input, grad_output.shape[1], grad_output.shape[2]),
            device=device, dtype=grad_output.dtype
        )

        # Local gradients
        grad_embedding.index_add_(0, comm_dict['local_indices_torch'], grad_output[comm_dict['is_local']])

        # Remote gradients
        p2p_ops = []
        recv_grad_bufs = {}

        # RECV: Gradients for nodes SENT in forward
        for target_rank, nodes in comm_dict['nodes_to_send'].items():
            if nodes:
                buf = torch.empty((len(nodes), grad_output.shape[1], grad_output.shape[2]), 
                                 device=device, dtype=grad_output.dtype)
                recv_grad_bufs[target_rank] = buf
                p2p_ops.append(dist.P2POp(dist.irecv, buf, target_rank))

        # SEND: Gradients for nodes RECEIVED in forward
        if len(comm_dict['remote_indices_torch']) > 0:
            remote_grads = grad_output[comm_dict['is_remote']]
            
            # Reconstruct the 'all_received' buffer shape to undo the mapping
            # remote_indices_torch maps: cat_buffer -> edge_indices
            # We need to put edge_grads back into cat_buffer positions
            cat_grad_buffer = torch.zeros(
                (len(comm_dict['remote_indices_torch']), grad_output.shape[1], grad_output.shape[2]),
                device=device, dtype=grad_output.dtype
            )
            cat_grad_buffer.index_add_(0, comm_dict['remote_indices_torch'], remote_grads)
            
            start_idx = 0
            for source_rank, nodes in comm_dict['nodes_to_recv'].items():
                if nodes:
                    count = len(nodes)
                    send_buf = cat_grad_buffer[start_idx : start_idx + count].contiguous()
                    p2p_ops.append(dist.P2POp(dist.isend, send_buf, source_rank))
                    start_idx += count

        # Execute Communication
        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        # Accumulate received gradients
        for target_rank, buf in recv_grad_bufs.items():
            indices = comm_dict['indices_to_send'][target_rank]
            grad_embedding.index_add_(0, indices, buf)

        return grad_embedding, None, None

def flatten_embedding(embedding):
    """
    Flattens the embedding for communication 
    """

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.contiguous().view(-1)

    return cp.from_dlpack(embedding.detach()) # convert to cupy array

def dtype_converter(input_dtype, input_library, output_library):

    """
    Converts a datatype between torch, numpy, and MPI.

    Parameters:
        input_dtype: The input datatype (e.g., torch.float32, np.float32, MPI.FLOAT).
        input_library: The library of the input datatype ('torch', 'numpy', 'cupy' or 'mpi').
        output_library: The target library for conversion ('torch', 'numpy', 'cupy' or 'mpi').

    Returns:
        The equivalent datatype in the target library.

    Raises:
        ValueError: If the conversion is not supported.
        NameError: If MPI is not imported and 'mpi' conversion is requested.
    """

    # Use string representations of MPI datatypes as intermediate keys
    mpi_types = {
        "MPI_FLOAT": MPI.FLOAT,
        "MPI_DOUBLE": MPI.DOUBLE,
    }

    # Mapping between libraries
    dtype_mapping = {
        'torch': {
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float64: "float64",
        },
        'numpy': {
            np.float16: "float16",
            np.float32: "float32",
            np.float64: "float64",
        },
        'cupy': {
            cp.float16: "float16",
            cp.float32: "float32",
            cp.float64: "float64",
        },
        'mpi': {
            "MPI_FLOAT": "float32",
            "MPI_DOUBLE": "float64",
        },
    }

    reverse_mapping = {
        'torch': {v: k for k, v in dtype_mapping['torch'].items()},
        'numpy': {v: k for k, v in dtype_mapping['numpy'].items()},
        'mpi': {v: k for k, v in dtype_mapping['mpi'].items()},
        'cupy': {v: k for k, v in dtype_mapping['cupy'].items()},
    }

    if input_library not in dtype_mapping or output_library not in reverse_mapping:
        raise ValueError(f"Unsupported library. Supported libraries are 'torch', 'numpy', and 'mpi'.")

    # Convert input dtype to an intermediate string representation
    if input_library == 'mpi' and isinstance(input_dtype, MPI.Datatype):
        input_dtype = next((key for key, val in mpi_types.items() if val == input_dtype), None)
        if input_dtype is None:
            raise ValueError(f"Unsupported MPI datatype: {input_dtype}")
    else:
        input_dtype = dtype_mapping[input_library].get(input_dtype)
        if input_dtype is None:
            raise ValueError(f"Unsupported input datatype for {input_library}: {input_dtype}")

    # Map intermediate string to the output library's dtype
    result = reverse_mapping[output_library].get(input_dtype)
    if output_library == 'mpi' and isinstance(result, str):
        result = mpi_types.get(result)

    if result is None:
        raise ValueError(f"Conversion from {input_library} to {output_library} failed for dtype: {input_dtype}")

    return result