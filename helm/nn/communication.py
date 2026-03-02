import torch
import torch.distributed as dist
from mpi4py import MPI
import cupy as cp
import numpy as np
from cupy import cuda
from cupy.cuda import nccl
from cupyx.distributed import NCCLBackend
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

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

def exchange_nodes(embedding, num_edges, communication_dict, comm, rank, world_size):
    """
    Exchange node embeddings between different processes using NCCL backend, according to communication_dict.
    """

    dist.barrier()
    comm = comm_nccl.comm
    stream = comm_nccl.stream

    num_nodes = embedding.shape[0]
    num_coefficients = embedding.shape[1]
    num_channels = embedding.shape[2]
    # new_embedding = torch.zeros(
    #                                 num_nodes,
    #                                 num_coefficients,
    #                                 num_channels,
    #                                 device=embedding.device,
    #                                 dtype=embedding.dtype,
    #                             )
    
    # Get precomputed communication plan
    is_local = communication_dict['is_local']
    is_remote = communication_dict['is_remote']
    local_indices = communication_dict['local_indices']
    local_indices_torch = communication_dict['local_indices_torch']
    remote_indices = communication_dict['remote_indices']
    remote_indices_torch = communication_dict['remote_indices_torch']
    nodes_to_send = communication_dict['nodes_to_send']                
    indices_to_send = communication_dict['indices_to_send']            
    indices_to_send_cp = communication_dict['indices_to_send_cp']  
    nodes_to_recv = communication_dict['nodes_to_recv']           
    
    # dist.barrier()
    # print(f"Rank {rank}: local indices: {local_indices}, remote indices: {remote_indices}", flush=True)
    # print(f"Rank {rank}: is local: {is_local}, is remote: {is_remote}", flush=True)
    # dist.barrier()

    # dist.barrier()
    # print(f"Rank {rank}: nodes to send: ", nodes_to_send, flush=True)
    # dist.barrier()

    # dist.barrier()
    # print(f"Rank {rank}: nodes to receive: ", nodes_to_recv, flush=True)
    # dist.barrier()

    # dist.barrier()
    # print(f"Rank {rank}: messages to send: ", {k: len(v) for k, v in messages_to_send.items()}, flush=True)
    # dist.barrier()
    # print(f"Rank {rank}: messages to receive: ", {k: len(v) for k, v in messages_to_recv.items()}, flush=True)
    # dist.barrier()

    with torch.no_grad():

        # --> Send/Receive embeddings
        num_nodes_to_recv = len(remote_indices)

        if num_nodes_to_recv:

            dist.barrier()
            print(f"Rank {rank}: Starting preparation for communication of embeddings. Number of messages to send: {sum(len(v) for v in nodes_to_send.values())}, number of messages to receive: {num_nodes_to_recv}", flush=True)
            dist.barrier()

            # Prepare buffers for sends and recvs
            cupy_buffer_dtype = dtype_converter(embedding.dtype, input_library='torch', output_library='cupy')
            recv_bufs = cp.empty(
                num_nodes_to_recv * num_coefficients * num_channels,
                dtype=cupy_buffer_dtype
            )

            dist.barrier()
            print(f"Rank {rank}: Starting communication of embeddings. Number of messages to send: {sum(len(v) for v in nodes_to_send.values())}, number of messages to receive: {num_nodes_to_recv}", flush=True)
            dist.barrier()

            sendbufs = {}
            for target_rank, nodes in nodes_to_send.items():
                if nodes:
                    sendbufs[target_rank] = flatten_embedding(embedding[indices_to_send[target_rank]])

            cp.cuda.runtime.deviceSynchronize()
            nccl.groupStart()

            # Sends
            for target_rank, nodes in nodes_to_send.items():
                if nodes:
                    comm.send(sendbufs[target_rank], target_rank, stream=stream)

            # Recvs
            recv_pointer = 0
            for i, (source_rank, nodes) in enumerate(nodes_to_recv.items()):
                if nodes:
                    start_idx = recv_pointer
                    end_idx = start_idx + len(nodes) * num_coefficients * num_channels
                    comm.recv(recv_bufs[start_idx:end_idx], source_rank, stream=stream)
                    recv_pointer = end_idx

            nccl.groupEnd()
            cp.cuda.runtime.deviceSynchronize()
            # NOTE: This is a blocking operation and such no overlap currently

            dist.barrier()
            print(f"Rank {rank}: Finished communication of embeddings.", flush=True)
            dist.barrier()

        # --> Slot in the local embeddings 
        edge_embeddings = torch.empty((num_edges, embedding.shape[1], embedding.shape[2]), device=embedding.device, dtype=embedding.dtype)

    edge_embeddings[is_local] = embedding[local_indices_torch]
    cp.cuda.runtime.deviceSynchronize()

    # dist.barrier()
    # print(f"Rank {rank}: Finished slotting in local embeddings.", flush=True)
    # dist.barrier()

    if num_nodes_to_recv:
        with torch.no_grad():
            received_embeddings = from_dlpack(recv_bufs.toDlpack()).reshape(num_nodes_to_recv, num_coefficients, num_channels)
        
        edge_embeddings[is_remote] = received_embeddings[remote_indices_torch]    # needs gradients
    
    # dist.barrier()
    # print(f"Rank {rank}: Finished slotting in remote embeddings.", flush=True)
    # dist.barrier()
    
    return edge_embeddings

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