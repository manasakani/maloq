import torch
import torch.nn as nn
import torch.distributed as dist
from e3nn.o3 import Linear
from transformer_block import NodeBlockV2, EdgeBlockV2
from SO3 import SO3_Rotation, SO3_Embedding

import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
     from torch_scatter import scatter
     import dgl
from mpi4py import MPI

import time
import numpy as np

# Borrowed from mace-ocp (https://github.com/ACEsuit/mace-ocp.git)
class GaussianSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_gaussians=50, basis_width_scalar=1.0
    ):
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians

        # will create a set of Gaussian basis functions with centers at each value of offset:
        offset = torch.linspace(start, stop, num_gaussians)

        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )

        self.register_buffer("offset", offset)

    def forward(self, dist):
        # the input dist is a tensor of scalar distances with shape (num_edges,)
        # self.offset is a tensor of shape (num_gaussians,)
        # the output dist will be a tensor of shape (num_edges, num_gaussians) containing the scalar distance to each 
        # of the "num_gaussians" Gaussian centers, for each edge in the input tensor

        # for each distance, find the scalar distance to each Gaussian center:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        
        # apply the Gaussian function to each distance:
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

# Note: we use Gate activation in all cases
class SO2Net(torch.nn.Module):

    def __init__(
        self,
        num_layers,                                             # num_MP_layers
        lmax, 
        mmax, 
        mappingReduced,                                         # SO3.CoefficientMappingModule(lmax, mmax)
        sphere_channels,
        edge_channels_list,                                     # [sphere_channels, sphere_channels, sphere_channels]  
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        ffn_hidden_channels, 
        irreps_in,
        irreps_out
    ):
        super(SO2Net, self).__init__()

        self.lmax = lmax
        self.mmax = mmax
    
        ffn_activation =    'scaled_silu'                   # activation function used in the feedforward network
        norm_type      =    'layer_norm_sh'                 # normalizes l=0 and l>0 coefficients separately

        self.sphere_channels    =   sphere_channels
        attn_hidden_channels    =   attn_hidden_channels
        num_heads               =   num_heads
        attn_alpha_channels     =   attn_alpha_channels
        attn_value_channels     =   attn_value_channels
        ffn_hidden_channels     =   ffn_hidden_channels
        attn_activation         =   'scaled_silu'
        use_attn_renorm         =   True

        use_m_share_rad         =   True                    # (?) share the radial part of the edge embedding for all m values

        max_num_elements        =   100                     # maximum number of elements which can exist in the dataset (used for the embedding layer)
        use_atom_edge_embedding =   True

        self.output_channels    =   edge_channels_list[-1]  # last entry of edge_channels_list is used for the output channels between each layer 

        self.distance_expansion = GaussianSmearing(
                                0.0,                        # start
                                5,                          # stop
                                edge_channels_list[0],      # num_gaussians used to expand the distance
                                2.0,                        # basis_width_scalar
                                )

        sphere_channels_all = self.output_channels
        self.sphere_embedding = nn.Embedding(max_num_elements, sphere_channels_all)

        self.node_lin = Linear(irreps_in=irreps_in, irreps_out=irreps_out, biases=True)
        self.edge_lin = Linear(irreps_in=irreps_in, irreps_out=irreps_out, biases=True)
        self.num_layers = num_layers

        self.SO3_rotation = nn.ModuleList()
        self.SO3_rotation.append(SO3_Rotation(lmax))

        self.blocks = nn.ModuleList()
    
        for i in range(num_layers):

            block1 = NodeBlockV2(
                        self.sphere_channels,
                        attn_hidden_channels,
                        num_heads,
                        attn_alpha_channels,
                        attn_value_channels,
                        ffn_hidden_channels,
                        self.sphere_channels, 
                        lmax,
                        mmax,
                        self.SO3_rotation,
                        mappingReduced,
                        max_num_elements,
                        edge_channels_list,
                        use_atom_edge_embedding,
                        use_m_share_rad,
                        attn_activation,
                        use_attn_renorm,
                        ffn_activation,
                        norm_type,
                        )
            

            self.blocks.append(block1)

            block2 = EdgeBlockV2(
                        self.sphere_channels,
                        attn_hidden_channels,
                        num_heads,
                        attn_alpha_channels,
                        attn_value_channels,
                        ffn_hidden_channels,
                        self.sphere_channels, 
                        lmax,
                        mmax,
                        self.SO3_rotation,
                        mappingReduced,
                        max_num_elements,
                        edge_channels_list,
                        use_atom_edge_embedding,
                        use_m_share_rad,
                        attn_activation,
                        use_attn_renorm,
                        ffn_activation,
                        norm_type,
                        )

            self.blocks.append(block2)


    def forward(self, batch):

        device = batch.y.device
        dtype = batch.y.dtype
                         
                                                                            # note: the batch size dimension multiplies the # nodes and # edges
        atomic_numbers = batch.x                                            # shape = (num_nodes) = [3]
        edge_distance = batch.edge_attr[:,0]                                # shape = (num_edges) = [6]
        edge_distance_vec = batch.edge_attr[:, [2, 3, 1]]                   # shape = (num_edges, 3) = [6, 3]
        edge_index = batch.edge_index                                       # shape = (2, num_edges) = [2, 6]

        num_subgraph_nodes = len(atomic_numbers)
        num_subgraph_edges = len(edge_distance)

        # *** SPLIT THE NODES AND EDGES BETWEEN PROCESSES ***

        rank = dist.get_rank()
        size = dist.get_world_size()
        comm = MPI.COMM_WORLD
        
        num_subgraph_nodes_local = num_subgraph_nodes // size
        num_subgraph_edges_local = num_subgraph_edges // size

        start_node = rank * num_subgraph_nodes_local
        end_node = start_node + num_subgraph_nodes_local
        start_edge = rank * num_subgraph_edges_local
        end_edge = start_edge + num_subgraph_edges_local

        if rank == size - 1:
            num_subgraph_nodes_local += num_subgraph_nodes % size
            end_node += num_subgraph_nodes % size
        if rank == size - 1:
            num_subgraph_edges_local += num_subgraph_edges % size
            end_edge += num_subgraph_edges % size

        # Initialise the node embeddings with atomic_numbers
        # length of angular momentum coefficients = (lmax+1)^2 = (4+1)^2 = 25 = 1(l=0) + 3(l=1) + 5(l=2) + 7(l=3) + 9(l=4)
        # total node embedding = (num atoms, num coefficients, sphere_channels) = (3, 25, 64)
        # total edge embedding = (num edges, num coefficients, sphere_channels) = (6, 25, 64)
        node_embedding_local = SO3_Embedding(num_subgraph_nodes_local, self.lmax, self.sphere_channels, device, dtype) # [number of atoms, number of coefficients, number of channels]
        edge_embedding_local = SO3_Embedding(num_subgraph_edges_local, self.lmax, self.sphere_channels, device, dtype) # [number of edges, number of coefficients, number of channels]
        
        print(f"Rank {rank} of {size}, start_node: {start_node}, end_node: {end_node}, start_edge: {start_edge}, end_edge: {end_edge}", flush=True)
        print(f"Rank {rank} of {size}, node_embedding.shape: {node_embedding_local.embedding.shape}, edge_embedding.shape: {edge_embedding_local.embedding.shape}", flush=True)

        # Initialize the l = 0, m = 0 coefficients of each embedding:

        offset_res = 0
        node_element_embedding_local = self.sphere_embedding(atomic_numbers[start_node:end_node])
        edge_distance_embedding_local = self.distance_expansion(edge_distance[start_edge:end_edge])
        node_embedding_local.embedding[:, offset_res, :] = node_element_embedding_local
        edge_embedding_local.embedding[:, offset_res, :] = edge_distance_embedding_local
        
        # Create 3D rotation matrices for each of the edges - note that all the edges are needed for a deterministic rotation matrix:
        edge_rot_mat_local = init_edge_rot_mat(edge_distance_vec)[start_edge:end_edge, :, :]                 # shape = (num_edges, 3, 3) = [6, 3, 3]
    
        edge_index_local = edge_index[:, start_edge:end_edge]                                                # shape = (2, num_edges) = [2, 6]

        # **** TEMP: allgatherv them back for debug **** 
        # NOTE: allgatherv expects a flattened numpy array, so all the data needs to be flattened

        # # ___ nodes & edges ___
        
        # node_embedding_local_np = node_embedding_local.flatten_embedding()
        # edge_embedding_local_np = edge_embedding_local.flatten_embedding()

        # local_node_size = len(node_embedding_local_np)
        # local_edge_size = len(edge_embedding_local_np)

        # all_node_counts = comm.allgather(local_node_size)
        # all_edge_counts = comm.allgather(local_edge_size)

        # displacements_nodes = np.cumsum([0] + all_node_counts[:-1])
        # displacements_edges = np.cumsum([0] + all_edge_counts[:-1])

        # recvbuf_nodes = np.empty(sum(all_node_counts), dtype=np.float64)
        # recvbuf_edges = np.empty(sum(all_edge_counts), dtype=np.float64)

        # comm.Allgatherv(node_embedding_local_np, [recvbuf_nodes, all_node_counts, displacements_nodes, MPI.DOUBLE])
        # comm.Allgatherv(edge_embedding_local_np, [recvbuf_edges, all_edge_counts, displacements_edges, MPI.DOUBLE])

        # num_subgraph_nodes_global = comm.allreduce(num_subgraph_nodes_local)
        # num_subgraph_edges_global = comm.allreduce(num_subgraph_edges_local)
        
        # node_embedding = SO3_Embedding(num_subgraph_nodes_global, self.lmax, self.sphere_channels, device, dtype)
        # node_embedding.unflatten_embedding(recvbuf_nodes)

        # edge_embedding = SO3_Embedding(num_subgraph_edges_global, self.lmax, self.sphere_channels, device, dtype)
        # edge_embedding.unflatten_embedding(recvbuf_edges)

        # # ___ edge rotation matrices ___

        # rot_mat_local_np = edge_rot_mat_local.cpu().detach().numpy().reshape(-1)
        # local_size = len(rot_mat_local_np)

        # all_counts = comm.allgather(local_size)
        # displacements = np.cumsum([0] + all_counts[:-1])

        # total_size = sum(all_counts)
        # recvbuf = np.empty(total_size, dtype=np.float64) # !!! DTYPES ARE HARDCODED TO FLOAT64 FOR H2O DEBUG EXAMPLE !!!

        # comm.Allgatherv(rot_mat_local_np, [recvbuf, all_counts, displacements, MPI.DOUBLE])
        # recvbuf_reshaped = recvbuf.reshape(-1, 3, 3)     # rotation matrices are always 3x3

        # edge_rot_mat = torch.tensor(recvbuf_reshaped, dtype=dtype, device=device)

        # # ___ edge distance embedding ___
        # edge_distance_embedding_shape = edge_distance_embedding.shape
        # edge_distance_embedding_np = edge_distance_embedding.cpu().detach().numpy().reshape(-1)
        # local_size = len(edge_distance_embedding_np)

        # all_counts = comm.allgather(local_size)
        # displacements = np.cumsum([0] + all_counts[:-1])

        # total_size = sum(all_counts)
        # recvbuf = np.empty(total_size, dtype=np.float64) # !!! DTYPES ARE HARDCODED TO FLOAT64 FOR H2O DEBUG EXAMPLE !!!
        # comm.Allgatherv(edge_distance_embedding_np, [recvbuf, all_counts, displacements, MPI.DOUBLE])
        # recvbuf_reshaped = recvbuf.reshape(-1, edge_distance_embedding_shape[1])

        # edge_distance_embedding = torch.tensor(recvbuf_reshaped, dtype=dtype, device=device)

        # print(f"Rank {rank}: Restored global node embedding shape: {node_embedding.embedding.shape}")
        # print(f"Rank {rank}: Restored global edge embedding shape: {edge_embedding.embedding.shape}")
        # print(f"Rank {rank}: Final edge rotation matrices gathered with shape {edge_rot_mat.shape}")
        # print(f"Rank {rank}: Final edge distance embedding gathered with shape {edge_distance_embedding.shape}")
        dist.barrier()        

        # **** TEMP END: allgather them back for debug: ****

        self.SO3_rotation[0].set_wigner(edge_rot_mat_local)                                              # set the rotation matrices for each of the edges in the edge list

        # Process the graph through the layers
        for i in range(self.num_layers):

            node_embedding_local = self.blocks[2*i](
                            node_embedding_local,                  # SO3_Embedding
                            atomic_numbers,
                            edge_distance_embedding_local,
                            edge_index_local,
                            edge_embedding_local,
                        )  
            
            edge_embedding_local = self.blocks[2*i+1](
                            node_embedding_local,                  # SO3_Embedding
                            atomic_numbers,
                            edge_distance_embedding_local,
                            edge_index_local,
                            edge_embedding_local,
                        )

        node_output = convert_to_irreps(node_embedding_local, self.output_channels, self.lmax, self.node_lin)
        edge_output = convert_to_irreps(edge_embedding_local, self.output_channels, self.lmax, self.edge_lin)

        return node_output, edge_output


def convert_to_irreps(input, output_channels, lmax, lin_node):
        
    """
    Converts the output irreps to the coupled space irrep representation needed to reconstruct the Hamiltonian using the linear layer from e3nn library 
    e.g. map 64x0e+64x1e+64x2e+64x3e+64x4e to 1x0e+1x1e+1x1e+1x0e+1x1e+1x2e+..+1x1e+1x2e+1x3e+1x4e

    """

    # prepare sorted_output:
    test_input = input.embedding.transpose(-1,-2) #rearrange from l major order into feature major order so that e.g. 64 x 1e can be extracted correctly after flattening the columns belonging to l = 1
    feature_size = test_input.shape[0]
    sorted_output = torch.zeros(feature_size, output_channels*((lmax+1)**2))
    device = input.embedding.device

    for l in range(lmax+1):
        start = l**2*output_channels
        end = l**2*output_channels+output_channels*(2*l+1)
        sorted_output[:,start:end] = torch.squeeze(test_input[:,:,l**2:l**2+(2*l+1)].reshape(feature_size, 1, -1))

    # convert:
    test_output = lin_node(sorted_output.to(device))
    
    return test_output
    

# Borrowed from EquiformerV2 (https://github.com/atomicarchitects/equiformer_v2.git)
def init_edge_rot_mat(edge_distance_vec):
    """
    Takes the edge distance vectors and returns the 3D rotation matrix for each edge
    """
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            "Error edge_vec_0_distance: {}".format(
                torch.min(edge_vec_0_distance)
            )
        )
        
    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))
    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
        -1, 1
    )
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
        -1, 1
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))

    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()