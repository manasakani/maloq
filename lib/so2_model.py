import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import math
import numpy as np
import dgl
import math
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.utils import degree

from e3nn.o3 import Irrep, Irreps, wigner_3j, matrix_to_angles, Linear, FullyConnectedTensorProduct, TensorProduct, SphericalHarmonics
from e3nn.nn import Extract
import networkx as nx
import matplotlib.pyplot as plt
from lib_equiformer.layer_norm import get_normalization_layer
from lib.transformer_block import FeedForwardNetwork
from lib.transformer_block import TransBlockV2,NodeBlockV2,EdgeBlockV2

import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
     from torch_scatter import scatter

# from edge_rot_mat import init_edge_rot_mat
from lib_equiformer.SO3 import CoefficientMappingModule, SO3_Rotation, SO3_Embedding
class GaussianSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_gaussians=50, basis_width_scalar=1.0
    ):
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

def convert_to_irreps(input,output_channels,lmax_list,lin_node):
        
        """
        Converts the output irreps to the coupled space irrep representation needed to reconstruct the Hamiltonian using the linear layer from e3nn library 
        e.g. map 64x0e+64x1e+64x2e+64x3e+64x4e to 1x0e+1x1e+1x1e+1x0e+1x1e+1x2e+..+1x1e+1x2e+1x3e+1x4e

        """
        test_input = input.embedding.transpose(-1,-2) #rearrange from l major order into feature major order so that e.g. 64 x 1e can be extracted correctly after flattening the columns belonging to l = 1
        feature_size = test_input.shape[0]
        sorted_output = torch.zeros(feature_size,output_channels*((lmax_list[0]+1)**2))
        device = input.embedding.device

        for l in range(lmax_list[0]+1):
                start = l**2*output_channels
                end = l**2*output_channels+output_channels*(2*l+1)
                sorted_output[:,start:end] = torch.squeeze(test_input[:,:,l**2:l**2+(2*l+1)].reshape(feature_size,1,-1))

        test_output = lin_node(sorted_output.to(device))

        return test_output
    
class SO2Net(torch.nn.Module):
    def __init__(
        self,
        num_layers, 
        lmax_list, 
        mmax_list, 
        mappingReduced, 
        sphere_channels,
        edge_channels_list,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        ffn_hidden_channels, 
        irreps_in,
        irreps_out
    ):
        super(SO2Net, self).__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
    
        ffn_activation='scaled_silu'
        use_grid_mlp=False
        use_sep_s2_act=False
        norm_type='layer_norm_sh'

        self.sphere_channels = sphere_channels
        attn_hidden_channels= attn_hidden_channels
        num_heads=num_heads
        attn_alpha_channels=attn_alpha_channels
        attn_value_channels=attn_value_channels
        ffn_hidden_channels=ffn_hidden_channels

        use_gate_act=True
        use_s2_act_attn=False
        attn_activation='scaled_silu'
        use_attn_renorm=True

        SO3_grid = None

        use_m_share_rad = True #Originally True

        max_num_elements = 100
        use_atom_edge_embedding = True

        alpha_drop=0,
        drop_path_rate=0
        proj_drop=0.0

        self.output_channels = edge_channels_list[-1] #last entry of edge_channels_list is used for the output channels between each layer 
        self.num_distance_basis = edge_channels_list[0] #first entry of edge_channels_list represents the number of distance basis functions

        self.distance_expansion = GaussianSmearing(
                                0.0,
                                5,
                                edge_channels_list[0],
                                2.0,
                            )

        self.num_resolutions = 1
        sphere_channels_all = self.num_resolutions*self.output_channels
        self.sphere_embedding = nn.Embedding(max_num_elements, sphere_channels_all)

        self.node_lin = Linear(irreps_in=irreps_in, irreps_out=irreps_out, biases=True)
        self.edge_lin = Linear(irreps_in=irreps_in, irreps_out=irreps_out, biases=True)
        self.num_layers = num_layers

        self.SO3_rotation = nn.ModuleList()
        self.SO3_rotation.append(SO3_Rotation(lmax_list[0]))

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
                        lmax_list,
                        mmax_list,
                        self.SO3_rotation,
                        mappingReduced,
                        SO3_grid,
                        max_num_elements,
                        edge_channels_list,
                        use_atom_edge_embedding,
                        use_m_share_rad,
                        attn_activation,
                        use_s2_act_attn,
                        use_attn_renorm,
                        ffn_activation,
                        use_gate_act,
                        use_grid_mlp,
                        use_sep_s2_act,
                        norm_type,
                        alpha_drop, 
                        drop_path_rate,
                        proj_drop
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
                        lmax_list,
                        mmax_list,
                        self.SO3_rotation,
                        mappingReduced,
                        SO3_grid,
                        max_num_elements,
                        edge_channels_list,
                        use_atom_edge_embedding,
                        use_m_share_rad,
                        attn_activation,
                        use_s2_act_attn,
                        use_attn_renorm,
                        ffn_activation,
                        use_gate_act,
                        use_grid_mlp,
                        use_sep_s2_act,
                        norm_type,
                        alpha_drop, 
                        drop_path_rate,
                        proj_drop
                        )


            self.blocks.append(block2)


    def forward(self, batch, total_num_nodes=None):

        # if the dataset was created using DGL dataloader, the input is a list of DGL graphs
        if isinstance(batch[0], dgl.DGLGraph):
            print("using DGL dataloader")
            # If batch is a list or tuple, process each graph individually 
            # needed for some samplers used by DGL
            if isinstance(batch, (list, tuple)):

                node_outputs = []
                edge_outputs = []

                for subgraph in batch:
                    node_output, edge_output = self.process_graph(subgraph, total_num_nodes)
                    node_outputs.append(node_output)
                    edge_outputs.append(edge_output)

                return node_outputs, edge_outputs
            else:
                # Process a single graph
                return self.process_graph(batch, total_num_nodes)
                
        # if the dataset was created using PyTorch Geometric dataloader, the input is a PyTorch Geometric data object
        else:
            print("using PyTorch Geometric dataloader")
            return self.forward_noDGL(batch)

    def process_graph(self, graph, total_num_nodes):
        """
        total_num_nodes = total number of nodes in the entire graph, from which this batch was extracted

        """
        # Extract features from the graph
        device = graph.device
        dtype = torch.float32

        # print("Node types in graph:", graph.ntypes)
        # print("Edge types in graph:", graph.etypes)

        # print("graph.nodes['_N']: ", graph.nodes['_N'])                             # labels of the input_nodes, with the ouput nodes ordered first
        # print("graph.ndata['feat']: ", graph.ndata['feat'])                         # features of the input_nodes, with the ouput nodes ordered first
        # print("graph.edata['edge_attr']: ", graph.edata['edge_attr'])               # features of the edges
        # print("graph.edata['label']: ", graph.edata['label'])                       # labels of the edges involved in this subgraph
        # print("graph.ndata['node_label']: ", graph.ndata['node_label'])             # labels of the nodes involved in this subgraph(?)

        # global_node_ids = graph.ndata[dgl.NID]
        # print("Global node IDs:", global_node_ids)

        # local_node_ids = graph.ndata['_ID']
        # print("Local node IDs:", local_node_ids)

        atomic_numbers = graph.ndata['feat']['_N'] 

        # print("atomic_numbers: ", atomic_numbers)
        # print("edge_attr: ", graph.edata['edge_attr'])

        edge_distance = graph.edata['edge_attr'][:, 0]
        edge_distance_vec = graph.edata['edge_attr'][:, [2, 3, 1]]                    # edge distance vector for each edge in the subgraph

        # print("edge distance: ", edge_distance)
        # print("edge distance vec: ", edge_distance_vec)

        u, v = graph.edges() 
        edge_index = torch.stack([u, v], dim=0)  #check if this is correct - flip the order of the nodes in the edge_index
        print("edge_index: ", edge_index)

        num_subgraph_nodes = len(atomic_numbers)
        num_subgraph_edges = len(edge_distance)

        # Initialize node and edge embeddings
        x = SO3_Embedding(num_subgraph_nodes, self.lmax_list, self.sphere_channels, device, dtype)
        # Initialize the l = 0, m = 0 coefficients for each resolution
        offset_res = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)        

        x.set_lmax_mmax(self.lmax_list, self.mmax_list)
        edge_distance_embedding = self.distance_expansion(edge_distance)

        edge_fea = SO3_Embedding(num_subgraph_edges, self.lmax_list, self.sphere_channels, device, dtype)

        offset_res = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                edge_fea.embedding[:, offset_res, :] = self.distance_expansion(edge_distance)

        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        self.SO3_rotation[0].set_wigner(edge_rot_mat)

        node_embedding = x
        edge_embedding = edge_fea

        for i in range(self.num_layers):

            node_embedding = self.blocks[2 * i](
                node_embedding,
                atomic_numbers,
                edge_distance_embedding,
                edge_index,
                edge_embedding,
                batch=None
            )

            edge_embedding = self.blocks[2 * i + 1](
                node_embedding,
                atomic_numbers,
                edge_distance_embedding,
                edge_index,
                edge_embedding,
                batch=None
            )

        node_output = convert_to_irreps(node_embedding, self.output_channels, self.lmax_list, self.node_lin)
        edge_output = convert_to_irreps(edge_embedding, self.output_channels, self.lmax_list, self.edge_lin)

        return node_output, edge_output


    def forward_noDGL(
        self,
        batch
    ):  
        device = batch.y.device
        # dtype = torch.float32
        dtype = batch.y.dtype
        atomic_numbers = batch.x
        edge_distance = batch.edge_attr[:,0]
        edge_index = batch.edge_index

        #initialise the node embedding with atomic_numbers
        x = SO3_Embedding(len(batch.x), self.lmax_list, self.sphere_channels, device, dtype) #first dimension is the number of atoms, second dimension is the number of coefficients, third dimension is the number of channels

        offset_res = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)        

        x.set_lmax_mmax(self.lmax_list,self.mmax_list)
        edge_distance_embedding = self.distance_expansion(edge_distance)

        edge_fea = SO3_Embedding(len(edge_index[0]), self.lmax_list, self.sphere_channels, device, dtype) #first dimension is the number of atoms, second dimension is the number of coefficients, 
        
        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each edge
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                edge_fea.embedding[:, offset_res, :] = self.distance_expansion(edge_distance)

        edge_distance_vec = batch.edge_attr[:, [2, 3, 1]]

        edge_rot_mat = init_edge_rot_mat(edge_distance_vec) #create rotation matrices 
        self.SO3_rotation[0].set_wigner(edge_rot_mat)
        
        node_embedding = x #initialise node embedding with atomic_numbers
        edge_embedding = edge_fea #initialise edge embedding with edge distance


        for i in range(self.num_layers):

            node_embedding = self.blocks[2*i](
                            node_embedding,                  # SO3_Embedding
                            atomic_numbers,
                            edge_distance_embedding,
                            edge_index,
                            edge_embedding,
                            batch=None    # for GraphDropPath
                        )  
            
            edge_embedding = self.blocks[2*i+1](
                            node_embedding,                  # SO3_Embedding
                            atomic_numbers,
                            edge_distance_embedding,
                            edge_index,
                            edge_embedding,
                            batch=None    # for GraphDropPath
                        )
            

        node_output = convert_to_irreps(node_embedding,self.output_channels,self.lmax_list,self.node_lin)
        edge_output = convert_to_irreps(edge_embedding,self.output_channels,self.lmax_list,self.edge_lin)

        return node_output, edge_output


def init_edge_rot_mat(edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    #assert torch.min(edge_vec_0_distance) < 0.0001
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

