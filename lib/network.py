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

import time

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

        # Initialise the node embedding with atomic_numbers
        # length of angular momentum coefficients = (lmax+1)^2 = (4+1)^2 = 25 = 1(l=0) + 3(l=1) + 5(l=2) + 7(l=3) + 9(l=4)
        # node embedding = (num atoms, num coefficients, sphere_channels) = (3, 25, 64)
        # edge embedding = (num edges, num coefficients, sphere_channels) = (6, 25, 64)
        node_embedding = SO3_Embedding(num_subgraph_nodes, self.lmax, self.sphere_channels, device, dtype) # [number of atoms, number of coefficients, number of channels]
        edge_embedding = SO3_Embedding(num_subgraph_edges, self.lmax, self.sphere_channels, device, dtype) # [number of edges, number of coefficients, number of channels]
        
        node_element_embedding = self.sphere_embedding(atomic_numbers)
        edge_distance_embedding = self.distance_expansion(edge_distance)

        # Initialize the l = 0, m = 0 coefficients of each embedding:
        offset_res = 0
        node_embedding.embedding[:, offset_res, :] = node_element_embedding
        edge_embedding.embedding[:, offset_res, :] = edge_distance_embedding

        # node_embedding.set_lmax_mmax([self.lmax], [self.mmax])
        # edge_embedding.set_lmax_mmax([self.lmax], [self.mmax])
        
        # Create 3D rotation matrices for each of the edges
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)                 # shape = (num_edges, 3, 3) = [6, 3, 3]
        self.SO3_rotation[0].set_wigner(edge_rot_mat)                       # set the rotation matrices for each of the edges in the edge list
        
        # Process the graph through the layers
        for i in range(self.num_layers):

            node_embedding = self.blocks[2*i](
                            node_embedding,                  # SO3_Embedding
                            atomic_numbers,
                            edge_distance_embedding,
                            edge_index,
                            edge_embedding,
                        )  
            
            edge_embedding = self.blocks[2*i+1](
                            node_embedding,                  # SO3_Embedding
                            atomic_numbers,
                            edge_distance_embedding,
                            edge_index,
                            edge_embedding,
                        )

        node_output = convert_to_irreps(node_embedding, self.output_channels, self.lmax, self.node_lin)
        edge_output = convert_to_irreps(edge_embedding, self.output_channels, self.lmax, self.edge_lin)

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