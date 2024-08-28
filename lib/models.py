import torch
import os
from typing import Union, Tuple
from math import ceil, sqrt
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import LayerNorm, PairNorm, InstanceNorm
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.nn.models.dimenet import BesselBasisLayer
import numpy as np
from scipy.special import comb
from lib import utils as utils

# Gaussian basis function implementation
class GaussianBasis(nn.Module):

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super(GaussianBasis, self).__init__()

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.
        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.
        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.
        """
        return self.gaussian_smearing(distances, self.offsets, self.width, centered=self.centered)

    def gaussian_smearing(self, distances, offset, widths, centered=False):
        if not centered:
            # compute width of Gaussian functions (using an overlap of 1 STDDEV)
            coeff = -0.5 / torch.pow(widths, 2)
            # Use advanced indexing to compute the individual components
            diff = distances[..., None] - offset
        else:
            # if Gaussian functions are centered, use offsets to compute widths
            coeff = -0.5 / torch.pow(offset, 2)
            # if Gaussian functions are centered, no offset is subtracted
            diff = distances[..., None]
        # compute smear distance values
        gauss = torch.exp(coeff * torch.pow(diff, 2))
        return gauss
     

# graph convolutional layer implementation using the MessagePassing base class in PyTorch Geometric
class CGConv(MessagePassing):

    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', normalization: str = None,
                 bias: bool = True, if_exp: bool = False, **kwargs):
        super(CGConv, self).__init__(aggr=aggr, flow="source_to_target", **kwargs)

        self.channels = channels                    # Number of input and output channels of the layer
        self.dim = dim                              # Dimension of the input features
        self.normalization = normalization          # Type of normalization applied after message passing
        self.if_exp = if_exp                        # Flag indicating whether to apply exponential decay to edge distances

        if isinstance(channels, int):
            channels = (channels, channels)

        # bias - Whether to include a bias term in linear layers

        self.lin_f = nn.Linear(sum(channels) + dim, channels[1], bias=bias) # Linear transformation applied to the input features
        self.lin_s = nn.Linear(sum(channels) + dim, channels[1], bias=bias) 

        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(channels[1], track_running_stats=True)
        elif self.normalization == 'LayerNorm':
            self.ln = LayerNorm(channels[1])
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(channels[1])
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(channels[1])
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(channels[1])
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(channels[1], 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.normalization == 'BatchNorm':
            self.bn.reset_parameters()

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor, batch, distance, size: Size = None) -> torch.Tensor:
        """
        Forward pass of the graph convolutional layer
        Args:
            x (Union[Tensor, PairTensor]): The input features.
            edge_index (Tensor): The edge indices.
            edge_attr (Tensor): The edge features.
            batch (Tensor): Batch vector
            distance (Tensor): Edge distances
            size (Size): The size of the graph.
        """

        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, distance=distance, size=size)
        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        elif self.normalization == 'LayerNorm':
            out = self.ln(out, batch)
        elif self.normalization == 'PairNorm':
            out = self.pn(out, batch)
        elif self.normalization == 'InstanceNorm':
            out = self.instance_norm(out, batch)
        elif self.normalization == 'GraphNorm':
            out = self.gn(out, batch)
        elif self.normalization == 'DiffGroupNorm':
            out = self.group_norm(out)
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor, distance) -> torch.Tensor:
        """
        Message passing function
        Args:
            x_i (Tensor): Source node features
            x_j (Tensor): Target node features
            edge_attr (Tensor): Edge features
            distance (Tensor): Edge distances
        """

        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-distance ** n / sigma ** n / 2).view(-1, 1)
        return out

    def __repr__(self):
        """
        String representation of the layer
        """
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels, self.dim)


# Graph attention network layer implementation
class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False, normalization: str = None,
                 dropout=0, bias=True, **kwargs):
        super(GAT_Crystal, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.neg_slope = 0.2
        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(heads)
        self.W = nn.Parameter(torch.Tensor(in_features + edge_dim, heads * out_features))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.normalization = normalization
        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(out_features, track_running_stats=True)
        elif self.normalization == 'LayerNorm':
            self.ln = LayerNorm(out_features)
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(out_features)
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(out_features)
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(out_features)
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(out_features, 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, batch, distance):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        elif self.normalization == 'LayerNorm':
            out = self.ln(out, batch)
        elif self.normalization == 'PairNorm':
            out = self.pn(out, batch)
        elif self.normalization == 'InstanceNorm':
            out = self.instance_norm(out, batch)
        elif self.normalization == 'GraphNorm':
            out = self.gn(out, batch)
        elif self.normalization == 'DiffGroupNorm':
            out = self.group_norm(out)
        return out

    def message(self, edge_index_i, x_i, x_j, size_i, index, ptr: OptTensor, edge_attr):
        x_i = torch.cat([x_i, edge_attr], dim=-1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        x_i = F.softplus(torch.matmul(x_i, self.W))
        x_j = F.softplus(torch.matmul(x_j, self.W))
        x_i = x_i.view(-1, self.heads, self.out_features)
        x_j = x_j.view(-1, self.heads, self.out_features)

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1))
        alpha = F.softplus(self.bn1(alpha))

        alpha = softmax(alpha, index, ptr, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out, x):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:
            aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:  aggr_out = aggr_out + self.bias
        return aggr_out



# Message passing layer implementation
class MPLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, if_exp, if_edge_update, normalization,
                 atom_update_net, gauss_stop, output_layer=False):
        super(MPLayer, self).__init__()

        if atom_update_net == 'CGConv':
            self.cgconv = CGConv(channels=in_atom_fea_len,
                                 dim=in_edge_fea_len,
                                 aggr='add',
                                 normalization=normalization,
                                 if_exp=if_exp)

        elif atom_update_net == 'GAT':
            self.cgconv = GAT_Crystal(
                in_features=in_atom_fea_len,
                out_features=in_atom_fea_len,
                edge_dim=in_edge_fea_len,
                heads=3,
                normalization=normalization
            )

        self.if_edge_update = if_edge_update
        self.atom_update_net = atom_update_net

        if if_edge_update:
            
            if output_layer:
                self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                                           nn.SiLU(),
                                           nn.Linear(128, out_edge_fea_len),
                                           )
            else:
                self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                                           nn.SiLU(),
                                           nn.Linear(128, out_edge_fea_len),
                                           nn.SiLU(),
                                           )

    def forward(self, atom_fea, edge_idx, edge_fea, batch, distance, edge_vec):
        """
        Forward pass of the message passing layer
        Args:
            atom_fea (Tensor): Atomic features
            edge_idx (Tensor): Edge indices
            edge_fea (Tensor): Edge features
            batch (Tensor): Batch vector
            distance (Tensor): Edge distances
            edge_vec (Tensor): Edge vectors
        """

        atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, distance)
        atom_fea_s = atom_fea
        
        if self.if_edge_update:
            row, col = edge_idx
            edge_fea = self.e_lin(torch.cat([atom_fea_s[row], atom_fea_s[col], edge_fea], dim=-1)) #concatenate the atomic number embeddings of two atoms and the features of the edge joining them 
            return atom_fea, edge_fea
        else:
            return atom_fea
        

# HGNN model implementation
class HGNN(nn.Module):
    def __init__(self, num_species, in_atom_fea_len, in_edge_fea_len, num_orbital, gauss_stop, num_MP_layers=1,
                 distance_expansion = 'GaussianBasis', if_exp='True', if_MultipleLinear = False, if_edge_update = True, if_lcmp = False,
                 normalization = 'LayerNorm', atom_update_net = 'CGConv', separate_onsite = False,
                 trainable_gaussians = False, type_affine = False, num_l=5):
        super(HGNN, self).__init__()
        self.num_species = num_species
        self.embed = nn.Embedding(num_species + 5, in_atom_fea_len)

        # pair-type aware affine
        if type_affine:
            self.type_affine = nn.Embedding(
                num_species ** 2, 2,
                _weight=torch.stack([torch.ones(num_species ** 2), torch.zeros(num_species ** 2)], dim=-1)
            )
        else:
            self.type_affine = None

        if if_edge_update or (if_edge_update is False and if_lcmp is False):
            distance_expansion_len = in_edge_fea_len
        else:
            distance_expansion_len = in_edge_fea_len - num_l ** 2

        if distance_expansion == 'GaussianBasis':
            self.distance_expansion = GaussianBasis(0.0, gauss_stop, distance_expansion_len, trainable=trainable_gaussians)
        else:
            raise ValueError('Unknown distance expansion function: {}'.format(distance_expansion))

        self.if_MultipleLinear = if_MultipleLinear
        self.if_edge_update = if_edge_update
        self.if_lcmp = if_lcmp
        self.atom_update_net = atom_update_net
        self.separate_onsite = separate_onsite
        self.num_MP_layers = num_MP_layers

        mp_output_edge_fea_len = in_edge_fea_len

        # Set up the network with X message passing layers
        if if_edge_update == True:
            if num_MP_layers == 1:
                self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
            elif num_MP_layers == 2:
                self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp2 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
            elif num_MP_layers == 3:
                self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp2 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp3 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
            elif num_MP_layers == 4:
                self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp2 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp3 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp4 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
            elif num_MP_layers == 5:
                self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp2 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp3 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp4 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                                atom_update_net, gauss_stop)
                self.mp5 = MPLayer(in_atom_fea_len, in_edge_fea_len, mp_output_edge_fea_len, if_exp, if_edge_update,
                                normalization, atom_update_net, gauss_stop)
            else:
                print('maximum number of Message Passing layers is 5')
        else:
            print('error')

        self.mp_output = MPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, if_exp, if_edge_update=True,
                                normalization=normalization, atom_update_net=atom_update_net,
                                gauss_stop=gauss_stop, output_layer=True)


    def forward(self, atom_attr, node_embedding_type, edge_idx, edge_attr, batch,
                sub_atom_idx=None, sub_edge_idx=None, sub_edge_ang=None, sub_index=None,
                huge_structure=False, output_final_layer_neuron=''):
        # batch_edge = batch[edge_idx[0]]

        # atom_fea0 = self.embed(atom_attr)

        # Set up the node embeddings
        if node_embedding_type == 'atomic_number':
            atom_fea0 = self.embed(atom_attr)

        elif node_embedding_type == 'SOAP':
            # Assuming SOAP features are stored in the atom_attr tensor
            atom_fea0 = atom_attr
        else:
            raise ValueError("Unsupported node embeddings type. Choose from 'atomic_number' or 'SOAP'.")

        # distance = edge_attr[:, 0]
        # edge_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]
        distance = edge_attr
        edge_vec = [0,0,0]
        if self.type_affine is None:
            edge_fea0 = self.distance_expansion(distance)
        else:
            affine_coeff = self.type_affine(self.num_species * atom_attr[edge_idx[0]] + atom_attr[edge_idx[1]])
            edge_fea0 = self.distance_expansion(distance * affine_coeff[:, 0] + affine_coeff[:, 1])


        if self.if_edge_update == True:
            if self.num_MP_layers == 1:
                atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            elif self.num_MP_layers == 2:
                atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            elif self.num_MP_layers == 3:
                atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
                atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            elif self.num_MP_layers == 4:
                atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
                atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea, edge_fea = self.mp4(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            elif self.num_MP_layers == 5:
                atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
                atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea, edge_fea = self.mp4(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
                atom_fea, edge_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)

            atom_fea, edge_fea = self.mp_output(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            out = edge_fea
        else:
            if self.num_MP_layers == 1:
                atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            elif self.num_MP_layers == 2:
                atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea = self.mp2(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea0 = atom_fea0 + atom_fea
            elif self.num_MP_layers == 3:
                atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea = self.mp2(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea0 = atom_fea0 + atom_fea
                atom_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            elif self.num_MP_layers == 4:
                atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea = self.mp2(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea0 = atom_fea0 + atom_fea
                atom_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea = self.mp4(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea0 = atom_fea0 + atom_fea
            elif self.num_MP_layers == 5:
                atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea = self.mp2(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea0 = atom_fea0 + atom_fea
                atom_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea = self.mp4(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                atom_fea0 = atom_fea0 + atom_fea
                atom_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)

            atom_fea_s = atom_fea
            atom_fea, edge_fea = self.mp_output(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            out = edge_fea

        return out
    

def init_model(restart_file, structures, node_embedding_type, graph_layer_mechanism, edge_fea_len, gauss_stop, num_MP_layers, device):
    '''
    Initialize the model
    '''

    if restart_file is None:

        # if soap is used as node embeddings, atom_fea_len is the size of the SOAP feature vector
        if node_embedding_type == 'SOAP':
            atom_fea_len = structures[0].soap_features.shape[1]
        else:
            # use the sum of the atomic numbers
            atom_fea_len = np.sum([utils.periodic_table[element] for element in structures[0].atomic_structure.get_chemical_symbols()])

        total_num_orbitals = structures[0].num_unique_orbitals

        if graph_layer_mechanism == 'GAT':
            model = HGNN(num_species = 72,
                         in_atom_fea_len = atom_fea_len,
                         num_orbital=total_num_orbitals**2,
                         in_edge_fea_len = edge_fea_len, 
                         gauss_stop = gauss_stop, atom_update_net='GAT',
                         num_MP_layers=num_MP_layers).to(device)
            
        elif graph_layer_mechanism == 'GCN':
            model = HGNN(num_species = 72,
                         in_atom_fea_len = atom_fea_len,
                         num_orbital=total_num_orbitals**2,
                         in_edge_fea_len = edge_fea_len, 
                         gauss_stop = gauss_stop,
                         num_MP_layers=num_MP_layers).to(device)
    
        else:
            raise NotImplementedError
        
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    else:
        print("Restarting training from a saved model...")
        model = torch.load(restart_file)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
        
    
    print(model)

    return model