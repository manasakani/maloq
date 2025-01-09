import torch
import torch.nn as nn
import math
import copy

from torch.nn import Linear
from SO3 import SO3_Embedding
from radial_function import RadialFunction

from torch import cuda
import os
DEBUG = os.environ.get("DEBUG", False)

class SO2_m_Convolution(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 Max degree (l) 
        mmax (int):                 Max order (m)
    """
    def __init__(
        self,
        m, 
        sphere_channels,
        m_output_channels,
        lmax, 
        mmax
    ):
        super(SO2_m_Convolution, self).__init__()
        
        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        num_channels = 0
        num_coefficents = 0

        if self.mmax >= self.m:
            num_coefficents = self.lmax - self.m + 1

        num_channels = num_channels + num_coefficents * self.sphere_channels

        assert num_channels > 0

        self.fc = Linear(num_channels, 
            2 * self.m_output_channels * (num_channels // self.sphere_channels), 
            bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))


    def forward(self, x_m):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.fc.out_features // 2)
        x_i = x_m.narrow(2, self.fc.out_features // 2, self.fc.out_features // 2)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1) #x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1) #x_r[:, 1] + x_i[:, 0]
        x_out = torch.cat((x_m_r, x_m_i), dim=1)
        
        return x_out


class SO2_Convolution(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 Max degree (l) 
        mmax (int):                 Max order (m) 
        mappingReduced (CoefficientMappingModule): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` (SO3_Embedding) and `extra_m0_features` (Tensor).
    """
    def __init__(
        self,
        sphere_channels,
        m_output_channels,
        lmax,
        mmax,
        mappingReduced,
        internal_weights=True,
        edge_channels_list=None,
        extra_m0_output_channels=None
    ):
        super(SO2_Convolution, self).__init__()

        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_rad = 0    # for radial function

        num_channels_m0 = 0
        num_coefficients = self.lmax + 1
        num_channels_m0 = num_channels_m0 + num_coefficients * self.sphere_channels             # m = 0 input block, size of [(l_max + 1) * 3/1*sphere_channels]

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (num_channels_m0 // self.sphere_channels) # m = 0 output block, size of [(l_max + 1) * m_output_channels]
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels             # m = 0 output block, size of [(l_max + 1) * m_output_channels + extra_m0_output_channels]
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)                                # Linear layer for m = 0 output block, dims [(l_max + 1) * 3/1*sphere_channels] -> [(l_max + 1) * m_output_channels]
        num_channels_rad = num_channels_rad + self.fc_m0.in_features                            # for radial function, size of [(l_max + 1) * 3/1*sphere_channels]

        # SO(2) convolution for non-zero m
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Convolution(
                    m, 
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax, 
                    self.mmax,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialFunction(self.edge_channels_list)


    def forward(self, x, x_edge):

        num_edges = len(x_edge)
        out = []

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_push("_m_primary")
        # Reshape the spherical harmonics based on m (order)
        x._m_primary(self.mappingReduced)

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("rad_func")
        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("m0")

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.embedding.narrow(1, 0, self.mappingReduced.m_size[0])               # m=0 coefficient block of the embeddings, shape [num_edges, (l_max + 1), 3/1*sphere_channels]
        x_0 = x_0.reshape(num_edges, -1)                                            # reshape the m=0 coefficients, shape [num_edges, (l_max + 1)*3/1*sphere_channels]
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0                                                    # multiply the input features with the radial function
        x_0 = self.fc_m0(x_0)                                                       # apply linear layer to the m=0 coefficients   
        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()

        x_0_extra = None
        # extract extra m0 features 
        if self.extra_m0_output_channels is not None:
            x_0_extra = x_0.narrow(-1, 0, self.extra_m0_output_channels)
            x_0 = x_0.narrow(-1, self.extra_m0_output_channels, (self.fc_m0.out_features - self.extra_m0_output_channels))
        
        x_0 = x_0.view(num_edges, -1, self.m_output_channels)
        #x.embedding[:, 0 : self.mappingReduced.m_size[0]] = x_0
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, self.mmax + 1):
            if DEBUG:
                cuda.synchronize()
                cuda.nvtx.range_push("m" + str(m))
            # Get the m order coefficients
            x_m = x.embedding.narrow(1, offset, 2 * self.mappingReduced.m_size[m])      # size of the m-th block of the embeddings, shape [num_edges, 2*(l_max - m + 1), 3/1*sphere_channels]
            x_m = x_m.reshape(num_edges, 2, -1)

            # Perform SO(2) convolution
            if self.rad_func is not None:
                x_edge_m = x_edge.narrow(1, offset_rad, self.so2_m_conv[m - 1].fc.in_features)
                x_edge_m = x_edge_m.reshape(num_edges, 1, self.so2_m_conv[m - 1].fc.in_features)
                x_m = x_m * x_edge_m
            x_m = self.so2_m_conv[m - 1](x_m)
            x_m = x_m.view(num_edges, -1, self.m_output_channels)
            #x.embedding[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = x_m
            out.append(x_m)
            offset = offset + 2 * self.mappingReduced.m_size[m]
            offset_rad = offset_rad + self.so2_m_conv[m - 1].fc.in_features

            if DEBUG:
                cuda.synchronize()
                cuda.nvtx.range_pop()

        out = torch.cat(out, dim=1)
        out_embedding = SO3_Embedding(
            0, 
            x.lmax, 
            self.m_output_channels, 
            device=x.device, 
            dtype=x.dtype
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax([self.lmax], [self.mmax])

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_push("_l_primary")

        # Reshape the spherical harmonics based on l (degree)
        out_embedding._l_primary(self.mappingReduced)

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()

        if self.extra_m0_output_channels is not None:
            return out_embedding, x_0_extra
        else:
            return out_embedding