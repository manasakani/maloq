"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .nn.activation import GateActivation, SeparableS2Activation#, SmoothLeakyReLU
from .nn.layer_norm import get_normalization_layer
from .nn.radial import PolynomialEnvelope
from .nn.so2_layers import SO2_Convolution
from .nn.so3_layers import SO3_Linear
from torch.utils.checkpoint import checkpoint

# Test equivariance
# import e3nn.o3
# from e3nn.util.test import equivariance_error
from e3nn.o3 import Irreps
from .common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
)


class Edgewise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list,
        mappingReduced,
        SO3_grid,
        cutoff,
        act_type="gate",
        include_edges=True
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.act_type = act_type
        self.include_edges = include_edges

        if self.act_type == "gate":
            # Get permutation to rearrange the gate scalars from l to m order
            l_to_m_permute = self.mappingReduced.l_harmonic[
                torch.argmax(self.mappingReduced.to_m, dim=1)
            ]

            self.act = GateActivation( # in m-major
                lmax=self.lmax, mmax=self.mmax, num_channels=self.hidden_channels, outer_dim='m', l_to_m_permute=l_to_m_permute
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")
        
        concat_size = 2

        self.so2_conv_1 = SO2_Convolution(
            concat_size * self.sphere_channels,  
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
        )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.sphere_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )

    
    def forward(
        self,
        x,
        x_message_edge,
        x_edge,
        forward_edges,
        edge_index,
        edge_mask,
        reverse_edge_map,
        wigner,
        wigner_inv,
        node_or_edge,
    ):
        if node_or_edge == 'node':
            return self.forward_node(x,
                                    x_message_edge,
                                    x_edge,
                                    forward_edges,
                                    edge_index,
                                    edge_mask,
                                    reverse_edge_map,
                                    wigner,
                                    wigner_inv
                                    )

        if node_or_edge == 'edge':
            return self.forward_edge(x,
                                    x_message_edge,
                                    x_edge,
                                    forward_edges,
                                    edge_index,
                                    edge_mask,
                                    reverse_edge_map,
                                    wigner,
                                    wigner_inv
                                    )

    def forward_node(
        self,
        x,
        x_message_edge,
        x_edge,
        forward_edges,
        edge_index,
        edge_mask,
        reverse_edge_map,
        wigner,
        wigner_inv
    ):

        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]

        # Create messages 
        x_message = torch.cat((x_source, x_target), dim=2) 

        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner, x_message)

        # SO2 convolutions + Gating
        x_message = torch.einsum("nac,ba->nbc", x_message, self.mappingReduced.to_m) # l-major -> m-major
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge) 
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)
        x_message = torch.einsum("nac,ab->nbc", x_message, self.mappingReduced.to_m) # m-major -> l-major

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        # aggregate messages
        new_embedding.index_add_(0, edge_index[1], x_message)    # if using the same, can just skip the if below

        return new_embedding
    
    def forward_edge(
        self,
        x,
        x_message_edge,
        x_edge,
        forward_edges,
        edge_index,
        edge_mask,
        reverse_edge_map,
        wigner,
        wigner_inv
    ):

        torch.cuda.nvtx.range_push("Create messages") # <--- START
        
        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]

        # Create regular messages
        x_message = torch.cat((x_source, x_target), dim=2) 

        torch.cuda.nvtx.range_pop() # <--- END

        # Rotate the irreps to align with the edge
        torch.cuda.nvtx.range_push("Rotate") # <--- START
        x_message = torch.bmm(wigner, x_message)
        torch.cuda.nvtx.range_pop() # <--- END

        torch.cuda.nvtx.range_push("l->m") # <--- START
        x_message = torch.einsum("nac,ba->nbc", x_message, self.mappingReduced.to_m) # l-major -> m-major
        torch.cuda.nvtx.range_pop() # <--- END

        # SO2 convolution #1
        torch.cuda.nvtx.range_push("So2 conv 1") # <--- START
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        torch.cuda.nvtx.range_pop() # <--- END
                    
        # Gate activation
        torch.cuda.nvtx.range_push("Gate") # <--- START
        x_message = self.act(x_0_gating, x_message)
        torch.cuda.nvtx.range_pop() # <--- END

        # SO2 convolution #2
        torch.cuda.nvtx.range_push("So2 conv 2") # <--- START
        x_message = self.so2_conv_2(x_message, x_edge)
        torch.cuda.nvtx.range_pop() # <--- END

        torch.cuda.nvtx.range_push("m->l") # <--- START
        x_message = torch.einsum("nac,ab->nbc", x_message, self.mappingReduced.to_m) # m-major -> l-major
        torch.cuda.nvtx.range_pop() # <--- END

        # Rotate back the irreps
        torch.cuda.nvtx.range_push("Rotate back") # <--- START
        x_message = torch.bmm(wigner_inv, x_message)
        torch.cuda.nvtx.range_pop() # <--- END

        # return new_embedding
        return x_message


class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.scalar_mlp = nn.Sequential(
            nn.Linear(
                self.sphere_channels,
                self.lmax * self.hidden_channels,
                bias=True, # False for antisymmetry
            ),
            nn.SiLU(), #Tanh() if for antisymmetry
        )

        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels, self.hidden_channels, lmax=self.lmax
        )
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        return self.so3_linear_2(x)


class eSEN_Block(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
        SO3_grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: str,
        act_type: str,
        mlp_type: str,
        include_edges=True,
        node_or_edge: str = 'node',  # 'node' or 'edge'
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        self.norm_2 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels 
        )

        self.edge_wise = Edgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            cutoff=cutoff,
            act_type=act_type,
            include_edges=include_edges
        )

        self.atom_wise = SpectralAtomwise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
        )

    def forward(
        self,
        x_message_node,
        x_message_edge,
        x_edge,
        forward_edges,
        edge_distance,
        edge_index,
        edge_mask,
        reverse_edge_map,
        wigner,
        wigner_inv,
        node_or_edge,
    ):

        if node_or_edge == 'node':
            x_res = x_message_node

            x_message_node = self.norm_1(x_message_node)

            x_message_node = self.edge_wise(
                x_message_node,
                x_message_edge,
                x_edge,
                forward_edges,
                edge_index,
                edge_mask,
                reverse_edge_map,
                wigner,
                wigner_inv,
                node_or_edge,
            )

            x_message_node = x_message_node + x_res
            x_res = x_message_node

            x_message_node = self.norm_2(x_message_node)
            x_message_node = self.atom_wise(x_message_node)
            return x_message_node + x_res
            
        else:
            x_res = x_message_edge

            torch.cuda.nvtx.range_push("Edgewise") # <--- START
            x_message_edge = self.edge_wise(
                x_message_node,
                x_message_edge,
                x_edge,
                forward_edges,
                edge_index,
                edge_mask,
                reverse_edge_map,
                wigner,
                wigner_inv,
                node_or_edge,
            )
            torch.cuda.nvtx.range_pop() # <--- END
            x_message_edge = self.norm_1(x_message_edge) 

            x_message_edge = x_message_edge + x_res
            x_res = x_message_edge 

            x_message_edge = self.norm_2(x_message_edge)

            torch.cuda.nvtx.range_push("Atomwise") # <--- START
            x_message_edge = self.atom_wise(x_message_edge)
            torch.cuda.nvtx.range_pop() # <--- END

            return x_message_edge + x_res
