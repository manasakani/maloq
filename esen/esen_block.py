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

        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax, mmax=self.mmax, num_channels=self.hidden_channels
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")

        self.so2_conv_1 = SO2_Convolution(
            3 * self.sphere_channels,  
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

        # self.so2_conv_3 = SO2_Convolution(
        #     self.sphere_channels,
        #     self.sphere_channels,
        #     self.lmax,
        #     self.mmax,
        #     self.mappingReduced,
        #     internal_weights=True,
        #     edge_channels_list=None,
        #     extra_m0_output_channels=None,
        # )

        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )

    
    def forward(
        self,
        x,
        x_message_edge,
        x_edge,
        edge_index,
        edge_mask,
        wigner,
        wigner_inv,
        node_or_edge,
    ):
        if node_or_edge == 'node':
            return self.forward_node(x,
                                    x_message_edge,
                                    x_edge,
                                    edge_index,
                                    edge_mask,
                                    wigner,
                                    wigner_inv
                                    )

        if node_or_edge == 'edge':
            return self.forward_edge(x,
                                    x_message_edge,
                                    x_edge,
                                    edge_index,
                                    edge_mask,
                                    wigner,
                                    wigner_inv
                                    )

    def forward_node(
        self,
        x,
        x_message_edge,
        x_edge,
        edge_index,
        edge_mask,
        wigner,
        wigner_inv
    ):

        x_source = x[edge_index[0][edge_mask]]
        x_target = x[edge_index[1][edge_mask]]
        x_message = torch.cat((x_source, x_message_edge, x_target), dim=2)

        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner, x_message)

        # SO2 convolution
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)

        # testing extra convolution for nodes:
        # x_message = self.so2_conv_3(x_message, x_edge)

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        ## DEBUG ###
        # reset backwards edges to rotation of forward edges
        # for forward_edge, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
        #     if i < j:
        #         mask = (edge_index[0] == j) & (edge_index[1] == i)
        #         indices = torch.nonzero(mask, as_tuple=False)
        #         index = indices[0].item() if indices.numel() > 0 else None
        #         x_message[forward_edge] = -1*x_message[index] 
        #         assert i == edge_index[1][index]
        #         assert j == edge_index[0][index]
        ## DEBUG ###

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        # aggregate messages
        new_embedding.index_add_(0, edge_index[1][edge_mask], x_message)

        if (~edge_mask).any():  # if we are ignoring half the edges
                new_embedding.index_add_(0, edge_index[0][edge_mask], -1*x_message)

        return new_embedding
    
    def forward_edge(
        self,
        x,
        x_message_edge,
        x_edge,
        edge_index,
        edge_mask,
        wigner,
        wigner_inv
    ):
        x_source = x[edge_index[0][edge_mask]]
        x_target = x[edge_index[1][edge_mask]]
        x_message = torch.cat((x_source, x_message_edge, x_target), dim=2)

        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner, x_message)

        # SO2 convolution
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        ## DEBUG ###
        # for forward_edge, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
        #     if i < j:
        #         mask = (edge_index[0] == j) & (edge_index[1] == i)
        #         indices = torch.nonzero(mask, as_tuple=False)
        #         index = indices[0].item() if indices.numel() > 0 else None
        #         x_message[forward_edge] = -1*x_message[index]   
        #         assert i == edge_index[1][index]
        #         assert j == edge_index[0][index]
        ## DEBUG ###

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
                bias=True,
            ),
            nn.SiLU(),
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
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(
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
        )

        self.norm_2 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        if mlp_type == "spectral":
            self.atom_wise = SpectralAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
            )
        else:
            raise ValueError(f"Unknown MLP type {mlp_type}")

    def forward(
        self,
        x_message_node,
        x_message_edge,
        x_edge,
        edge_distance,
        edge_index,
        edge_mask,
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
                edge_index,
                edge_mask,
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
            x_message_edge = self.norm_1(x_message_edge)

            x_message_edge = self.edge_wise(
                x_message_node,
                x_message_edge,
                x_edge,
                edge_index,
                edge_mask,
                wigner,
                wigner_inv,
                node_or_edge,
            )

            x_message_edge = x_message_edge + x_res

            x_res = x_message_edge
            x_message_edge = self.norm_2(x_message_edge)

            x_message_edge = self.atom_wise(x_message_edge)
            return x_message_edge + x_res