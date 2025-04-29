"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .nn.activation import GateActivation, SeparableS2Activation, SmoothLeakyReLU
from .nn.layer_norm import get_normalization_layer
from .nn.radial import PolynomialEnvelope
from .nn.so2_layers import SO2_Convolution
from .nn.so3_layers import SO3_Linear

class Edgewise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list,
        mappingReduced,
        cutoff,
        act_type="gate",
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.mappingReduced = mappingReduced
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

    
    def forward(
        self,
        x,
        x_message_edge,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        node_or_edge,
    ):
        if node_or_edge == 'node':
            return self.forward_node(x,
                                    x_message_edge,
                                    x_edge,
                                    edge_distance,
                                    edge_index,
                                    wigner,
                                    wigner_inv
                                    )

        if node_or_edge == 'edge':
            return self.forward_edge(x,
                                    x_message_edge,
                                    x_edge,
                                    edge_distance,
                                    edge_index,
                                    wigner,
                                    wigner_inv
                                    )

    def forward_node(
        self,
        x,
        x_message_edge,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv
    ):

        self.num_heads = 2 ###
        self.attn_alpha_channels = 16 ###
        self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels) ###
        self.alpha_act = SmoothLeakyReLU() ###

        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]
        x_message = torch.cat((x_source, x_message_edge, x_target), dim=2)

        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner, x_message)

        # SO2 convolution
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        #---
        # x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)
        # x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        # x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels) # for activation
        # x_0_alpha  = x_0_extra.narrow(1, 0, x_alpha_num_channels) # for attention weights, shape [E, num_heads * attn_alpha_channels]
        # x_message = self.act(x_0_gating, x_message)
        #---
        
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)

        #---
        # # Attention weights
        # start_attention = time.time()
        # x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels) # shape of [E, num_heads, attn_alpha_channels]
        # x_0_alpha = self.alpha_norm(x_0_alpha)
        # x_0_alpha = self.alpha_act(x_0_alpha)
        # alpha = torch.einsum('bik, ik -> bi', x_0_alpha, self.alpha_dot)

        # # Compute the softmax over the incoming edges
        # offset_local_dst_indices = partition.expand_edge_0["local_indices"]
        # alpha = torch_geometric.utils.softmax(alpha, offset_local_dst_indices)      # softmax over the incoming edges
        # alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)                 # shape of [E, 1, num_heads, 1]

        # # Attention weights * non-linear messages (weight each message by the corresponding attention weight)
        # attn = x_message                                                                      # shape of [E, (lmax+1)^2, # hidden channels]
        # attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)     # shape of [E, #channels, num_heads, attn_value_channels]
        # attn = attn * alpha
        # attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        # x_message = attn
        # end_attention = time.time()
        #---

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        # aggregate messages
        new_embedding.index_add_(0, edge_index[1], x_message)

        return new_embedding
    
    def forward_edge(
        self,
        x,
        x_message_edge,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv
    ):
        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]
        x_message = torch.cat((x_source, x_message_edge, x_target), dim=2)

        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner, x_message)

        # SO2 convolution
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

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


# class GridAtomwise(torch.nn.Module):
#     def __init__(
#         self,
#         sphere_channels: int,
#         hidden_channels: int,
#         lmax: int,
#         mmax: int,
#         SO3_grid,
#     ):
#         super().__init__()
#         self.sphere_channels = sphere_channels
#         self.hidden_channels = hidden_channels
#         self.lmax = lmax
#         self.mmax = mmax
#         self.SO3_grid = SO3_grid

#         self.grid_mlp = nn.Sequential(
#             nn.Linear(self.sphere_channels, self.hidden_channels, bias=False),
#             nn.SiLU(),
#             nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
#             nn.SiLU(),
#             nn.Linear(self.hidden_channels, self.sphere_channels, bias=False),
#         )

#     def forward(self, x):
#         # Project to grid
#         x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
#         # Perform point-wise operations
#         x_grid = self.grid_mlp(x_grid)
#         # Project back to spherical harmonic coefficients
#         return self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)


class eSEN_Block(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
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
                edge_distance,
                edge_index,
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
                edge_distance,
                edge_index,
                wigner,
                wigner_inv,
                node_or_edge,
            )
            x_message_edge = x_message_edge + x_res

            x_res = x_message_edge
            x_message_edge = self.norm_2(x_message_edge)
            x_message_edge = self.atom_wise(x_message_edge)
            return x_message_edge + x_res