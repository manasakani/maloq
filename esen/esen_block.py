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

# Test equivariance
# import e3nn.o3
# from e3nn.util.test import equivariance_error
from e3nn.o3 import Irreps

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

        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
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

        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]
        x_message = torch.cat((x_source, x_message_edge, x_target), dim=2)
        # x_message = torch.cat((x_source, x_target), dim=2)
        # x_message = x_source

        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner, x_message)

        # SO2 convolution
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)

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
        wigner,
        wigner_inv,
        node_or_edge,
    ):

        if node_or_edge == 'node':
            x = x_message_node
        else:
            x = x_message_edge

        x_res = x

        # #__ROTATION___
        # # Cartesian Rotation for the mol:
        # device="cuda:0"
        # alpha=230.0
        # beta=70.0
        # gamma=180.0
        # alpha_rad = torch.deg2rad(torch.tensor(alpha))
        # beta_rad = torch.deg2rad(torch.tensor(beta))
        # gamma_rad = torch.deg2rad(torch.tensor(gamma))
        # Rx = torch.tensor([[1, 0, 0], [0, torch.cos(alpha_rad), -torch.sin(alpha_rad)], [0, torch.sin(alpha_rad), torch.cos(alpha_rad)]])
        # Ry = torch.tensor([[torch.cos(beta_rad), 0, torch.sin(beta_rad)], [0, 1, 0], [-torch.sin(beta_rad), 0, torch.cos(beta_rad)]])
        # Rz = torch.tensor([[torch.cos(gamma_rad), -torch.sin(gamma_rad), 0], [torch.sin(gamma_rad), torch.cos(gamma_rad), 0], [0, 0, 1]])
        # R_cart = torch.matmul(Rz, torch.matmul(Ry, Rx)) 

        # # Spherical Rotation for Irreps:
        # internal_irreps = Irreps("1x0e+1x1e+1x2e+1x3e+1x4e")
        # R_sphere_in = internal_irreps.D_from_matrix(R_cart).to(device)
        # #__ROTATION___

        x = self.norm_1(x)

        # x = torch.matmul(R_sphere_in, x) # <-- Rotate first // forward commutator

        x = self.edge_wise(
            x_message_node,
            x_message_edge,
            x_edge,
            edge_distance,
            edge_index,
            wigner,
            wigner_inv,
            node_or_edge,
        )

        # x = torch.matmul(R_sphere_in, x) # <-- Rotate last // backward commutator
        
        print(x)
        exit()

        x = x + x_res
        x_res = x

        x = self.norm_2(x)
        x = self.atom_wise(x)
        return x + x_res