"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import torch

from .radial import PolynomialEnvelope, RadialMLP


class EdgeDegreeEmbedding(torch.nn.Module):
    """

    Args:
        sphere_channels (int):      Number of spherical channels

        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features

        rescale_factor (float):     Rescale the sum aggregation
        cutoff (float):             Cutoff distance for the radial function

        mappingReduced (CoefficientMapping): Class to convert l and m indices once node embedding is rotated
        out_mask (torch.Tensor):    Mask to select the output irreps
        # use_envelope (bool):        Whether to use envelope function
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        max_num_elements: int,
        edge_channels_list,
        rescale_factor,
        cutoff,
        mappingReduced,
        out_mask,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced

        self.m_0_num_coefficients: int = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents: int = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialMLP(self.edge_channels_list)

        self.rescale_factor = rescale_factor

        self.out_mask = out_mask

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        forward_edge_mask,
        wigner_inv,
        node_or_edge='node'
    ):
        # adds stuff to all the m=0s in all the ls

        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        x_edge_m_pad = torch.zeros(
            (
                x_edge_m_0.shape[0],
                (self.m_all_num_coefficents - self.m_0_num_coefficients),
                self.sphere_channels,
            ),
            device=x_edge_m_0.device,
            dtype=x_edge_m_0.dtype,
        )
        x_edge_embedding = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding = torch.einsum(
            "nac,ab->nbc", x_edge_embedding, self.mappingReduced.to_m
        )

        if node_or_edge == 'node':
            # Rotate back the irreps
            x_edge_embedding = torch.bmm(wigner_inv, x_edge_embedding)  # this rotation seems to be important

            x_edge_embedding = x_edge_embedding.to(x.dtype)

            x.index_add_(
                0, edge_index[1][forward_edge_mask], x_edge_embedding / self.rescale_factor
            )
            # if (~forward_edge_mask).any():  # if we are ignoring half the edges, add the other half
            #     x.index_add_(
            #     0, edge_index[0][forward_edge_mask], -1*x_edge_embedding / self.rescale_factor
            # )

            return x
        else:

            # Rotate back the irreps
            x_edge_embedding = torch.bmm(wigner_inv, x_edge_embedding)

            x_edge_embedding = x_edge_embedding.to(x.dtype)

            return x_edge_embedding
