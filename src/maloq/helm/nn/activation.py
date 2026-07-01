# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""
The codes in this file are adapted from fairchem (https://github.com/facebookresearch/fairchem).
See LICENSES/MIT-fairchem.md for license information.
"""

from __future__ import annotations

import torch


class GateActivation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int, num_channels: int, outer_dim='l', l_to_m_permute=None) -> None:
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        # if outer_dim == 'm', check that l_to_m_permute is provided
        if outer_dim == 'm':
            assert l_to_m_permute is not None, "l_to_m_permute must be provided when outer_dim is 'm'!"

        # compute `expand_index` based on `lmax` and `mmax`
        if outer_dim == 'l':
            num_components = 0
            for lval in range(1, self.lmax + 1):
                num_m_components = min((2 * lval + 1), (2 * self.mmax + 1))
                num_components = num_components + num_m_components
            expand_index = torch.zeros([num_components]).long()
            start_idx = 0
            for lval in range(1, self.lmax + 1):
                length = min((2 * lval + 1), (2 * self.mmax + 1))
                expand_index[start_idx : (start_idx + length)] = lval - 1
                start_idx = start_idx + length
            self.register_buffer("expand_index", expand_index)
            # print("expand_index (l):", self.expand_index)

        else:  # outer_dim == 'm'
            # Skip scalar (ℓ = 0, m = 0)
            num_components = l_to_m_permute.numel() - 1
            expand_index = torch.zeros(num_components, dtype=torch.long)

            for i in range(1, l_to_m_permute.numel()):
                l = l_to_m_permute[i]
                expand_index[i - 1] = l - 1 # -1 to account for scalar skip
            self.register_buffer("expand_index", expand_index)
            # print("expand_index (m):", self.expand_index)

        self.scalar_act = (
            torch.nn.SiLU()
            # torch.nn.Tanh()  # for antisym
        )  # SwiGLU(self.num_channels, self.num_channels)  # #
        # self.gate_act = torch.nn.Sigmoid()  # torch.nn.SiLU() # #
        self.gate_act = torch.nn.Tanh()  # torch.nn.SiLU() # #

    def forward(self, gating_scalars, input_tensors):
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """

        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )
        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )

        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)

        input_tensors_vectors = input_tensors.narrow(1, 1, input_tensors.shape[1] - 1)
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        return torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)


class S2Activation(torch.nn.Module):
    """
    Assume we only have one resolution
    """

    def __init__(self, lmax: int, mmax: int, SO3_grid) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()
        self.SO3_grid = SO3_grid

    def forward(self, inputs):
        to_grid_mat = self.SO3_grid["lmax_mmax"].get_to_grid_mat()
        from_grid_mat = self.SO3_grid["lmax_mmax"].get_from_grid_mat()
        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        return torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)


class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int, SO3_grid) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax, SO3_grid)

    def forward(self, input_scalars, input_tensors):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors)
        return torch.cat(
            (
                output_scalars,
                output_tensors.narrow(1, 1, output_tensors.shape[1] - 1),
            ),
            dim=1,
        )
