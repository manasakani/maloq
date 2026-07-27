# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Feature-local NTE node block used by :mod:`op_projection.backbone`.

The operator-projection trunk has no persistent edge stack. Pair geometry is
used transiently while updating node states.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ...helm.nn.activation import GateActivation
from ...helm.nn.layer_norm import get_normalization_layer
from ...helm.nn.radial import PolynomialEnvelope
from ...helm.nn.so2_layers import SO2_Convolution


class Edgewise(nn.Module):
    """Source-target SO(2) message transform for node or pair output."""

    def __init__(
        self,
        *,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        mapping,
        grids,
        cutoff: float,
        modulate_radial_features: bool,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mapping
        self.SO3_grid = grids
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.cutoff = float(cutoff)
        self.use_edge_envelope = True
        self.envelope = PolynomialEnvelope(exponent=5)
        self.use_edge_scalar_modulation = modulate_radial_features
        self.edge_scalar_modulator = None
        if modulate_radial_features:
            edge_feature_dim = self.edge_channels_list[0]
            self.edge_scalar_modulator = nn.Sequential(
                nn.Linear(self.sphere_channels, self.hidden_channels),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, edge_feature_dim),
            )

        l_to_m_permute = self.mappingReduced.l_harmonic[
            torch.argmax(self.mappingReduced.to_m, dim=1)
        ]
        self.act = GateActivation(
            lmax=self.lmax,
            mmax=self.mmax,
            num_channels=self.hidden_channels,
            outer_dim="m",
            l_to_m_permute=l_to_m_permute,
            gate_act_type="sigmoid",
        )
        extra_m0_output_channels = self.lmax * self.hidden_channels
        self.concat_size = 2
        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
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

    def _source_target(self, node_state, edge_index):
        return (
            node_state[edge_index[0]],
            node_state[edge_index[1]],
        )

    def _transform(
        self,
        source,
        target,
        radial_features,
        edge_distance,
        wigner,
        wigner_inv,
    ):
        message = torch.cat((source, target), dim=2)
        message = torch.bmm(wigner, message)
        message = torch.einsum(
            "nac,ba->nbc",
            message,
            self.mappingReduced.to_m,
        )
        message, gating = self.so2_conv_1(message, radial_features)
        message = self.act(gating, message)
        message = self.so2_conv_2(message, radial_features)
        envelope = self.envelope(edge_distance / self.cutoff)
        message = message * envelope.view(-1, 1, 1)
        message = torch.einsum(
            "nac,ab->nbc",
            message,
            self.mappingReduced.to_m,
        )
        return torch.bmm(wigner_inv, message)

    def node_messages(
        self,
        node_state,
        radial_features,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
    ):
        source, target = self._source_target(node_state, edge_index)
        messages = self._transform(
            source,
            target,
            radial_features,
            edge_distance,
            wigner,
            wigner_inv,
        )
        aggregated = torch.zeros(
            (node_state.shape[0],) + messages.shape[1:],
            dtype=messages.dtype,
            device=messages.device,
        )
        aggregated.index_add_(0, edge_index[1], messages)
        return aggregated

    def edge_messages(
        self,
        node_state,
        radial_features,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
    ):
        source, target = self._source_target(node_state, edge_index)
        first_radial_features = radial_features
        if self.edge_scalar_modulator is not None:
            scalar_pair = source[:, 0, :] * target[:, 0, :]
            modulation = self.edge_scalar_modulator(scalar_pair)
            first_radial_features = radial_features * modulation

        message = torch.cat((source, target), dim=2)
        message = torch.bmm(wigner, message)
        message = torch.einsum(
            "nac,ba->nbc",
            message,
            self.mappingReduced.to_m,
        )
        message, gating = self.so2_conv_1(message, first_radial_features)
        message = self.act(gating, message)
        # The selected experiment modulates only SO2 convolution 1.
        message = self.so2_conv_2(message, radial_features)
        envelope = self.envelope(edge_distance / self.cutoff)
        message = message * envelope.view(-1, 1, 1)
        message = torch.einsum(
            "nac,ab->nbc",
            message,
            self.mappingReduced.to_m,
        )
        return torch.bmm(wigner_inv, message)


class GridAtomwise(nn.Module):
    """Three-layer equivariant feed-forward network on an SO(3) grid."""

    def __init__(
        self,
        *,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        grids,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = lmax
        self.SO3_grid = grids
        self.grid_mlp = nn.Sequential(
            nn.Linear(sphere_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, sphere_channels, bias=False),
        )

    def forward(self, state):
        grid_state = self.SO3_grid["lmax_lmax"].to_grid(
            state,
            self.lmax,
            self.lmax,
        )
        grid_state = self.grid_mlp(grid_state)
        return self.SO3_grid["lmax_lmax"].from_grid(
            grid_state,
            self.lmax,
            self.lmax,
        )


class DegreeScale(nn.Module):
    """Fixed bounded per-degree residual scale."""

    def __init__(self, lmax: int) -> None:
        super().__init__()
        self.mode = "bounded_degree"
        self.init = 0.015625
        self.log_range = 4.1588830833596715
        self.raw = nn.Parameter(torch.zeros(lmax + 1))
        expand_index = torch.cat(
            [torch.full((2 * degree + 1,), degree) for degree in range(lmax + 1)]
        ).long()
        self.register_buffer("expand_index", expand_index, persistent=False)

    def degree_scales(self):
        return self.init * torch.exp(self.log_range * torch.tanh(self.raw))

    def forward(self, update):
        component_scales = self.degree_scales().index_select(
            0,
            self.expand_index,
        )
        return update * component_scales.view(1, -1, 1)


class _Block(nn.Module):
    """Shared construction for the three fixed block equations."""

    def __init__(
        self,
        *,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mapping,
        grids,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: str,
        modulate_radial_features: bool,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.norm_1 = get_normalization_layer(
            norm_type,
            lmax=lmax,
            num_channels=sphere_channels,
        )
        self.norm_2 = get_normalization_layer(
            norm_type,
            lmax=lmax,
            num_channels=sphere_channels,
        )
        self.post_residual_norm = nn.Identity()
        self.edge_wise = Edgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mapping=mapping,
            grids=grids,
            cutoff=cutoff,
            modulate_radial_features=modulate_radial_features,
        )
        self.atom_wise = GridAtomwise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            grids=grids,
        )
        self.edge_update_scale = DegreeScale(lmax)
        self.atom_update_scale = DegreeScale(lmax)

    def _atomwise(self, state):
        return self.atom_wise(self.norm_2(state))


class NodeBlock(_Block):
    """N <- N + Se·EdgeWise(N); N <- N + Sa·AtomWise(N)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(modulate_radial_features=False, **kwargs)

    def forward(
        self,
        node_state,
        edge_state,
        radial_features,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
    ):
        local = self.edge_wise.node_messages(
            self.norm_1(node_state),
            radial_features,
            edge_distance,
            edge_index,
            wigner,
            wigner_inv,
        )
        local = node_state + self.edge_update_scale(local)
        return local + self.atom_update_scale(self._atomwise(local))


class PairProjectionBlock(_Block):
    """Create a transient equivariant pair latent from node latents.

    The returned tensor is meant to be consumed immediately by an operator
    projection and discarded. It is deliberately not part of the backbone
    output contract.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(modulate_radial_features=True, **kwargs)
        # Pair projection has no pre-existing edge state and therefore no
        # edge residual to scale. The inherited scale remains in the state
        # layout but is intentionally excluded from optimization/DDP.
        self.edge_update_scale.requires_grad_(False)

    def forward(
        self,
        node_state,
        radial_features,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
    ):
        return self.from_normalized_nodes(
            self.norm_1(node_state),
            radial_features,
            edge_distance,
            edge_index,
            wigner,
            wigner_inv,
        )

    def from_normalized_nodes(
        self,
        normalized_node_state,
        radial_features,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
    ):
        """Project one edge chunk after normalizing nodes once per callback."""
        local = self.edge_wise.edge_messages(
            normalized_node_state,
            radial_features,
            edge_distance,
            edge_index,
            wigner,
            wigner_inv,
        )
        return local + self.atom_update_scale(self._atomwise(local))


__all__ = ["NodeBlock", "PairProjectionBlock"]
