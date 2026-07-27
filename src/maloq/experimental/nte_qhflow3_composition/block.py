# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""
The codes in this file are adapted from fairchem (https://github.com/facebookresearch/fairchem).
See LICENSES/MIT-fairchem.md for license information.

This feature-local copy preserves the selector-heavy NTE/QHFlow3 composition experiments.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn

from ...helm.nn.activation import GateActivation, SeparableS2Activation#, SmoothLeakyReLU
from ...helm.nn.layer_norm import get_normalization_layer
from ...helm.nn.radial import PolynomialEnvelope
from ...helm.nn.so2_layers import SO2_Convolution
from ...helm.nn.so3_layers import SO3_Linear
from ...helm.nn.communication import exchange_nodes
from ...helm.nn.communication import ExchangeNodes
from torch.utils.checkpoint import checkpoint

import torch.distributed as dist
from mpi4py import MPI

from e3nn.o3 import Irreps
from ...helm.common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
)

# from torch.autograd import gradcheck

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
        message_type,
        act_type="gate",
        gate_act_type="tanh",
        include_edges=True,
        use_edge_envelope=False,
        use_edge_scalar_modulation=False,
    ):
        super().__init__()

        # self.rank = dist.get_rank()
        # self.world_size = dist.get_world_size()
        # self.comm = MPI.COMM_WORLD

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.act_type = act_type
        self.gate_act_type = gate_act_type
        self.include_edges = include_edges
        self.cutoff = float(cutoff)
        self.use_edge_envelope = bool(use_edge_envelope)
        self.envelope = PolynomialEnvelope(exponent=5)
        self.use_edge_scalar_modulation = bool(use_edge_scalar_modulation)
        self.edge_scalar_modulator = None
        if self.use_edge_scalar_modulation:
            edge_feature_dim = self.edge_channels_list[0]
            self.edge_scalar_modulator = nn.Sequential(
                nn.Linear(self.sphere_channels, self.hidden_channels),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, edge_feature_dim),
            )

        if self.act_type == "gate":
            # Get permutation to rearrange the gate scalars from l to m order
            l_to_m_permute = self.mappingReduced.l_harmonic[
                torch.argmax(self.mappingReduced.to_m, dim=1)
            ]

            self.act = GateActivation( # in m-major
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=self.hidden_channels,
                outer_dim='m',
                l_to_m_permute=l_to_m_permute,
                gate_act_type=self.gate_act_type,
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")

        if message_type == "source-target-message":
            self.concat_size = 3
        elif message_type == "source-target":
            self.concat_size = 2
        else:
            raise ValueError(f"Unknown message type {message_type}")

        self.so2_conv_1 = SO2_Convolution(
            self.concat_size * self.sphere_channels,
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
        partition,
    ):
        if node_or_edge == 'node':

            return self.forward_node(x,
                                    x_message_edge,
                                    x_edge,
                                    edge_distance,
                                    edge_index,
                                    wigner,
                                    wigner_inv,
                                    partition
                                    )

        if node_or_edge == 'edge':

            return self.forward_edge(x,
                                    x_message_edge,
                                    x_edge,
                                    edge_distance,
                                    edge_index,
                                    wigner,
                                    wigner_inv,
                                    partition
                                    )

    def forward_node(
        self,
        x,
        x_message_edge,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        partition
    ):

        #  Communicate edge embeddings between partitions to assemble the messages
        if partition:
            local_num_edges = edge_index.shape[1]
            x_source = ExchangeNodes.apply(
                                            x,
                                            local_num_edges,
                                            partition.expand_edge_0
                                        )

            # Due to the `incoming edge` distribution, 'target' nodes are locally owned by this rank
            local_indices_torch = partition.expand_edge_1['local_indices_torch']
            x_target = x[local_indices_torch]

        else:
            x_source = x[edge_index[0]]
            x_target = x[edge_index[1]]

        # Create messages
        if self.concat_size == 3:
            x_message = torch.cat((x_source, x_target, x_message_edge), dim=2)
        else:
            x_message = torch.cat((x_source, x_target), dim=2)

        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner, x_message)

        # SO2 convolutions + Gating
        x_message = torch.einsum("nac,ba->nbc", x_message, self.mappingReduced.to_m)   # l-major -> m-major
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)                     # SO2 Convolution #1 (embedding dim 2E -> H)
        x_message = self.act(x_0_gating, x_message)                                    # Gate activation
        x_message = self.so2_conv_2(x_message, x_edge)                                 # SO2 Convolution #2 (embedding dim H -> E)
        if self.use_edge_envelope:
            envelope = self.envelope(edge_distance / self.cutoff)
            x_message = x_message * envelope.view(-1, 1, 1)
        x_message = torch.einsum("nac,ab->nbc", x_message, self.mappingReduced.to_m)   # m-major -> l-major

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        # aggregate messages
        if partition:
            is_local = partition.reduce_edge['is_local']
            local_indices = partition.reduce_edge['local_indices']
            new_embedding.index_add_(0, local_indices, x_message)
        else:
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
        wigner_inv,
        partition
    ):

        #  Communicate the edge embeddings between partitions to assemble the messages
        if partition:
            local_num_edges = edge_index.shape[1]
            x_source = ExchangeNodes.apply(
                                            x,
                                            local_num_edges,
                                            partition.expand_edge_0
                                        )

            # Due to the `incoming edge` distribution, 'target' nodes are locally owned by this rank
            local_indices_torch = partition.expand_edge_1['local_indices_torch']
            x_target = x[local_indices_torch]

        else:
            x_source = x[edge_index[0]]
            x_target = x[edge_index[1]]

        # Create messages
        if self.concat_size == 3:
            x_message = torch.cat((x_source, x_target, x_message_edge), dim=2)
        else:
            x_message = torch.cat((x_source, x_target), dim=2)
        modulated_x_edge = x_edge
        if self.use_edge_scalar_modulation:
            scalar_pair_features = x_source[:, 0, :] * x_target[:, 0, :]
            modulation = self.edge_scalar_modulator(scalar_pair_features)
            modulated_x_edge = x_edge * modulation


        # Rotate the irreps to align with the edge
        x_message = torch.bmm(wigner, x_message)

        # SO2 convolutions + Gating
        x_message = torch.einsum("nac,ba->nbc", x_message, self.mappingReduced.to_m) # l-major -> m-major
        x_message, x_0_gating = self.so2_conv_1(x_message, modulated_x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)
        if self.use_edge_envelope:
            envelope = self.envelope(edge_distance / self.cutoff)
            x_message = x_message * envelope.view(-1, 1, 1)
        x_message = torch.einsum("nac,ab->nbc", x_message, self.mappingReduced.to_m) # m-major -> l-major

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        return x_message


class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        gate_act_type: str = "tanh",
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
            lmax=self.lmax,
            mmax=self.lmax,
            num_channels=self.hidden_channels,
            gate_act_type=gate_act_type,
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        return self.so3_linear_2(x)


class GridAtomwise(torch.nn.Module):
    """QHFlow3-style atomwise feed-forward network on an SO(3) grid."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid
        self.grid_mlp = nn.Sequential(
            nn.Linear(sphere_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, sphere_channels, bias=False),
        )

    def forward(self, x):
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(
            x,
            self.lmax,
            self.lmax,
        )
        x_grid = self.grid_mlp(x_grid)
        return self.SO3_grid["lmax_lmax"].from_grid(
            x_grid,
            self.lmax,
            self.lmax,
        )


class DegreeLayerScale(torch.nn.Module):
    """Equivariant per-degree scale for one residual-update branch."""

    def __init__(
        self,
        lmax: int,
        mode: str = "none",
        init: float = 1.0,
        log_range: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"none", "bounded_degree"}:
            raise ValueError(
                "residual_update_scale_mode must be 'none' or "
                f"'bounded_degree', got {mode!r}."
            )
        if init <= 0.0:
            raise ValueError("residual_update_scale_init must be positive.")
        if log_range < 0.0:
            raise ValueError("residual_update_scale_log_range cannot be negative.")
        self.mode = mode
        self.init = float(init)
        self.log_range = float(log_range)
        if self.mode == "bounded_degree":
            self.raw = nn.Parameter(torch.zeros(lmax + 1))
            expand_index = torch.cat(
                [torch.full((2 * degree + 1,), degree) for degree in range(lmax + 1)]
            ).long()
            self.register_buffer("expand_index", expand_index, persistent=False)
        else:
            self.register_parameter("raw", None)

    def degree_scales(self):
        if self.raw is None:
            return None
        return self.init * torch.exp(self.log_range * torch.tanh(self.raw))

    def forward(self, update):
        scales = self.degree_scales()
        if scales is None:
            return update
        component_scales = scales.index_select(0, self.expand_index)
        return update * component_scales.view(1, -1, 1)


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
        message_type: str,
        gate_act_type: str = "tanh",
        include_edges=True,
        node_or_edge: str = 'node',  # 'node' or 'edge'
        use_edge_envelope: bool = False,
        use_edge_scalar_modulation: bool = False,
        residual_update_scale_mode: str = "none",
        residual_update_scale_init: float = 1.0,
        residual_update_scale_log_range: float = 0.0,
        atom_norm_type: str | None = None,
        post_residual_norm_type: str | None = None,
        atomwise_output_mode: str = "residual_scaled",
        edge_norm1_position: str = "post_edgewise",
        edgewise_output_mode: str = "residual_scaled",
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        if edgewise_output_mode not in {"residual_scaled", "direct"}:
            raise ValueError(
                "edgewise_output_mode must be 'residual_scaled' or 'direct', "
                f"got {edgewise_output_mode!r}."
            )
        self.edgewise_output_mode = edgewise_output_mode
        if atomwise_output_mode not in {"residual_scaled", "direct"}:
            raise ValueError(
                "atomwise_output_mode must be 'residual_scaled' or 'direct', "
                f"got {atomwise_output_mode!r}."
            )
        self.atomwise_output_mode = atomwise_output_mode
        if edge_norm1_position not in {"post_edgewise", "pre_node"}:
            raise ValueError(
                "edge_norm1_position must be 'post_edgewise' or 'pre_node', "
                f"got {edge_norm1_position!r}."
            )
        self.edge_norm1_position = edge_norm1_position

        self.norm_1 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        self.norm_2 = get_normalization_layer(
            norm_type if atom_norm_type is None else atom_norm_type,
            lmax=self.lmax,
            num_channels=sphere_channels,
        )
        self.post_residual_norm = (
            nn.Identity()
            if post_residual_norm_type is None
            else get_normalization_layer(
                post_residual_norm_type,
                lmax=self.lmax,
                num_channels=sphere_channels,
            )
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
            message_type=message_type,
            act_type=act_type,
            gate_act_type=gate_act_type,
            include_edges=include_edges,
            use_edge_envelope=use_edge_envelope,
            use_edge_scalar_modulation=(
                use_edge_scalar_modulation and node_or_edge == 'edge'
            ),
        )

        if mlp_type == "spectral":
            self.atom_wise = SpectralAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                gate_act_type=gate_act_type,
            )
        elif mlp_type == "grid":
            self.atom_wise = GridAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        else:
            raise ValueError(f"Unknown MLP type {mlp_type!r}")

        scale_kwargs = {
            "lmax": lmax,
            "mode": residual_update_scale_mode,
            "init": residual_update_scale_init,
            "log_range": residual_update_scale_log_range,
        }
        self.edge_update_scale = DegreeLayerScale(**scale_kwargs)
        self.atom_update_scale = DegreeLayerScale(**scale_kwargs)

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
        partition,
        system_node_embedding=None,
    ):

        if node_or_edge == 'node':
            x_res = x_message_node

            x_message_node = self.norm_1(x_message_node)
            if system_node_embedding is not None:
                if tuple(system_node_embedding.shape) != (
                    x_message_node.shape[0],
                    x_message_node.shape[2],
                ):
                    raise ValueError(
                        "system_node_embedding must have shape "
                        f"{(x_message_node.shape[0], x_message_node.shape[2])}, "
                        f"got {tuple(system_node_embedding.shape)}."
                    )
                x_message_node[:, 0, :] = (
                    x_message_node[:, 0, :] + system_node_embedding
                )

            x_message_node = self.edge_wise(
                x_message_node,
                x_message_edge,
                x_edge,
                edge_distance,
                edge_index,
                wigner,
                wigner_inv,
                node_or_edge,
                partition
            )


            x_message_node = self.edge_update_scale(x_message_node) + x_res
            x_res = x_message_node

            x_message_node = self.norm_2(x_message_node)
            x_message_node = self.atom_wise(x_message_node)


            return self.atom_update_scale(x_message_node) + x_res

        else:
            x_res = x_message_edge
            edgewise_node = x_message_node
            edge_norm1_position = getattr(
                self,
                "edge_norm1_position",
                "post_edgewise",
            )
            if edge_norm1_position == "pre_node":
                edgewise_node = self.norm_1(edgewise_node)

            torch.cuda.nvtx.range_push("Edgewise") # <--- START
            x_message_edge = self.edge_wise(
                edgewise_node,
                x_message_edge,
                x_edge,
                edge_distance,
                edge_index,
                wigner,
                wigner_inv,
                node_or_edge,
                partition
            )
            torch.cuda.nvtx.range_pop() # <--- END
            if edge_norm1_position == "post_edgewise":
                x_message_edge = self.norm_1(x_message_edge)

            if getattr(
                self,
                "edgewise_output_mode",
                "residual_scaled",
            ) == "residual_scaled":
                x_message_edge = self.edge_update_scale(x_message_edge) + x_res
            x_res = x_message_edge

            x_message_edge = self.norm_2(x_message_edge)

            torch.cuda.nvtx.range_push("Atomwise") # <--- START
            x_message_edge = self.atom_wise(x_message_edge)
            torch.cuda.nvtx.range_pop() # <--- END

            if getattr(
                self,
                "atomwise_output_mode",
                "residual_scaled",
            ) == "direct":
                return self.post_residual_norm(x_message_edge)
            x_message_edge = self.atom_update_scale(x_message_edge) + x_res
            return self.post_residual_norm(x_message_edge)

    def forward_qhflow3_pair(
        self,
        x_message_node,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        partition,
    ):
        """Compute one independent QHFlow3-style pair branch.

        Unlike the native recurrent eSEN edge path, this branch normalizes the
        final node state before constructing pair messages, does not add an
        incoming edge residual, and returns the atomwise pair update directly.
        Multiple branches are summed by the backbone before final edge
        normalization.
        """

        normalized_node = self.norm_1(x_message_node)
        pair_message = self.edge_wise(
            normalized_node,
            None,
            x_edge,
            edge_distance,
            edge_index,
            wigner,
            wigner_inv,
            "edge",
            partition,
        )
        pair_message = self.norm_2(pair_message)
        return self.atom_wise(pair_message)
