# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""
The codes in this file are adapted from fairchem (https://github.com/facebookresearch/fairchem).
See LICENSES/MIT-fairchem.md for license information.
"""

from __future__ import annotations

import matplotlib.pyplot as plt # remove
import os, sys
import torch
import math
import torch.nn as nn
import e3nn
from e3nn.o3 import Irreps, Irrep
from e3nn.o3 import Linear as e3nn_Linear
from e3nn.nn import Gate
from torch.nn import Linear
import numpy as np
from abc import ABCMeta, abstractmethod
from torch.utils.checkpoint import checkpoint
import time
from .nn.so3_layers import SO3_Linear

import torch.distributed as dist
from mpi4py import MPI

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
e3nn.set_optimization_defaults(jit_script_fx=False)

from .common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
    eulers_to_wigner,
    init_edge_rot_euler_angles
)
from .common.so3 import (
    CoefficientMapping,
    SO3_Grid,
)
from .esen_block import eSEN_Block
from .nn.embedding import EdgeDegreeEmbedding
from .nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .nn.radial import EnvelopedBesselBasis, GaussianSmearing
from .nn.so2_layers import SO2_Convolution
from .nn.so3_layers import SO3_Linear
from .nn.activation import GateActivation
from .nte_conditioning import NTEMatrixConditioning
from .qhflow3_clean import (
    GridAtomwise as QHFlow3GridAtomwise,
    MuonVisibleIrrepLinear,
    eSCNMD_Block as QHFlow3NodeBlock,
    eSCNMD_Block_xy2 as QHFlow3PairBlock,
)
from .qhf_layer.layer_norm import (
    get_normalization_layer as get_qhflow3_normalization_layer,
)
from .qhf_layer.radial import GaussianSmearing as QHFlow3GaussianSmearing

from .common.irreps_utils import get_reduced_to_all_indices, get_parity_multiplier, get_product_irreps, get_subspace_remix_permutation


class QHFlow3IrrepLinear(nn.Module):
    """Apply QHFlow3's e3nn projection to native eSEN embeddings.

    eSEN stores features as ``[sample, l-major coefficient, channel]``, while
    :class:`MuonVisibleIrrepLinear` follows e3nn's flattened
    degree/channel/m ordering.  This adapter performs the exact layout
    conversion in both directions and keeps the wrapped path weights visible
    to the shape-based Muon router as ``[degree, output, input]``.  With raw
    path weight ``M[l, out, in]``, e3nn applies the degreewise channel map
    ``M / sqrt(in_features)`` independently to every magnetic component.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lmax: int,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lmax = int(lmax)
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("Projection channel counts must be positive.")
        if self.lmax < 0:
            raise ValueError("lmax must be non-negative.")

        irreps_in = Irreps(
            [
                (
                    self.in_features,
                    (degree, 1 if degree % 2 == 0 else -1),
                )
                for degree in range(self.lmax + 1)
            ]
        )
        irreps_out = Irreps(
            [
                (
                    self.out_features,
                    (degree, 1 if degree % 2 == 0 else -1),
                )
                for degree in range(self.lmax + 1)
            ]
        )
        self.linear = MuonVisibleIrrepLinear(irreps_in, irreps_out)
        expected_shape = (
            self.lmax + 1,
            self.out_features,
            self.in_features,
        )
        if tuple(self.linear.weight.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected QHFlow3 projection path layout: "
                f"{tuple(self.linear.weight.shape)} != {expected_shape}."
            )

    @property
    def weight(self) -> nn.Parameter:
        """Return the sole registered path tensor without registering an alias."""
        return self.linear.weight

    def _native_to_e3nn(self, features: torch.Tensor) -> torch.Tensor:
        blocks = [
            features[:, degree**2 : (degree + 1) ** 2, :]
            .transpose(1, 2)
            .reshape(features.shape[0], -1)
            for degree in range(self.lmax + 1)
        ]
        return torch.cat(blocks, dim=1)

    def _e3nn_to_native(self, features: torch.Tensor) -> torch.Tensor:
        blocks = []
        offset = 0
        for degree in range(self.lmax + 1):
            multiplicity = 2 * degree + 1
            width = self.out_features * multiplicity
            block = features[:, offset : offset + width].reshape(
                features.shape[0],
                self.out_features,
                multiplicity,
            )
            blocks.append(block.transpose(1, 2))
            offset += width
        return torch.cat(blocks, dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected_shape = (
            (self.lmax + 1) ** 2,
            self.in_features,
        )
        if features.ndim != 3 or tuple(features.shape[1:]) != expected_shape:
            raise ValueError(
                "QHFlow3IrrepLinear expects native eSEN features shaped "
                f"[N, {expected_shape[0]}, {expected_shape[1]}], got "
                f"{tuple(features.shape)}."
            )
        projected = self.linear(self._native_to_e3nn(features))
        return self._e3nn_to_native(projected)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, lmax={self.lmax})"
        )


def _qhflow3_irrep_projection_with_legacy_rng(
    in_features: int,
    out_features: int,
    lmax: int,
) -> QHFlow3IrrepLinear:
    """Build the QHFlow3 operator without shifting downstream initialization.

    The ablation intentionally changes projection math and parameterization,
    not the initialization of the following edge projection or matrix head.
    Preserve the QHFlow3 normal weight sampled from the entry RNG state, then
    advance the global CPU RNG exactly as the replaced legacy ``SO3_Linear``
    constructor would have done.
    """
    with torch.random.fork_rng(devices=[]):
        projection = QHFlow3IrrepLinear(
            in_features,
            out_features,
            lmax,
        )
    # ``SO3_Linear`` consumes a normal draw for parameter construction and a
    # uniform draw for its final initialization. Discarding this temporary
    # module reproduces that exact legacy CPU RNG trajectory.
    SO3_Linear(
        in_features,
        out_features,
        lmax=lmax,
        bias=False,
    )
    return projection


@registry.register_model("esen_backbone")
class eSEN_Backbone(nn.Module):
    def __init__(
        self,
        irreps_out,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        max_neighbors: int = 300,
        cutoff = 10.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = False,
        regress_forces: bool = True,
        regress_stress: bool = False,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        gate_act_type: str = "tanh",
        mlp_type: str = "spectral",
        gaussian_width = 1.0,
        include_edges=True,
        open_shell=False,
        wigner_backend: str = "torch",
        distributed_graph_training=False,
        message_type='source-target',
        message_passing_schedule: str = "interleaved",
        initial_edge_state_mode: str = "edge_degree",
        num_edge_layers: int | None = None,
        output_sphere_channels: int | None = None,
        nte_output_projection_mode: str = "so3_linear",
        use_edge_envelope: bool = False,
        use_edge_scalar_modulation: bool = False,
        residual_update_scale_mode: str = "none",
        residual_update_scale_init: float = 1.0,
        residual_update_scale_log_range: float = 0.0,
        unscaled_node_layers: tuple[int, ...] = (),
        repeat_system_embedding_each_node_block: bool = False,
        node_stack_mode: str = "nte",
        edge_stack_mode: str = "recurrent",
        qhflow3_layer_gaussian_width: float = 2.0,
        qhflow3_layer_grid_ffn_chunk_size: int | None = 512,
        qhflow3_exact_pair_rng_aligned: bool = False,
        edge_atom_norm_type: str | None = None,
        edge_post_residual_norm_type: str | None = None,
        edge_atomwise_output_mode: str = "residual_scaled",
        edge_norm1_position: str = "post_edgewise",
        direct_edgewise_layers: tuple[int, ...] = (),
        input_conditioning: str = "none",
        conditioning_basis: str = "def2-svp",
        conditioning_delta_learning: bool = False,
        conditioning_delta_target: str = "fock_matrix",
    ):
        super().__init__()

        assert wigner_backend in ("torch", "triton"), \
            f"wigner_backend must be 'torch' or 'triton', got '{wigner_backend}'"
        self.wigner_backend = wigner_backend
        self._wigner_buf = None  # upper-bound pre-allocated, grown only when num_edges exceeds capacity

        if message_passing_schedule not in {"interleaved", "node_then_edge"}:
            raise ValueError(
                "message_passing_schedule must be 'interleaved' or "
                f"'node_then_edge', got {message_passing_schedule!r}."
            )
        self.message_passing_schedule = message_passing_schedule
        if initial_edge_state_mode not in {"edge_degree", "zero"}:
            raise ValueError(
                "initial_edge_state_mode must be 'edge_degree' or 'zero', "
                f"got {initial_edge_state_mode!r}."
            )
        self.initial_edge_state_mode = initial_edge_state_mode
        if node_stack_mode not in {"nte", "qhflow3_exact"}:
            raise ValueError(
                "node_stack_mode must be 'nte' or 'qhflow3_exact', "
                f"got {node_stack_mode!r}."
            )
        self.node_stack_mode = node_stack_mode
        if nte_output_projection_mode not in {
            "so3_linear",
            "qhflow3_irrep_linear",
        }:
            raise ValueError(
                "nte_output_projection_mode must be 'so3_linear' or "
                f"'qhflow3_irrep_linear', got {nte_output_projection_mode!r}."
            )
        self.nte_output_projection_mode = nte_output_projection_mode
        parallel_edge_stack_modes = {
            "nte_parallel",
            "qhflow3_parallel",
            "qhflow3_exact_parallel",
        }
        if edge_stack_mode not in {"recurrent", *parallel_edge_stack_modes}:
            raise ValueError(
                "edge_stack_mode must be 'recurrent', 'nte_parallel', "
                "'qhflow3_parallel', or 'qhflow3_exact_parallel', "
                f"got {edge_stack_mode!r}."
            )
        if edge_stack_mode in parallel_edge_stack_modes and message_type != "source-target":
            raise ValueError(
                "Parallel edge branches require message_type='source-target'."
            )
        if (
            edge_stack_mode in parallel_edge_stack_modes
            and message_passing_schedule != "node_then_edge"
        ):
            raise ValueError(
                "Parallel edge branches require "
                "message_passing_schedule='node_then_edge'."
            )
        if self.initial_edge_state_mode == "zero":
            if not include_edges:
                raise ValueError(
                    "initial_edge_state_mode='zero' requires include_edges=True."
                )
            if message_passing_schedule != "node_then_edge":
                raise ValueError(
                    "initial_edge_state_mode='zero' requires "
                    "message_passing_schedule='node_then_edge'."
                )
            if edge_stack_mode != "recurrent":
                raise ValueError(
                    "initial_edge_state_mode='zero' requires "
                    "edge_stack_mode='recurrent'."
                )
            if message_type != "source-target":
                raise ValueError(
                    "initial_edge_state_mode='zero' requires "
                    "message_type='source-target'."
                )
        self.edge_stack_mode = edge_stack_mode
        self.qhflow3_exact_pair_rng_aligned = bool(
            qhflow3_exact_pair_rng_aligned
        )
        if (
            self.qhflow3_exact_pair_rng_aligned
            and edge_stack_mode != "qhflow3_exact_parallel"
        ):
            raise ValueError(
                "qhflow3_exact_pair_rng_aligned requires "
                "edge_stack_mode='qhflow3_exact_parallel'."
            )
        self.uses_qhflow3_exact_layers = (
            node_stack_mode == "qhflow3_exact"
            or edge_stack_mode == "qhflow3_exact_parallel"
        )
        if self.uses_qhflow3_exact_layers and mlp_type != "grid":
            raise ValueError(
                "Exact QHFlow3 layer transplants require mlp_type='grid'."
            )
        if self.uses_qhflow3_exact_layers and distributed_graph_training:
            raise ValueError(
                "Exact QHFlow3 layer transplants do not support distributed "
                "graph training."
            )
        if (
            node_stack_mode == "qhflow3_exact"
            and message_passing_schedule != "node_then_edge"
        ):
            raise ValueError(
                "The exact QHFlow3 node stack requires "
                "message_passing_schedule='node_then_edge'."
            )
        self.qhflow3_layer_gaussian_width = float(
            qhflow3_layer_gaussian_width
        )
        if self.qhflow3_layer_gaussian_width <= 0.0:
            raise ValueError("qhflow3_layer_gaussian_width must be positive.")
        self.qhflow3_layer_grid_ffn_chunk_size = (
            None
            if qhflow3_layer_grid_ffn_chunk_size is None
            else int(qhflow3_layer_grid_ffn_chunk_size)
        )
        if (
            self.qhflow3_layer_grid_ffn_chunk_size is not None
            and self.qhflow3_layer_grid_ffn_chunk_size <= 0
        ):
            raise ValueError(
                "qhflow3_layer_grid_ffn_chunk_size must be positive."
            )
        valid_norm_types = {"layer_norm", "layer_norm_sh", "rms_norm_sh"}
        for option_name, option_value in (
            ("edge_atom_norm_type", edge_atom_norm_type),
            ("edge_post_residual_norm_type", edge_post_residual_norm_type),
        ):
            if option_value is not None and option_value not in valid_norm_types:
                raise ValueError(
                    f"{option_name} must be None or one of "
                    f"{sorted(valid_norm_types)}, got {option_value!r}."
                )
        self.edge_atom_norm_type = edge_atom_norm_type
        self.edge_post_residual_norm_type = edge_post_residual_norm_type
        if edge_atomwise_output_mode not in {"residual_scaled", "direct"}:
            raise ValueError(
                "edge_atomwise_output_mode must be "
                "'residual_scaled' or 'direct', "
                f"got {edge_atomwise_output_mode!r}."
            )
        self.edge_atomwise_output_mode = edge_atomwise_output_mode
        if edge_norm1_position not in {"post_edgewise", "pre_node"}:
            raise ValueError(
                "edge_norm1_position must be 'post_edgewise' or 'pre_node', "
                f"got {edge_norm1_position!r}."
            )
        self.edge_norm1_position = edge_norm1_position
        self.repeat_system_embedding_each_node_block = bool(
            repeat_system_embedding_each_node_block
        )
        if (
            self.repeat_system_embedding_each_node_block
            and input_conditioning != "qhflow3_exact"
        ):
            raise ValueError(
                "repeat_system_embedding_each_node_block requires "
                "input_conditioning='qhflow3_exact'."
            )
        self.unscaled_node_layers = tuple(int(index) for index in unscaled_node_layers)
        if len(set(self.unscaled_node_layers)) != len(self.unscaled_node_layers):
            raise ValueError("unscaled_node_layers must not contain duplicates.")
        invalid_unscaled_layers = [
            index
            for index in self.unscaled_node_layers
            if index < 1 or index > num_layers
        ]
        if invalid_unscaled_layers:
            raise ValueError(
                "unscaled_node_layers uses 1-based node-block indices in "
                f"[1, {num_layers}], got {invalid_unscaled_layers}."
            )
        self.use_edge_envelope = bool(use_edge_envelope)
        self.use_edge_scalar_modulation = bool(use_edge_scalar_modulation)
        self.residual_update_scale_mode = residual_update_scale_mode
        self.residual_update_scale_init = float(residual_update_scale_init)
        self.residual_update_scale_log_range = float(
            residual_update_scale_log_range
        )
        if input_conditioning not in {
            "none",
            "overlap",
            "qhflow3_exact",
        }:
            raise ValueError(
                "input_conditioning must be 'none', 'overlap', or "
                f"'qhflow3_exact', got {input_conditioning!r}."
            )
        self.input_conditioning = input_conditioning
        self.input_conditioner = (
            None
            if input_conditioning == "none"
            else NTEMatrixConditioning(
                mode=input_conditioning,
                basis=conditioning_basis,
                hidden_size=sphere_channels,
                delta_learning=conditioning_delta_learning,
                delta_target=conditioning_delta_target,
            )
        )

        if not include_edges:
            print("Note: Initializing eSEN backbone without edge_embeddings!")

        # if dist.is_initialized():
        #     self.rank = dist.get_rank()
        #     self.world_size = dist.get_world_size()
        #     # self.comm = MPI.COMM_WORLD
        # else:
        #     self.rank = 0
        #     self.world_size = 1
        #     # self.comm = None

        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.output_sphere_channels = int(
            sphere_channels
            if output_sphere_channels is None
            else output_sphere_channels
        )
        if self.output_sphere_channels <= 0:
            raise ValueError("output_sphere_channels must be positive.")
        self.gaussian_width = gaussian_width

        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress
        self.mlp_type = mlp_type
        self.include_edges = include_edges      # whether to use embeddings for the edges as well

        if open_shell:                                      # output two sets of labels for alpha/beta fock
            self.num_spins = 2
        else:
            self.num_spins = 1

        # rotation utils
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        # lmax_lmax for node, lmax_mmax for edge
        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        # atom embedding
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )
        #  For charge: possible values are -10 to +10 (21 values)
        self.abs_max_charge = 10
        self.charge_embedding = nn.Embedding(
            2*self.abs_max_charge + 1, self.sphere_channels
        )
        self.max_spin_multiplicity = 11
        self.spin_embedding = nn.Embedding(
            self.max_spin_multiplicity, self.sphere_channels
        )

        # # These three will be combined together with a linear layer:
        self.scalar_node_embedding = nn.Linear(3 * self.sphere_channels, self.sphere_channels)

        # edge distance embedding
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                self.gaussian_width,
            )
        elif self.distance_function == "bessel":
            self.distance_expansion = EnvelopedBesselBasis(
                num_radial=self.num_distance_basis,
                cutoff=cutoff,
            )
            self.distance_expansion.offset = [self.cutoff]
            self.distance_expansion.num_output = self.num_distance_basis
        else:
            raise ValueError("Unknown distance function")
        self.qhflow3_layer_distance_expansion = (
            QHFlow3GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                self.qhflow3_layer_gaussian_width,
            )
            if self.uses_qhflow3_exact_layers
            else None
        )

        # equivariant initial embedding
        # self.element_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels) # for antisym
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)

        # nn.init.uniform_(self.element_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
                sphere_channels=self.sphere_channels,
                lmax=self.lmax,
                mmax=self.mmax,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                rescale_factor=5.0,
                cutoff=self.cutoff,
                mappingReduced=self.mappingReduced,
                out_mask=self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
                    self.lmax, self.mmax
                )
            )

        self.num_layers = num_layers
        self.num_edge_layers = int(
            num_layers if num_edge_layers is None else num_edge_layers
        )
        if self.num_edge_layers <= 0:
            raise ValueError("num_edge_layers must be positive.")
        if (
            self.message_passing_schedule == "interleaved"
            and self.num_edge_layers != self.num_layers
        ):
            raise ValueError(
                "The interleaved schedule requires num_edge_layers == num_layers."
            )
        self.direct_edgewise_layers = tuple(
            int(index) for index in direct_edgewise_layers
        )
        if len(set(self.direct_edgewise_layers)) != len(
            self.direct_edgewise_layers
        ):
            raise ValueError("direct_edgewise_layers must not contain duplicates.")
        if any(
            index < 1 or index > self.num_edge_layers
            for index in self.direct_edgewise_layers
        ):
            raise ValueError(
                "direct_edgewise_layers must contain 1-based indices within "
                "num_edge_layers."
            )
        if (
            self.initial_edge_state_mode == "zero"
            and 1 in self.direct_edgewise_layers
        ):
            raise ValueError(
                "initial_edge_state_mode='zero' is redundant with "
                "direct_edgewise_layers containing EdgeBlock 1."
            )
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type
        self.gate_act_type = gate_act_type

        # Initialize the blocks for each layer
        self.node_blocks = nn.ModuleList()

        if self.include_edges:
            self.edge_blocks = nn.ModuleList()


        block_kwargs = {
            "gate_act_type": self.gate_act_type,
            "use_edge_envelope": self.use_edge_envelope,
            "use_edge_scalar_modulation": self.use_edge_scalar_modulation,
            "residual_update_scale_mode": self.residual_update_scale_mode,
            "residual_update_scale_init": self.residual_update_scale_init,
            "residual_update_scale_log_range": self.residual_update_scale_log_range,
        }

        for layer_index in range(1, self.num_layers + 1):
            node_block_kwargs = dict(block_kwargs)
            if layer_index in self.unscaled_node_layers:
                node_block_kwargs["residual_update_scale_mode"] = "none"
            if self.node_stack_mode == "qhflow3_exact":
                node_block = QHFlow3NodeBlock(
                    self.sphere_channels,
                    self.hidden_channels,
                    self.lmax,
                    self.mmax,
                    self.mappingReduced,
                    self.SO3_grid,
                    self.edge_channels_list,
                    self.cutoff,
                    self.norm_type,
                    self.act_type,
                    self.mlp_type,
                    activation_checkpoint_chunk_size=None,
                )
            else:
                node_block = eSEN_Block(
                    self.sphere_channels,
                    self.hidden_channels,
                    self.lmax,
                    self.mmax,
                    self.mappingReduced,
                    self.SO3_grid,
                    self.edge_channels_list,
                    self.cutoff,
                    self.norm_type,
                    self.act_type,
                    self.mlp_type,
                    message_type,
                    include_edges=self.include_edges,
                    node_or_edge='node',
                    **node_block_kwargs,
                )
            self.node_blocks.append(node_block)

        if self.include_edges:
            for edge_layer_index in range(1, self.num_edge_layers + 1):
                if self.edge_stack_mode == "qhflow3_exact_parallel":
                    edge_block = QHFlow3PairBlock(
                        self.sphere_channels,
                        self.hidden_channels,
                        self.lmax,
                        self.mmax,
                        self.mappingReduced,
                        self.SO3_grid,
                        self.edge_channels_list,
                        self.cutoff,
                        self.norm_type,
                        self.act_type,
                        self.mlp_type,
                        activation_checkpoint_chunk_size=None,
                        rng_align_dead_fc2=(
                            self.qhflow3_exact_pair_rng_aligned
                        ),
                    )
                else:
                    edge_block = eSEN_Block(
                        self.sphere_channels,
                        self.hidden_channels,
                        self.lmax,
                        self.mmax,
                        self.mappingReduced,
                        self.SO3_grid,
                        self.edge_channels_list,
                        self.cutoff,
                        self.norm_type,
                        self.act_type,
                        self.mlp_type,
                        message_type,
                        include_edges=self.include_edges,
                        node_or_edge='edge',
                        atom_norm_type=self.edge_atom_norm_type,
                        post_residual_norm_type=self.edge_post_residual_norm_type,
                        edgewise_output_mode=(
                            "direct"
                            if edge_layer_index in self.direct_edgewise_layers
                            else "residual_scaled"
                        ),
                        atomwise_output_mode=self.edge_atomwise_output_mode,
                        edge_norm1_position=self.edge_norm1_position,
                        **block_kwargs,
                    )
                self.edge_blocks.append(edge_block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels
        )
        self.qhflow3_pair_norm = (
            get_qhflow3_normalization_layer(
                self.norm_type,
                lmax=self.lmax,
                num_channels=self.sphere_channels,
            )
            if self.edge_stack_mode == "qhflow3_exact_parallel"
            else None
        )
        if self.qhflow3_layer_grid_ffn_chunk_size is not None:
            exact_layer_blocks = list(self.node_blocks)
            if self.include_edges:
                exact_layer_blocks.extend(self.edge_blocks)
            for block in exact_layer_blocks:
                for module in block.modules():
                    if isinstance(module, QHFlow3GridAtomwise):
                        module.grid_ffn_chunk_size = (
                            self.qhflow3_layer_grid_ffn_chunk_size
                        )
        if self.output_sphere_channels == self.sphere_channels:
            self.node_output_projection = nn.Identity()
            self.edge_output_projection = nn.Identity()
        elif self.nte_output_projection_mode == "qhflow3_irrep_linear":
            self.node_output_projection = _qhflow3_irrep_projection_with_legacy_rng(
                self.sphere_channels,
                self.output_sphere_channels,
                self.lmax,
            )
            self.edge_output_projection = _qhflow3_irrep_projection_with_legacy_rng(
                self.sphere_channels,
                self.output_sphere_channels,
                self.lmax,
            )
        else:
            self.node_output_projection = SO3_Linear(
                self.sphere_channels,
                self.output_sphere_channels,
                lmax=self.lmax,
                bias=False,
            )
            self.edge_output_projection = SO3_Linear(
                self.sphere_channels,
                self.output_sphere_channels,
                lmax=self.lmax,
                bias=False,
            )

    def _get_rotmat_and_wigner(self, edge_distance_vecs):
        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
            for l in range(self.lmax + 1)
        ]

        if self.wigner_backend == "triton":
            from .triton_kernels import edge_vec_to_wigner_fused
            num_edges = edge_distance_vecs.shape[0]
            out_dim = (self.lmax + 1) ** 2
            if self._wigner_buf is None or num_edges > self._wigner_buf.shape[0]:
                self._wigner_buf = torch.zeros(
                    num_edges, out_dim, out_dim,
                    device=edge_distance_vecs.device,
                    dtype=torch.float32,
                )
            wigner = edge_vec_to_wigner_fused(
                edge_distance_vecs, Jd_buffers, lmax=self.lmax,
                out=self._wigner_buf[:num_edges],
            )
        else:
            euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
            wigner = eulers_to_wigner(
                euler_angles,
                0,
                self.lmax,
                Jd_buffers,
            )
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        return wigner, wigner_inv

    def _to_qhflow3_wigner(self, wigner, wigner_inv):
        """Convert l-major Wigner matrices to QHFlow3's m-major contract."""
        to_m = self.mappingReduced.to_m.to(wigner.dtype)
        qhflow3_wigner = torch.einsum("mk,nkj->nmj", to_m, wigner)
        qhflow3_wigner_inv = torch.einsum(
            "njk,mk->njm", wigner_inv, to_m
        )
        return qhflow3_wigner, qhflow3_wigner_inv

    def _run_message_passing(
        self,
        x_message_node,
        x_message_edge,
        x_edge,
        graph_dict,
        wigner,
        wigner_inv,
        system_node_embedding=None,
        qhflow3_x_edge=None,
        qhflow3_wigner=None,
        qhflow3_wigner_inv=None,
    ):
        def update_node(block, node_state, edge_state):
            if self.node_stack_mode == "qhflow3_exact":
                if (
                    qhflow3_x_edge is None
                    or qhflow3_wigner is None
                    or qhflow3_wigner_inv is None
                ):
                    raise ValueError(
                        "Exact QHFlow3 node blocks require QHFlow3 edge and "
                        "Wigner inputs."
                    )
                return block(
                    node_state,
                    qhflow3_x_edge,
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    qhflow3_wigner,
                    qhflow3_wigner_inv,
                    sys_node_embedding=system_node_embedding,
                    node_offset=0,
                )
            return block(
                node_state,
                edge_state,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner,
                wigner_inv,
                node_or_edge='node',
                partition=graph_dict["partition"],
                system_node_embedding=system_node_embedding,
            )

        def update_edge(block, node_state, edge_state):
            return block(
                node_state,
                edge_state,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner,
                wigner_inv,
                node_or_edge='edge',
                partition=graph_dict["partition"],
            )

        if not self.include_edges:
            for node_block in self.node_blocks:
                x_message_node = update_node(node_block, x_message_node, None)
        elif self.message_passing_schedule == "interleaved":
            for node_block, edge_block in zip(self.node_blocks, self.edge_blocks):
                x_message_node = update_node(
                    node_block, x_message_node, x_message_edge
                )
                x_message_edge = update_edge(
                    edge_block, x_message_node, x_message_edge
                )
        elif self.edge_stack_mode == "recurrent":
            for node_block in self.node_blocks:
                x_message_node = update_node(
                    node_block, x_message_node, x_message_edge
                )
            if self.initial_edge_state_mode == "zero":
                # Keep the edge-degree state available to every NodeBlock and
                # ablate only EdgeBlock 1's incoming residual boundary.
                x_message_edge = torch.zeros_like(x_message_edge)
            if self.include_edges:
                for edge_block in self.edge_blocks:
                    x_message_edge = update_edge(
                        edge_block, x_message_node, x_message_edge
                    )
        elif self.edge_stack_mode == "nte_parallel":
            for node_block in self.node_blocks:
                x_message_node = update_node(
                    node_block, x_message_node, x_message_edge
                )
            initial_edge_state = x_message_edge
            edge_branches = [
                update_edge(edge_block, x_message_node, initial_edge_state)
                for edge_block in self.edge_blocks
            ]
            x_message_edge = torch.stack(edge_branches, dim=0).sum(dim=0)
        elif self.edge_stack_mode == "qhflow3_exact_parallel":
            for node_block in self.node_blocks:
                x_message_node = update_node(
                    node_block, x_message_node, x_message_edge
                )
            if (
                qhflow3_x_edge is None
                or qhflow3_wigner is None
                or qhflow3_wigner_inv is None
            ):
                raise ValueError(
                    "Exact QHFlow3 pair blocks require QHFlow3 edge and "
                    "Wigner inputs."
                )
            pair_branches = [
                edge_block(
                    x_message_node,
                    qhflow3_x_edge,
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    qhflow3_wigner,
                    qhflow3_wigner_inv,
                    node_offset=0,
                )
                for edge_block in self.edge_blocks
            ]
            x_message_edge = torch.stack(pair_branches, dim=0).sum(dim=0)
        else:  # NTE primitives arranged in QHFlow3-style parallel branches.
            for node_block in self.node_blocks:
                x_message_node = update_node(
                    node_block, x_message_node, x_message_edge
                )
            pair_branches = [
                edge_block.forward_qhflow3_pair(
                    x_message_node,
                    x_edge,
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    wigner,
                    wigner_inv,
                    graph_dict["partition"],
                )
                for edge_block in self.edge_blocks
            ]
            x_message_edge = torch.stack(pair_branches, dim=0).sum(dim=0)
        return x_message_node, x_message_edge


    @conditional_grad(torch.enable_grad())
    def forward(self, batch, batch_index=None, output_dir=None):


        distributed_graph_training = batch.distributed_graph_training if "distributed_graph_training" in batch else False

        data_dict = {
            "pos": batch.pos,
            "edge_index": batch.edge_index.squeeze(0).reshape(2, -1),       # composed of local fraction of global node indices
            "edge_dist": batch.edge_attr,
            "nedges": len(batch.edge_index[0]),
            "natoms": len(batch.pos),
            "atomic_numbers": batch.atomic_numbers,                         # always global
            "charges": batch.charge,
            "spin_multiplicity": batch.spin_multiplicity,
            "num_atoms_in_molecule": batch.num_atoms_in_molecule if not distributed_graph_training else None
        }

        # The input edges are in xyz coordinates, we need to rotate them to the yzx coordinates expected by e3nn to be consistent with the data
        edge_distance_vec = data_dict["edge_dist"][:, [2, 3, 1]]
        edge_distance = data_dict["edge_dist"][:, 0]

        graph_dict = {
            "edge_index": data_dict["edge_index"],
            "edge_distance": edge_distance,
            "edge_distance_vec": edge_distance_vec,
            "partition": batch.fock_target_object[0].domain if distributed_graph_training else None
        }


        wigner, wigner_inv = self._get_rotmat_and_wigner(
            graph_dict["edge_distance_vec"]
        )
        qhflow3_wigner = None
        qhflow3_wigner_inv = None
        if self.uses_qhflow3_exact_layers:
            qhflow3_wigner, qhflow3_wigner_inv = (
                self._to_qhflow3_wigner(wigner, wigner_inv)
            )

        # --> Rotation test:
        # rotated_edges_to_z_axis = torch.bmm(wigner[:, 1:4, 1:4], graph_dict["edge_distance_vec"].unsqueeze(-1)).squeeze(-1)
        # print("Rotated edges to z-axis: ", rotated_edges_to_z_axis) # only middle components should be nonzero (equal to distance)

        ###############################################################
        # Initialize node and edge embeddings
        ###############################################################


        # x_message: [data_dict["pos"].shape[0] = #nodes, self.sph_feature_size = (l_max+1)**2, self.sphere_channels = E]
        x_message_node = torch.zeros(
            data_dict["pos"].shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=data_dict["pos"].device,
            dtype=data_dict["pos"].dtype,
        )
        # set l = 0 components to the element embeddings + charge + spin:

        if distributed_graph_training:

            local_node_indices = graph_dict['partition'].local_node_indices

            # Double check these for distributed case!!!
            atom_charges = data_dict["charges"] + self.abs_max_charge
            element_emb = self.sphere_embedding(data_dict["atomic_numbers"][local_node_indices])
            charge_emb = self.charge_embedding(atom_charges)
            spin_emb = self.spin_embedding(data_dict["spin_multiplicity"])

        else:

            # Seperate batch nodes into their molecules
            molecule_indices = torch.cat([
                torch.full((data_dict['natoms'],), i, dtype=torch.long, device=data_dict["pos"].device)
                for i, data_dict['natoms'] in enumerate(data_dict["num_atoms_in_molecule"])
            ])
            atom_charges = data_dict["charges"][molecule_indices] + self.abs_max_charge         # shape: [total_num_atoms]
            atom_spins = data_dict["spin_multiplicity"][molecule_indices]                       # shape: [total_num_atoms]

            element_emb = self.sphere_embedding(data_dict["atomic_numbers"])
            charge_emb = self.charge_embedding(atom_charges)
            spin_emb = self.spin_embedding(atom_spins)

        # x_message_node[:, :, 0, :] = element_emb + charge_emb + spin_emb # dims: [spin, nodes, l, E]

        # Concatenate along the last dimension
        combined_emb = torch.cat([element_emb, charge_emb, spin_emb], dim=-1) # [num_atoms, 3 * sphere_channels]
        final_emb = self.scalar_node_embedding(combined_emb)                  # [num_atoms, sphere_channels]
        if self.input_conditioner is not None:
            if distributed_graph_training:
                raise ValueError(
                    "NTE matrix input conditioning does not support "
                    "distributed graph training."
                )
            final_emb = self.input_conditioner(
                batch,
                atom_embedding=element_emb,
                base_scalar=final_emb,
                molecule_indices=molecule_indices,
            )
        x_message_node[:, 0, :] = final_emb

        if self.include_edges:
            # x_message_edge: [#edges considered, self.sph_feature_size = (l_max+1)**2, self.sphere_channels = E]
            x_message_edge = torch.zeros(
                data_dict["nedges"],
                self.sph_feature_size,
                self.num_distance_basis, #self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            # set l = 0 components to the distance expansion
            x_message_edge[:, 0, :] = self.distance_expansion(graph_dict["edge_distance"]) # maybe remove
        else:
            x_message_edge = None

        # edge embedding: [num_edges, num gaussian basis functions]
        # source_embedding, target_embedding: [num_edges, self.sphere_channels]
        # x_edge: [num_edges, num gaussian basis functions + 2*self.sphere_channels]

        edge_distance_embedding = self.distance_expansion(graph_dict["edge_distance"])

        source_embedding = self.source_embedding(
            data_dict["atomic_numbers"][graph_dict["edge_index"][0]]
        )

        target_embedding = self.target_embedding(
            data_dict["atomic_numbers"][graph_dict["edge_index"][1]]
        )

        x_edge = torch.cat((source_embedding, edge_distance_embedding, target_embedding), dim=1) 
        qhflow3_x_edge = None
        if self.uses_qhflow3_exact_layers:
            qhflow3_distance_embedding = (
                self.qhflow3_layer_distance_expansion(
                    graph_dict["edge_distance"]
                )
            )
            qhflow3_x_edge = torch.cat(
                (qhflow3_distance_embedding, source_embedding, target_embedding),
                dim=1,
            )

        # do edge degree embeddings for both nodes and edges:
        x_message_node = self.edge_degree_embedding( 
            x_message_node,
            x_edge,
            graph_dict["edge_distance"],
            graph_dict["edge_index"],
            wigner_inv,
            node_or_edge='node',
            partition=graph_dict["partition"]
        )

        if self.include_edges:
            x_message_edge = self.edge_degree_embedding(
                x_message_node,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner_inv,
                node_or_edge='edge',
                partition=None
            )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        system_node_embedding = None
        if self.repeat_system_embedding_each_node_block:
            system_node_embedding = self.input_conditioner.system_embedding(
                x_message_node.shape[0],
                device=x_message_node.device,
                dtype=x_message_node.dtype,
            )
        x_message_node, x_message_edge = self._run_message_passing(
            x_message_node,
            x_message_edge,
            x_edge,
            graph_dict,
            wigner,
            wigner_inv,
            system_node_embedding=system_node_embedding,
            qhflow3_x_edge=qhflow3_x_edge,
            qhflow3_wigner=qhflow3_wigner,
            qhflow3_wigner_inv=qhflow3_wigner_inv,
        )

        # Final layer norm
        x_message_node = self.norm(x_message_node)
        x_message_node = self.node_output_projection(x_message_node)

        if self.include_edges:
            if self.qhflow3_pair_norm is None:
                x_message_edge = self.norm(x_message_edge)
            else:
                # QHFlow3 has a pair-stack norm with separate affine weights.
                x_message_edge = self.qhflow3_pair_norm(x_message_edge)
            x_message_edge = self.edge_output_projection(x_message_edge)

        # Return the output
        if self.include_edges: # all we need for the fock output head
            out = {
                    "node_embeddings": x_message_node,
                    "edge_embeddings": x_message_edge,
                }
        else:
            out = {
                    "node_embeddings": x_message_node, 
                    "x_edge": x_edge,
                    "wigner": wigner,
                    "wigner_inv": wigner_inv
                }
        out.update(graph_dict)
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)



@registry.register_model("fock_irreps_head")
class Fock_Irreps_Head(nn.Module):
    """
    Takes an input irrep like 64x0e+64x1e+64x2e ... and nonlinearly maps it to the output irreps of arbitrary size and l-multiplicity
    """
    def __init__(self, irreps_in,
                irreps_out,
                lmax,
                sphere_channels,
                reduce_edge=False,
                ls_list=None,
                open_shell=False,
                reduce_node=False,
                reduce_node_intra=False,
                orbital_basis=None):

        super().__init__()

        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.reduce_node = reduce_node                      # take advantage of 'inter'-orbital interaction symmetry within node blocks
        self.reduce_node_intra = reduce_node_intra          # take advantage of 'intra'-orbital interaction symmetry within node blocks
        self.reduce_edge = reduce_edge                      # take advantage of edge interaction symmetry (ei, ej) vs (ej, ei) 
        self.irreps_out = irreps_out
        self.ls_list = ls_list                              # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        
        if open_shell:                                      # output two sets of labels for alpha/beta fock
            self.num_spins = 2
        else:
            self.num_spins = 1

        print("Ls_list for output head: ", self.ls_list)

        # -------- Handle irrep reductions for output head --------

        print("IRREP SYMMETRY REDUCTIONS: reduce_node = ", self.reduce_node, 
                                        " reduce_node_intra = ", self.reduce_node_intra, 
                                        " reduce_edge = ", self.reduce_edge, flush=True)

        # We make a new list of irreps (irreps_nodereduced) which contains only the unique irreps in the node blocks
        if self.reduce_node:
            self.reduced_to_all_indices = get_reduced_to_all_indices(self.ls_list, reduce_node_intra=self.reduce_node_intra)
            parity_multiplier = torch.asarray(get_parity_multiplier(self.ls_list, reduce_node_intra=self.reduce_node_intra), dtype=torch.float32)
            self.register_buffer("parity_multiplier", parity_multiplier) # so it can be used in the forward pass and is on the correct device/dtype

            self.make_irreps_nodereduced()   
            print("Reduced set of node irreps: ", self.irreps_nodereduced, flush=True)
        
        # We seperate the edge irreps into the alpha and beta sets
        if self.reduce_edge:
            self.make_irreps_edgereduced()     
            not_antisym = False  # reduce edges requires that operations preserve odd-ness, using this flag for biases
            print("Reduced set of edge irreps for alpha (symmetric) space: ", self.irreps_edgereduced_alpha, flush=True)
            print("Reduced set of edge irreps for beta (antisymmetric) space: ", self.irreps_edgereduced_beta, flush=True)
        else:
            not_antisym = True
        
        # store the output permutation which will be used to permute the output irreps to match the order of the labels in the data
        self.output_permutation = {} 
        if reduce_node and reduce_edge:
            self.node_lin_out_layers = nn.ModuleList()
            self.edge_lin_out_layers = nn.ModuleDict({
                'alpha': nn.ModuleList(),
                'beta': nn.ModuleList()
            })
            self.output_permutation['node'] = self.get_output_permutation(self.irreps_nodereduced)
            self.output_permutation['edge_alpha'] = self.get_output_permutation(self.irreps_edgereduced_alpha)
            self.output_permutation['edge_beta'] = self.get_output_permutation(self.irreps_edgereduced_beta)
            self.output_permutation['remix_subspaces'] = get_subspace_remix_permutation(self.irreps_edgereduced_alpha, self.irreps_edgereduced_beta, self.ls_list)

        elif reduce_node:
            self.node_lin_out_layers = nn.ModuleList()
            self.edge_lin_out_layers = nn.ModuleList()
            self.output_permutation['node'] = self.get_output_permutation(self.irreps_nodereduced)
            self.output_permutation['edge'] = self.get_output_permutation(irreps_out)

        elif reduce_edge and not reduce_node:
            raise NotImplementedError("The case of not reducing node irreps but reducing edge irreps is currently not implemented")
        
        else:
            self.lin_out_layers = nn.ModuleList()
            self.output_permutation['common'] = self.get_output_permutation(irreps_out)
        
        # -------- Prepare gating layers --------

        # 1. Apply a linear layer to convert the number of input scalars to the number of required gating scalars
        # the number of input scalars is equal to sphere_channels
        # the output 'irreps_gates' are the gating scalars

        irreps_scalars, irreps_gated = self.split_irreps(irreps_in)
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        # --> gate with learnable parameters by outputting more random scalars:
        input_scalars_irreps = Irreps(f"{self.sphere_channels}x0e")
        combined_output_scalars = Irreps(f"{irreps_scalars.num_irreps + irreps_gated.num_irreps}x0e")
        self.lin_scalars_learnable = nn.ModuleList()
        self.gate = nn.ModuleList()

        for spin in range(self.num_spins):

            # if using edge reduction, need to maintain antisymmetry of negative edges
            self.lin_scalars_learnable.append(e3nn_Linear(irreps_in=input_scalars_irreps, irreps_out=combined_output_scalars, biases=not_antisym))
            # this returns the irreps_gates

            # 2. Apply the gating to the other ls (need to pass in a stack of [l=0, l~=0])
            act_gates = [torch.sigmoid] * len(irreps_gates) if not_antisym else [torch.tanh] * len(irreps_gates)
            self.gate.append(Gate(irreps_scalars=Irreps(),
                                    act_scalars=[],
                                    irreps_gates=irreps_gates,
                                    act_gates=act_gates,
                                    irreps_gated=irreps_gated
                                ))

            # now we have the [l=0s, gated l>0s] in a stack, and we just need to map them to the output irrep order:
            irreps_in_simplified = (irreps_scalars + self.gate[spin].irreps_out).simplify()
            assert irreps_in_simplified == (irreps_scalars+self.gate[spin].irreps_out), "The irreps_in for the output linear layer should not change when simplified!"
            
            if reduce_node and reduce_edge:
                node_lin_layers = nn.ModuleList()
                edge_lin_layers_alpha = nn.ModuleList()
                edge_lin_layers_beta = nn.ModuleList()

                for l in range(0, self.lmax+1):
                    mul_in = self.sphere_channels

                    # for nodes:
                    mul_out = self.irreps_nodereduced.count('{}e'.format(l))
                    irreps_in_l = f"{mul_in}x{l}e"
                    irreps_out_l = f"{mul_out}x{l}e"
                    print("Creating node linear output map for l = ", l, " with irreps_in = ", irreps_in_l, " and irreps_out = ", irreps_out_l, flush=True)
                    node_lin_layers.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l, biases=True))

                    # for edges:
                    mul_out_alpha = self.irreps_edgereduced_alpha.count('{}e'.format(l))
                    mul_out_beta = self.irreps_edgereduced_beta.count('{}e'.format(l))
                    irreps_in_l = f"{mul_in}x{l}e"
                    irreps_out_l_alpha = f"{mul_out_alpha}x{l}e"
                    irreps_out_l_beta = f"{mul_out_beta}x{l}e"

                    print("Creating edge linear output map for l = ", l, " with irreps_in = ", irreps_in_l, " and irreps_out_alpha = ", irreps_out_l_alpha, " and irreps_out_beta = ", irreps_out_l_beta, flush=True)
                    edge_lin_layers_alpha.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l_alpha, biases=not_antisym))
                    edge_lin_layers_beta.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l_beta, biases=not_antisym)) 

                self.node_lin_out_layers.append(node_lin_layers)
                self.edge_lin_out_layers['alpha'].append(edge_lin_layers_alpha)
                self.edge_lin_out_layers['beta'].append(edge_lin_layers_beta)
            
            elif reduce_node:
                node_lin_layers = nn.ModuleList()
                edge_lin_layers = nn.ModuleList()
                for l in range(0, self.lmax+1):
                    mul_in = self.sphere_channels

                    # for nodes:
                    mul_out = self.irreps_nodereduced.count('{}e'.format(l))
                    irreps_in_l = f"{mul_in}x{l}e"
                    irreps_out_l = f"{mul_out}x{l}e"
                    print("Creating node linear output map for l = ", l, " with irreps_in = ", irreps_in_l, " and irreps_out = ", irreps_out_l, flush=True)
                    node_lin_layers.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l, biases=True))

                    # for edges:
                    mul_out = self.irreps_out.count('{}e'.format(l))
                    irreps_in_l = f"{mul_in}x{l}e"
                    irreps_out_l = f"{mul_out}x{l}e"
                    print("Creating edge linear output map for l = ", l, " with irreps_in = ", irreps_in_l, " and irreps_out = ", irreps_out_l, flush=True)
                    edge_lin_layers.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l, biases=True))

                self.node_lin_out_layers.append(node_lin_layers)
                self.edge_lin_out_layers.append(edge_lin_layers)
            else:

                # create single linear layers up to lmax:
                lin_layers = nn.ModuleList()
                for l in range(0, self.lmax+1):
                    mul_in = self.sphere_channels
                    mul_out = irreps_out.count('{}e'.format(l))
                    irreps_in_l = f"{mul_in}x{l}e"
                    irreps_out_l = f"{mul_out}x{l}e"
                    print("Creating linear output map for l = ", l, " with irreps_in = ", irreps_in_l, " and irreps_out = ", irreps_out_l, flush=True)
                    lin_layers.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l, biases=True))

                self.lin_out_layers.append(lin_layers)
    
    def make_irreps_nodereduced(self):
        """
        Create the list of irreps for the node blocks after taking into account the reduction from only considering unique orbital interactions.
        The lower triangle irreps are the same as the upper triangle, so we only need to keep one of them. 
        For the diagonal blocks, we only keep the even irreps, since the odd irreps should be zero due to symmetry
        """

        irreps_nodereduced = []
        for i, l1 in enumerate(self.ls_list):
            for j, l2 in enumerate(self.ls_list):

                # if this is an orbital self-interaction within the node block, we add the even irreps
                if i == j and l1 == l2:
                    if self.reduce_node_intra:
                        product_irreps = str(get_product_irreps(l1, l2, 'even'))
                    else:
                        product_irreps = str(get_product_irreps(l1, l2))

                    irreps_nodereduced.append(product_irreps)

                # this is an upper-triangle off-diag interaction within the node block, we add all the required irreps
                if i < j:
                    product_irreps = str(get_product_irreps(l1, l2))
                    irreps_nodereduced.append(product_irreps)
                    irrep_len = sum([2*l + 1 for l in Irreps(product_irreps).ls])

        # Now we can project to this reduced set of irreps, and expand it out later
        self.irreps_nodereduced = Irreps('+'.join(irreps_nodereduced))
    
    def make_irreps_edgereduced(self):
        """
        Create the list of irreps for the edge blocks
        """
        
        irreps_edgereduced_alpha = []
        irreps_edgereduced_beta = []

        rolling_irrep_ptr = 0
        parity_flip_indices = []

        for i, l1 in enumerate(self.ls_list):
            for j, l2 in enumerate(self.ls_list):
                product_irreps = str(get_product_irreps(l1, l2))

                # diagonal block - even irreps go to the alpha space, odd irreps go to the beta space
                if i == j and l1 == l2:
                    even_irreps = str(get_product_irreps(l1, l2, 'even'))
                    odd_irreps = str(get_product_irreps(l1, l2, 'odd'))

                    if even_irreps != '':
                        irreps_edgereduced_alpha.append(even_irreps)
                    if odd_irreps != '':
                        irreps_edgereduced_beta.append(odd_irreps)
                    
                    rolling_irrep_ptr += sum([2*l + 1 for l in Irreps(product_irreps).ls])

                # upper triangle irreps go to the alpha space
                if i < j:
                    irreps_edgereduced_alpha.append(product_irreps)

                    input_parity = (l1 + l2) % 2
                    for mul_ir in Irreps(product_irreps):
                        ir = mul_ir.ir

                        # if input parity is even, then flip the odd components of the product irreps 
                        # and if input parity is odd, then flip the even components of the product irreps
                        if input_parity == 0 and ir.l % 2 == 1:
                            parity_flip_indices.extend(list(range(rolling_irrep_ptr, rolling_irrep_ptr + (2*ir.l + 1))))
                            rolling_irrep_ptr += (2*ir.l + 1)
                        elif input_parity == 1 and ir.l % 2 == 0:
                            parity_flip_indices.extend(list(range(rolling_irrep_ptr, rolling_irrep_ptr + (2*ir.l + 1))))
                            rolling_irrep_ptr += (2*ir.l + 1)
                        else:
                            rolling_irrep_ptr += (2*ir.l + 1)
                
                # lower triangle irreps go to the beta space
                if j < i:
                    irreps_edgereduced_beta.append(product_irreps)
                    rolling_irrep_ptr += sum([2*l + 1 for l in Irreps(product_irreps).ls])
                
        self.irreps_edgereduced_alpha = Irreps('+'.join(irreps_edgereduced_alpha))  # targets for alpha (includes ei + ej):
        self.irreps_edgereduced_beta = Irreps('+'.join(irreps_edgereduced_beta))    # targets for beta (includes ei - ej):
        self.parity_flip_indices = parity_flip_indices

        # --- Second Pass: Create indices to pair up the alpha and beta spaces ---
        off_diag_irrep_indices = {'alpha': [], 'beta': []}

        alpha_blocks = {}
        beta_blocks = {}

        alpha_track = 0
        beta_track = 0

        for i, l1 in enumerate(self.ls_list):
            for j, l2 in enumerate(self.ls_list):
                product_irreps = str(get_product_irreps(l1, l2))

                if i == j and l1 == l2:
                    even_irreps = str(get_product_irreps(l1, l2, 'even'))
                    odd_irreps = str(get_product_irreps(l1, l2, 'odd'))
                    
                    if even_irreps != '':
                        alpha_track += sum([2*l + 1 for l in Irreps(even_irreps).ls])
                    if odd_irreps != '':
                        beta_track += sum([2*l + 1 for l in Irreps(odd_irreps).ls])

                if i < j and product_irreps != '':
                    dim = sum([2*l + 1 for l in Irreps(product_irreps).ls])
                    alpha_blocks[(i, j)] = np.arange(alpha_track, alpha_track + dim)
                    alpha_track += dim
                
                if j < i and product_irreps != '':
                    dim = sum([2*l + 1 for l in Irreps(product_irreps).ls])
                    beta_blocks[(i, j)] = np.arange(beta_track, beta_track + dim)
                    beta_track += dim

        # --- Pair them up ---
        for (i, j), indices in alpha_blocks.items():
            off_diag_irrep_indices['alpha'].append(indices)
            off_diag_irrep_indices['beta'].append(beta_blocks[(j, i)])

        # check that the length of the sub-arrays in off_diag_irrep_indices['alpha'] are the same as the length of the sub arrays in off_diag_irrep_indices['beta']:
        # print("off_diag_irrep_indices['alpha']: ", off_diag_irrep_indices['alpha'], flush=True)
        # print("off_diag_irrep_indices['beta ']: ", off_diag_irrep_indices['beta'], flush=True)
        for alpha_indices, beta_indices in zip(off_diag_irrep_indices['alpha'], off_diag_irrep_indices['beta']):
            assert len(alpha_indices) == len(beta_indices), "The length of the alpha and beta irreps should be the same for each off-diagonal block, but got {} and {}! Something is wrong with the indexing in make_irreps_edgereduced()".format(len(alpha_indices), len(beta_indices))

        # if they match irrep-for-irrep, then we combine them into a single list which will be used to index the alpha and beta edge output components 
        # (eg, to extract Vsp and Vps, so that we can create Vsp+Vps and Vsp-Vps):
        off_diag_irrep_indices['alpha'] = np.concatenate(off_diag_irrep_indices['alpha']) if len(off_diag_irrep_indices['alpha']) > 0 else np.array([], dtype=np.int64)
        off_diag_irrep_indices['beta'] = np.concatenate(off_diag_irrep_indices['beta']) if len(off_diag_irrep_indices['beta']) > 0 else np.array([], dtype=np.int64)

        self.off_diag_irrep_indices = off_diag_irrep_indices

    def get_output_permutation(self, output_irreps):
        """
        Get the permutation that reorders irreps from sorted-by-l order to irreps_out order.
        Initially, we have the output irreps in sorted order: (omol) 114x0e+ 229x1e+239x2e+161x3e+73x4e+24x5e+4x6e
        We want to permute them to the order they appear in irreps_out.
        """
        input_irreps_simplified = output_irreps.sort()[0].simplify()
        total_dim = sum(mul * (2 * ir.l + 1) for mul, ir in input_irreps_simplified)
        # print("Total output dim: ", total_dim, flush=True)

        sorted_irreps, permutation, inverse_permutation = output_irreps.sort()
        # Inverse permutation: output irreps -> sorted irreps
        # Permutation: sorted irreps -> output irreps

        req_permutation = list(inverse_permutation)

        # augment each element of permutation by the total dimension of all previous irreps:
        tensor_inverse_permutation = torch.zeros(total_dim, dtype=torch.long)
        sorted_tensor_pos = 0

        # For each irrep in sorted order, find where it should go in original order
        for sorted_irrep_idx, (_, ir) in enumerate(sorted_irreps):
            ir_dim = 2 * ir.l + 1
            # print("Handling irrep: ", ir, " of dim ", ir_dim, "in sorted position ", sorted_irrep_idx, flush=True)
            # print("This should go to unsorted position ", req_permutation[sorted_irrep_idx], flush=True)

            # Find which original irrep this corresponds to in the output_irreps
            original_irrep_idx = req_permutation[sorted_irrep_idx]

            # Calculate tensor position in original order for this irrep
            original_tensor_pos = 0
            for i in range(original_irrep_idx):
                _, orig_ir = output_irreps[i]
                original_tensor_pos += 2 * orig_ir.l + 1

            # Map tensor elements: original[orig_pos] = sorted[sorted_pos]
            for j in range(ir_dim):
                tensor_inverse_permutation[original_tensor_pos + j] = sorted_tensor_pos + j

            sorted_tensor_pos += ir_dim

        # print("tensor_inverse_permutation: ", [tensor_inverse_permutation[i].item() for i in range(len(tensor_inverse_permutation))], flush=True)
        return tensor_inverse_permutation


    def split_irreps(self, irreps):
        scalars = []
        gated = []
        for mul, ir in irreps:
            if ir.l == 0:
                scalars.append((mul, ir))
            else:
                gated.append((mul, ir))
        return Irreps(scalars), Irreps(gated)


    def forward(self, emb, batch):

        node_embeddings = emb["node_embeddings"]
        edge_embeddings = emb["edge_embeddings"]
        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)

        if self.reduce_edge:
            edge_embeddings_alpha, edge_embeddings_beta, half_mask, reverse_indices = self.create_half_antisym_edge_pairs(
                    edge_embeddings, edge_index
                )
            
        node_outputs = []
        edge_outputs = []
        for spin in range(self.num_spins):
            node_embeddings_spin = self.stack_irreps(node_embeddings)
            node_output = self.process(node_embeddings_spin, 'node', spin)      # gating and linear layers to produce the output irreps in sorted order
            
            if self.reduce_node:
                node_output = node_output[:, self.output_permutation['node']]                      # permute to the correct order of output irreps
            else:
                node_output = node_output[:, self.output_permutation['common']]                      

            if self.reduce_edge:
                edge_embeddings_spin_alpha = self.stack_irreps(edge_embeddings_alpha)
                edge_embeddings_spin_beta = self.stack_irreps(edge_embeddings_beta)

                edge_output_alpha = self.process(edge_embeddings_spin_alpha, 'edge_alpha', spin)
                edge_output_beta = self.process(edge_embeddings_spin_beta, 'edge_beta', spin)

                # We only computed half the edges for alpha, and beta, since the other half were simple + and - multiplications. Now we reconstruct all of the edges.
                edge_output_alpha, edge_output_beta = self.half_edges_to_full(edge_index, reverse_indices, half_mask, edge_output_alpha, edge_output_beta)

                # recover the original order within the alpha and beta subspaces
                edge_output_alpha = edge_output_alpha[:, self.output_permutation['edge_alpha']]
                edge_output_beta = edge_output_beta[:, self.output_permutation['edge_beta']]

                # extract Vpa and Vpb from the edge outputs
                Vpa = edge_output_alpha[:, self.off_diag_irrep_indices['alpha']].clone() 
                Vpb = edge_output_beta[:, self.off_diag_irrep_indices['beta']].clone() 

                edge_output_alpha[:, self.off_diag_irrep_indices['alpha']] = Vpa + Vpb  # build Vsp in place
                edge_output_beta[:, self.off_diag_irrep_indices['beta']] = Vpa - Vpb    # build Vps in place
                edge_output = torch.cat((edge_output_alpha, edge_output_beta), dim=-1)  

                # permute to the correct order of output irreps, now that we have combined the alpha and beta subspaces back together:
                edge_output = edge_output[:, self.output_permutation['remix_subspaces']] 

                # add a -1 factor to:
                # [odd output irreps from even off-diag input irrep interactions, like p-p, d-d, f-f, p-f]
                # [even output irreps from odd off-diag input irrep interactions, like p-d, d-f]
                edge_output[:, self.parity_flip_indices] = -1*edge_output[:, self.parity_flip_indices]  

            else:
                edge_embeddings_spin = self.stack_irreps(edge_embeddings)
                edge_output = self.process(edge_embeddings_spin, 'edge', spin)  # gating and linear layers to produce the output irreps in sorted order
                
                # if we don't use reduce_node or reduce_edge, there is a single common output permutation
                if self.reduce_node:
                    edge_output = edge_output[:, self.output_permutation['edge']]                  # permute to the correct order of output irreps
                else:
                    edge_output = edge_output[:, self.output_permutation['common']]                  


            # augment the node irreps back to the full irrep list (containing the lower triangle of orbital interactions and odd self-interaction irreps)
            # using edge_output to infer the total size of the output embeddings
            if self.reduce_node:
                node_output = self.expand_reduced_node(node_output)
                        
            node_outputs.append(node_output)
            edge_outputs.append(edge_output)

        # Stack along spin dimension
        node_outputs = torch.stack(node_outputs, dim=0)  # [spin, num_nodes, ...]
        edge_outputs = torch.stack(edge_outputs, dim=0)  # [spin, num_edges, ...]
        return node_outputs, edge_outputs


    def stack_irreps(self, x_message):
        # input = x_message = [num_atoms/edges (batch_size), (lmax+1)**2, sphere_channels]

        x_message_T = x_message.transpose(-1,-2) # rearrange dimensions from [l, E] to [E, l]
        batch_size = x_message_T.shape[0]

        # group all the different ls so l_sorted output looks like sphere_channels*0e + sphere_channels*1e + sphere_channels*2e ...
        l_sorted_output = torch.zeros(batch_size, self.sphere_channels*((self.lmax+1)**2), device=x_message.device)
        for l in range(self.lmax+1):
            start = (l**2)*self.sphere_channels
            end = (l**2)*self.sphere_channels + self.sphere_channels*(2*l+1)
            l_sorted_output[:,start:end] = torch.squeeze(x_message_T[:, :, (l**2):(l**2)+(2*l+1)].reshape(batch_size, 1, -1))

        return l_sorted_output

    def process(self, x, node_or_edge, spin):

        # 1. Extract the scalar components, which are the first # sphere_channels elements of this tensor
        x_scalars = x[:, :self.sphere_channels]
        x_nonscalars = x[:, self.sphere_channels:]

        # 2. Prepare some scalars for gating - gate with learnable scalars: the first 'sphere_channels' scalars are the l=0, and others are used for gating
        all_scalars = self.lin_scalars_learnable[spin](x_scalars)

        transformed_l0_scalars = all_scalars[:, :self.sphere_channels]
        gating_scalars = all_scalars[:, self.sphere_channels:]

        # take abs of the gating scalars to preserve antisymmetry, since they will multiply antisymmetric data ( need even*odd=odd )
        if self.reduce_edge:
            gating_scalars = torch.abs(gating_scalars)  

        # 3. Gate the l>0 irreps:
        x_gated = self.gate[spin](torch.cat([gating_scalars, x_nonscalars], dim=1))
        x_gated = torch.cat([transformed_l0_scalars, x_gated], dim=1)   # use the transformed scalars as the output

        # 4. Apply linear map to output irreps
        if not self.reduce_node:

            # pass each irrep of x_out through its own SO3_Linear layer, where x_out followed simplified irreps_in (self.sphere_channels*0e + sphere_channels*1e + sphere_channels*2e ...)
            x_out_list = []
            irrep_end_track = 0
            batch_size = x_gated.shape[0]
            for l in range(0, self.lmax+1):

                # sphere channels is the multiplicity of this l in the input
                start_idx = irrep_end_track
                end_idx = start_idx + self.sphere_channels  * (2*l + 1)
                irrep_end_track = end_idx

                x_l = x_gated[:, start_idx:end_idx]  # extract the l-th irrep component
                x_l_out = self.lin_out_layers[spin][l](x_l)  # apply the Linear layer for degree l
                x_out_list.append(x_l_out)

            # concatenate all the l outputs back together
            x_out = torch.cat(x_out_list, dim=1)

        else:
            x_out_list = []
            irrep_end_track = 0
            batch_size = x_gated.shape[0]
            for l in range(0, self.lmax+1):

                # sphere channels is the multiplicity of this l in the input
                start_idx = irrep_end_track
                end_idx = start_idx + self.sphere_channels  * (2*l + 1)
                irrep_end_track = end_idx

                x_l = x_gated[:, start_idx:end_idx]  # extract the l-th irrep component
                if node_or_edge == 'node':
                    x_l_out = self.node_lin_out_layers[spin][l](x_l)  # apply the Linear layer for degree l
                if node_or_edge == 'edge':
                    x_l_out = self.edge_lin_out_layers[spin][l](x_l) 
                if node_or_edge == 'edge_alpha':
                    x_l_out = self.edge_lin_out_layers['alpha'][spin][l](x_l)
                if node_or_edge == 'edge_beta':
                    x_l_out = self.edge_lin_out_layers['beta'][spin][l](x_l)
                x_out_list.append(x_l_out)

            # concatenate all the l outputs back together 
            x_out = torch.cat(x_out_list, dim=1)

        return x_out


    def expand_reduced_node(self, node_output):
        """
        Expand irreps_nodereduced to irreps_out, by adding the previously-removed irreps back in:
        1. The odd irreps for the orbital self-interactions
        2. The 'lower triangle' of inter-orbital interactions on this node (eg the p-s to s-p)
        """

        expanded_node_output = node_output[:, self.reduced_to_all_indices] * self.parity_multiplier
        return expanded_node_output
    
    def create_antisym_edge_pairs(self, edge_embeddings, edge_index):
        """
        Create alpha and beta edge embeddings from the symmetric and antisymmetric pairs of original edge embeddings
        """
        # Get the sorting permutation that groups original edges sequentially: (i, j)
        _, perm_orig = torch.sort(edge_index[0] * (edge_index.max() + 1) + edge_index[1])
        
        # Get the sorting permutation that groups flipped edges sequentially: (j, i)
        _, perm_flip = torch.sort(edge_index[1] * (edge_index.max() + 1) + edge_index[0])
        
        # Map directly from original array index to the corresponding flipped array index
        # We invert the original permutation array to discover the alignment mapping
        inv_perm_orig = torch.argsort(perm_orig)
        reverse_edge_indices = perm_flip[inv_perm_orig]

        # Construct symmetric and antisymmetric spaces
        backward_edge_embeddings = edge_embeddings[reverse_edge_indices]
        
        edge_embeddings_alpha = edge_embeddings + backward_edge_embeddings
        edge_embeddings_beta = edge_embeddings - backward_edge_embeddings

        return edge_embeddings_alpha, edge_embeddings_beta


    def create_half_antisym_edge_pairs(self, edge_embeddings, edge_index):
        """
        Creates alpha and beta embeddings ONLY for the canonical half of the edges.
        """
        _, perm_orig = torch.sort(edge_index[0] * (edge_index.max() + 1) + edge_index[1])
        _, perm_flip = torch.sort(edge_index[1] * (edge_index.max() + 1) + edge_index[0])
        inv_perm_orig = torch.argsort(perm_orig)
        reverse_edge_indices = perm_flip[inv_perm_orig]

        # Create a boolean mask tracking the canonical half (where current idx < pair idx)
        indices = torch.arange(edge_index.shape[1], device=edge_index.device)
        half_mask = indices < reverse_edge_indices

        # Slice out the half spaces
        half_orig = edge_embeddings[half_mask]
        half_flip = edge_embeddings[reverse_edge_indices[half_mask]]
        
        # Construct symmetric and antisymmetric spaces only for this half
        edge_embeddings_alpha_half = half_orig + half_flip
        edge_embeddings_beta_half = half_orig - half_flip

        return edge_embeddings_alpha_half, edge_embeddings_beta_half, half_mask, reverse_edge_indices

    def half_edges_to_full(self, edge_index, reverse_indices, half_mask, edge_output_alpha_half, edge_output_beta_half):
        """
        After processing only the half of the edges for alpha and beta (created by create_half_antisym_edge_pairs), 
        we need to reconstruct the full edge output tensors by filling in the other half using the symmetry and antisymmetry properties.
        """

        E = edge_index.shape[1]

        # Allocate full tensors matching the processed feature shapes
        edge_output_alpha_full = torch.zeros((E,) + edge_output_alpha_half.shape[1:], dtype=edge_output_alpha_half.dtype, device=edge_output_alpha_half.device)
        edge_output_beta_full = torch.zeros((E,) + edge_output_beta_half.shape[1:], dtype=edge_output_beta_half.dtype, device=edge_output_beta_half.device)

        # Fetch the destination slots for the other directional half
        flipped_indices = reverse_indices[half_mask]

        # Populate alpha symmetrically: slots (i, j) and (j, i) get identical values
        edge_output_alpha_full[half_mask] = edge_output_alpha_half
        edge_output_alpha_full[flipped_indices] = edge_output_alpha_half

        # Populate beta antisymmetrically: slot (j, i) gets the negated value of (i, j)
        edge_output_beta_full[half_mask] = edge_output_beta_half
        edge_output_beta_full[flipped_indices] = -edge_output_beta_half

        return edge_output_alpha_full, edge_output_beta_full


@registry.register_model("esen_linear_energy_head")
class HELM_Energy_Head(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.lmax = backbone.lmax
        self.mmax = backbone.mmax
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)  # need to re-create mapping reduced to avoid inplace modification error!
        self.edge_channels_list = backbone.edge_channels_list
        extra_m0_output_channels = self.lmax * self.hidden_channels

        l_to_m_permute = self.mappingReduced.l_harmonic[
                torch.argmax(self.mappingReduced.to_m, dim=1)
            ]

        self.act = GateActivation( # in m-major
                lmax=self.lmax, mmax=self.mmax, num_channels=self.hidden_channels, outer_dim='m', l_to_m_permute=l_to_m_permute
            )

        multiplier = 2 
        self.so2_conv_1 = SO2_Convolution(
            multiplier*self.sphere_channels,
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

        # self.linear = nn.Linear(backbone.sphere_channels, 1, bias=True)

        # make this a two linear layers with nonlinearity in between:
        # self.linear = nn.Sequential(
        #     nn.Linear(self.sphere_channels, 2*self.sphere_channels, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(2*self.sphere_channels, 2*self.sphere_channels, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(2*self.sphere_channels, 1, bias=True),
        # )
        h = 2*self.sphere_channels
        self.linear = nn.Sequential(
            nn.Linear(self.sphere_channels, h, bias=True),
            nn.SiLU(),
            nn.Linear(h, 1, bias=True),
        )


    def forward(self, emb: dict[str, torch.Tensor], batch):

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)

        # Trim the embeddings to the chosen lmax (not used)
        nodes = emb["node_embeddings"]#[:, :(self.lmax+1)**2, :]
        # edges = emb["edge_embeddings"]#[:, :(self.lmax+1)**2, :]
        x_edge = emb["x_edge"]
        wigner = emb["wigner"]#[:, :(self.lmax+1)**2, :(self.lmax+1)**2]
        wigner_inv = emb["wigner_inv"]#[:, :(self.lmax+1)**2, :(self.lmax+1)**2]

        # Create the messages for the last convolution:
        x_source = nodes[edge_index[0]]
        x_target = nodes[edge_index[1]]
        x_message = torch.cat((x_source, x_target), dim=2)

        # -----------------
        # Rotate the irreps
        x_message = torch.bmm(wigner, x_message)

        # Apply the SO2 convolution to the messages
        x_message = torch.einsum("nac,ba->nbc", x_message, self.mappingReduced.to_m)   # l-major -> m-major
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)
        x_message = torch.einsum("nac,ab->nbc", x_message, self.mappingReduced.to_m)   # m-major -> l-major

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (nodes.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        # aggregate messages
        new_embedding.index_add_(0, edge_index[1], x_message) # only for the first row
        energies = new_embedding.narrow(1, 0, 1)
        energies = self.linear(energies)

        energies = energies.squeeze(-1).squeeze(-1)

        molecule_indices = torch.cat([
            torch.full((num_atoms,), i, dtype=torch.long, device=energies.device)
            for i, num_atoms in enumerate(batch.num_atoms_in_molecule)
        ])

        # atom-resolved energy -> molecule-resolved energy
        num_molecules = batch.num_atoms_in_molecule.size(0)
        energies_per_molecule = torch.zeros(num_molecules, device=energies.device)
        energies_per_molecule = energies_per_molecule.scatter_add(0, molecule_indices, energies)

        return {"energies": energies_per_molecule}


@registry.register_model("esen_linear_force_head")
class HELM_Force_Head(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.lmax = backbone.lmax
        self.mmax = backbone.mmax
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)  # need to re-create mapping reduced to avoid inplace modification error!
        self.edge_channels_list = backbone.edge_channels_list
        extra_m0_output_channels = self.lmax * self.hidden_channels

        l_to_m_permute = self.mappingReduced.l_harmonic[
                torch.argmax(self.mappingReduced.to_m, dim=1)
            ]

        self.act = GateActivation( # in m-major
                lmax=self.lmax, mmax=self.mmax, num_channels=self.hidden_channels, outer_dim='m', l_to_m_permute=l_to_m_permute
            )

        multiplier = 2 
        self.so2_conv_1 = SO2_Convolution(
            multiplier*self.sphere_channels,
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
            1,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )


    def forward(self, emb: dict[str, torch.Tensor], batch):

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)

        nodes = emb["node_embeddings"]
        x_edge = emb["x_edge"]
        wigner = emb["wigner"]
        wigner_inv = emb["wigner_inv"]

        # Create the messages for the last convolution:
        x_source = nodes[edge_index[0]]
        x_target = nodes[edge_index[1]]
        x_message = torch.cat((x_source, x_target), dim=2)

        # -----------------
        # Rotate the irreps
        x_message = torch.bmm(wigner, x_message)

        # Apply the SO2 convolution to the messages
        x_message = torch.einsum("nac,ba->nbc", x_message, self.mappingReduced.to_m)   # l-major -> m-major
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)
        x_message = torch.einsum("nac,ab->nbc", x_message, self.mappingReduced.to_m)   # m-major -> l-major

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (nodes.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        # aggregate messages
        new_embedding.index_add_(0, edge_index[1], x_message) 
        forces = new_embedding.narrow(1, 1, 3)
        forces = forces.squeeze(-1)

        return {"forces": forces}


@registry.register_model("esen_linear_force_head")
class HELM_Simple_Force_Head(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, emb: dict[str, torch.Tensor], batch):

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)

        forces = self.linear(emb["node_embeddings"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        return {"forces": forces}
