# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""
Experimental configurable NTE/QHFlow3 composition backbone.

The implementation is adapted from fairchem; see LICENSES/MIT-fairchem.md.
"""

from __future__ import annotations

import matplotlib.pyplot as plt # remove
import os, sys
from pathlib import Path
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
from ...helm.nn.so3_layers import SO3_Linear

import torch.distributed as dist
from mpi4py import MPI

from fairchem.core.common.utils import conditional_grad
e3nn.set_optimization_defaults(jit_script_fx=False)

from ...helm.common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
    eulers_to_wigner,
    init_edge_rot_euler_angles
)
from ...helm.common.so3 import (
    CoefficientMapping,
    SO3_Grid,
)
from .block import eSEN_Block
from ...helm.nn.embedding import EdgeDegreeEmbedding
from ...helm.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from ...helm.nn.radial import EnvelopedBesselBasis, GaussianSmearing
from ...helm.nn.so2_layers import SO2_Convolution
from ...helm.nn.so3_layers import SO3_Linear
from ...helm.nn.activation import GateActivation
from ...helm.nn.matrix_embedding import MatrixEmbedding
from ...helm.qhflow3 import (
    GridAtomwise as QHFlow3GridAtomwise,
    MuonVisibleIrrepLinear,
    eSCNMD_Block as QHFlow3NodeBlock,
    eSCNMD_Block_xy2 as QHFlow3PairBlock,
)
from ...helm.qhf_layer.layer_norm import (
    get_normalization_layer as get_qhflow3_normalization_layer,
)
from ...helm.qhf_layer.radial import GaussianSmearing as QHFlow3GaussianSmearing

from ...helm.common.irreps_utils import get_reduced_to_all_indices, get_parity_multiplier, get_product_irreps, get_subspace_remix_permutation


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


class ConfigurableNTEBackbone(nn.Module):
    """Selector-compatible backbone for historical NTE/QHFlow3 ablations.

    Canonical MALOQ remains in :mod:`maloq.helm.esen_osh`; this class is
    activated only through the explicit feature workflow.
    """

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
        atom_scalar_embedding_mode: str = "element_charge_spin",
        wigner_backend: str = "torch",
        distributed_graph_training=False,
        message_type='source-target',
        message_passing_schedule: str = "interleaved",
        initial_edge_state_mode: str = "edge_degree",
        num_edge_layers: int | None = None,
        output_sphere_channels: int | None = None,
        nte_output_projection_mode: str = "so3_linear",
        output_norm_sharing: str = "shared",
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
        direct_atomwise_layers: tuple[int, ...] = (),
        input_conditioning: str = "none",
        conditioning_basis: str = "def2-svp",
        conditioning_delta_learning: bool = False,
        conditioning_delta_target: str = "fock_matrix",
    ):
        super().__init__()
        if atom_scalar_embedding_mode not in {
            "element_charge_spin",
            "element_only",
        }:
            raise ValueError(
                "atom_scalar_embedding_mode must be 'element_charge_spin' "
                f"or 'element_only', got {atom_scalar_embedding_mode!r}."
            )
        self.atom_scalar_embedding_mode = atom_scalar_embedding_mode

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
        if output_norm_sharing not in {"shared", "separate"}:
            raise ValueError(
                "output_norm_sharing must be 'shared' or 'separate', "
                f"got {output_norm_sharing!r}."
            )
        if output_norm_sharing == "separate" and not include_edges:
            raise ValueError(
                "output_norm_sharing='separate' requires include_edges=True."
            )
        if (
            output_norm_sharing == "separate"
            and edge_stack_mode == "qhflow3_exact_parallel"
        ):
            raise ValueError(
                "output_norm_sharing='separate' is redundant with "
                "edge_stack_mode='qhflow3_exact_parallel', which already "
                "uses a separate QHFlow3 pair norm."
            )
        self.output_norm_sharing = output_norm_sharing
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
            else MatrixEmbedding(
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
        Jd_list = torch.load(Path(__file__).resolve().parents[2] / "helm" / "Jd.pt")
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

        # Paper-era HELM used only the element embedding.  The charge/spin
        # modules were introduced later for open-shell training, so omit the
        # modules entirely in element-only mode to preserve that architecture.
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )
        if self.atom_scalar_embedding_mode == "element_charge_spin":
            # Charge values -10..10 and multiplicities 0..10.
            self.abs_max_charge = 10
            self.charge_embedding = nn.Embedding(
                2 * self.abs_max_charge + 1,
                self.sphere_channels,
            )
            self.max_spin_multiplicity = 11
            self.spin_embedding = nn.Embedding(
                self.max_spin_multiplicity,
                self.sphere_channels,
            )
            self.scalar_node_embedding = nn.Linear(
                3 * self.sphere_channels,
                self.sphere_channels,
            )
        else:
            self.charge_embedding = None
            self.spin_embedding = None
            self.scalar_node_embedding = None

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
        self.direct_atomwise_layers = tuple(
            int(index) for index in direct_atomwise_layers
        )
        if len(set(self.direct_atomwise_layers)) != len(
            self.direct_atomwise_layers
        ):
            raise ValueError("direct_atomwise_layers must not contain duplicates.")
        if any(
            index < 1 or index > self.num_edge_layers
            for index in self.direct_atomwise_layers
        ):
            raise ValueError(
                "direct_atomwise_layers must contain 1-based indices within "
                "num_edge_layers."
            )
        if self.direct_atomwise_layers and not include_edges:
            raise ValueError(
                "direct_atomwise_layers requires include_edges=True."
            )
        if (
            self.direct_atomwise_layers
            and self.edge_stack_mode in {
                "qhflow3_parallel",
                "qhflow3_exact_parallel",
            }
        ):
            raise ValueError(
                "direct_atomwise_layers requires a native eSEN edge-block "
                "forward path, not a QHFlow3 pair stack."
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
                        atomwise_output_mode=(
                            "direct"
                            if edge_layer_index in self.direct_atomwise_layers
                            else self.edge_atomwise_output_mode
                        ),
                        edge_norm1_position=self.edge_norm1_position,
                        **block_kwargs,
                    )
                self.edge_blocks.append(edge_block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels
        )
        self.edge_norm = (
            get_normalization_layer(
                self.norm_type,
                lmax=self.lmax,
                num_channels=self.sphere_channels,
            )
            if self.output_norm_sharing == "separate"
            else None
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
            from ...helm.triton_kernels import edge_vec_to_wigner_fused
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
            "num_atoms_in_molecule": batch.num_atoms_in_molecule if not distributed_graph_training else None
        }
        if self.atom_scalar_embedding_mode == "element_charge_spin":
            data_dict["charges"] = batch.charge
            data_dict["spin_multiplicity"] = batch.spin_multiplicity

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
        molecule_indices = None
        if distributed_graph_training:
            local_node_indices = graph_dict['partition'].local_node_indices
            element_emb = self.sphere_embedding(
                data_dict["atomic_numbers"][local_node_indices]
            )
        else:
            element_emb = self.sphere_embedding(data_dict["atomic_numbers"])
            if (
                self.atom_scalar_embedding_mode == "element_charge_spin"
                or self.input_conditioner is not None
            ):
                molecule_indices = torch.cat(
                    [
                        torch.full(
                            (int(molecule_size),),
                            molecule_index,
                            dtype=torch.long,
                            device=data_dict["pos"].device,
                        )
                        for molecule_index, molecule_size in enumerate(
                            data_dict["num_atoms_in_molecule"]
                        )
                    ]
                )

        if self.atom_scalar_embedding_mode == "element_charge_spin":
            assert self.charge_embedding is not None
            assert self.spin_embedding is not None
            assert self.scalar_node_embedding is not None
            if distributed_graph_training:
                atom_charges = data_dict["charges"] + self.abs_max_charge
                atom_spins = data_dict["spin_multiplicity"]
            else:
                assert molecule_indices is not None
                atom_charges = (
                    data_dict["charges"][molecule_indices]
                    + self.abs_max_charge
                )
                atom_spins = data_dict["spin_multiplicity"][molecule_indices]
            combined_emb = torch.cat(
                [
                    element_emb,
                    self.charge_embedding(atom_charges),
                    self.spin_embedding(atom_spins),
                ],
                dim=-1,
            )
            final_emb = self.scalar_node_embedding(combined_emb)
        else:
            final_emb = element_emb

        if self.input_conditioner is not None:
            if distributed_graph_training:
                raise ValueError(
                    "NTE matrix input conditioning does not support "
                    "distributed graph training."
                )
            assert molecule_indices is not None
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
            if self.edge_norm is not None:
                x_message_edge = self.edge_norm(x_message_edge)
            elif self.qhflow3_pair_norm is None:
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
