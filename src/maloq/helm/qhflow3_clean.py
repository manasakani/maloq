"""Clean QHFlow3 feature backbone.

This file is a side-by-side readable version of the active
``qhflow3.py::QHFlow3StrongCondFeatures`` density path.  It keeps the same
tensor-expansion feature contract, but removes the QHFlow2/QHFlow3 inheritance
stack and the inactive SelfNet/PairNet branches.

Comparison map against ``qhflow3.py``:

* ``QHFlow3StrongCond.__init__`` -> ``QHFlow3CleanFeatures.__init__``
* ``_install_strong_conditioned_full_flow_injection`` ->
  ``_process_through_main_layers``
* ``_qhflow3_backbone_output`` -> ``forward``

The lower-level eSCN math modules are kept local in this file so the active path
can be inspected without jumping through the old monolithic backbone class.
"""

from __future__ import annotations

import functools
import math
import copy
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import Irreps, Linear
from torch.profiler import record_function

from ..fock_utils import basis_sets
from .qhf_layer.embedding import (
    ChgSpinEmbedding,
    EdgeDegreeEmbedding,
)
from .qhf_layer.activation import (
    GateActivation,
    SeparableS2Activation_M,
)
from .qhf_layer.layer_norm import (
    get_normalization_layer,
)
from .qhf_layer.radial import (
    GaussianSmearing,
    PolynomialEnvelope,
)
from .qhf_layer.so2_layers import (
    SO2_Convolution,
)
from .qhf_layer.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from .qhf_layer.so3 import (
    CoefficientMapping,
    SO3_Grid,
)


class _NoGraphParallel:
    @staticmethod
    def initialized() -> bool:
        return False


gp_utils = _NoGraphParallel()


@dataclass
class BackboneOutput:
    node_feats: torch.Tensor
    edge_feats: torch.Tensor
    extra: dict[str, Any] = field(default_factory=dict)


_QHFLOW3_CLEAN_JD_PATH = (
    Path(__file__).resolve().parent / "qhf_layer/Jd.pt"
)


def construct_o3irrps(dim: int, order: int) -> str:
    return "+".join(
        f"{dim}x{l}e" if l % 2 == 0 else f"{dim}x{l}o"
        for l in range(order + 1)
    )


def construct_o3irrps_base(dim: int, order: int) -> str:
    return "+".join(f"{dim}x{l}e" for l in range(order + 1))


class MuonVisibleIrrepLinear(nn.Module):
    """An e3nn Linear with the same math and a Muon-visible weight layout.

    ``e3nn.o3.Linear`` stores all path weights in one flat parameter, so the
    shape-based Muon router treats an otherwise matrix-valued projection as an
    AdamW auxiliary parameter. This wrapper keeps e3nn's compiled operation
    unchanged but stores the paths as ``[path, output, input]``. Flattening the
    transposed view reconstructs e3nn's original weight order exactly.

    Initialization consumes the same random values in the same order as the
    corresponding internally weighted e3nn Linear. Before the optimizer takes
    a step, mapped weights, outputs, input gradients, and weight gradients are
    therefore identical.
    """

    def __init__(self, irreps_in: Irreps | str, irreps_out: Irreps | str) -> None:
        super().__init__()
        self.linear = Linear(
            irreps_in,
            irreps_out,
            internal_weights=False,
            shared_weights=True,
            biases=False,
        )
        path_shapes = [
            instruction.path_shape
            for instruction in self.linear.instructions
            if instruction.i_in >= 0
        ]
        if not path_shapes:
            raise ValueError("Muon-visible irrep projection needs a weighted path.")
        if len(set(path_shapes)) != 1:
            raise ValueError(
                "Muon-visible irrep projection requires one common matrix shape "
                f"across paths, got {path_shapes}."
            )
        input_channels, output_channels = path_shapes[0]
        path_major_weight = torch.randn(
            len(path_shapes),
            input_channels,
            output_channels,
        )
        self.weight = nn.Parameter(
            path_major_weight.transpose(1, 2).contiguous()
        )
        if self.weight.numel() != self.linear.weight_numel:
            raise RuntimeError(
                "Muon-visible weight size does not match the e3nn Linear "
                f"contract: {self.weight.numel()} != {self.linear.weight_numel}."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat_weight = self.weight.transpose(1, 2).reshape(-1)
        return self.linear(features, flat_weight)


def get_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_positions: int = 2000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb,
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def cutoff_function(x: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(x)
    x_masked = torch.where(x < cutoff, x, zeros)
    denominator = (cutoff - x_masked) * (cutoff + x_masked)
    exponential = torch.exp(-(x_masked**2) / denominator)
    return torch.where(x < cutoff, exponential, zeros)


def softplus_inverse(x: float | torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


class ExponentialBernsteinRadialBasisFunctions(nn.Module):
    def __init__(
        self,
        num_basis_functions: int,
        cutoff: float,
        ini_alpha: float = 0.5,
        fix_alpha: bool = True,
    ) -> None:
        super().__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        self.fix_alpha = fix_alpha
        self._precompute_coefficients(num_basis_functions)
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.register_parameter(
            "_alpha",
            nn.Parameter(torch.tensor(1.0, dtype=torch.float32)),
        )
        self.reset_parameters()

    def _precompute_coefficients(self, num_basis_functions: int) -> None:
        log_factorial = np.zeros(num_basis_functions)
        for i in range(2, num_basis_functions):
            log_factorial[i] = log_factorial[i - 1] + np.log(i)

        v_indices = np.arange(0, num_basis_functions)
        n_indices = (num_basis_functions - 1) - v_indices
        log_binomial = (
            log_factorial[-1]
            - log_factorial[v_indices]
            - log_factorial[n_indices]
        )

        self.register_buffer(
            "log_binomial_coeff",
            torch.tensor(log_binomial, dtype=torch.float32),
        )
        self.register_buffer("n_indices", torch.tensor(n_indices, dtype=torch.float32))
        self.register_buffer("v_indices", torch.tensor(v_indices, dtype=torch.float32))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self._alpha.data.fill_(softplus_inverse(self.ini_alpha))

    @property
    def alpha(self) -> float | torch.Tensor:
        if self.fix_alpha:
            return 1.0
        return F.softplus(self._alpha)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        negative_alpha_r = -alpha * distances
        one_minus_exp = -torch.expm1(negative_alpha_r)
        log_terms = (
            self.log_binomial_coeff
            + self.n_indices * negative_alpha_r
            + self.v_indices * torch.log(one_minus_exp)
        )
        rbf_values = torch.exp(log_terms)
        cutoff_values = cutoff_function(distances, self.cutoff)
        return cutoff_values * rbf_values


class ParamContraction(nn.Module):
    """Contraction layer with one learned parameter tensor per CG path."""

    def __init__(
        self,
        irrep_in_1: str | Irreps,
        irrep_in_2: str | Irreps,
        irrep_out: str | Irreps,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.irrep_in_1 = Irreps(irrep_in_1)
        self.irrep_in_2 = Irreps(irrep_in_2)
        self.irrep_out = Irreps(irrep_out)
        self.instructions = self.get_contraction_path(
            self.irrep_in_1,
            self.irrep_in_2,
            self.irrep_out,
        )
        self.use_bias = use_bias
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions)
        self.num_bias = sum(prod(ins[-1][2:]) for ins in self.instructions)
        self.path_counts = self._count_paths()

        if self.num_path_weight > 0:
            self.path_weights = nn.Parameter(torch.rand(self.num_path_weight))
        if self.num_bias > 0 and self.use_bias:
            self.bias_weights = nn.Parameter(torch.rand(self.num_bias))
        self.num_weights = self.num_path_weight + self.num_bias

    def _count_paths(self) -> torch.Tensor:
        counts = torch.zeros(len(self.irrep_out), dtype=torch.float32)
        for ins in self.instructions:
            idx_out = ins[2]
            counts[idx_out] += 1 * ins[-1][0] * ins[-1][1]
        return counts.clamp(min=1)

    def forward(
        self,
        x_in: torch.Tensor,
        weights: torch.Tensor | None = None,
        bias_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_num = x_in.shape[0]
        x_in_blocks = [
            [x_in[:, s1, s2] for s2 in self.irrep_in_2.slices()]
            for s1 in self.irrep_in_1.slices()
        ]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            idx_in1, idx_in2, idx_out = ins[0], ins[1], ins[2]
            l_in1, l_in2, l_out = ins[3], ins[4], ins[5]
            mul_tuple = [ins[-1][2], ins[-1][0], ins[-1][1]]
            mul_ir_in1 = self.irrep_in_1[idx_in1]
            mul_ir_in2 = self.irrep_in_2[idx_in2]

            x1_reshaped = x_in_blocks[idx_in1][idx_in2].reshape(
                batch_num,
                mul_ir_in1.mul,
                mul_ir_in1.ir.dim,
                mul_ir_in2.mul,
                mul_ir_in2.ir.dim,
            )
            w3j_matrix = o3.wigner_3j(l_in1, l_in2, l_out).to(x_in.device).type(
                x1_reshaped.dtype,
            )

            if weights is None:
                weight = self.path_weights[
                    flat_weight_index : flat_weight_index + prod(ins[-1])
                ].reshape(mul_tuple)
                result = torch.einsum("buivj, ijk, wuv -> bwk", x1_reshaped, w3j_matrix, weight)
                if self.use_bias:
                    bias_weight = self.bias_weights[
                        bias_weight_index : bias_weight_index + prod(ins[-1][2:])
                    ].reshape(ins[-1][2:])
                    bias_weight_index += prod(ins[-1][2:])
                    result = result + bias_weight.unsqueeze(-1)
            else:
                weight = weights[
                    :, flat_weight_index : flat_weight_index + prod(ins[-1])
                ].reshape([-1] + mul_tuple)
                result = torch.einsum("buivj, ijk, wuv -> bwk", x1_reshaped, w3j_matrix, weight)
                if self.use_bias and bias_weights is not None:
                    bias_weight = bias_weights[
                        :, bias_weight_index : bias_weight_index + prod(ins[-1][2:])
                    ].reshape([-1] + ins[-1][2:])
                    bias_weight_index += prod(ins[-1][2:])
                    result = result + bias_weight.unsqueeze(-1)

            flat_weight_index += prod(ins[-1])
            result = result * ((2 * l_out + 1) ** 0.5)

            if idx_out in outputs:
                outputs[idx_out] = outputs[idx_out] + result
            else:
                outputs[idx_out] = result

        final_tensors = []
        path_counts = self.path_counts.to(x_in.device)
        for i, mul_ir in enumerate(self.irrep_out):
            if i in outputs:
                normalized_output = outputs[i] / path_counts[i]
                final_tensors.append(normalized_output.reshape(batch_num, -1))
            else:
                final_tensors.append(
                    torch.zeros(batch_num, mul_ir.dim, device=x_in.device, dtype=x_in.dtype),
                )
        return torch.cat(final_tensors, dim=-1)

    @staticmethod
    def get_contraction_path(
        irrep_in_1: Irreps,
        irrep_in_2: Irreps,
        irrep_out: Irreps,
    ) -> list[list[Any]]:
        instructions = []
        for i, (num_in1, ir_in1) in enumerate(irrep_in_1):
            for j, (num_in2, ir_in2) in enumerate(irrep_in_2):
                for k, (num_out, ir_out) in enumerate(irrep_out):
                    if ir_out in ir_in1 * ir_in2:
                        instructions.append(
                            [
                                i,
                                j,
                                k,
                                ir_in1.l,
                                ir_in2.l,
                                ir_out.l,
                                [num_in1, num_in2, num_out],
                            ],
                        )
        return instructions


def _cast_batch_floating_tensors(batch: Any, dtype: torch.dtype) -> None:
    keys = batch.keys() if hasattr(batch, "keys") else vars(batch).keys()
    for key in keys:
        value = getattr(batch, key)
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != dtype:
            setattr(batch, key, value.to(dtype=dtype))


@functools.lru_cache(maxsize=32)
def _escn_m2c_perm(lmax: int, channels: int, device: torch.device) -> torch.Tensor:
    idx_in_list = [0] * (((lmax + 1) ** 2) * channels)
    for l_value in range(lmax + 1):
        width = 2 * l_value + 1
        l_offset = l_value * l_value
        for m_idx in range(width):
            for channel in range(channels):
                out = l_offset * channels + channel * width + m_idx
                idx_in_list[out] = (l_offset + m_idx) * channels + channel
    return torch.tensor(idx_in_list, dtype=torch.long, device=device)


def _escn_to_e3nn_flat(x: torch.Tensor, lmax: int) -> torch.Tensor:
    perm = _escn_m2c_perm(lmax, x.shape[-1], x.device)
    return x.reshape(x.shape[0], -1).index_select(1, perm)


def _transpose_indices_from_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int | None = None,
) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E).")
    num_edges = int(edge_index.shape[1])
    if num_edges == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
    edge_hash = src * int(num_nodes) + dst
    reverse_hash = dst * int(num_nodes) + src
    sort_order = torch.argsort(edge_hash)
    sorted_hash = edge_hash.index_select(0, sort_order)
    pos = torch.searchsorted(sorted_hash, reverse_hash)
    safe_pos = pos.clamp(max=num_edges - 1)
    found = (pos < num_edges) & (sorted_hash.index_select(0, safe_pos) == reverse_hash)
    if not bool(found.all().item()):
        raise ValueError("edge_index must contain transpose pairs for Hermitian symmetrization.")
    return sort_order.index_select(0, safe_pos)


class Edgewise(torch.nn.Module):
    """Node-message eSCN edge update used inside each node block.

    It gathers source/target node spherical tensors for every directed edge,
    rotates them into the edge frame, applies two SO(2) convolutions with a
    gated nonlinearity, applies the smooth radial envelope, rotates back, and
    scatters messages to destination nodes.  This is the geometric message
    passing core of ``eSCNMD_Block``.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        cutoff: float,
        activation_checkpoint_chunk_size: int | None,
        act_type: Literal["gate", "s2"] = "gate", # Mostly used as default "gate"
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.act_type = act_type

        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=self.hidden_channels,
                m_prime=True,
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        elif self.act_type == "s2":
            self.act = SeparableS2Activation_M(
                lmax=self.lmax,
                mmax=self.mmax,
                SO3_grid=self.SO3_grid,
                to_m=self.mappingReduced.to_m,
            )
            extra_m0_output_channels = self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")

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

        self.cutoff = cutoff
        self.envelope = PolynomialEnvelope(exponent=5)
        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax,
            self.mmax,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_and_M_mapping: torch.Tensor,
        wigner_and_M_mapping_inv: torch.Tensor,
        node_offset: int = 0,
        return_x_message: bool = False,
    ):
        if self.activation_checkpoint_chunk_size is None:
            return self.forward_chunk(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                node_offset,
                return_x_message=return_x_message,
            )
        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size,
            dim=1,
        )
        wigner_partitions = wigner_and_M_mapping.split(
            self.activation_checkpoint_chunk_size,
            dim=0,
        )
        wigner_inv_partitions = wigner_and_M_mapping_inv.split(
            self.activation_checkpoint_chunk_size,
            dim=0,
        )
        edge_distance_parititons = edge_distance.split(
            self.activation_checkpoint_chunk_size,
            dim=0,
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)
        new_embeddings = []
        x_messages = []

        for idx in range(len(edge_index_partitions)):
            res = torch.utils.checkpoint.checkpoint(
                self.forward_chunk,
                x,
                x_edge_partitions[idx],
                edge_distance_parititons[idx],
                edge_index_partitions[idx],
                wigner_partitions[idx],
                wigner_inv_partitions[idx],
                node_offset,
                use_reentrant=False,
                return_x_message=return_x_message,
            )
            if return_x_message:
                new_embeddings.append(res[0])
                x_messages.append(res[1])
            else:
                new_embeddings.append(res)

            if len(new_embeddings) > 8:
                new_embeddings = [torch.stack(new_embeddings).sum(axis=0)]
                if return_x_message:
                    x_messages = [torch.cat(x_messages, dim=0)]
        if return_x_message:
            return torch.stack(new_embeddings).sum(axis=0), torch.cat(x_messages, dim=0)
        return torch.stack(new_embeddings).sum(axis=0)

    def forward_chunk(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_and_M_mapping: torch.Tensor,
        wigner_and_M_mapping_inv: torch.Tensor,
        node_offset: int = 0,
        return_x_message: bool = False,
    ):
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region_sum_grad(x, dim=0)
            x_source = x_full[edge_index[0]]
            x_target = x_full[edge_index[1]]
        else:
            x_source = x[edge_index[0]]
            x_target = x[edge_index[1]]

        x_message = torch.cat((x_source, x_target), dim=2)

        with record_function("SO2Conv"):
            x_message = torch.bmm(wigner_and_M_mapping, x_message)
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
            x_message = self.act(x_0_gating, x_message)
            x_message = self.so2_conv_2(x_message, x_edge)
            dist_scaled = edge_distance / self.cutoff
            env = self.envelope(dist_scaled)
            x_message = x_message * env.view(-1, 1, 1)
            x_message = torch.bmm(wigner_and_M_mapping_inv, x_message)

        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)
        if return_x_message:
            return new_embedding, x_message
        return new_embedding


class Edgewise_xy(torch.nn.Module):
    """Pair-feature eSCN edge update used by the xy block.

    This is the off-node/pair counterpart of ``Edgewise``.  It computes directed
    edge messages and additionally modulates the two SO(2) convolutions with
    scalar source-target products.  The returned ``xy`` tensor is what the
    tensor-expansion head later consumes as edge latent features.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        cutoff: float,
        activation_checkpoint_chunk_size: int | None,
        act_type: Literal["gate", "s2"] = "gate",
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.act_type = act_type

        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=self.hidden_channels,
                m_prime=True,
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        elif self.act_type == "s2":
            self.act = SeparableS2Activation_M(
                lmax=self.lmax,
                mmax=self.mmax,
                SO3_grid=self.SO3_grid,
                to_m=self.mappingReduced.to_m,
            )
            extra_m0_output_channels = self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")
        self.len_edge_channels = self.edge_channels_list[0]

        # The two linear layers are used to compute the scalar gating values from the
        # scalar source-target products.  The first linear layer computes the gating
        # values for the first SO(2) convolution, and the second linear layer computes
        # the gating values for the second SO(2) convolution.  The gating values are
        # then used to modulate the output of the SO(2) convolutions.
        self.fc1 = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.len_edge_channels),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.len_edge_channels),
        )

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

        self.cutoff = cutoff
        self.envelope = PolynomialEnvelope(exponent=5)
        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax,
            self.mmax,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_and_M_mapping: torch.Tensor,
        wigner_and_M_mapping_inv: torch.Tensor,
        node_offset: int = 0,
        return_x_message: bool = False,
    ):
        if self.activation_checkpoint_chunk_size is None:
            return self.forward_chunk(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                node_offset,
                return_x_message=return_x_message,
            )
        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size,
            dim=1,
        )
        wigner_partitions = wigner_and_M_mapping.split(
            self.activation_checkpoint_chunk_size,
            dim=0,
        )
        wigner_inv_partitions = wigner_and_M_mapping_inv.split(
            self.activation_checkpoint_chunk_size,
            dim=0,
        )
        edge_distance_parititons = edge_distance.split(
            self.activation_checkpoint_chunk_size,
            dim=0,
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)
        new_embeddings = []
        x_messages = []

        for idx in range(len(edge_index_partitions)):
            res = torch.utils.checkpoint.checkpoint(
                self.forward_chunk,
                x,
                x_edge_partitions[idx],
                edge_distance_parititons[idx],
                edge_index_partitions[idx],
                wigner_partitions[idx],
                wigner_inv_partitions[idx],
                node_offset,
                use_reentrant=False,
                return_x_message=return_x_message,
            )
            if return_x_message:
                new_embeddings.append(res[0])
                x_messages.append(res[1])
            else:
                new_embeddings.append(res)

            if len(new_embeddings) > 8:
                new_embeddings = [torch.stack(new_embeddings).sum(axis=0)]
                if return_x_message:
                    x_messages = [torch.cat(x_messages, dim=0)]
        if return_x_message:
            return torch.stack(new_embeddings).sum(axis=0), torch.cat(x_messages, dim=0)
        return torch.stack(new_embeddings).sum(axis=0)

    def forward_chunk(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_and_M_mapping: torch.Tensor,
        wigner_and_M_mapping_inv: torch.Tensor,
        node_offset: int = 0,
        return_x_message: bool = False,
    ):
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region_sum_grad(x, dim=0)
            x_source = x_full[edge_index[0]]
            x_target = x_full[edge_index[1]]
        else:
            x_source = x[edge_index[0]]
            x_target = x[edge_index[1]]

        x_message = torch.cat((x_source, x_target), dim=2)
        x_s0 = x_source[:, 0, :] * x_target[:, 0, :]

        with record_function("SO2Conv"):
            x_message = torch.bmm(wigner_and_M_mapping, x_message)
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge * self.fc1(x_s0))
            x_message = self.act(x_0_gating, x_message)
            # BUG: so2_conv_2 has internal_weights=True, so its forward path
            # ignores x_edge. The fc2 branch below is therefore dead: it does
            # not affect x_message and its parameters receive no gradients.
            x_message = self.so2_conv_2(x_message, x_edge * self.fc2(x_s0))
            dist_scaled = edge_distance / self.cutoff
            env = self.envelope(dist_scaled)
            x_message = x_message * env.view(-1, 1, 1)
            x_message = torch.bmm(wigner_and_M_mapping_inv, x_message)

        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)
        if return_x_message:
            return new_embedding, x_message
        return new_embedding


class GridAtomwise(torch.nn.Module):
    """Per-node/pair equivariant FFN applied through the SO3 grid.

    eSCN alternates geometric message passing with an atomwise feed-forward
    block.  This block projects spherical coefficients to a grid, runs an MLP
    independently at grid points, and projects back to spherical coefficients.
    The default QHFlow3 clean path uses this grid variant.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid: SO3_Grid,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.grid_mlp = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.sphere_channels, bias=False),
        )

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        x_grid = self.grid_mlp(x_grid)
        return self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)

    def compile_fixed_chunk(self, chunk_size: int) -> None:
        """Compile the repeated full-size pair chunk without changing state keys."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        object.__setattr__(
            self,
            "_compiled_forward_chunk",
            torch.compile(self._forward_chunk, mode="reduce-overhead"),
        )
        self.compiled_chunk_size = int(chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compiled_forward = getattr(self, "_compiled_forward_chunk", None)
        if compiled_forward is not None and x.shape[0] == self.compiled_chunk_size:
            # reduce-overhead uses CUDA Graphs whose output storage is reused on
            # the next chunk invocation. Keep each chunk alive independently.
            return compiled_forward(x).clone()
        chunk_size = getattr(self, "grid_ffn_chunk_size", None)
        if not chunk_size or x.shape[0] <= chunk_size:
            return self._forward_chunk(x)
        outputs = [
            torch.utils.checkpoint.checkpoint(
                self._forward_chunk,
                x_chunk,
                use_reentrant=False,
            )
            for x_chunk in x.split(chunk_size, dim=0)
        ]
        return torch.cat(outputs, dim=0)


class eSCNMD_Block(torch.nn.Module):
    """One node-update eSCN layer.

    This block is used in ``QHFlow3ESCNBackboneHam.blocks``.  It performs:
    normalization -> edgewise geometric message passing -> residual add ->
    normalization -> grid atomwise FFN -> residual add.  The output remains a
    node spherical tensor with shape ``[num_nodes, (lmax+1)^2, channels]``.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"],
        act_type: Literal["gate", "s2"],
        ff_type: Literal["spectral", "grid"],
        activation_checkpoint_chunk_size: int | None,
    ) -> None:
        super().__init__()
        if ff_type != "grid":
            raise ValueError("QHFlow3 clean keeps only the grid atomwise eSCN path.")
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(
            norm_type,
            lmax=self.lmax,
            num_channels=sphere_channels,
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
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
        )
        self.norm_2 = get_normalization_layer(
            norm_type,
            lmax=self.lmax,
            num_channels=sphere_channels,
        )
        self.atom_wise = GridAtomwise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            SO3_grid=SO3_grid,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_and_M_mapping: torch.Tensor,
        wigner_and_M_mapping_inv: torch.Tensor,
        sys_node_embedding: torch.Tensor | None = None,
        node_offset: int = 0,
    ) -> torch.Tensor:
        x_res = x
        x = self.norm_1(x)
        if sys_node_embedding is not None:
            x[:, 0, :] = x[:, 0, :] + sys_node_embedding

        with record_function("edgewise"):
            x = self.edge_wise(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                node_offset,
            )
            x = x + x_res

        x_res = x
        x = self.norm_2(x)
        with record_function("atomwise"):
            x = self.atom_wise(x)
            x = x + x_res
        return x


class eSCNMD_Block_xy2(torch.nn.Module):
    """One pair/edge-latent eSCN layer.

    This block runs after node updates to produce directed pair features
    ``xy_embedding``.  It reuses normalized node states to form edge messages,
    optionally adds a recurrent pair state, then applies the same grid atomwise
    FFN over edge spherical tensors.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"],
        act_type: Literal["gate", "s2"],
        ff_type: Literal["spectral", "grid"],
        activation_checkpoint_chunk_size: int | None,
    ) -> None:
        super().__init__()
        if ff_type != "grid":
            raise ValueError("QHFlow3 clean keeps only the grid atomwise eSCN path.")
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.atom_wise_residual = False

        self.norm_1 = get_normalization_layer(
            norm_type,
            lmax=self.lmax,
            num_channels=sphere_channels,
        )
        self.edge_wise = Edgewise_xy(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            cutoff=cutoff,
            act_type=act_type,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
        )
        self.norm_2 = get_normalization_layer(
            norm_type,
            lmax=self.lmax,
            num_channels=sphere_channels,
        )
        self.atom_wise = GridAtomwise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            SO3_grid=SO3_grid,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_and_M_mapping: torch.Tensor,
        wigner_and_M_mapping_inv: torch.Tensor,
        sys_node_embedding: torch.Tensor | None = None,
        node_offset: int = 0,
        pair_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        x = self.norm_1(x)
        if sys_node_embedding is not None:
            x[:, 0, :] = x[:, 0, :] + sys_node_embedding

        with record_function("edgewise_xy"):
            _, xy = self.edge_wise(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                node_offset,
                return_x_message=True,
            )
        xy = xy.to(dtype)

        xy_res = xy
        if pair_state is not None:
            if pair_state.shape != xy.shape:
                raise ValueError(
                    f"pair_state shape mismatch: expected {tuple(xy.shape)}, "
                    f"got {tuple(pair_state.shape)}",
                )
            xy = xy + pair_state.to(device=xy.device, dtype=xy.dtype)
        xy = self.norm_2(xy)

        with record_function("atomwise"):
            xy = self.atom_wise(xy)
            if self.atom_wise_residual:
                xy = xy + xy_res
        xy = xy.to(dtype)
        return xy


# Copied from the vendored qhflow2_legacy eSCN backbone and kept local so
# QHFlow3CleanFeatures no longer imports the vendor backbone class itself.
class QHFlow3ESCNBackboneHam(nn.Module):
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        num_distance_basis: int = 512,
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
        activation_checkpointing: bool = False,
        chg_spin_emb_type: Literal["pos_emb", "lin_emb", "rand_emb"] = "pos_emb",
        cs_emb_grad: bool = False,
        use_block_S: bool = True,
        use_block_H: bool = True,
        use_time_embedding: bool = True,
        num_ham_gnn_layers: int = 2,
        xy_pair_state_mode: Literal["residual_sum", "recurrent"] = "residual_sum",
    ) -> None:
        super().__init__()
        if xy_pair_state_mode not in {"residual_sum", "recurrent"}:
            raise ValueError(
                "xy_pair_state_mode must be 'residual_sum' or 'recurrent', "
                f"got {xy_pair_state_mode!r}.",
            )
        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.use_block_S = use_block_S
        self.use_block_H = use_block_H
        self.use_time_embedding = use_time_embedding
        self.num_ham_gnn_layers = num_ham_gnn_layers
        # QHFlow3's default off-diagonal feature contract is `residual_sum`:
        # each xy2 block reads the final node state and its output is summed.
        # The `recurrent` mode is an opt-in ablation used by maloq_fixed: it
        # preserves QHFlow3's wigner/M mapping, edge-degree envelope, and xy2
        # tensor-expansion contract, but threads the previous xy2 output as the
        # next pair state.  This tests recurrent pair dynamics without feeding a
        # HELM/MALOQ edge latent directly into the QHFlow3 head.
        self.xy_pair_state_mode = xy_pair_state_mode

        activation_checkpoint_chunk_size = None
        if activation_checkpointing:
            activation_checkpoint_chunk_size = ESCNMD_DEFAULT_EDGE_CHUNK_SIZE

        self.chg_spin_emb_type = chg_spin_emb_type
        self.cs_emb_grad = cs_emb_grad

        # rotation utils
        Jd_list = torch.load(_QHFLOW3_CLEAN_JD_PATH)
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
        # charge / spin embedding
        self.charge_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "charge",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )
        self.spin_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "spin",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )
        self.node_feats_H_embedding = nn.Linear(self.sphere_channels, self.sphere_channels)
        if self.use_block_H:
            self.node_feats_H_init_embedding = nn.Linear(self.sphere_channels, self.sphere_channels)
        if self.use_block_S:
            self.node_feats_S_embedding = nn.Linear(self.sphere_channels, self.sphere_channels)

        matrix_mix_cnt = 2
        if self.use_time_embedding:
            matrix_mix_cnt += 1
        if self.use_block_H:
            matrix_mix_cnt += 1
        if self.use_block_S:
            matrix_mix_cnt += 1
        self.mix_matrix = nn.Linear(matrix_mix_cnt * self.sphere_channels, self.sphere_channels)
        self.mix_csd = nn.Linear(2 * self.sphere_channels, self.sphere_channels)

        # edge distance embedding
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.num_distance_basis = num_distance_basis
        self.distance_expansion = GaussianSmearing(
            0.0,
            self.cutoff,
            self.num_distance_basis,
            2.0,
        )

        # equivariant initial embedding
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
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
            rescale_factor=5.0,  # NOTE: sqrt avg degree
            cutoff=self.cutoff,
            mappingReduced=self.mappingReduced,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
        )

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type
        self.ff_type = ff_type

        # Initialize the blocks for each layer
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = eSCNMD_Block(
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
                self.ff_type,
                activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
            )
            self.blocks.append(block)
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )
        self.xy_blocks = nn.ModuleList()
        for _ in range(self.num_ham_gnn_layers):
            self.xy_blocks.append(
                eSCNMD_Block_xy2(
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
                    self.ff_type,
                    activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
                )
            )
        self.xy_norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)

    def _get_rotmat_and_wigner(
        self,
        edge_distance_vecs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
            for l in range(self.lmax + 1)
        ]

        with record_function("obtain rotmat wigner original"):
            euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
            wigner = eulers_to_wigner(
                euler_angles,
                0,
                self.lmax,
                Jd_buffers,
            )
            wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        # select subset of coefficients we are using
        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, self.coefficient_index)
            wigner_inv = wigner_inv.index_select(2, self.coefficient_index)

        wigner_and_M_mapping = torch.einsum(
            "mk,nkj->nmj", self.mappingReduced.to_m.to(wigner.dtype), wigner
        )
        wigner_and_M_mapping_inv = torch.einsum(
            "njk,mk->njm", wigner_inv, self.mappingReduced.to_m.to(wigner_inv.dtype)
        )
        return wigner_and_M_mapping, wigner_and_M_mapping_inv

    def csd_embedding(self, charge, spin):
        with record_function("charge spin embeddings"):
            chg_emb = self.charge_embedding(charge)
            spin_emb = self.spin_embedding(spin)
            return torch.nn.SiLU()(self.mix_csd(torch.cat((chg_emb, spin_emb), dim=1)))

    def _build_edge_inputs(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distance: torch.Tensor,
    ) -> torch.Tensor:
        edge_distance_embedding = self.distance_expansion(edge_distance)
        source_embedding = self.source_embedding(
            atomic_numbers[edge_index[0]],
        ).squeeze(1)
        target_embedding = self.target_embedding(
            atomic_numbers[edge_index[1]],
        ).squeeze(1)
        return torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding),
            dim=1,
        )

    def _run_xy_stack(
        self,
        x_message: torch.Tensor,
        x_edge: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_and_M_mapping: torch.Tensor,
        wigner_and_M_mapping_inv: torch.Tensor,
        node_offset: int = 0,
    ) -> torch.Tensor:
        xy_pair_message = None
        for i, xy_block in enumerate(self.xy_blocks):
            with record_function(f"xy message passing layer {i}"):
                xy_pair_message_res = xy_block(
                    x_message,
                    x_edge,
                    edge_distance,
                    edge_index,
                    wigner_and_M_mapping,
                    wigner_and_M_mapping_inv,
                    node_offset=node_offset,
                    pair_state=(
                        xy_pair_message
                        if self.xy_pair_state_mode == "recurrent"
                        else None
                    ),
                )
            if self.xy_pair_state_mode == "recurrent":
                xy_pair_message = xy_pair_message_res
            else:
                xy_pair_message = (
                    xy_pair_message_res
                    if xy_pair_message is None
                    else xy_pair_message + xy_pair_message_res
                )
        if xy_pair_message is None:
            raise ValueError("QHFlow3 clean requires at least one xy_block.")
        return self.xy_norm(xy_pair_message)

    def forward(self, data_dict: Any, ham_features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"].squeeze()
        data_dict["batch_full"] = data_dict["batch"]
        num_nodes = data_dict["atomic_numbers"].shape[0]
        data_dict["charge"] = torch.zeros(
            num_nodes, dtype=torch.long, device=data_dict["pos"].device
        )
        data_dict["spin"] = torch.zeros(
            num_nodes, dtype=torch.long, device=data_dict["pos"].device
        )

        _, node_feats_H, node_feats_H_init, node_feats_S = ham_features

        matrix_features = {
            "node_feats_H": node_feats_H,
            "node_feats_H_init": node_feats_H_init if self.use_block_H else None,
            "node_feats_S": node_feats_S if self.use_block_S else None,
        }
        for feature_name, feature in matrix_features.items():
            if feature is None:
                continue
            if feature.shape[0] != num_nodes:
                raise ValueError(
                    f"{feature_name} must contain one row per atom; got "
                    f"{feature.shape[0]} rows for {num_nodes} atoms."
                )

        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
        )

        node_feats_H = self.node_feats_H_embedding(node_feats_H)
        if self.use_block_H:
            node_feats_H_init = self.node_feats_H_init_embedding(node_feats_H_init)
        if self.use_block_S:
            node_feats_S = self.node_feats_S_embedding(node_feats_S)

        if "edge_index" not in data_dict:
            raise ValueError("QHFlow3 clean expects precomputed edge_index.")
        graph_dict = {
            "edge_index": data_dict["edge_index"],
            "edge_distance": data_dict["edge_distance"],
            "edge_distance_vec": data_dict["edge_distance_vec"],
            "node_offset": 0,
        }
        if gp_utils.initialized():
            graph_dict = self._init_gp_partitions(
                graph_dict, data_dict["atomic_numbers_full"]
            )
            node_partition = graph_dict["node_partition"]
            data_dict["atomic_numbers"] = data_dict["atomic_numbers_full"][
                node_partition
            ]
            data_dict["batch"] = data_dict["batch_full"][node_partition]
            node_feats_H = node_feats_H.index_select(0, node_partition)
            if self.use_block_H:
                node_feats_H_init = node_feats_H_init.index_select(0, node_partition)
            if self.use_block_S:
                node_feats_S = node_feats_S.index_select(0, node_partition)
        else:
            graph_dict["edge_distance_vec_full"] = graph_dict["edge_distance_vec"]
            graph_dict["edge_distance_full"] = graph_dict["edge_distance"]
            graph_dict["edge_index_full"] = graph_dict["edge_index"]

        if graph_dict["edge_index"].numel() == 0:
            raise ValueError(
                f"No edges found in input system, this means either you have a single atom in the system or the atoms are farther apart than the radius cutoff of the model of {self.cutoff} Angstroms. We don't know how to handle this case. Check the positions of system: {data_dict['pos']}"
            )

        with record_function("obtain wigner"):
            (wigner_and_M_mapping_full, wigner_and_M_mapping_inv_full) = (
                self._get_rotmat_and_wigner(
                    graph_dict["edge_distance_vec_full"],
                )
            )
            if gp_utils.initialized():
                wigner_and_M_mapping = wigner_and_M_mapping_full[
                    graph_dict["edge_partition"]
                ]
                wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full[
                    graph_dict["edge_partition"]
                ]
            else:
                wigner_and_M_mapping = wigner_and_M_mapping_full
                wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        # import pdb; pdb.set_trace()

        with record_function("atom embedding"):
            x_message = torch.zeros(
                data_dict["atomic_numbers"].shape[0],
                self.sph_feature_size,
                self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            x_message_original = self.sphere_embedding(data_dict["atomic_numbers"])
            x_message[:, 0, :] = x_message_original[:, 0, :]

        if self.use_time_embedding:
            time_message = get_time_embedding(data_dict["t"], self.sphere_channels)[data_dict["batch"]]
            x_message[:, 0, :] = x_message[:, 0, :] + time_message

        # ParamContraction emits one feature row per atom. ``batch`` contains
        # graph IDs and is only valid for graph-level conditions such as time.
        matrix_l_len = node_feats_H.shape[1]
        matrix_mix_list = [x_message_original[:, 0, :], node_feats_H[:, 0, :]]
        x_message[:, 0:matrix_l_len, :] = (
            x_message[:, 0:matrix_l_len, :] + node_feats_H[:, 0:matrix_l_len, :]
        )
        if self.use_block_H:
            matrix_mix_list.append(node_feats_H_init[:, 0, :])
            x_message[:, 0:matrix_l_len, :] = (
                x_message[:, 0:matrix_l_len, :]
                + node_feats_H_init[:, 0:matrix_l_len, :]
            )
        if self.use_block_S:
            matrix_mix_list.append(node_feats_S[:, 0, :])
            x_message[:, 0:matrix_l_len, :] = (
                x_message[:, 0:matrix_l_len, :] + node_feats_S[:, 0:matrix_l_len, :]
            )
        if self.use_time_embedding:
            matrix_mix_list.append(time_message)

        node_feats_matrix = self.mix_matrix(torch.cat(matrix_mix_list, dim=1))
        x_message[:, 0, :] = x_message[:, 0, :] + node_feats_matrix

        sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        # edge degree embedding
        with record_function("edge embedding"):
            x_edge = self._build_edge_inputs(
                data_dict["atomic_numbers_full"],
                graph_dict["edge_index"],
                graph_dict["edge_distance"],
            )
            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner_and_M_mapping_inv,
                graph_dict["node_offset"],
            )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            with record_function(f"message passing layer {i}"):
                x_message = self.blocks[i](
                    x_message,
                    x_edge,
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    wigner_and_M_mapping,
                    wigner_and_M_mapping_inv,
                    sys_node_embedding=sys_node_embedding,
                    node_offset=graph_dict["node_offset"],
                )

        pair_edge_index = data_dict.get("pair_edge_index", graph_dict["edge_index"])
        if pair_edge_index is graph_dict["edge_index"]:
            pair_edge_distance = graph_dict["edge_distance"]
            pair_x_edge = x_edge
            pair_wigner = wigner_and_M_mapping
            pair_wigner_inv = wigner_and_M_mapping_inv
        else:
            if gp_utils.initialized():
                raise NotImplementedError(
                    "separate QHFlow3 pair graphs are not implemented with graph parallelism."
                )
            pair_edge_vec = data_dict.get("pair_edge_distance_vec")
            if pair_edge_vec is None:
                pair_edge_vec = (
                    data_dict["pos"][pair_edge_index[0]]
                    - data_dict["pos"][pair_edge_index[1]]
                )
            pair_edge_distance = data_dict.get("pair_edge_distance")
            if pair_edge_distance is None:
                pair_edge_distance = pair_edge_vec.norm(dim=-1)
            pair_wigner, pair_wigner_inv = self._get_rotmat_and_wigner(pair_edge_vec)
            pair_x_edge = self._build_edge_inputs(
                data_dict["atomic_numbers_full"],
                pair_edge_index,
                pair_edge_distance,
            )

        pair_chunk_size = data_dict.get("pair_chunk_size")
        checkpoint_pair_chunks = bool(data_dict.get("checkpoint_pair_chunks", False))
        n_pair_edges = int(pair_edge_index.shape[1])
        if n_pair_edges == 0:
            raise ValueError("QHFlow3 clean requires at least one pair edge.")
        if pair_chunk_size is None or n_pair_edges <= int(pair_chunk_size):
            xy_pair_message = self._run_xy_stack(
                x_message,
                pair_x_edge,
                pair_edge_distance,
                pair_edge_index,
                pair_wigner,
                pair_wigner_inv,
                graph_dict["node_offset"],
            )
        else:
            xy_parts = []
            for start in range(0, n_pair_edges, int(pair_chunk_size)):
                stop = min(start + int(pair_chunk_size), n_pair_edges)
                args = (
                    x_message,
                    pair_x_edge[start:stop],
                    pair_edge_distance[start:stop],
                    pair_edge_index[:, start:stop],
                    pair_wigner[start:stop],
                    pair_wigner_inv[start:stop],
                    0,
                )
                if checkpoint_pair_chunks and torch.is_grad_enabled():
                    xy_part = torch.utils.checkpoint.checkpoint(
                        self._run_xy_stack,
                        *args,
                        use_reentrant=False,
                    )
                else:
                    xy_part = self._run_xy_stack(*args)
                xy_parts.append(xy_part)
            xy_pair_message = torch.cat(xy_parts, dim=0)

        # Final layer norm
        x_message = self.norm(x_message)

        out = {
            "node_embedding": x_message,
            "xy_embedding": xy_pair_message,
            "batch": data_dict["batch"],
            "node_attr_R_init": x_message_original.squeeze(),
            "node_attr_mixed": node_feats_matrix.squeeze(),
        }
        return out

    def _init_gp_partitions(self, graph_dict, atomic_numbers_full):
        """Graph Parallel
        This creates the required partial tensors for each rank given the full tensors.
        The tensors are split on the dimension along the node index using node_partition.
        """
        edge_index = graph_dict["edge_index"]
        edge_distance = graph_dict["edge_distance"]
        edge_distance_vec_full = graph_dict["edge_distance_vec"]

        node_partition = torch.tensor_split(
            torch.arange(len(atomic_numbers_full)).to(atomic_numbers_full.device),
            gp_utils.get_gp_world_size(),
        )[gp_utils.get_gp_rank()]

        assert (
            node_partition.numel() > 0
        ), "Looks like there is no atoms in this graph paralell partition. Cannot proceed"
        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),  # TODO: 0 or 1?
            )
        )[0]

        # full versions of data
        graph_dict["edge_distance_vec_full"] = edge_distance_vec_full
        graph_dict["edge_distance_full"] = edge_distance
        graph_dict["edge_index_full"] = edge_index
        graph_dict["edge_partition"] = edge_partition
        graph_dict["node_partition"] = node_partition

        # gp versions of data
        graph_dict["edge_index"] = edge_index[:, edge_partition]
        graph_dict["edge_distance"] = edge_distance[edge_partition]
        graph_dict["edge_distance_vec"] = edge_distance_vec_full[edge_partition]
        graph_dict["node_offset"] = node_partition.min().item()

        return graph_dict


class _ScalarChannelFiLM(nn.Module):
    """FiLM scalar or degree-structured channels with a zero-init start."""

    def __init__(
        self,
        cond_dim: int,
        hidden_size: int,
        mlp_hidden_dim: int | None = None,
        use_gate: bool = False,
        lmax: int | None = None,
        scale_sharing: str = "scalar",
    ) -> None:
        super().__init__()
        if scale_sharing not in {"scalar", "shared", "degree"}:
            raise ValueError(
                "scale_sharing must be 'scalar', 'shared', or 'degree', got "
                f"{scale_sharing!r}."
            )
        if scale_sharing in {"shared", "degree"} and lmax is None:
            raise ValueError("lmax is required when scale_sharing is not 'scalar'.")
        self.hidden_size = int(hidden_size)
        self.use_gate = bool(use_gate)
        self.lmax = None if lmax is None else int(lmax)
        self.scale_sharing = str(scale_sharing)
        self.scale_width = self.hidden_size
        if self.scale_sharing == "degree":
            self.scale_width *= int(self.lmax) + 1
        out_dim = self.scale_width + self.hidden_size + (1 if self.use_gate else 0)
        width = int(mlp_hidden_dim or hidden_size)
        self.net = nn.Sequential(
            nn.Linear(int(cond_dim), width),
            nn.SiLU(),
            nn.Linear(width, out_dim),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _scale_so3(self, scale: torch.Tensor) -> torch.Tensor:
        if self.scale_sharing == "scalar":
            return scale.unsqueeze(1)
        if self.scale_sharing == "shared":
            pieces = [
                scale.unsqueeze(1).expand(-1, 2 * degree + 1, -1)
                for degree in range(int(self.lmax) + 1)
            ]
            return torch.cat(pieces, dim=1)
        per_degree = scale.reshape(scale.shape[0], int(self.lmax) + 1, self.hidden_size)
        pieces = [
            per_degree[:, degree : degree + 1, :].expand(-1, 2 * degree + 1, -1)
            for degree in range(int(self.lmax) + 1)
        ]
        return torch.cat(pieces, dim=1)

    def _scale_flat(self, scale: torch.Tensor) -> torch.Tensor:
        if self.scale_sharing == "scalar":
            return scale
        if self.scale_sharing == "shared":
            pieces = [
                scale.unsqueeze(-1)
                .expand(-1, self.hidden_size, 2 * degree + 1)
                .reshape(scale.shape[0], -1)
                for degree in range(int(self.lmax) + 1)
            ]
            return torch.cat(pieces, dim=-1)
        per_degree = scale.reshape(scale.shape[0], int(self.lmax) + 1, self.hidden_size)
        pieces = [
            per_degree[:, degree, :]
            .unsqueeze(-1)
            .expand(-1, self.hidden_size, 2 * degree + 1)
            .reshape(scale.shape[0], -1)
            for degree in range(int(self.lmax) + 1)
        ]
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        cond: torch.Tensor,
        *,
        channel_dim: int = -1,
    ) -> torch.Tensor:
        if cond.shape[-1] != self.net[0].in_features:
            raise ValueError(
                f"conditioning dim mismatch: expected {self.net[0].in_features}, "
                f"got {cond.shape[-1]}",
            )

        params = self.net(cond.to(device=features.device, dtype=features.dtype))
        scale = params[..., : self.scale_width]
        shift = params[..., self.scale_width : self.scale_width + self.hidden_size]

        if channel_dim == -1:
            if self.scale_sharing == "scalar":
                scalar = features[..., : self.hidden_size]
                out = torch.cat(
                    [
                        scalar * (1.0 + scale) + shift,
                        features[..., self.hidden_size :],
                    ],
                    dim=-1,
                )
            else:
                flat_scale = self._scale_flat(scale)
                if flat_scale.shape[-1] > features.shape[-1]:
                    raise ValueError(
                        "degree/shared FiLM scale is wider than feature width: "
                        f"{flat_scale.shape[-1]} > {features.shape[-1]}",
                    )
                prefix = features[..., : flat_scale.shape[-1]]
                out = torch.cat(
                    [
                        prefix * (1.0 + flat_scale),
                        features[..., flat_scale.shape[-1] :],
                    ],
                    dim=-1,
                )
                out = out.clone()
                out[..., : self.hidden_size] = out[..., : self.hidden_size] + shift
        elif channel_dim == 1:
            if self.scale_sharing == "scalar":
                scalar = features[:, 0, :]
                first = (scalar * (1.0 + scale) + shift).unsqueeze(1)
                out = torch.cat([first, features[:, 1:, :]], dim=1)
            else:
                so3_scale = self._scale_so3(scale)
                if so3_scale.shape[1] > features.shape[1]:
                    raise ValueError(
                        "degree/shared FiLM scale has more components than features: "
                        f"{so3_scale.shape[1]} > {features.shape[1]}",
                    )
                prefix = features[:, : so3_scale.shape[1], :]
                out = torch.cat(
                    [
                        prefix * (1.0 + so3_scale),
                        features[:, so3_scale.shape[1] :, :],
                    ],
                    dim=1,
                )
                out = out.clone()
                out[:, 0, :] = out[:, 0, :] + shift
        else:
            raise ValueError(f"unsupported channel_dim={channel_dim}")

        if self.use_gate:
            gate = torch.tanh(params[..., -1:])
            while gate.dim() < out.dim():
                gate = gate.unsqueeze(1)
            out = out * (1.0 + gate)
        return out


class QHFlow3CleanFeatures(nn.Module):
    """QHFlow3 strong-conditioned feature backbone without QHFlow3 inheritance.

    Parity target: ``QHFlow3StrongCondFeatures`` with the current density
    settings (SelfNet/PairNet disabled, tensor-expansion head external).  The
    forward contract matches that target exactly: it emits
    ``BackboneOutput(node_feats=fii, edge_feats=fij)`` for a separate matrix
    head.  The implementation is intentionally linear in this file:

    1. prepare the same sorted full-edge batch as QHFlow3;
    2. contract the same diagonal matrix/overlap blocks;
    3. apply node/context/final FiLM conditioning;
    4. run the eSCN trunk without inactive legacy refinement layers;
    5. expose the final latent features to the tensor-expansion head.
    """

    def __init__(
        self,
        sh_lmax: int = 4,
        hidden_size: int = 128,
        bottle_hidden_size: int = 64,
        num_gnn_layers: int = 3,
        num_ham_gnn_layers: int = 2,
        max_radius: float = 12.0,
        radius_embed_dim: int = 32,
        escn_edge_channels: int = 128,
        escn_num_distance_basis: int = 512,
        use_block_S: bool = True,
        use_block_H: bool = False,
        basis: str = "def2-svp",
        basis_elements: list[int] | None = None,
        esen_max_radius: float = 15.0,
        expand_lmax: int | None = None,
        grid_resolution: int | None = None,
        init_diag_attr: str = "diagonal_init_dm",
        default_hamiltonian_input: str = "init_ham",
        muonize_output_projection: bool = False,
        module_dtype: str | None = "float32",
        time_condition_dim: int | None = None,
        matrix_condition_dim: int | None = None,
        condition_hidden_dim: int | None = None,
        use_node_time_conditioning: bool = True,
        use_node_matrix_conditioning: bool = True,
        use_context_conditioning: bool = True,
        use_final_feature_conditioning: bool = True,
        final_condition_gate: bool = True,
        final_condition_scale_sharing: str = "scalar",
    ) -> None:
        super().__init__()
        self._init_escn_trunk(
            sh_lmax=sh_lmax,
            hidden_size=hidden_size,
            bottle_hidden_size=bottle_hidden_size,
            num_gnn_layers=num_gnn_layers,
            num_ham_gnn_layers=num_ham_gnn_layers,
            max_radius=max_radius,
            radius_embed_dim=radius_embed_dim,
            escn_edge_channels=escn_edge_channels,
            escn_num_distance_basis=escn_num_distance_basis,
            use_block_S=use_block_S,
            use_block_H=use_block_H,
            basis=basis,
            basis_elements=basis_elements,
            esen_max_radius=esen_max_radius,
            expand_lmax=expand_lmax,
            grid_resolution=grid_resolution,
            init_diag_attr=init_diag_attr,
            default_hamiltonian_input=default_hamiltonian_input,
            muonize_output_projection=muonize_output_projection,
            module_dtype=module_dtype,
        )

        hidden_size = int(self.hidden_size)
        self.time_condition_dim = int(time_condition_dim or hidden_size)
        self.matrix_condition_dim = int(matrix_condition_dim or hidden_size)
        self.use_node_time_conditioning = bool(use_node_time_conditioning)
        self.use_node_matrix_conditioning = bool(use_node_matrix_conditioning)
        self.use_context_conditioning = bool(use_context_conditioning)

        node_cond_dim = (
            (self.time_condition_dim if self.use_node_time_conditioning else 0)
            + (self.matrix_condition_dim if self.use_node_matrix_conditioning else 0)
        )
        if node_cond_dim <= 0 and self.use_context_conditioning:
            raise ValueError("context conditioning needs at least one node condition source")

        self.node_condition_film = _ScalarChannelFiLM(
            cond_dim=max(node_cond_dim, 1),
            hidden_size=hidden_size,
            mlp_hidden_dim=condition_hidden_dim,
            use_gate=False,
        )
        self.node_context_film = _ScalarChannelFiLM(
            cond_dim=max(node_cond_dim, 1),
            hidden_size=hidden_size,
            mlp_hidden_dim=condition_hidden_dim,
            use_gate=False,
        )
        self.use_final_feature_conditioning = bool(use_final_feature_conditioning)
        self.final_condition_scale_sharing = str(final_condition_scale_sharing)
        self.final_node_condition_film = _ScalarChannelFiLM(
            cond_dim=max(node_cond_dim, 1),
            hidden_size=hidden_size,
            mlp_hidden_dim=condition_hidden_dim,
            use_gate=final_condition_gate,
            lmax=self.order,
            scale_sharing=self.final_condition_scale_sharing,
        )

    def _init_escn_trunk(
        self,
        sh_lmax: int = 4,
        hidden_size: int = 128,
        bottle_hidden_size: int = 32,
        num_gnn_layers: int = 3,
        num_ham_gnn_layers: int = 2,
        max_radius: float = 12.0,
        radius_embed_dim: int = 32,
        escn_edge_channels: int = 128,
        escn_num_distance_basis: int = 512,
        use_block_S: bool = True,
        use_block_H: bool = True,
        basis: str = "def2-svp",
        basis_elements: list[int] | None = None,
        esen_max_radius: float = 5.0,
        expand_lmax: int | None = None,
        grid_resolution: int | None = None,
        init_diag_attr: str = "diagonal_init_ham",
        default_hamiltonian_input: str = "init_ham",
        muonize_output_projection: bool = False,
        module_dtype: str | None = "float32",
    ) -> None:
        if default_hamiltonian_input not in {"init_ham", "zero"}:
            raise ValueError(
                "default_hamiltonian_input must be 'init_ham' | 'zero', "
                f"got {default_hamiltonian_input!r}.",
            )

        self.init_diag_attr = init_diag_attr
        self.default_hamiltonian_input = default_hamiltonian_input

        self.order = int(sh_lmax)
        self.expand_lmax = int(expand_lmax if expand_lmax is not None else sh_lmax)
        self.hidden_size = int(hidden_size)
        self.bottle_hidden_size = int(bottle_hidden_size)
        self.radius_embed_dim = int(radius_embed_dim)
        self.max_radius = float(max_radius)
        self.num_gnn_layers = int(num_gnn_layers)
        self.num_ham_gnn_layers = int(num_ham_gnn_layers)
        self.use_block_S = bool(use_block_S)
        self.use_block_H = bool(use_block_H)
        self.esen_max_radius = float(esen_max_radius)
        self.basis = basis
        self.basis_elements = None if basis_elements is None else [int(z) for z in basis_elements]
        self.grid_resolution = grid_resolution
        self.escn_edge_channels = escn_edge_channels
        self.escn_num_distance_basis = escn_num_distance_basis
        self.muonize_output_projection = bool(muonize_output_projection)

        if self.basis_elements is not None:
            raise ValueError(
                "The SC26 QHFlow3 port selects its fixed element set from "
                "basis; basis_elements overrides are not supported."
            )
        if basis == "def2-svp":
            self.output_irrep = o3.Irreps("3x0e + 2x1e + 1x2e")
            self.output_matrix_dim = 14
        elif basis == "def2-svp-nabla":
            self.output_irrep = o3.Irreps("5x0e + 4x1e + 3x2e")
            self.output_matrix_dim = 32
        elif basis == "def2-tzvp":
            self.output_irrep = o3.Irreps("5x0e + 5x1e + 2x2e + 1x3e")
            self.output_matrix_dim = 37
        else:
            raise ValueError(f"Invalid basis: {basis}")

        self.input_irrep = o3.Irreps(f"{self.hidden_size}x0e")
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hidden_size, order=self.order))
        self.expand_bottle_irrep = o3.Irreps(
            construct_o3irrps(self.bottle_hidden_size, order=self.expand_lmax),
        )
        self.hidden_irrep_base = o3.Irreps(
            construct_o3irrps_base(self.hidden_size, order=self.order),
        )

        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(
            self.radius_embed_dim,
            self.max_radius,
        )
        self.contraction_layer_H = ParamContraction(
            self.output_irrep,
            self.output_irrep,
            self.input_irrep,
        )
        if self.use_block_S:
            self.contraction_layer_S = ParamContraction(
                self.output_irrep,
                self.output_irrep,
                self.input_irrep,
            )
        if self.use_block_H:
            self.contraction_layer_H_init = ParamContraction(
                self.output_irrep,
                self.output_irrep,
                self.input_irrep,
            )

        self.node_attr_backbone = QHFlow3ESCNBackboneHam(
            lmax=self.order,
            mmax=self.order,
            sphere_channels=self.hidden_size,
            hidden_channels=self.hidden_size,
            num_layers=self.num_gnn_layers,
            use_block_S=self.use_block_S,
            use_block_H=self.use_block_H,
            use_time_embedding=True,
            num_ham_gnn_layers=self.num_ham_gnn_layers,
            cutoff=self.esen_max_radius,
            edge_channels=self.escn_edge_channels,
            num_distance_basis=self.escn_num_distance_basis,
            grid_resolution=self.grid_resolution,
        )
        output_projection_type = (
            MuonVisibleIrrepLinear
            if self.muonize_output_projection
            else Linear
        )
        self.output_ii = output_projection_type(
            self.hidden_irrep,
            self.expand_bottle_irrep,
        )
        self.output_ij = output_projection_type(
            self.hidden_irrep,
            self.expand_bottle_irrep,
        )

        self._device_initialized = False

        if module_dtype is not None:
            if module_dtype == "float64":
                self.double()
            elif module_dtype == "float32":
                self.float()
            else:
                raise ValueError(
                    "module_dtype must be None | 'float32' | 'float64', "
                    f"got {module_dtype!r}",
                )

    def _process_input_matrices(
        self,
        data: Any,
        H: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        node_feats_h = self.contraction_layer_H(H)
        node_feats_h_init = None
        if self.use_block_H:
            node_feats_h_init = self.contraction_layer_H_init(getattr(data, self.init_diag_attr))
        node_feats_s = None
        if self.use_block_S:
            node_feats_s = self.contraction_layer_S(getattr(data, "diagonal_overlap"))
        return node_feats_h, node_feats_h_init, node_feats_s

    def _ensure_device(self, device: torch.device) -> None:
        if not self._device_initialized:
            self.to(device)
            self._device_initialized = True

    @staticmethod
    def _bridge_batch(batch: Any) -> None:
        if batch.atoms.dim() == 1:
            batch.atoms = batch.atoms.unsqueeze(-1)
        if not hasattr(batch, "overlap") and hasattr(batch, "diagonal_overlap"):
            batch.overlap = batch.diagonal_overlap

        num_graphs = batch.ptr.shape[0] - 1
        existing_t = getattr(batch, "t", None)
        if existing_t is None:
            batch.t = torch.ones(num_graphs, device=batch.pos.device, dtype=batch.pos.dtype)
        elif existing_t.dim() == 0:
            batch.t = existing_t.expand(num_graphs).contiguous()
        elif existing_t.shape[0] == batch.atoms.shape[0]:
            batch.t = existing_t[batch.ptr[:-1]].contiguous()

    def _prepare_qhflow3_edge_batch_and_run(
        self,
        batch: Any,
        H: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        edge_index_orig = batch.edge_index_full
        num_nodes = batch.pos.shape[0]
        order = torch.argsort(edge_index_orig[1] * num_nodes + edge_index_orig[0])
        edge_index = edge_index_orig[:, order]
        inv_order = torch.argsort(order)

        batch["atomic_numbers"] = batch.atoms
        batch["edge_index"] = edge_index
        edge_distance_vec_xyz = (
            batch.pos[edge_index[0]] - batch.pos[edge_index[1]]
        )
        # MALOQ's matrix/AO convention represents Cartesian vectors as
        # (y, z, x). Match eSEN_Backbone's edge_dist[:, [2, 3, 1]]
        # conversion so geometry and matrix irreps transform under the same
        # SO(3) representation.
        batch["edge_distance_vec"] = edge_distance_vec_xyz[:, [1, 2, 0]]
        batch["edge_distance"] = batch["edge_distance_vec"].norm(dim=-1)
        batch["full_edge_index"] = edge_index
        batch["full_edge_distance_vec"] = batch["edge_distance_vec"]
        batch["full_edge_distance"] = batch["edge_distance"]
        batch["full_edge_attr"] = (
            self.distance_expansion(batch["full_edge_distance"].unsqueeze(-1))
            .squeeze()
            .type(batch.pos.type())
        )
        batch["transpose_edge_index"] = _transpose_indices_from_edge_index(
            edge_index,
            num_nodes=num_nodes,
        )
        batch["pbc"] = torch.zeros(
            len(batch),
            3,
            dtype=torch.bool,
            device=batch.pos.device,
        )

        node_feats_h, node_feats_h_init, node_feats_s = self._process_input_matrices(
            batch,
            H,
        )
        batch["node_feats_H"] = node_feats_h
        batch["node_feats_H_init"] = node_feats_h_init
        batch["node_feats_S"] = node_feats_s

        fii, fij = self._process_through_main_layers(batch)
        transpose_sorted = batch["transpose_edge_index"]
        return {
            "fii": fii,
            "fij": fij[inv_order],
            "node_attr_init": batch["node_attr_R_init"],
            "full_edge_index": edge_index_orig,
            "transpose_edge_index": order[transpose_sorted[inv_order]],
        }

    def _run_qhflow3_feature_path(
        self,
        batch: Any,
        H: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Mirror QHFlow3's feature-output path without legacy inheritance."""

        model_dtype = next(self.parameters()).dtype
        _cast_batch_floating_tensors(batch, model_dtype)
        self._ensure_device(batch.pos.device)
        self._bridge_batch(batch)
        if H is None:
            if self.default_hamiltonian_input == "zero":
                H = torch.zeros(
                    batch.atoms.shape[0],
                    self.output_matrix_dim,
                    self.output_matrix_dim,
                    device=batch.pos.device,
                    dtype=batch.pos.dtype,
                )
            elif hasattr(batch, "init_ham_t"):
                H = batch.init_ham_t
            elif hasattr(batch, "diagonal_init_ham"):
                H = batch.diagonal_init_ham
            else:
                H = torch.zeros(
                    batch.atoms.shape[0],
                    self.output_matrix_dim,
                    self.output_matrix_dim,
                    device=batch.pos.device,
                    dtype=batch.pos.dtype,
                )
        elif isinstance(H, (tuple, list)):
            H = H[0]
        if H.is_floating_point() and H.dtype != model_dtype:
            H = H.to(dtype=model_dtype)
        result = self._prepare_qhflow3_edge_batch_and_run(batch, H)
        return result

    @staticmethod
    def _build_edge_context(
        node_context: torch.Tensor,
        full_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        full_dst, full_src = full_edge_index
        return torch.cat([node_context[full_dst], node_context[full_src]], dim=-1)

    @staticmethod
    def _node_time_cond(data: Any, num_items: int) -> torch.Tensor | None:
        t_cond = getattr(data, "t_cond", None)
        if t_cond is None:
            return None
        if t_cond.shape[0] == num_items:
            return t_cond
        batch_index = getattr(data, "batch", None)
        ptr = getattr(data, "ptr", None)
        if batch_index is not None and ptr is not None and num_items == ptr.numel() - 1:
            return t_cond.index_select(0, ptr[:-1].to(device=t_cond.device))
        raise ValueError(
            "batch.t_cond must be per-node or per-graph for QHFlow3 time "
            f"conditioning; got shape {tuple(t_cond.shape)} for {num_items} items.",
        )

    @staticmethod
    def _node_matrix_cond(data: Any, num_items: int) -> torch.Tensor | None:
        h_cond = getattr(data, "h_cond", None)
        if h_cond is None:
            return None
        if h_cond.shape[0] == num_items:
            return h_cond
        ptr = getattr(data, "ptr", None)
        if ptr is not None and num_items == ptr.numel() - 1:
            return h_cond.index_select(0, ptr[:-1].to(device=h_cond.device))
        raise ValueError(
            "batch.h_cond must be per-node or per-graph for QHFlow3 matrix "
            f"conditioning; got shape {tuple(h_cond.shape)} for {num_items} items.",
        )

    @staticmethod
    def _zeros_like_condition(
        reference: torch.Tensor,
        length: int,
        dim: int,
    ) -> torch.Tensor:
        return reference.new_zeros(length, int(dim))

    def _node_condition(self, data: Any, num_items: int) -> torch.Tensor | None:
        time_cond = (
            self._node_time_cond(data, num_items)
            if self.use_node_time_conditioning
            else None
        )
        matrix_cond = (
            self._node_matrix_cond(data, num_items)
            if self.use_node_matrix_conditioning
            else None
        )
        if time_cond is None and matrix_cond is None:
            return None
        reference = time_cond if time_cond is not None else matrix_cond
        if reference is None:
            return None
        parts = []
        if self.use_node_time_conditioning:
            parts.append(
                time_cond
                if time_cond is not None
                else self._zeros_like_condition(reference, num_items, self.time_condition_dim)
            )
        if self.use_node_matrix_conditioning:
            parts.append(
                matrix_cond
                if matrix_cond is not None
                else self._zeros_like_condition(reference, num_items, self.matrix_condition_dim)
            )
        return torch.cat(parts, dim=-1)

    def _apply_node_condition(
        self,
        data: Any,
        node_feats_h: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._node_condition(data, node_feats_h.shape[0])
        if cond is None:
            return node_feats_h
        return self.node_condition_film(node_feats_h, cond, channel_dim=1)

    def _apply_context_condition(self, data: Any, node_context: torch.Tensor) -> torch.Tensor:
        if not self.use_context_conditioning:
            return node_context
        cond = self._node_condition(data, node_context.shape[0])
        if cond is None:
            return node_context
        return self.node_context_film(node_context, cond, channel_dim=-1)

    def _apply_final_node_condition(self, data: Any, features: torch.Tensor) -> torch.Tensor:
        if not self.use_final_feature_conditioning or features.shape[-1] < self.hidden_size:
            return features
        cond = self._node_condition(data, features.shape[0])
        if cond is None:
            return features
        return self.final_node_condition_film(features, cond, channel_dim=-1)

    def _process_through_main_layers(self, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        node_attr_r = None
        num_nodes = data["node_feats_H"].shape[0]
        node_feats_h = data["node_feats_H"].reshape(num_nodes, -1, self.hidden_size)
        node_feats_h = self._apply_node_condition(data, node_feats_h)
        if self.use_block_H:
            node_feats_h_init = data["node_feats_H_init"].reshape(
                num_nodes,
                -1,
                self.hidden_size,
            )
        else:
            node_feats_h_init = None
        if self.use_block_S:
            node_feats_s = data["node_feats_S"].reshape(num_nodes, -1, self.hidden_size)
        else:
            node_feats_s = None

        middle_features = self.node_attr_backbone(
            data,
            [node_attr_r, node_feats_h, node_feats_h_init, node_feats_s],
        )
        node_attr_r = _escn_to_e3nn_flat(middle_features["node_embedding"], self.order)
        data["node_attr_R"] = node_attr_r
        node_context = middle_features["node_embedding"].narrow(1, 0, 1).squeeze()
        data["node_attr_R_init"] = self._apply_context_condition(data, node_context)

        fii = self._apply_final_node_condition(data, node_attr_r)
        fii = self.output_ii(fii)
        fij = self.output_ij(_escn_to_e3nn_flat(middle_features["xy_embedding"], self.order))
        return fii, fij

    def forward(self, batch: Any, H: torch.Tensor | None = None) -> BackboneOutput:
        result = self._run_qhflow3_feature_path(batch, H=H)
        required = ("fii", "fij", "node_attr_init")
        if any(key not in result for key in required):
            raise RuntimeError(
                f"{self.__class__.__name__} needs eSCN feature outputs "
                f"{required}, but got {sorted(result.keys())}.",
            )
        node_context = result["node_attr_init"]
        full_edge_index = result["full_edge_index"]
        edge_context = self._build_edge_context(node_context, full_edge_index)
        return BackboneOutput(
            node_feats=result["fii"],
            edge_feats=result["fij"],
            extra={
                "node_context": node_context,
                "edge_context": edge_context,
                "full_edge_index": result.get("full_edge_index"),
            },
        )


def _flat_to_esen_embeddings(
    flat: torch.Tensor,
    lmax: int,
    channels: int,
) -> torch.Tensor:
    """Convert QHFlow3 channel-major irreps to the native MALOQ head layout."""
    embeddings = flat.new_zeros(flat.shape[0], (lmax + 1) ** 2, channels)
    for degree in range(lmax + 1):
        start = (degree**2) * channels
        width = channels * (2 * degree + 1)
        block = flat[:, start : start + width].reshape(
            flat.shape[0], channels, 2 * degree + 1
        )
        embeddings[:, degree**2 : (degree + 1) ** 2, :] = block.transpose(1, 2)
    return embeddings


def _orbital_masks_for_basis(
    basis: str,
) -> tuple[dict[int, torch.Tensor], int]:
    """Map each element's compact AO order into QHFlow3's padded AO grid."""
    orbital_bases = {
        "def2-svp": basis_sets.orbital_basis_def2_svp_QM7,
        "def2-svp-nabla": basis_sets.orbital_basis_def2_svp_nabla,
    }
    try:
        orbital_basis = orbital_bases[basis]
    except KeyError as exc:
        raise ValueError(
            f"The MALOQ QHFlow3 bridge does not define AO masks for {basis!r}."
        ) from exc

    max_shell_counts = {
        degree: max(shells.count(degree) for shells in orbital_basis.values())
        for degree in range(max(max(shells) for shells in orbital_basis.values()) + 1)
    }
    degree_starts: dict[int, int] = {}
    matrix_dim = 0
    for degree, shell_count in max_shell_counts.items():
        degree_starts[degree] = matrix_dim
        matrix_dim += shell_count * (2 * degree + 1)

    masks: dict[int, torch.Tensor] = {}
    for atomic_number, shells in orbital_basis.items():
        shell_offsets = {degree: 0 for degree in max_shell_counts}
        indices: list[int] = []
        for degree in shells:
            width = 2 * degree + 1
            start = degree_starts[degree] + shell_offsets[degree] * width
            indices.extend(range(start, start + width))
            shell_offsets[degree] += 1
        masks[int(atomic_number)] = torch.tensor(indices, dtype=torch.long)
    return masks, matrix_dim


class QHFlow3MaloqBackbone(QHFlow3CleanFeatures):
    """Headless QHFlow3 trunk bridged to MALOQ's native matrix loaders.

    Absolute-target runs use zero matrix features and optionally the real
    overlap. Delta-learning runs condition the trunk on the source initial
    density, initial Hamiltonian, and optionally the overlap. The primary
    matrix matches the target type and the other initial matrix is the
    auxiliary block input. Pair features follow the loader's local directed
    graph, so matrix blocks not emitted by that graph are omitted exactly as
    they are for MALOQ backbones.
    """

    def __init__(self, **kwargs: Any) -> None:
        grid_ffn_chunk_size = kwargs.pop("grid_ffn_chunk_size", 512)
        delta_learning = bool(kwargs.pop("delta_learning", False))
        delta_target = kwargs.pop("delta_target", "density_matrix")
        if delta_target not in {"fock_matrix", "density_matrix"}:
            raise ValueError(
                "delta_target must be 'fock_matrix' or 'density_matrix'."
            )
        kwargs.setdefault("basis", "def2-svp")
        kwargs.setdefault(
            "default_hamiltonian_input",
            "init_ham" if delta_learning else "zero",
        )
        kwargs.setdefault("init_diag_attr", "diagonal_aux_matrix")
        kwargs.setdefault("use_block_S", True)
        kwargs.setdefault("use_block_H", delta_learning)
        # The lmax=4 default 10x11 grid leaves visible SO(3) aliasing in the
        # pair GridAtomwise path.  A 48x48 grid keeps the unmodified QHFlow3
        # nonlinearity while bringing both node and pair covariance below the
        # float32 1e-4 regression tolerance for general 3D rotations.
        kwargs.setdefault("grid_resolution", 48)
        super().__init__(**kwargs)
        self._orbital_masks, bridge_matrix_dim = _orbital_masks_for_basis(
            self.basis
        )
        if bridge_matrix_dim != self.output_matrix_dim:
            raise RuntimeError(
                f"QHFlow3 {self.basis} AO mask dimension {bridge_matrix_dim} "
                f"does not match its contraction dimension {self.output_matrix_dim}."
            )
        self.delta_learning = delta_learning
        self.delta_target = delta_target
        self.grid_ffn_chunk_size = (
            None if grid_ffn_chunk_size is None else int(grid_ffn_chunk_size)
        )
        if self.grid_ffn_chunk_size is not None:
            if self.grid_ffn_chunk_size <= 0:
                raise ValueError("grid_ffn_chunk_size must be positive.")
            for module in self.modules():
                if isinstance(module, GridAtomwise):
                    module.grid_ffn_chunk_size = self.grid_ffn_chunk_size

    def _matrix_blocks(
        self,
        batch: Any,
        attribute: str,
        matrix_label: str,
    ) -> torch.Tensor:
        matrices = getattr(batch, attribute, None)
        if matrices is None:
            raise ValueError(
                f"QHFlow3 requires {attribute} from the native matrix loader."
            )
        if not isinstance(matrices, (list, tuple)):
            matrices = [matrices]
        if not hasattr(batch, "ptr"):
            raise ValueError("QHFlow3 requires PyG batch pointers.")

        ptr = batch.ptr.detach().cpu().tolist()
        if len(matrices) != len(ptr) - 1:
            raise ValueError(
                f"Expected {len(ptr) - 1} {matrix_label} matrices, "
                f"got {len(matrices)}."
            )

        all_blocks = []
        for graph_index, matrix in enumerate(matrices):
            atoms = batch.atomic_numbers[ptr[graph_index] : ptr[graph_index + 1]]
            atoms_cpu = [int(value) for value in atoms.detach().cpu().tolist()]
            masks = []
            for atomic_number in atoms_cpu:
                if atomic_number not in self._orbital_masks:
                    raise ValueError(
                        f"QHFlow3 basis {self.basis!r} does not support "
                        f"Z={atomic_number}."
                    )
                masks.append(self._orbital_masks[atomic_number])

            sizes = [int(mask.numel()) for mask in masks]
            offsets = np.cumsum([0, *sizes]).tolist()
            matrix_tensor = torch.as_tensor(
                matrix,
                device=batch.pos.device,
                dtype=batch.pos.dtype,
            )
            expected = offsets[-1]
            if tuple(matrix_tensor.shape) != (expected, expected):
                raise ValueError(
                    f"{matrix_label} shape {tuple(matrix_tensor.shape)} does not match "
                    f"the {expected} {self.basis} orbitals."
                )

            blocks = matrix_tensor.new_zeros(
                len(atoms_cpu), self.output_matrix_dim, self.output_matrix_dim
            )
            for atom_index, mask_cpu in enumerate(masks):
                mask = mask_cpu.to(device=matrix_tensor.device)
                start, stop = offsets[atom_index], offsets[atom_index + 1]
                blocks[atom_index][mask[:, None], mask[None, :]] = matrix_tensor[
                    start:stop, start:stop
                ]
            all_blocks.append(blocks)
        return torch.cat(all_blocks, dim=0)

    def _overlap_blocks(self, batch: Any) -> torch.Tensor:
        return self._matrix_blocks(batch, "overlap_matrix", "overlap")

    @staticmethod
    def _validate_local_pair_graph(batch: Any) -> None:
        edge_index = batch.edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("QHFlow3 edge_index must have shape (2, E).")
        if edge_index.shape[1] == 0:
            raise ValueError("QHFlow3 requires at least one local directed pair.")

        num_nodes = int(batch.atomic_numbers.shape[0])
        if (
            int(edge_index.min().item()) < 0
            or int(edge_index.max().item()) >= num_nodes
        ):
            raise ValueError("QHFlow3 edge_index contains an out-of-range atom index.")

        source, target = edge_index
        if bool((source == target).any().item()):
            raise ValueError("QHFlow3 local pair graph must not contain self edges.")
        source_graph = batch.batch.index_select(0, source)
        target_graph = batch.batch.index_select(0, target)
        if not bool((source_graph == target_graph).all().item()):
            raise ValueError("QHFlow3 local pair edges must stay within each molecule.")

        edge_hash = source * num_nodes + target
        if int(torch.unique(edge_hash).numel()) != int(edge_hash.numel()):
            raise ValueError(
                "QHFlow3 local pair graph contains duplicate directed edges."
            )
        _transpose_indices_from_edge_index(edge_index, num_nodes=num_nodes)

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        self._validate_local_pair_graph(batch)
        original_atomic_numbers = batch.atomic_numbers
        original_edge_index = batch.edge_index
        batch.atoms = batch.atomic_numbers
        batch.edge_index_full = batch.edge_index
        if self.use_block_S:
            batch.diagonal_overlap = self._overlap_blocks(batch)
        initial_density_blocks = None
        if self.delta_learning:
            initial_density_blocks = self._matrix_blocks(
                batch,
                "initial_density_matrix",
                "initial density",
            )
            initial_hamiltonian_blocks = self._matrix_blocks(
                batch,
                "initial_hamiltonian",
                "initial Hamiltonian",
            )
            if self.delta_target == "density_matrix":
                primary_matrix_blocks = initial_density_blocks
                batch.diagonal_aux_matrix = initial_hamiltonian_blocks
            else:
                primary_matrix_blocks = initial_hamiltonian_blocks
                batch.diagonal_aux_matrix = initial_density_blocks
        else:
            primary_matrix_blocks = None

        try:
            output = super().forward(batch, H=primary_matrix_blocks)
            return {
                "node_embeddings": _flat_to_esen_embeddings(
                    output.node_feats,
                    self.expand_lmax,
                    self.bottle_hidden_size,
                ),
                "edge_embeddings": _flat_to_esen_embeddings(
                    output.edge_feats,
                    self.expand_lmax,
                    self.bottle_hidden_size,
                ),
            }
        finally:
            # The clean QHFlow3 path sorts edges for its pair layers and expands
            # atomic numbers to [N, 1].  Its returned pair features are restored
            # to the incoming order; restore the shared batch as well so the
            # native MALOQ head sees the loader's original contract.
            batch.atomic_numbers = original_atomic_numbers
            batch.edge_index = original_edge_index
