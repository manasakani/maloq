# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Clean, fixed eSEN implementation for MALOQ-NTE-V2.

This module intentionally has no compatibility selectors for historical NTE
or QHFlow3 ablations.  The architecture is the one selected by the NablaDFT
``Edge2+InitEnv+Atom2Direct`` experiment:

* QHF matrix conditioning is applied once at the input.
* Native NTE NodeBlocks run before a recurrent edge stack.
* The initial edge block returns EdgeWise directly and adds a scaled
  AtomWise residual.
* Each refinement block scales EdgeWise locally, returns AtomWise directly,
  and adds the incoming edge state only after AtomWise.
* Node and edge outputs use separate norms and degreewise irrep projections.

Historical selector-based configurations use the explicit :mod:`maloq.experimental.nte_qhflow3_composition` workflow.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear
from fairchem.core.common.registry import registry

from .common.rotation import eulers_to_wigner, init_edge_rot_euler_angles
from .common.so3 import CoefficientMapping, SO3_Grid
from .esen_block_v2 import (
    EdgeRefinementBlock,
    InitialEdgeBlock,
    NodeBlock,
)
from .nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .nn.matrix_embedding import MatrixEmbedding
from .nn.radial import GaussianSmearing, PolynomialEnvelope, RadialMLP
from .nn.so3_layers import SO3_Linear


MALOQ_NTE_V2_ARCHITECTURE = "MALOQ-NTE-V2"


class InitialEdgeEmbedding(nn.Module):
    """Envelope-weighted edge-degree initialization for nodes and pairs."""

    def __init__(
        self,
        *,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        cutoff: float,
        mapping,
        output_mask,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mapping
        self.m_0_num_coefficients = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialMLP(self.edge_channels_list)
        self.rescale_factor = 5.0
        self.out_mask = output_mask
        self.cutoff = float(cutoff)
        self.use_envelope = True
        self.envelope = PolynomialEnvelope(exponent=5)

    def _edge_features(
        self,
        radial_features,
        edge_distance,
        wigner_inv,
        dtype,
    ):
        m0_features = self.rad_func(radial_features).reshape(
            -1,
            self.m_0_num_coefficients,
            self.sphere_channels,
        )
        padding = torch.zeros(
            (
                m0_features.shape[0],
                self.m_all_num_coefficents - self.m_0_num_coefficients,
                self.sphere_channels,
            ),
            device=m0_features.device,
            dtype=m0_features.dtype,
        )
        edge_features = torch.cat((m0_features, padding), dim=1)
        edge_features = torch.einsum(
            "nac,ab->nbc",
            edge_features,
            self.mappingReduced.to_m,
        )
        edge_features = torch.bmm(wigner_inv, edge_features)
        envelope = self.envelope(edge_distance / self.cutoff)
        return (edge_features * envelope.view(-1, 1, 1)).to(dtype)

    def node_embeddings(
        self,
        node_state,
        radial_features,
        edge_distance,
        edge_index,
        wigner_inv,
    ):
        edge_features = self._edge_features(
            radial_features,
            edge_distance,
            wigner_inv,
            node_state.dtype,
        )
        node_state.index_add_(
            0,
            edge_index[1],
            edge_features / self.rescale_factor,
        )
        return node_state

    def edge_embeddings(
        self,
        node_state,
        radial_features,
        edge_distance,
        wigner_inv,
    ):
        return self._edge_features(
            radial_features,
            edge_distance,
            wigner_inv,
            node_state.dtype,
        )


class _MuonVisibleIrrepLinear(nn.Module):
    """e3nn Linear with one explicit degree/output/input weight tensor."""

    def __init__(self, irreps_in: Irreps, irreps_out: Irreps) -> None:
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
        input_channels, output_channels = path_shapes[0]
        path_major_weight = torch.randn(
            len(path_shapes),
            input_channels,
            output_channels,
        )
        self.weight = nn.Parameter(path_major_weight.transpose(1, 2).contiguous())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat_weight = self.weight.transpose(1, 2).reshape(-1)
        return self.linear(features, flat_weight)


class IrrepProjection(nn.Module):
    """Degreewise projection between native eSEN channel layouts."""

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
        self.linear = _MuonVisibleIrrepLinear(irreps_in, irreps_out)

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight

    def _native_to_e3nn(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                features[:, degree**2 : (degree + 1) ** 2, :]
                .transpose(1, 2)
                .reshape(features.shape[0], -1)
                for degree in range(self.lmax + 1)
            ],
            dim=1,
        )

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
        projected = self.linear(self._native_to_e3nn(features))
        return self._e3nn_to_native(projected)


def _build_output_projection(
    in_features: int,
    out_features: int,
    lmax: int,
) -> IrrepProjection:
    """Build the fixed degreewise projection with stable initialization."""
    with torch.random.fork_rng(devices=[]):
        projection = IrrepProjection(
            in_features,
            out_features,
            lmax,
        )
    # Preserve the reference initialization sequence used by the promoted
    # architecture so matched reruns start from the same seeded distribution.
    SO3_Linear(
        in_features,
        out_features,
        lmax=lmax,
        bias=False,
    )
    return projection


@registry.register_model("maloq_nte_v2_backbone")
class MaloqNTEV2Backbone(nn.Module):
    """Independent fixed backbone for MALOQ-NTE-V2."""

    architecture = MALOQ_NTE_V2_ARCHITECTURE

    def __init__(
        self,
        irreps_out,
        *,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        cutoff: float = 10.0,
        edge_channels: int = 128,
        num_distance_basis: int = 512,
        num_layers: int = 3,
        num_edge_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        gaussian_width: float = 1.0,
        open_shell: bool = False,
        wigner_backend: str = "torch",
        output_sphere_channels: int = 64,
        conditioning_basis: str = "def2-svp",
        conditioning_delta_learning: bool = False,
        conditioning_delta_target: str = "fock_matrix",
    ) -> None:
        super().__init__()
        del irreps_out
        if wigner_backend not in {"torch", "triton"}:
            raise ValueError(
                f"wigner_backend must be 'torch' or 'triton', got {wigner_backend!r}."
            )
        if output_sphere_channels <= 0:
            raise ValueError("output_sphere_channels must be positive.")

        self.wigner_backend = wigner_backend
        self._wigner_buf = None
        self.max_num_elements = int(max_num_elements)
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.sphere_channels = int(sphere_channels)
        self.output_sphere_channels = int(output_sphere_channels)
        self.hidden_channels = int(hidden_channels)
        self.norm_type = norm_type
        self.cutoff = float(cutoff)
        self.edge_channels = int(edge_channels)
        self.num_distance_basis = int(num_distance_basis)
        self.gaussian_width = float(gaussian_width)
        self.num_layers = int(num_layers)
        self.num_edge_layers = int(num_edge_layers)
        if self.num_edge_layers < 2:
            raise ValueError("MALOQ-NTE-V2 requires at least two edge layers.")
        self.num_spins = 2 if open_shell else 1

        # Matrix conditioning is an explicit component of the fixed V2 model.
        self.input_conditioner = MatrixEmbedding(
            mode="qhflow3_exact",
            basis=conditioning_basis,
            hidden_size=self.sphere_channels,
            delta_learning=conditioning_delta_learning,
            delta_target=conditioning_delta_target,
        )

        jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for degree in range(self.lmax + 1):
            self.register_buffer(f"Jd_{degree}", jd_list[degree])
        self.sph_feature_size = (self.lmax + 1) ** 2
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)
        self.SO3_grid = nn.ModuleDict(
            {
                "lmax_lmax": SO3_Grid(
                    self.lmax,
                    self.lmax,
                    resolution=grid_resolution,
                    rescale=True,
                ),
                "lmax_mmax": SO3_Grid(
                    self.lmax,
                    self.mmax,
                    resolution=grid_resolution,
                    rescale=True,
                ),
            }
        )

        self.sphere_embedding = nn.Embedding(
            self.max_num_elements,
            self.sphere_channels,
        )
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

        self.distance_expansion = GaussianSmearing(
            0.0,
            self.cutoff,
            self.num_distance_basis,
            self.gaussian_width,
        )
        self.source_embedding = nn.Embedding(
            self.max_num_elements,
            self.edge_channels,
        )
        self.target_embedding = nn.Embedding(
            self.max_num_elements,
            self.edge_channels,
        )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]
        self.edge_degree_embedding = InitialEdgeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            edge_channels_list=self.edge_channels_list,
            cutoff=self.cutoff,
            mapping=self.mappingReduced,
            output_mask=self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
                self.lmax, self.mmax
            ),
        )

        block_kwargs = {
            "sphere_channels": self.sphere_channels,
            "hidden_channels": self.hidden_channels,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "mapping": self.mappingReduced,
            "grids": self.SO3_grid,
            "edge_channels_list": self.edge_channels_list,
            "cutoff": self.cutoff,
            "norm_type": self.norm_type,
        }
        self.node_blocks = nn.ModuleList(
            [NodeBlock(**block_kwargs) for _ in range(self.num_layers)]
        )
        self.edge_blocks = nn.ModuleList(
            [
                InitialEdgeBlock(**block_kwargs),
                *(
                    EdgeRefinementBlock(**block_kwargs)
                    for _ in range(self.num_edge_layers - 1)
                ),
            ]
        )
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )
        self.edge_norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )
        self.node_output_projection = _build_output_projection(
            self.sphere_channels,
            self.output_sphere_channels,
            self.lmax,
        )
        self.edge_output_projection = _build_output_projection(
            self.sphere_channels,
            self.output_sphere_channels,
            self.lmax,
        )

    def _get_rotmat_and_wigner(self, edge_distance_vecs):
        jd_buffers = [
            getattr(self, f"Jd_{degree}").type(edge_distance_vecs.dtype)
            for degree in range(self.lmax + 1)
        ]
        if self.wigner_backend == "triton":
            from .triton_kernels import edge_vec_to_wigner_fused

            num_edges = edge_distance_vecs.shape[0]
            out_dim = (self.lmax + 1) ** 2
            if self._wigner_buf is None or num_edges > self._wigner_buf.shape[0]:
                self._wigner_buf = torch.zeros(
                    num_edges,
                    out_dim,
                    out_dim,
                    device=edge_distance_vecs.device,
                    dtype=torch.float32,
                )
            wigner = edge_vec_to_wigner_fused(
                edge_distance_vecs,
                jd_buffers,
                lmax=self.lmax,
                out=self._wigner_buf[:num_edges],
            )
        else:
            euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
            wigner = eulers_to_wigner(
                euler_angles,
                0,
                self.lmax,
                jd_buffers,
            )
        return wigner, torch.transpose(wigner, 1, 2).contiguous()

    def _run_message_passing(
        self,
        node_state,
        edge_state,
        x_edge,
        graph,
        wigner,
        wigner_inv,
    ):
        block_args = (
            x_edge,
            graph["edge_distance"],
            graph["edge_index"],
            wigner,
            wigner_inv,
        )
        for block in self.node_blocks:
            node_state = block(node_state, edge_state, *block_args)
        for block in self.edge_blocks:
            edge_state = block(node_state, edge_state, *block_args)
        return node_state, edge_state

    def forward(self, batch, batch_index=None, output_dir=None):
        del batch_index, output_dir
        if getattr(batch, "distributed_graph_training", False):
            raise ValueError("MALOQ-NTE-V2 supports data-parallel training only.")

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)
        edge_distance = batch.edge_attr[:, 0]
        graph = {
            "edge_index": edge_index,
            "edge_distance": edge_distance,
            "edge_distance_vec": batch.edge_attr[:, [2, 3, 1]],
            "partition": None,
        }
        wigner, wigner_inv = self._get_rotmat_and_wigner(graph["edge_distance_vec"])

        node_state = torch.zeros(
            batch.pos.shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=batch.pos.device,
            dtype=batch.pos.dtype,
        )
        molecule_indices = torch.repeat_interleave(
            torch.arange(
                len(batch.num_atoms_in_molecule),
                device=batch.pos.device,
            ),
            batch.num_atoms_in_molecule.to(batch.pos.device),
        )
        element_embedding = self.sphere_embedding(batch.atomic_numbers)
        atom_charges = batch.charge[molecule_indices] + self.abs_max_charge
        atom_spins = batch.spin_multiplicity[molecule_indices]
        base_scalar = self.scalar_node_embedding(
            torch.cat(
                [
                    element_embedding,
                    self.charge_embedding(atom_charges),
                    self.spin_embedding(atom_spins),
                ],
                dim=-1,
            )
        )
        node_state[:, 0, :] = self.input_conditioner(
            batch,
            atom_embedding=element_embedding,
            base_scalar=base_scalar,
            molecule_indices=molecule_indices,
        )

        distance_embedding = self.distance_expansion(edge_distance)
        source_embedding = self.source_embedding(batch.atomic_numbers[edge_index[0]])
        target_embedding = self.target_embedding(batch.atomic_numbers[edge_index[1]])
        x_edge = torch.cat(
            (source_embedding, distance_embedding, target_embedding),
            dim=1,
        )
        node_state = self.edge_degree_embedding.node_embeddings(
            node_state,
            x_edge,
            edge_distance,
            edge_index,
            wigner_inv,
        )
        edge_state = self.edge_degree_embedding.edge_embeddings(
            node_state,
            x_edge,
            edge_distance,
            wigner_inv,
        )
        node_state, edge_state = self._run_message_passing(
            node_state,
            edge_state,
            x_edge,
            graph,
            wigner,
            wigner_inv,
        )
        node_state = self.node_output_projection(self.norm(node_state))
        edge_state = self.edge_output_projection(self.edge_norm(edge_state))
        return {
            "node_embeddings": node_state,
            "edge_embeddings": edge_state,
            **graph,
        }

    @property
    def num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        no_weight_decay = []
        parameter_names = {name for name, _ in self.named_parameters()}
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
                    global_name = f"{module_name}.{parameter_name}"
                    if global_name in parameter_names:
                        no_weight_decay.append(global_name)
        return set(no_weight_decay)


__all__ = [
    "InitialEdgeEmbedding",
    "IrrepProjection",
    "MALOQ_NTE_V2_ARCHITECTURE",
    "MaloqNTEV2Backbone",
]
