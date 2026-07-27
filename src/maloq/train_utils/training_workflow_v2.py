# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Matched MALOQ/NTEV2/QHFlow3 comparison workflow.

The canonical :mod:`training_workflow` remains the original MALOQ execution
path.  This module opts into the two comparison backbones and the
Muon-visible matrix head without exposing historical NTE selector branches.
"""

from __future__ import annotations

from ..helm.esen_osh_v2 import MaloqNTEV2Backbone
from ..helm.nn.muon_fock_head import MuonFockIrrepsHead
from ..helm.qhflow3 import QHFlow3Backbone
from .training_workflow import TrainingWorkflow
from .training_workflow_fixed import TrainingWorkflowFixedMixin


MALOQ_NTE_V2_BACKBONE_TYPE = "maloq_nte_v2"
QHFLOW3_BACKBONE_TYPE = "qhflow3"


class TrainingWorkflowV2(TrainingWorkflow):
    """Workflow for controlled MALOQ, NTEV2, and QHFlow3 comparisons."""

    SUPPORTED_BACKBONE_TYPES = frozenset(
        {"esen", MALOQ_NTE_V2_BACKBONE_TYPE, QHFLOW3_BACKBONE_TYPE}
    )
    SUPPORTED_HEAD_TYPES = frozenset({"maloq", "maloq_muon"})
    DEFAULTS = TrainingWorkflow.DEFAULTS | {
        "num_edge_layers": None,
        "output_l_embedding_dim": None,
        "qhflow3_max_radius": 12.0,
        "qhflow3_radius_embed_dim": 32,
        "qhflow3_grid_resolution": 48,
        "qhflow3_grid_ffn_chunk_size": 512,
        "qhflow3_use_overlap": True,
        "qhflow3_muonize_output_projection": False,
        "muon_output_projection_policy": "shape_muon",
    }

    def _configured_edge_layers(self) -> int:
        configured = self.config.get("num_edge_layers")
        return int(self.config["num_mp_layers"] if configured is None else configured)

    def _supports_atom_scalar_embedding(self):
        return self.config["backbone_type"] == "esen"

    def _validate_backbone_feature_config(self):
        super()._validate_backbone_feature_config()
        c = self.config
        backbone_type = c["backbone_type"]
        edge_layers = self._configured_edge_layers()

        if edge_layers <= 0:
            raise ValueError("num_edge_layers must be positive.")
        if backbone_type == "esen" and edge_layers != int(c["num_mp_layers"]):
            raise ValueError(
                "Original MALOQ has one edge update per message-passing "
                "layer; num_edge_layers must equal num_mp_layers."
            )
        if backbone_type == MALOQ_NTE_V2_BACKBONE_TYPE and edge_layers < 2:
            raise ValueError("MALOQ-NTE-V2 requires at least two edge layers.")

        if backbone_type in {
            MALOQ_NTE_V2_BACKBONE_TYPE,
            QHFLOW3_BACKBONE_TYPE,
        }:
            if c["distribute_graphs"]:
                raise ValueError(
                    f"{backbone_type} supports data parallelism only; "
                    "set distribute_graphs=False."
                )
            if "matrix" not in c["loss_target"]:
                raise ValueError(
                    f"{backbone_type} requires a matrix-valued loss target."
                )
            output_channels = c.get("output_l_embedding_dim")
            if output_channels is None or int(output_channels) <= 0:
                raise ValueError(
                    f"{backbone_type} requires positive output_l_embedding_dim."
                )

        if c["head_type"] == "maloq_muon":
            if "matrix" not in c["loss_target"]:
                raise ValueError("head_type='maloq_muon' requires a matrix target.")
            if c["optimizer_type"] != "muon":
                raise ValueError(
                    "head_type='maloq_muon' requires optimizer_type='muon'."
                )
            if c["reduce_edge"]:
                raise ValueError(
                    "The Muon-visible MALOQ head requires reduce_edge=False."
                )

        projection_policy = c["muon_output_projection_policy"]
        if projection_policy not in {"shape_muon", "adamw"}:
            raise ValueError(
                "muon_output_projection_policy must be 'shape_muon' or 'adamw'."
            )
        if projection_policy == "adamw" and backbone_type != MALOQ_NTE_V2_BACKBONE_TYPE:
            raise ValueError(
                "muon_output_projection_policy='adamw' is defined only for "
                "MALOQ-NTE-V2 output projections."
            )

        if c["qhflow3_muonize_output_projection"]:
            if backbone_type != QHFLOW3_BACKBONE_TYPE:
                raise ValueError(
                    "qhflow3_muonize_output_projection requires "
                    "backbone_type='qhflow3'."
                )
            if c["optimizer_type"] != "muon":
                raise ValueError(
                    "qhflow3_muonize_output_projection requires optimizer_type='muon'."
                )
        grid_resolution = c["qhflow3_grid_resolution"]
        if grid_resolution is not None and int(grid_resolution) <= 0:
            raise ValueError("qhflow3_grid_resolution must be positive.")
        grid_chunk = c["qhflow3_grid_ffn_chunk_size"]
        if grid_chunk is not None and int(grid_chunk) <= 0:
            raise ValueError("qhflow3_grid_ffn_chunk_size must be positive.")

    def _uses_matrix_input_conditioning(self):
        return self.config["backbone_type"] in {
            MALOQ_NTE_V2_BACKBONE_TYPE,
            QHFLOW3_BACKBONE_TYPE,
        }

    def _needs_delta_auxiliary_matrix(self):
        return self.config["backbone_type"] in {
            MALOQ_NTE_V2_BACKBONE_TYPE,
            QHFLOW3_BACKBONE_TYPE,
        }

    def _build_backbone(self, required_irreps):
        c = self.config
        backbone_type = c["backbone_type"]
        delta_learning = bool(c.get("delta_learning", False))
        edge_layers = self._configured_edge_layers()
        basis = "def2-svp-nabla" if c["dataset_name"] == "nablaDFT" else "def2-svp"

        if backbone_type == "esen":
            return super()._build_backbone(required_irreps)
        if backbone_type == MALOQ_NTE_V2_BACKBONE_TYPE:
            return MaloqNTEV2Backbone(
                required_irreps,
                sphere_channels=c["l_embedding_dim"],
                hidden_channels=c["hidden_dim"],
                lmax=required_irreps.lmax,
                mmax=required_irreps.lmax,
                cutoff=c["rcut_gaussian"],
                grid_resolution=c["esen_grid_resolution"],
                edge_channels=c["l_embedding_dim"],
                num_layers=c["num_mp_layers"],
                num_edge_layers=edge_layers,
                num_distance_basis=c["num_distance_basis"],
                gaussian_width=c["gaussian_width"],
                open_shell=c["open_shell"],
                wigner_backend=c.get("wigner_backend", "torch"),
                output_sphere_channels=c["output_l_embedding_dim"],
                conditioning_basis=basis,
                conditioning_delta_learning=delta_learning,
                conditioning_delta_target=c["loss_target"],
            ).to(self.device)
        if backbone_type == QHFLOW3_BACKBONE_TYPE:
            return QHFlow3Backbone(
                sh_lmax=required_irreps.lmax,
                hidden_size=c["l_embedding_dim"],
                bottle_hidden_size=c["output_l_embedding_dim"],
                num_gnn_layers=c["num_mp_layers"],
                num_ham_gnn_layers=edge_layers,
                max_radius=c["qhflow3_max_radius"],
                radius_embed_dim=c["qhflow3_radius_embed_dim"],
                escn_edge_channels=c["hidden_dim"],
                escn_num_distance_basis=c["num_distance_basis"],
                esen_max_radius=c["rcut_gaussian"],
                grid_resolution=c["qhflow3_grid_resolution"],
                grid_ffn_chunk_size=c["qhflow3_grid_ffn_chunk_size"],
                basis=basis,
                delta_learning=delta_learning,
                delta_target=c["loss_target"],
                default_hamiltonian_input=("init_ham" if delta_learning else "zero"),
                use_block_S=c["qhflow3_use_overlap"],
                use_block_H=delta_learning,
                muonize_output_projection=(c["qhflow3_muonize_output_projection"]),
            ).to(self.device)
        raise AssertionError(f"Unhandled backbone_type={backbone_type!r}")

    def _head_channels(self, backbone):
        if self.config["backbone_type"] == "esen":
            return super()._head_channels(backbone)
        return int(self.config["output_l_embedding_dim"])

    def _build_matrix_head(
        self,
        *,
        irreps_in,
        required_irreps,
        head_channels,
        orb_basis,
        ls_list,
    ):
        if self.config["head_type"] == "maloq":
            return super()._build_matrix_head(
                irreps_in=irreps_in,
                required_irreps=required_irreps,
                head_channels=head_channels,
                orb_basis=orb_basis,
                ls_list=ls_list,
            )
        c = self.config
        return MuonFockIrrepsHead(
            irreps_in=irreps_in,
            irreps_out=required_irreps,
            lmax=required_irreps.lmax,
            sphere_channels=head_channels,
            reduce_edge=c["reduce_edge"],
            open_shell=c["open_shell"],
            ls_list=ls_list,
            reduce_node=c["reduce_node"],
            reduce_node_intra=c["reduce_node_intra"],
            orbital_basis=orb_basis,
        )

    def _collect_output_projection_adamw_parameters(self, backbone):
        if self.config["backbone_type"] != MALOQ_NTE_V2_BACKBONE_TYPE:
            return []
        parameters = []
        for module_name in (
            "node_output_projection",
            "edge_output_projection",
        ):
            module = getattr(backbone, module_name, None)
            weight = getattr(module, "weight", None)
            if weight is not None and weight.requires_grad:
                parameters.append(weight)
        return parameters

    def _architecture_name(self, backbone):
        if self.config["backbone_type"] == QHFLOW3_BACKBONE_TYPE:
            return "QHFlow3"
        return super()._architecture_name(backbone)

    def _edge_layer_count(self, backbone):
        if self.config["backbone_type"] == "esen":
            return super()._edge_layer_count(backbone)
        return self._configured_edge_layers()

    def _backbone_summary(self, backbone):
        c = self.config
        backbone_type = c["backbone_type"]
        if backbone_type == "esen":
            return super()._backbone_summary(backbone)
        if backbone_type == MALOQ_NTE_V2_BACKBONE_TYPE:
            edge_layers = self._configured_edge_layers()
            return {
                "message_passing_schedule": "node_then_edge",
                "initial_edge_state_mode": "edge_degree",
                "initial_edge_degree_envelope": True,
                "post_atomwise_edge_residual_layers": list(range(2, edge_layers + 1)),
                "mlp_type": "grid",
                "matrix_conditioning": "qhflow3_exact",
                "output_projection": "qhflow3_irrep_linear",
                "output_norm_sharing": "separate",
                "esen_grid_resolution": c["esen_grid_resolution"],
            }
        return {
            "message_passing_schedule": "qhflow3_node_then_pair",
            "initial_edge_state_mode": None,
            "mlp_type": "grid",
            "esen_grid_resolution": None,
            "qhflow3_primary_matrix_input": (
                (
                    "initial_density_matrix"
                    if c["loss_target"] == "density_matrix"
                    else "initial_hamiltonian"
                )
                if c.get("delta_learning", False)
                else "zero"
            ),
            "qhflow3_auxiliary_matrix_input": (
                (
                    "initial_hamiltonian"
                    if c["loss_target"] == "density_matrix"
                    else "initial_density_matrix"
                )
                if c.get("delta_learning", False)
                else None
            ),
            "qhflow3_basis": backbone.basis,
            "qhflow3_overlap_input": (
                "native_loader_atom_diagonal_blocks"
                if c["qhflow3_use_overlap"]
                else "disabled"
            ),
            "qhflow3_grid_resolution": (
                None
                if c["qhflow3_grid_resolution"] is None
                else int(c["qhflow3_grid_resolution"])
            ),
            "qhflow3_grid_ffn_chunk_size": (c["qhflow3_grid_ffn_chunk_size"]),
            "qhflow3_output_projection_optimizer": (
                "muon" if c["qhflow3_muonize_output_projection"] else "adamw"
            ),
        }


class TrainingWorkflowV2Fixed(
    TrainingWorkflowFixedMixin,
    TrainingWorkflowV2,
):
    """Comparison workflow with atomic epoch-boundary resume."""
