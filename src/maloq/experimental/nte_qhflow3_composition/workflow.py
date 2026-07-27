"""Explicit workflow adapters for configurable NTE/QHFlow3 experiments."""

from __future__ import annotations

from maloq.train_utils.training_workflow import (
    TrainingWorkflow as CanonicalTrainingWorkflow,
)
from maloq.train_utils.training_workflow_fixed import (
    TrainingWorkflowFixed as CanonicalTrainingWorkflowFixed,
)

from .backbone import ConfigurableNTEBackbone
from .config import PROFILE_ID, SELECTOR_DEFAULTS, validate_selector_config


class _ConfigurableNTEWorkflowMixin:
    DEFAULTS = CanonicalTrainingWorkflow.DEFAULTS | SELECTOR_DEFAULTS

    def _validate_backbone_feature_config(self):
        validate_selector_config(self.config)
        if self.config.get("backbone_type") != "esen":
            super()._validate_backbone_feature_config()

    def _needs_delta_auxiliary_matrix(self):
        return (
            super()._needs_delta_auxiliary_matrix()
            or self.config["nte_input_conditioning"] == "qhflow3_exact"
        )

    def _uses_matrix_input_conditioning(self):
        return self.config["nte_input_conditioning"] != "none"

    def _backbone_summary(self, backbone):
        summary = super()._backbone_summary(backbone)
        if self.config.get("backbone_type") != "esen":
            return summary
        c = self.config
        summary.update(
            experimental_feature="nte_qhflow3_composition",
            experimental_profile=PROFILE_ID,
            message_passing_schedule=c["message_passing_schedule"],
            initial_edge_state_mode=c["initial_edge_state_mode"],
            initial_edge_degree_envelope=bool(
                getattr(backbone, "initial_edge_degree_envelope", False)
            ),
            post_atomwise_edge_residual_layers=list(
                getattr(backbone, "post_atomwise_edge_residual_layers", ())
            ),
            gate_act_type=c["gate_act_type"],
            residual_update_scale_mode=c["residual_update_scale_mode"],
            residual_update_scale_init=c["residual_update_scale_init"],
            residual_update_scale_log_range=(
                c["residual_update_scale_log_range"]
            ),
            unscaled_node_layers=list(c["unscaled_node_layers"]),
            repeat_system_embedding_each_node_block=bool(
                c["repeat_system_embedding_each_node_block"]
            ),
            node_stack_mode=c["node_stack_mode"],
            edge_stack_mode=c["edge_stack_mode"],
            qhflow3_layer_gaussian_width=c["qhflow3_layer_gaussian_width"],
            qhflow3_layer_grid_ffn_chunk_size=(
                c["qhflow3_layer_grid_ffn_chunk_size"]
            ),
            qhflow3_exact_pair_rng_aligned=bool(
                c["qhflow3_exact_pair_rng_aligned"]
            ),
            edge_atom_norm_type=c["edge_atom_norm_type"],
            edge_post_residual_norm_type=c["edge_post_residual_norm_type"],
            direct_edgewise_layers=list(c["direct_edgewise_layers"]),
            direct_atomwise_layers=list(c["direct_atomwise_layers"]),
            edge_atomwise_output_mode=c["edge_atomwise_output_mode"],
            edge_norm1_position=c["edge_norm1_position"],
            nte_output_projection_mode=c["nte_output_projection_mode"],
            output_norm_sharing=c["output_norm_sharing"],
            nte_output_projection_rng_contract=(
                "legacy_so3_linear_aligned"
                if c["nte_output_projection_mode"] == "qhflow3_irrep_linear"
                else "native_so3_linear"
            ),
            nte_input_conditioning=c["nte_input_conditioning"],
        )
        return summary

    def _build_esen_backbone(self, required_irreps):
        c = self.config
        delta_learning = c.get("delta_learning", False)
        return ConfigurableNTEBackbone(
            required_irreps,
            sphere_channels=c["l_embedding_dim"],
            hidden_channels=c["hidden_dim"],
            lmax=required_irreps.lmax,
            mmax=required_irreps.lmax,
            cutoff=c["rcut_gaussian"],
            grid_resolution=c["esen_grid_resolution"],
            edge_channels=c["l_embedding_dim"],
            num_layers=c["num_mp_layers"],
            act_type="gate",
            mlp_type=c["mlp_type"],
            gate_act_type=c["gate_act_type"],
            num_distance_basis=c["num_distance_basis"],
            gaussian_width=c["gaussian_width"],
            include_edges=c["include_edges"],
            open_shell=c["open_shell"],
            atom_scalar_embedding_mode=c["atom_scalar_embedding_mode"],
            wigner_backend=c.get("wigner_backend", "torch"),
            distributed_graph_training=c["distribute_graphs"],
            message_type=c["message_type"],
            message_passing_schedule=c["message_passing_schedule"],
            initial_edge_state_mode=c["initial_edge_state_mode"],
            num_edge_layers=c["num_edge_layers"],
            output_sphere_channels=c["output_l_embedding_dim"],
            nte_output_projection_mode=c["nte_output_projection_mode"],
            output_norm_sharing=c["output_norm_sharing"],
            use_edge_envelope=c["use_edge_envelope"],
            use_edge_scalar_modulation=c["use_edge_scalar_modulation"],
            residual_update_scale_mode=c["residual_update_scale_mode"],
            residual_update_scale_init=c["residual_update_scale_init"],
            residual_update_scale_log_range=c["residual_update_scale_log_range"],
            unscaled_node_layers=c["unscaled_node_layers"],
            repeat_system_embedding_each_node_block=(
                c["repeat_system_embedding_each_node_block"]
            ),
            node_stack_mode=c["node_stack_mode"],
            edge_stack_mode=c["edge_stack_mode"],
            qhflow3_layer_gaussian_width=c["qhflow3_layer_gaussian_width"],
            qhflow3_layer_grid_ffn_chunk_size=(
                c["qhflow3_layer_grid_ffn_chunk_size"]
            ),
            qhflow3_exact_pair_rng_aligned=c["qhflow3_exact_pair_rng_aligned"],
            edge_atom_norm_type=c["edge_atom_norm_type"],
            edge_post_residual_norm_type=c["edge_post_residual_norm_type"],
            direct_edgewise_layers=c["direct_edgewise_layers"],
            direct_atomwise_layers=c["direct_atomwise_layers"],
            edge_atomwise_output_mode=c["edge_atomwise_output_mode"],
            edge_norm1_position=c["edge_norm1_position"],
            input_conditioning=c["nte_input_conditioning"],
            conditioning_basis=(
                "def2-svp-nabla"
                if c["dataset_name"] == "nablaDFT"
                else "def2-svp"
            ),
            conditioning_delta_learning=delta_learning,
            conditioning_delta_target=c["loss_target"],
        ).to(self.device)


class TrainingWorkflow(_ConfigurableNTEWorkflowMixin, CanonicalTrainingWorkflow):
    """Training workflow that explicitly enables the ablation backbone."""


class TrainingWorkflowFixed(
    _ConfigurableNTEWorkflowMixin,
    CanonicalTrainingWorkflowFixed,
):
    """Restart-safe workflow for fresh feature-owned experiment runs."""
