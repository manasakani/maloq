"""Workflow adapter for symmetry-reduced node and edge matrix outputs."""

from __future__ import annotations

from maloq.train_utils.training_workflow_v2 import TrainingWorkflowV2Fixed

from .head import SymmetryReducedMuonFockHead


class CoupledHamiltonianSymmetryWorkflow(TrainingWorkflowV2Fixed):
    """Matched V2 workflow with fixed coupled-irrep symmetry reduction."""

    feature_profile = "node_intra_edge_pair_irrep_reduction_v1"

    def _validate_backbone_feature_config(self):
        super()._validate_backbone_feature_config()
        config = self.config
        if config["loss_target"] not in {"fock_matrix", "density_matrix"}:
            raise ValueError(
                "Coupled Hamiltonian symmetry requires a matrix-valued target."
            )
        if config["head_type"] != "maloq_muon":
            raise ValueError(
                "Coupled Hamiltonian symmetry requires head_type='maloq_muon'."
            )
        enabled_legacy_reductions = [
            name
            for name in ("reduce_node", "reduce_node_intra", "reduce_edge")
            if config[name]
        ]
        if enabled_legacy_reductions:
            raise ValueError(
                "The experimental head owns its fixed symmetry reduction; "
                "legacy reduction flags must remain disabled, got "
                f"{enabled_legacy_reductions}."
            )

    def _build_matrix_head(
        self,
        *,
        irreps_in,
        required_irreps,
        head_channels,
        orb_basis,
        ls_list,
    ):
        return SymmetryReducedMuonFockHead(
            irreps_in=irreps_in,
            irreps_out=required_irreps,
            lmax=required_irreps.lmax,
            sphere_channels=head_channels,
            ls_list=ls_list,
            open_shell=self.config["open_shell"],
            orbital_basis=orb_basis,
        )

    def _backbone_summary(self, backbone):
        summary = super()._backbone_summary(backbone)
        summary.update(
            output_symmetry=self.feature_profile,
            node_symmetry="upper_triangle_plus_even_diagonal_irreps",
            edge_symmetry="reverse_pair_alpha_beta_irreps",
            symmetry_space="reduced_coupled_irreps",
            symmetry_reduction=True,
        )
        return summary


__all__ = ["CoupledHamiltonianSymmetryWorkflow"]
