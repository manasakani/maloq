"""Symmetry-reduced Muon head for onsite and reverse-edge matrices."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from e3nn.o3 import Irreps
from e3nn.o3 import Linear as E3Linear
from torch import Tensor, nn

from maloq.helm.esen_osh import Fock_Irreps_Head


class _SemanticIrrepLinear(nn.Module):
    """Store degree-wise e3nn maps as one Muon-visible semantic matrix."""

    SEMANTIC_LAYOUT_VERSION = 1

    def __init__(
        self,
        irreps_out: Irreps | str,
        input_channels: int,
        input_lmax: int,
        source_layers: Sequence[E3Linear],
    ) -> None:
        super().__init__()
        self.irreps_out = Irreps(irreps_out)
        self.input_channels = int(input_channels)
        self.input_lmax = int(input_lmax)
        if len(source_layers) != self.input_lmax + 1:
            raise ValueError(
                "Expected one source e3nn linear per input degree, got "
                f"{len(source_layers)} for lmax={self.input_lmax}."
            )

        rows_by_degree: dict[int, list[int]] = {
            degree: [] for degree in range(self.input_lmax + 1)
        }
        semantic_row = 0
        for multiplicity, irrep in self.irreps_out:
            if irrep.l > self.input_lmax:
                raise ValueError(
                    f"Output degree {irrep.l} exceeds input lmax={self.input_lmax}."
                )
            rows_by_degree[irrep.l].extend(
                range(semantic_row, semantic_row + multiplicity)
            )
            semantic_row += multiplicity
        all_rows = [row for rows in rows_by_degree.values() for row in rows]
        if sorted(all_rows) != list(range(self.irreps_out.num_irreps)):
            raise RuntimeError("Semantic output rows are not a complete bijection.")

        weight = source_layers[0].weight.new_empty(
            self.irreps_out.num_irreps,
            self.input_channels,
        )
        scalar_bias: Tensor | None = None
        external_layers = []
        degrees = []
        for degree, rows in rows_by_degree.items():
            if not rows:
                continue
            source = source_layers[degree]
            expected_out = len(rows)
            expected_in_irreps = Irreps(f"{self.input_channels}x{degree}e")
            expected_out_irreps = Irreps(f"{expected_out}x{degree}e")
            if source.irreps_in != expected_in_irreps:
                raise ValueError(
                    f"Unexpected degree-{degree} input irreps {source.irreps_in}."
                )
            if source.irreps_out != expected_out_irreps:
                raise ValueError(
                    f"Unexpected degree-{degree} output irreps {source.irreps_out}."
                )
            instruction_indices = [
                index
                for index, instruction in enumerate(source.instructions)
                if instruction.i_in >= 0
            ]
            if len(instruction_indices) != 1:
                raise ValueError(
                    "Symmetry-reduced Muon head expects one e3nn instruction "
                    f"for degree {degree}."
                )
            source_view = source.weight_view_for_instruction(instruction_indices[0])
            if tuple(source_view.shape) != (self.input_channels, expected_out):
                raise ValueError(
                    f"Unexpected degree-{degree} weight shape {source_view.shape}."
                )
            weight[rows] = source_view.detach().T

            source_bias = getattr(source, "bias", None)
            source_bias_count = 0 if source_bias is None else source_bias.numel()
            use_bias = degree == 0 and source_bias_count == expected_out
            if degree == 0 and source_bias_count not in {0, expected_out}:
                raise ValueError(
                    "Unexpected scalar bias size "
                    f"{source_bias_count}; expected 0 or {expected_out}."
                )
            if use_bias:
                scalar_bias = source_bias.detach().clone()

            self.register_buffer(
                f"_rows_l{degree}",
                torch.tensor(rows, dtype=torch.long),
                persistent=False,
            )
            external_layers.append(
                E3Linear(
                    expected_in_irreps,
                    expected_out_irreps,
                    biases=use_bias,
                    internal_weights=False,
                    shared_weights=True,
                )
            )
            degrees.append(degree)

        self.weight = nn.Parameter(weight)
        if scalar_bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(scalar_bias)
        self.external_layers = nn.ModuleList(external_layers)
        self._degrees = tuple(degrees)
        self.register_buffer(
            "_semantic_layout_version",
            torch.tensor(self.SEMANTIC_LAYOUT_VERSION, dtype=torch.int64),
        )

    def forward(self, features: Tensor) -> Tensor:
        expected_dim = self.input_channels * (self.input_lmax + 1) ** 2
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(
                "Semantic irrep features must have shape "
                f"[items, {expected_dim}], got {tuple(features.shape)}."
            )

        outputs = []
        for layer_index, degree in enumerate(self._degrees):
            rows = getattr(self, f"_rows_l{degree}")
            start = degree**2 * self.input_channels
            width = (2 * degree + 1) * self.input_channels
            degree_features = features[:, start : start + width]
            degree_weight = self.weight.index_select(0, rows).T.reshape(-1)
            degree_bias = self.bias if degree == 0 else None
            outputs.append(
                self.external_layers[layer_index](
                    degree_features,
                    degree_weight,
                    degree_bias,
                )
            )
        return torch.cat(outputs, dim=-1)


class SymmetryReducedMuonFockHead(Fock_Irreps_Head):
    """Fixed symmetry-adapted head with no public reduction branches.

    Onsite blocks learn the unique orbital upper triangle and only even-L
    diagonal channels. Reverse-directed edges learn exchange-even alpha and
    exchange-odd beta channels. The canonical coupled-irrep reconstruction
    expands both representations to the full matrix target.
    """

    symmetry_profile = "node_intra_edge_pair_irrep_reduction_v1"

    def __init__(
        self,
        *,
        irreps_in,
        irreps_out,
        lmax: int,
        sphere_channels: int,
        ls_list: Sequence[int] | Tensor,
        open_shell: bool,
        orbital_basis,
    ) -> None:
        super().__init__(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            lmax=lmax,
            sphere_channels=sphere_channels,
            reduce_edge=True,
            ls_list=ls_list,
            open_shell=open_shell,
            reduce_node=True,
            reduce_node_intra=True,
            orbital_basis=orbital_basis,
        )

        node_layers = nn.ModuleList(
            [
                _SemanticIrrepLinear(
                    self.irreps_nodereduced,
                    self.sphere_channels,
                    self.lmax,
                    self.node_lin_out_layers[spin],
                )
                for spin in range(self.num_spins)
            ]
        )
        edge_alpha_layers = nn.ModuleList(
            [
                _SemanticIrrepLinear(
                    self.irreps_edgereduced_alpha,
                    self.sphere_channels,
                    self.lmax,
                    self.edge_lin_out_layers["alpha"][spin],
                )
                for spin in range(self.num_spins)
            ]
        )
        edge_beta_layers = nn.ModuleList(
            [
                _SemanticIrrepLinear(
                    self.irreps_edgereduced_beta,
                    self.sphere_channels,
                    self.lmax,
                    self.edge_lin_out_layers["beta"][spin],
                )
                for spin in range(self.num_spins)
            ]
        )
        del self.node_lin_out_layers
        del self.edge_lin_out_layers
        self.node_semantic_layers = node_layers
        self.edge_alpha_semantic_layers = edge_alpha_layers
        self.edge_beta_semantic_layers = edge_beta_layers

    def semantic_matrix_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only node/edge path-by-channel matrices for Muon."""
        for layers in (
            self.node_semantic_layers,
            self.edge_alpha_semantic_layers,
            self.edge_beta_semantic_layers,
        ):
            for layer in layers:
                yield layer.weight

    def gate_matrix_parameters(self) -> Iterator[nn.Parameter]:
        """The scalar gate remains on the auxiliary AdamW route."""
        return iter(())

    def process(self, x, node_or_edge, spin):
        x_scalars = x[:, : self.sphere_channels]
        x_nonscalars = x[:, self.sphere_channels :]
        all_scalars = self.lin_scalars_learnable[spin](x_scalars)
        transformed_l0_scalars = all_scalars[:, : self.sphere_channels]
        gating_scalars = torch.abs(all_scalars[:, self.sphere_channels :])
        x_gated = self.gate[spin](torch.cat([gating_scalars, x_nonscalars], dim=1))
        x_gated = torch.cat([transformed_l0_scalars, x_gated], dim=1)

        if node_or_edge == "node":
            return self.node_semantic_layers[spin](x_gated)
        if node_or_edge == "edge_alpha":
            return self.edge_alpha_semantic_layers[spin](x_gated)
        if node_or_edge == "edge_beta":
            return self.edge_beta_semantic_layers[spin](x_gated)
        raise ValueError(f"Unsupported symmetry-reduced output kind {node_or_edge!r}.")


__all__ = ["SymmetryReducedMuonFockHead"]
