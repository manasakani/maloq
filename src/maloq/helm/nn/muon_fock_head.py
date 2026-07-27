# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Muon-compatible reparameterization of the native MALOQ matrix head."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from e3nn.o3 import Irreps
from e3nn.o3 import Linear as E3Linear
from torch import nn

from ..esen_osh import Fock_Irreps_Head


class SemanticIrrepLinear(nn.Module):
    """Apply degree-wise e3nn linears stored as one path-by-channel matrix.

    ``e3nn.o3.Linear`` stores its shared weights as flat vectors. That is
    functionally correct, but it hides the output-path/input-channel matrix
    from Muon. This module preserves e3nn's external-weight forward exactly
    while materializing the parameter as ``(semantic path, input channel)``.
    Rows follow the original block-first output-irrep order and are explicitly
    gathered for degree-first execution.
    """

    SEMANTIC_LAYOUT_VERSION = 1

    def __init__(
        self,
        irreps_out: Irreps | str,
        input_channels: int,
        source_layers: Sequence[E3Linear],
    ) -> None:
        super().__init__()
        self.irreps_out = Irreps(irreps_out)
        self.input_channels = int(input_channels)
        if len(source_layers) != self.irreps_out.lmax + 1:
            raise ValueError(
                "Expected one source e3nn linear per output degree, got "
                f"{len(source_layers)} for lmax={self.irreps_out.lmax}."
            )

        rows_by_degree: dict[int, list[int]] = {
            degree: [] for degree in range(self.irreps_out.lmax + 1)
        }
        semantic_row = 0
        for multiplicity, irrep in self.irreps_out:
            rows_by_degree[irrep.l].extend(
                range(semantic_row, semantic_row + multiplicity)
            )
            semantic_row += multiplicity
        all_rows = [row for rows in rows_by_degree.values() for row in rows]
        if sorted(all_rows) != list(range(self.irreps_out.num_irreps)):
            raise RuntimeError("MALOQ semantic head rows are not a complete bijection.")

        weight = source_layers[0].weight.new_empty(
            self.irreps_out.num_irreps,
            self.input_channels,
        )
        scalar_bias = source_layers[0].weight.new_empty(0)
        external_layers = []
        self._degrees = tuple(
            degree for degree, rows in rows_by_degree.items() if rows
        )
        for degree in self._degrees:
            rows = rows_by_degree[degree]
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
            weight_instruction_indices = [
                index
                for index, instruction in enumerate(source.instructions)
                if instruction.i_in >= 0
            ]
            if len(weight_instruction_indices) != 1:
                raise ValueError(
                    "Muon-compatible MALOQ expects one e3nn instruction per degree."
                )
            source_view = source.weight_view_for_instruction(
                weight_instruction_indices[0]
            )
            if tuple(source_view.shape) != (self.input_channels, expected_out):
                raise ValueError(
                    f"Unexpected degree-{degree} weight shape {source_view.shape}."
                )
            weight[rows] = source_view.detach().T
            if degree == 0:
                scalar_bias = source.bias.detach().clone()
            self.register_buffer(
                f"_rows_l{degree}",
                torch.tensor(rows, dtype=torch.long),
                persistent=False,
            )
            external_layers.append(
                E3Linear(
                    expected_in_irreps,
                    expected_out_irreps,
                    biases=degree == 0,
                    internal_weights=False,
                    shared_weights=True,
                )
            )

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(scalar_bias)
        self.external_layers = nn.ModuleList(external_layers)
        self.register_buffer(
            "_semantic_layout_version",
            torch.tensor(self.SEMANTIC_LAYOUT_VERSION, dtype=torch.int64),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected_dim = self.input_channels * (self.irreps_out.lmax + 1) ** 2
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(
                "Semantic MALOQ features must have shape "
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


class MuonFockIrrepsHead(Fock_Irreps_Head):
    """Native gated MALOQ head with Muon-visible semantic output matrices."""

    SEMANTIC_MUON_ROUTING = "semantic_global_node_edge"
    SEMANTIC_GATE_ROUTING = "semantic_scalar_gate"

    def __init__(self, *args, muonize_gate: bool = False, **kwargs) -> None:
        if kwargs.get("reduce_edge", False):
            raise ValueError("maloq_muon currently requires reduce_edge=False.")
        super().__init__(*args, **kwargs)
        self.muonize_gate = bool(muonize_gate)

        if self.muonize_gate:
            gate_layers = nn.ModuleList(
                [
                    SemanticIrrepLinear(
                        source.irreps_out,
                        self.sphere_channels,
                        (source,),
                    )
                    for source in self.lin_scalars_learnable
                ]
            )
            del self.lin_scalars_learnable
            self.gate_semantic_layers = gate_layers

        if self.reduce_node:
            node_layers = nn.ModuleList(
                [
                    SemanticIrrepLinear(
                        self.irreps_nodereduced,
                        self.sphere_channels,
                        self.node_lin_out_layers[spin],
                    )
                    for spin in range(self.num_spins)
                ]
            )
            edge_layers = nn.ModuleList(
                [
                    SemanticIrrepLinear(
                        self.irreps_out,
                        self.sphere_channels,
                        self.edge_lin_out_layers[spin],
                    )
                    for spin in range(self.num_spins)
                ]
            )
            del self.node_lin_out_layers
            del self.edge_lin_out_layers
            self.node_semantic_layers = node_layers
            self.edge_semantic_layers = edge_layers
        else:
            common_layers = nn.ModuleList(
                [
                    SemanticIrrepLinear(
                        self.irreps_out,
                        self.sphere_channels,
                        self.lin_out_layers[spin],
                    )
                    for spin in range(self.num_spins)
                ]
            )
            del self.lin_out_layers
            self.common_semantic_layers = common_layers

    def semantic_matrix_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the node/edge global path-by-channel contraction matrices."""
        if self.reduce_node:
            for layers in (self.node_semantic_layers, self.edge_semantic_layers):
                for layer in layers:
                    yield layer.weight
        else:
            for layer in self.common_semantic_layers:
                yield layer.weight

    def gate_matrix_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the scalar/gate projection matrices materialized for Muon."""
        if not self.muonize_gate:
            return
        for layer in self.gate_semantic_layers:
            yield layer.weight

    def process(self, x, node_or_edge, spin):
        x_scalars = x[:, : self.sphere_channels]
        x_nonscalars = x[:, self.sphere_channels :]
        if self.muonize_gate:
            all_scalars = self.gate_semantic_layers[spin](x_scalars)
        else:
            all_scalars = self.lin_scalars_learnable[spin](x_scalars)
        transformed_l0_scalars = all_scalars[:, : self.sphere_channels]
        gating_scalars = all_scalars[:, self.sphere_channels :]
        x_gated = self.gate[spin](
            torch.cat([gating_scalars, x_nonscalars], dim=1)
        )
        x_gated = torch.cat([transformed_l0_scalars, x_gated], dim=1)

        if not self.reduce_node:
            return self.common_semantic_layers[spin](x_gated)
        if node_or_edge == "node":
            return self.node_semantic_layers[spin](x_gated)
        if node_or_edge == "edge":
            return self.edge_semantic_layers[spin](x_gated)
        raise ValueError(f"Unsupported semantic MALOQ output kind {node_or_edge!r}.")
