# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Static tensor-expansion head with semantic path-by-channel parameters."""

from __future__ import annotations

from collections.abc import Iterator
import math

import torch
from e3nn.o3 import Irreps
from torch import nn

from .common.irreps_utils import (
    get_parity_multiplier,
    get_product_irreps,
    get_reduced_to_all_indices,
)


class SemanticPathContraction(nn.Module):
    """Contract equivariant channels with one semantic matrix.

    Rows follow the original block-first ``irreps_out`` order, while input
    features and execution are degree-first.  Degree row indices and output
    component indices therefore perform the SC26 equivalent of the corrected
    ``path_offsets`` scatter.  Raw matrix concatenation must not be used here:
    it mixes path rows and channels when adjacent paths have different sizes.
    """

    PATH_LAYOUT = "path_offsets"
    PATH_LAYOUT_VERSION = 1
    CHANNEL_REDUCTION = "mean"

    def __init__(
        self,
        irreps_out: Irreps | str,
        input_channels: int,
        *,
        init_mode: str = "zero",
        init_std: float = 1.0,
        gate_degrees: tuple[int, ...] = (),
        gate_activation: str = "none",
        gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.irreps_out = Irreps(irreps_out)
        self.input_channels = int(input_channels)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if init_mode not in {"zero", "normal"}:
            raise ValueError("init_mode must be 'zero' or 'normal'.")
        if init_std <= 0.0:
            raise ValueError("init_std must be positive.")

        rows_by_degree: dict[int, list[int]] = {
            degree: [] for degree in range(self.irreps_out.lmax + 1)
        }
        components_by_degree: dict[int, list[int]] = {
            degree: [] for degree in range(self.irreps_out.lmax + 1)
        }
        semantic_row = 0
        component_start = 0
        for multiplicity, irrep in self.irreps_out:
            width = irrep.dim
            for copy_index in range(multiplicity):
                rows_by_degree[irrep.l].append(semantic_row)
                row_component_start = component_start + copy_index * width
                components_by_degree[irrep.l].extend(
                    range(row_component_start, row_component_start + width)
                )
                semantic_row += 1
            component_start += multiplicity * width

        all_rows = [row for rows in rows_by_degree.values() for row in rows]
        all_components = [
            component
            for components in components_by_degree.values()
            for component in components
        ]
        if sorted(all_rows) != list(range(self.irreps_out.num_irreps)):
            raise RuntimeError("Static TE semantic rows are not a complete bijection.")
        if sorted(all_components) != list(range(self.irreps_out.dim)):
            raise RuntimeError("Static TE output components are not a complete bijection.")

        self._degrees = tuple(
            degree for degree, rows in rows_by_degree.items() if rows
        )
        self.gate_degrees = tuple(int(degree) for degree in gate_degrees)
        if len(set(self.gate_degrees)) != len(self.gate_degrees):
            raise ValueError("gate_degrees must not contain duplicates.")
        if any(degree not in self._degrees for degree in self.gate_degrees):
            raise ValueError(
                "Every gate degree must be present in irreps_out; got "
                f"{self.gate_degrees} for degrees {self._degrees}."
            )
        if gate_activation not in {"none", "residual_tanh", "sigmoid"}:
            raise ValueError(
                "gate_activation must be 'none', 'residual_tanh', or 'sigmoid'."
            )
        if self.gate_degrees and gate_activation == "none":
            raise ValueError("A non-empty gate_degrees requires an active gate.")
        if not self.gate_degrees and gate_activation != "none":
            raise ValueError("An active gate requires non-empty gate_degrees.")
        self.gate_activation = gate_activation
        self._gate_index_by_degree = {
            degree: index for index, degree in enumerate(self.gate_degrees)
        }
        for degree in self._degrees:
            self.register_buffer(
                f"_rows_l{degree}",
                torch.tensor(rows_by_degree[degree], dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_components_l{degree}",
                torch.tensor(components_by_degree[degree], dtype=torch.long),
                persistent=False,
            )

        weight = torch.zeros(self.irreps_out.num_irreps, self.input_channels)
        if init_mode == "normal":
            nn.init.normal_(weight, std=float(init_std))
        self.weight = nn.Parameter(weight)
        scalar_rows = len(rows_by_degree.get(0, ()))
        self.bias = nn.Parameter(torch.zeros(scalar_rows))
        if self.gate_degrees:
            self.degree_gate = nn.Linear(
                self.input_channels,
                len(self.gate_degrees),
                bias=True,
            )
            nn.init.zeros_(self.degree_gate.weight)
            if gate_activation == "residual_tanh":
                if not 0.0 < float(gate_init) < 2.0:
                    raise ValueError(
                        "residual_tanh gate_init must be strictly between 0 and 2."
                    )
                raw_gate_init = math.atanh(float(gate_init) - 1.0)
            else:
                if not 0.0 < float(gate_init) < 1.0:
                    raise ValueError(
                        "sigmoid gate_init must be strictly between 0 and 1."
                    )
                raw_gate_init = math.log(float(gate_init) / (1.0 - float(gate_init)))
            nn.init.constant_(self.degree_gate.bias, raw_gate_init)
        else:
            self.degree_gate = None
        self.register_buffer(
            "_path_layout_version",
            torch.tensor(self.PATH_LAYOUT_VERSION, dtype=torch.int64),
        )

    def _degree_gate_values(self, embeddings: torch.Tensor) -> torch.Tensor | None:
        """Return scalar per-item gates computed only from invariant l=0 channels."""
        if self.degree_gate is None:
            return None
        raw_gate = self.degree_gate(embeddings[:, 0, :])
        if self.gate_activation == "residual_tanh":
            return 1.0 + torch.tanh(raw_gate)
        return torch.sigmoid(raw_gate)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected_components = (self.irreps_out.lmax + 1) ** 2
        if embeddings.ndim != 3:
            raise ValueError(
                "Static TE embeddings must have shape [items, spherical, channels]."
            )
        if embeddings.shape[1] < expected_components:
            raise ValueError(
                "Static TE embeddings do not contain every requested degree: "
                f"{embeddings.shape[1]} < {expected_components}."
            )
        if embeddings.shape[2] != self.input_channels:
            raise ValueError(
                "Static TE channel mismatch: "
                f"{embeddings.shape[2]} != {self.input_channels}."
            )

        output = embeddings.new_empty(embeddings.shape[0], self.irreps_out.dim)
        channel_scale = float(self.input_channels)
        degree_gates = self._degree_gate_values(embeddings)
        for degree in self._degrees:
            rows = getattr(self, f"_rows_l{degree}")
            components = getattr(self, f"_components_l{degree}")
            degree_features = embeddings[
                :, degree**2 : (degree + 1) ** 2, :
            ]
            degree_weight = self.weight.index_select(0, rows)
            degree_output = torch.einsum(
                "nmc,rc->nrm",
                degree_features,
                degree_weight,
            ) / channel_scale
            gate_index = self._gate_index_by_degree.get(degree)
            if degree_gates is not None and gate_index is not None:
                degree_output = degree_output * degree_gates[
                    :, gate_index, None, None
                ]
            if degree == 0:
                degree_output = degree_output + self.bias[None, :, None] / channel_scale
            output.index_copy_(1, components, degree_output.flatten(1))
        return output


class StaticTensorExpansionHead(nn.Module):
    """Static TE head for QH9Stable's coupled-irrep contract."""

    def __init__(
        self,
        *,
        irreps_out: Irreps | str,
        lmax: int,
        sphere_channels: int,
        ls_list: list[int],
        reduce_node: bool = True,
        reduce_node_intra: bool = True,
        reduce_edge: bool = False,
        open_shell: bool = False,
        init_mode: str = "zero",
        init_std: float = 1.0,
        gate_degrees: tuple[int, ...] = (),
        gate_activation: str = "none",
        gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        if reduce_edge:
            raise ValueError("Static TE currently requires reduce_edge=False.")
        if open_shell:
            raise ValueError("Static TE currently supports closed-shell data only.")
        self.num_spins = 1
        self.lmax = int(lmax)
        self.sphere_channels = int(sphere_channels)
        self.irreps_out = Irreps(irreps_out)
        self.reduce_node = bool(reduce_node)
        self.reduce_node_intra = bool(reduce_node_intra)
        self.ls_list = [int(degree) for degree in ls_list]
        if self.lmax != self.irreps_out.lmax:
            raise ValueError(
                f"Static TE lmax mismatch: {self.lmax} != {self.irreps_out.lmax}."
            )

        node_irreps = self.irreps_out
        if self.reduce_node:
            reduced_to_all = torch.as_tensor(
                get_reduced_to_all_indices(
                    self.ls_list,
                    reduce_node_intra=self.reduce_node_intra,
                ),
                dtype=torch.long,
            )
            parity_multiplier = torch.as_tensor(
                get_parity_multiplier(
                    self.ls_list,
                    reduce_node_intra=self.reduce_node_intra,
                ),
                dtype=torch.float32,
            )
            self.register_buffer("reduced_to_all_indices", reduced_to_all)
            self.register_buffer("parity_multiplier", parity_multiplier)
            node_irreps = self._make_reduced_node_irreps()

        self.node_contraction = SemanticPathContraction(
            node_irreps,
            self.sphere_channels,
            init_mode=init_mode,
            init_std=init_std,
            gate_degrees=gate_degrees,
            gate_activation=gate_activation,
            gate_init=gate_init,
        )
        self.edge_contraction = SemanticPathContraction(
            self.irreps_out,
            self.sphere_channels,
            init_mode=init_mode,
            init_std=init_std,
            gate_degrees=gate_degrees,
            gate_activation=gate_activation,
            gate_init=gate_init,
        )

    def _make_reduced_node_irreps(self) -> Irreps:
        reduced: list[str] = []
        for row, row_degree in enumerate(self.ls_list):
            for column, column_degree in enumerate(self.ls_list):
                if row == column:
                    parity = "even" if self.reduce_node_intra else None
                    reduced.append(
                        str(get_product_irreps(row_degree, column_degree, parity))
                    )
                elif row < column:
                    reduced.append(str(get_product_irreps(row_degree, column_degree)))
        return Irreps("+".join(reduced))

    def semantic_matrix_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the matrices whose row/column geometry Muon should see."""
        yield self.node_contraction.weight
        yield self.edge_contraction.weight

    def forward(self, embeddings, batch):
        del batch
        node_output = self.node_contraction(embeddings["node_embeddings"])
        edge_output = self.edge_contraction(embeddings["edge_embeddings"])
        if self.reduce_node:
            node_output = (
                node_output[:, self.reduced_to_all_indices]
                * self.parity_multiplier.to(dtype=node_output.dtype)
            )
        return node_output.unsqueeze(0), edge_output.unsqueeze(0)
