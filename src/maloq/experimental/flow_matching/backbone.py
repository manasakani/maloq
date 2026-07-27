"""Feature-local node/edge/time conditioning for native MALOQ embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear

from maloq.helm.qhflow3 import QHFlow3Backbone


def _so3_compatible_irreps(irreps: Irreps) -> Irreps:
    """Relabel matrix irreps for the proper-rotation native tensor layout.

    Canonical MALOQ matrix targets currently label every coupled output with
    even parity. Native MALOQ embeddings use parity ``(-1)^l``. Proper
    rotations do not depend on this label, so relabeling permits a complete
    SO(3)-equivariant projection without claiming reflection equivariance.
    """

    return Irreps(
        [
            (multiplicity, (irrep.l, 1 if irrep.l % 2 == 0 else -1))
            for multiplicity, irrep in irreps
        ]
    )


def _flat_to_native_embeddings(
    flat: torch.Tensor,
    *,
    lmax: int,
    channels: int,
) -> torch.Tensor:
    expected = channels * (lmax + 1) ** 2
    if flat.ndim != 2 or flat.shape[-1] != expected:
        raise ValueError(f"Projected flow must have shape [items, {expected}].")
    embeddings = flat.new_zeros(flat.shape[0], (lmax + 1) ** 2, channels)
    for degree in range(lmax + 1):
        start = degree**2 * channels
        width = channels * (2 * degree + 1)
        block = flat[:, start : start + width].reshape(
            flat.shape[0], channels, 2 * degree + 1
        )
        embeddings[:, degree**2 : (degree + 1) ** 2, :] = block.transpose(1, 2)
    return embeddings


def _dimension(
    base: nn.Module,
    explicit: int | None,
    *,
    attributes: tuple[str, ...],
    name: str,
    minimum: int,
) -> int:
    value = explicit
    if value is None:
        for attribute in attributes:
            candidate = getattr(base, attribute, None)
            if candidate is not None:
                value = int(candidate)
                break
    if value is None or int(value) < minimum:
        choices = ", ".join(attributes)
        raise ValueError(
            f"{name} must be >= {minimum} or exposed by base as one of: {choices}."
        )
    return int(value)


def _floating_reference(base: nn.Module) -> torch.Tensor | None:
    for tensor in base.parameters():
        if tensor.is_floating_point():
            return tensor
    for tensor in base.buffers():
        if tensor.is_floating_point():
            return tensor
    return None


class FlowConditionedBackbone(nn.Module):
    """Condition native node/edge embeddings on the complete flow state.

    ``base`` must return ``node_embeddings`` and ``edge_embeddings`` with
    native shape ``[items, (lmax + 1)^2, channels]``. Node and edge coupled
    states use separate SO(3)-equivariant projections. Incoming edge state is
    degree-normalized into destination nodes, while graph time is added only
    to the invariant ``l=0`` component of both outputs.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        flow_irreps: Irreps | str,
        embedding_lmax: int | None = None,
        embedding_channels: int | None = None,
        edge_flow_scale: float = 1.0,
        node_flow_scale: float = 1.0,
        incident_edge_scale: float = 1.0,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Module):
            raise TypeError("base must be a torch.nn.Module.")
        self.base = base
        self.embedding_lmax = _dimension(
            base,
            embedding_lmax,
            attributes=("expand_lmax", "lmax"),
            name="embedding_lmax",
            minimum=0,
        )
        self.embedding_channels = _dimension(
            base,
            embedding_channels,
            attributes=(
                "bottle_hidden_size",
                "output_sphere_channels",
                "sphere_channels",
            ),
            name="embedding_channels",
            minimum=1,
        )
        self.flow_irreps = Irreps(flow_irreps)
        self.flow_so3_irreps = _so3_compatible_irreps(self.flow_irreps)
        if self.flow_irreps.lmax > self.embedding_lmax:
            raise ValueError("Flow-state lmax cannot exceed the native embedding lmax.")
        self.embedding_irreps = Irreps(
            [
                (
                    self.embedding_channels,
                    (degree, 1 if degree % 2 == 0 else -1),
                )
                for degree in range(self.embedding_lmax + 1)
            ]
        )
        self.node_flow_projection = Linear(
            self.flow_so3_irreps,
            self.embedding_irreps,
            biases=False,
        )
        self.edge_flow_projection = Linear(
            self.flow_so3_irreps,
            self.embedding_irreps,
            biases=False,
        )
        self.node_time_projection = nn.Linear(1, self.embedding_channels)
        self.edge_time_projection = nn.Linear(1, self.embedding_channels)
        self.edge_flow_scale = self._finite_scale(edge_flow_scale, "edge_flow_scale")
        self.node_flow_scale = self._finite_scale(node_flow_scale, "node_flow_scale")
        self.incident_edge_scale = self._finite_scale(
            incident_edge_scale,
            "incident_edge_scale",
        )
        self.time_scale = self._finite_scale(time_scale, "time_scale")

        reference = _floating_reference(self.base)
        if reference is not None:
            for module in (
                self.node_flow_projection,
                self.edge_flow_projection,
                self.node_time_projection,
                self.edge_time_projection,
            ):
                module.to(dtype=reference.dtype, device=reference.device)

    @staticmethod
    def _finite_scale(value: float, name: str) -> float:
        value = float(value)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{name} must be finite.")
        return value

    @property
    def architecture(self) -> str:
        return str(getattr(self.base, "architecture", type(self.base).__name__))

    @property
    def basis(self) -> str:
        return self.base.basis

    @property
    def flow_basis(self) -> str | None:
        return getattr(self.base, "basis", None)

    @property
    def output_matrix_dim(self) -> int:
        return int(self.base.output_matrix_dim)

    @property
    def expand_lmax(self) -> int:
        return self.embedding_lmax

    @property
    def bottle_hidden_size(self) -> int:
        return self.embedding_channels

    @property
    def default_hamiltonian_input(self) -> str:
        return self.base.default_hamiltonian_input

    @default_hamiltonian_input.setter
    def default_hamiltonian_input(self, value: str) -> None:
        self.base.default_hamiltonian_input = value

    @property
    def delta_learning(self) -> bool:
        return bool(getattr(self.base, "delta_learning", False))

    @property
    def delta_target(self) -> str | None:
        return getattr(self.base, "delta_target", None)

    @property
    def grid_ffn_chunk_size(self) -> int | None:
        return getattr(self.base, "grid_ffn_chunk_size", None)

    @property
    def node_output_projection(self) -> nn.Module:
        return self.base.node_output_projection

    @property
    def edge_output_projection(self) -> nn.Module:
        return self.base.edge_output_projection

    @property
    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _project_state(
        self,
        state: torch.Tensor,
        projection: Linear,
    ) -> torch.Tensor:
        return _flat_to_native_embeddings(
            projection(state),
            lmax=self.embedding_lmax,
            channels=self.embedding_channels,
        )

    @staticmethod
    def _node_graph_index(
        batch: Any,
        *,
        node_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        graph_index = getattr(batch, "batch", None)
        if isinstance(graph_index, torch.Tensor) and graph_index.numel() == node_count:
            graph_index = graph_index.reshape(-1)
        else:
            ptr = getattr(batch, "ptr", None)
            if not isinstance(ptr, torch.Tensor) or ptr.ndim != 1 or ptr.numel() < 2:
                raise ValueError(
                    "Flow conditioning requires node-level batch indices or ptr."
                )
            counts = ptr[1:] - ptr[:-1]
            if int(counts.sum().item()) != node_count:
                raise ValueError("batch.ptr does not match the node count.")
            graph_index = torch.arange(
                counts.numel(),
                dtype=torch.long,
                device=ptr.device,
            ).repeat_interleave(counts.to(dtype=torch.long))
        if (
            graph_index.dtype != torch.long
            or graph_index.device != device
            or graph_index.shape != (node_count,)
        ):
            raise ValueError("Node graph indices must be a same-device long tensor.")
        return graph_index

    def condition_embeddings(
        self,
        *,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        node_flow_state: torch.Tensor,
        edge_flow_state: torch.Tensor,
        graph_time: torch.Tensor,
        node_graph_index: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_tail = (
            (self.embedding_lmax + 1) ** 2,
            self.embedding_channels,
        )
        if (
            not isinstance(node_embeddings, torch.Tensor)
            or not isinstance(edge_embeddings, torch.Tensor)
            or node_embeddings.ndim != 3
            or edge_embeddings.ndim != 3
            or node_embeddings.shape[1:] != expected_tail
            or edge_embeddings.shape[1:] != expected_tail
        ):
            raise ValueError(
                "Backbone embeddings must have native shape "
                f"[items, {expected_tail[0]}, {expected_tail[1]}]."
            )
        if (
            not node_embeddings.is_floating_point()
            or not edge_embeddings.is_floating_point()
        ):
            raise TypeError("Backbone embeddings must be floating point.")
        if (
            node_embeddings.device != edge_embeddings.device
            or node_embeddings.dtype != edge_embeddings.dtype
        ):
            raise ValueError("Node and edge embeddings must share dtype/device.")
        projection_reference = next(self.node_time_projection.parameters())
        if (
            projection_reference.device != node_embeddings.device
            or projection_reference.dtype != node_embeddings.dtype
        ):
            raise ValueError(
                "Flow conditioner modules and embeddings must share dtype/device."
            )
        for state, rows, name in (
            (node_flow_state, node_embeddings.shape[0], "node_flow_t"),
            (edge_flow_state, edge_embeddings.shape[0], "edge_flow_t"),
        ):
            if (
                not isinstance(state, torch.Tensor)
                or state.ndim != 2
                or state.shape != (rows, self.flow_irreps.dim)
            ):
                raise ValueError(
                    f"batch.{name} must have shape [{rows}, {self.flow_irreps.dim}]."
                )
            if not state.is_floating_point():
                raise TypeError(f"batch.{name} must be floating point.")
            if (
                state.device != node_embeddings.device
                or state.dtype != node_embeddings.dtype
            ):
                raise ValueError(
                    f"batch.{name} and embeddings must share dtype/device."
                )
        if node_embeddings.shape[0] == 0:
            raise ValueError("Flow conditioning requires at least one node.")
        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.dtype != torch.long
            or edge_index.device != node_embeddings.device
            or edge_index.shape != (2, edge_embeddings.shape[0])
        ):
            raise ValueError(
                "batch.edge_index must be a same-device long tensor of shape [2, E]."
            )
        if edge_index.numel() and (
            int(edge_index.min().item()) < 0
            or int(edge_index.max().item()) >= node_embeddings.shape[0]
        ):
            raise ValueError("batch.edge_index contains an out-of-range node index.")
        if (
            not isinstance(node_graph_index, torch.Tensor)
            or node_graph_index.dtype != torch.long
            or node_graph_index.device != node_embeddings.device
            or node_graph_index.shape != (node_embeddings.shape[0],)
        ):
            raise ValueError("Node graph indices must be a same-device long tensor.")
        if (
            not isinstance(graph_time, torch.Tensor)
            or graph_time.ndim != 1
            or graph_time.numel() == 0
            or not graph_time.is_floating_point()
            or graph_time.dtype != node_embeddings.dtype
            or graph_time.device != node_embeddings.device
            or not bool(torch.isfinite(graph_time).all().item())
        ):
            raise ValueError(
                "batch.t must be finite, floating, same dtype/device, and per-graph."
            )
        expected_graphs = int(node_graph_index.max().item()) + 1
        if int(node_graph_index.min().item()) < 0 or graph_time.shape != (
            expected_graphs,
        ):
            raise ValueError("Node graph indices are incompatible with batch.t.")

        source_graph = node_graph_index.index_select(0, edge_index[0])
        destination_graph = node_graph_index.index_select(0, edge_index[1])
        if not torch.equal(source_graph, destination_graph):
            raise ValueError("Every directed edge must stay within one graph.")

        node_projected = self._project_state(
            node_flow_state,
            self.node_flow_projection,
        )
        edge_projected = self._project_state(
            edge_flow_state,
            self.edge_flow_projection,
        )
        conditioned_node = node_embeddings + self.node_flow_scale * node_projected
        conditioned_edge = edge_embeddings + self.edge_flow_scale * edge_projected

        destination = edge_index[1]
        incident = torch.zeros_like(node_embeddings)
        incident.index_add_(0, destination, edge_projected)
        degree = torch.zeros(
            node_embeddings.shape[0],
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        degree.index_add_(
            0,
            destination,
            torch.ones_like(destination, dtype=degree.dtype),
        )
        incident = incident / degree.clamp_min(1).reshape(-1, 1, 1)
        conditioned_node = conditioned_node + self.incident_edge_scale * incident

        node_time = self.node_time_projection(
            graph_time.index_select(0, node_graph_index).unsqueeze(-1)
        )
        edge_time = self.edge_time_projection(
            graph_time.index_select(0, source_graph).unsqueeze(-1)
        )
        node_time_embedding = torch.zeros_like(node_embeddings)
        edge_time_embedding = torch.zeros_like(edge_embeddings)
        node_time_embedding[:, 0, :] = node_time
        edge_time_embedding[:, 0, :] = edge_time
        conditioned_node = conditioned_node + self.time_scale * node_time_embedding
        conditioned_edge = conditioned_edge + self.time_scale * edge_time_embedding
        return conditioned_node, conditioned_edge

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        flow_time = getattr(batch, "t", None)
        flow_edge_index = getattr(batch, "edge_index", None)
        if isinstance(flow_time, torch.Tensor):
            # Some native backbones (currently QHFlow3) contain their own
            # direct-model time path. Keep that path at its canonical
            # endpoint value so every architecture receives the varying flow
            # time exactly once through this common conditioner.
            batch.t = torch.ones_like(flow_time)
        try:
            output = self.base(batch)
        finally:
            if isinstance(flow_time, torch.Tensor):
                batch.t = flow_time
            # QHFlow3 sorts and replaces edge_index internally. Flow states
            # and targets retain the loader's canonical edge-row order.
            if isinstance(flow_edge_index, torch.Tensor):
                batch.edge_index = flow_edge_index
        if not isinstance(output, Mapping):
            raise TypeError("Flow-conditioned backbone must return a mapping.")
        node_embeddings = output.get("node_embeddings")
        edge_embeddings = output.get("edge_embeddings")
        if not isinstance(node_embeddings, torch.Tensor):
            raise TypeError("Backbone output requires tensor node_embeddings.")
        if not isinstance(edge_embeddings, torch.Tensor):
            raise TypeError("Backbone output requires tensor edge_embeddings.")
        conditioning_time = flow_time
        if isinstance(flow_time, torch.Tensor) and flow_time.is_floating_point():
            conditioning_time = flow_time.to(
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            )
        node_graph_index = self._node_graph_index(
            batch,
            node_count=node_embeddings.shape[0],
            device=node_embeddings.device,
        )
        node_embeddings, edge_embeddings = self.condition_embeddings(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            node_flow_state=getattr(batch, "node_flow_t", None),
            edge_flow_state=getattr(batch, "edge_flow_t", None),
            graph_time=conditioning_time,
            node_graph_index=node_graph_index,
            edge_index=flow_edge_index,
        )
        return {
            **output,
            "node_embeddings": node_embeddings,
            "edge_embeddings": edge_embeddings,
        }


class FlowConditionedQHFlow3Backbone(FlowConditionedBackbone):
    """Compatibility name for the canonical QHFlow3 flow wrapper."""

    def __init__(
        self,
        base: QHFlow3Backbone,
        *,
        flow_irreps: Irreps | str,
        edge_flow_scale: float = 1.0,
        node_flow_scale: float = 1.0,
        incident_edge_scale: float = 1.0,
        time_scale: float = 1.0,
    ) -> None:
        if not isinstance(base, QHFlow3Backbone):
            raise TypeError("base must be a canonical QHFlow3Backbone.")
        super().__init__(
            base,
            flow_irreps=flow_irreps,
            embedding_lmax=base.expand_lmax,
            embedding_channels=base.bottle_hidden_size,
            edge_flow_scale=edge_flow_scale,
            node_flow_scale=node_flow_scale,
            incident_edge_scale=incident_edge_scale,
            time_scale=time_scale,
        )
