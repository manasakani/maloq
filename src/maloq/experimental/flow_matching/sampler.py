"""Fixed-step endpoint Euler sampling for joint node and edge flows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import FlowMatchingConfig
from .objective import EndpointFlowMatcher, _validate_graph_index, broadcast_mask


@dataclass(frozen=True)
class EndpointPrediction:
    """Clean node and directed-edge endpoints at one Euler evaluation."""

    node: Tensor
    edge: Tensor


@dataclass(frozen=True)
class EndpointEulerResult:
    """Joint node and directed-edge states after endpoint Euler integration."""

    node: Tensor
    edge: Tensor
    times: Tensor


EndpointPredictor = Callable[[Tensor, Tensor, Tensor], EndpointPrediction]
StateProjector = Callable[[Tensor, Tensor], EndpointPrediction]


class EndpointEulerSampler:
    """Integrate node and edge states with endpoint-derived velocities."""

    def __init__(self, config: FlowMatchingConfig):
        self.config = config
        self.matcher = EndpointFlowMatcher(config)

    def sample(
        self,
        node_source: Tensor,
        edge_source: Tensor,
        *,
        node_graph_index: Tensor,
        edge_graph_index: Tensor,
        predict_endpoint: EndpointPredictor,
        node_mask: Tensor | None = None,
        edge_mask: Tensor | None = None,
        project_state: StateProjector | None = None,
    ) -> EndpointEulerResult:
        self._validate_source(node_source, name="node_source")
        self._validate_source(edge_source, name="edge_source")
        if (
            node_source.dtype != edge_source.dtype
            or node_source.device != edge_source.device
        ):
            raise ValueError("Node and edge sources must share dtype/device.")

        node_graph_index = _validate_graph_index(
            node_graph_index,
            entry_count=node_source.shape[0],
            device=node_source.device,
            name="node_graph_index",
        )
        edge_graph_index = _validate_graph_index(
            edge_graph_index,
            entry_count=edge_source.shape[0],
            device=edge_source.device,
            name="edge_graph_index",
        )
        num_graphs = int(node_graph_index.max().item()) + 1
        if int(edge_graph_index.max().item()) + 1 != num_graphs:
            raise ValueError(
                "Node and edge graph indices must describe the same graph batch."
            )

        node_valid = broadcast_mask(node_mask, node_source, entry_dim=0)
        edge_valid = broadcast_mask(edge_mask, edge_source, entry_dim=0)
        node_state = torch.where(
            node_valid,
            node_source,
            torch.zeros_like(node_source),
        )
        edge_state = torch.where(
            edge_valid,
            edge_source,
            torch.zeros_like(edge_source),
        )
        times = torch.linspace(
            self.config.time_min,
            1.0,
            self.config.num_ode_steps + 1,
            device=node_source.device,
            dtype=node_source.dtype,
        )
        for step in range(self.config.num_ode_steps):
            graph_time = times[step].expand(num_graphs)
            prediction = predict_endpoint(node_state, edge_state, graph_time)
            if not isinstance(prediction, EndpointPrediction):
                raise TypeError("predict_endpoint must return EndpointPrediction.")
            self._validate_prediction(
                prediction.node,
                reference=node_state,
                name="node",
            )
            self._validate_prediction(
                prediction.edge,
                reference=edge_state,
                name="edge",
            )
            node_velocity = self.matcher.derived_velocity(
                prediction.node,
                node_state,
                graph_time,
                graph_index=node_graph_index,
                entry_dim=0,
            )
            edge_velocity = self.matcher.derived_velocity(
                prediction.edge,
                edge_state,
                graph_time,
                graph_index=edge_graph_index,
                entry_dim=0,
            )
            step_size = times[step + 1] - times[step]
            node_state = node_state + step_size * node_velocity
            edge_state = edge_state + step_size * edge_velocity
            node_state = torch.where(
                node_valid,
                node_state,
                torch.zeros_like(node_state),
            )
            edge_state = torch.where(
                edge_valid,
                edge_state,
                torch.zeros_like(edge_state),
            )
            if project_state is not None:
                projected = project_state(node_state, edge_state)
                if not isinstance(projected, EndpointPrediction):
                    raise TypeError("project_state must return EndpointPrediction.")
                self._validate_prediction(
                    projected.node,
                    reference=node_state,
                    name="projected node state",
                )
                self._validate_prediction(
                    projected.edge,
                    reference=edge_state,
                    name="projected edge state",
                )
                node_state = torch.where(
                    node_valid,
                    projected.node,
                    torch.zeros_like(projected.node),
                )
                edge_state = torch.where(
                    edge_valid,
                    projected.edge,
                    torch.zeros_like(projected.edge),
                )
        return EndpointEulerResult(
            node=node_state,
            edge=edge_state,
            times=times,
        )

    @staticmethod
    def _validate_source(source: Tensor, *, name: str) -> None:
        if source.ndim < 2 or not source.is_floating_point():
            raise ValueError(
                f"{name} must be a floating-point entry tensor with at least 2 dims."
            )

    @staticmethod
    def _validate_prediction(
        prediction: Tensor,
        *,
        reference: Tensor,
        name: str,
    ) -> None:
        if (
            prediction.shape != reference.shape
            or prediction.dtype != reference.dtype
            or prediction.device != reference.device
        ):
            raise ValueError(
                f"Predicted {name} endpoint must match its flow state "
                "shape/dtype/device."
            )
