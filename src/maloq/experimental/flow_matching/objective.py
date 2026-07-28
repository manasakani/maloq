"""Joint node/edge corruption with clean-endpoint supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import FlowMatchingConfig


def _validate_graph_index(
    graph_index: Tensor,
    *,
    entry_count: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if not isinstance(graph_index, Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if graph_index.ndim != 1 or graph_index.numel() != entry_count:
        raise ValueError(f"{name} must contain one index per state entry.")
    if graph_index.device != device:
        raise ValueError(f"{name} and state must share a device.")
    if graph_index.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(f"{name} must contain integer indices.")
    index = graph_index.to(dtype=torch.long)
    if index.numel() == 0 or int(index.min().item()) < 0:
        raise ValueError(f"{name} must be non-empty and non-negative.")
    graph_count = int(index.max().item()) + 1
    expected = torch.arange(graph_count, device=device)
    if not torch.equal(index.unique(sorted=True), expected):
        raise ValueError(f"{name} must contain contiguous graph indices from zero.")
    return index


def normalize_entry_dim(state: Tensor, entry_dim: int) -> int:
    if state.ndim == 0:
        raise ValueError("Flow state must have at least one dimension.")
    normalized = entry_dim if entry_dim >= 0 else state.ndim + entry_dim
    if not 0 <= normalized < state.ndim:
        raise ValueError(
            f"entry_dim={entry_dim} is invalid for a {state.ndim}D tensor."
        )
    return normalized


def expand_per_graph(
    values: Tensor,
    state: Tensor,
    *,
    graph_index: Tensor | None,
    entry_dim: int,
) -> Tensor:
    """Broadcast one scalar per molecular graph over state entries."""
    entry_dim = normalize_entry_dim(state, entry_dim)
    values = values.reshape(-1)
    if values.numel() == 0:
        raise ValueError("At least one per-graph value is required.")
    if values.device != state.device:
        raise ValueError("Per-graph values and state must share a device.")
    entry_count = state.shape[entry_dim]
    if graph_index is None:
        if values.numel() == 1:
            gathered = values.expand(entry_count)
        elif values.numel() == entry_count:
            gathered = values
        else:
            raise ValueError(
                "graph_index is required unless one global value or one "
                "value per entry is provided."
            )
    else:
        index = _validate_graph_index(
            graph_index,
            entry_count=entry_count,
            device=state.device,
            name="graph_index",
        )
        if int(index.max().item()) >= values.numel():
            raise ValueError("graph_index contains an out-of-range value.")
        gathered = values[index]
    shape = [1] * state.ndim
    shape[entry_dim] = entry_count
    return gathered.reshape(shape)


def broadcast_mask(
    mask: Tensor | None,
    state: Tensor,
    *,
    entry_dim: int,
) -> Tensor:
    """Return a boolean validity mask broadcast to ``state.shape``."""
    if mask is None:
        return torch.ones_like(state, dtype=torch.bool)
    if not isinstance(mask, Tensor) or mask.dtype is not torch.bool:
        raise TypeError("Flow masks must be boolean tensors.")
    if mask.device != state.device:
        raise ValueError("Flow mask and state must share a device.")
    entry_dim = normalize_entry_dim(state, entry_dim)
    if mask.ndim == 1 and mask.numel() == state.shape[entry_dim]:
        shape = [1] * state.ndim
        shape[entry_dim] = mask.numel()
        mask = mask.reshape(shape)
    try:
        return torch.broadcast_to(mask, state.shape)
    except RuntimeError as error:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot broadcast to "
            f"state shape {tuple(state.shape)}."
        ) from error


@dataclass(frozen=True)
class EndpointFlowSample:
    """One corrupted matrix-block state in coupled-irrep coordinates."""

    source: Tensor
    clean_endpoint: Tensor
    time: Tensor
    state: Tensor
    mask: Tensor


@dataclass(frozen=True)
class JointEndpointFlowSample:
    """Node and directed-edge flow samples sharing one graph time."""

    node: EndpointFlowSample
    edge: EndpointFlowSample
    time: Tensor


class EndpointFlowMatcher:
    r"""Implement the active QHFlow2 endpoint parameterization safely.

    Every node or directed-edge state follows

    .. math::
       z_0 \sim \mathcal{N}_{\mathrm{irrep}}(0,\sigma^2 I),\qquad
       z_t=(1-t)z_0+tz_1.

    The model predicts clean :math:`z_1` directly for both node and edge
    blocks. Both state families are corrupted and integrated.
    """

    def __init__(self, config: FlowMatchingConfig):
        self.config = config

    def sample_time(
        self,
        num_graphs: int,
        *,
        reference: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if num_graphs <= 0:
            raise ValueError("num_graphs must be positive.")
        uniform = torch.rand(
            (num_graphs,),
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )
        return (
            uniform * (self.config.time_max - self.config.time_min)
            + self.config.time_min
        )

    def sample_coupled_prior(
        self,
        clean_target: Tensor,
        *,
        mask: Tensor | None,
        entry_dim: int = 0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample the basis-free default Gaussian coupled prior.

        Structured priors need the basis transform and are constructed by the
        feature-owned prior factory in the trainer/loader integration.
        """
        if self.config.prior_type != "coupled_irrep_gaussian":
            raise ValueError(
                "Tensor Expansion prior sampling requires a basis_transform; "
                "use build_coupled_prior() or provide an explicit source."
            )
        if not clean_target.is_floating_point():
            raise TypeError("Coupled-irrep targets must be floating point.")
        valid = broadcast_mask(mask, clean_target, entry_dim=entry_dim)
        source = (
            torch.randn(
                clean_target.shape,
                dtype=clean_target.dtype,
                device=clean_target.device,
                generator=generator,
            )
            * self.config.sigma
        )
        return torch.where(valid, source, torch.zeros_like(source))

    def corrupt(
        self,
        clean_target: Tensor,
        *,
        time: Tensor | None = None,
        source: Tensor | None = None,
        mask: Tensor | None = None,
        graph_index: Tensor | None = None,
        entry_dim: int = 0,
        generator: torch.Generator | None = None,
    ) -> EndpointFlowSample:
        """Interpolate a node or edge state and retain its clean endpoint."""
        entry_dim = normalize_entry_dim(clean_target, entry_dim)
        valid = broadcast_mask(mask, clean_target, entry_dim=entry_dim)
        clean = torch.where(
            valid,
            clean_target,
            torch.zeros_like(clean_target),
        )
        if source is None:
            source = self.sample_coupled_prior(
                clean,
                mask=valid,
                entry_dim=entry_dim,
                generator=generator,
            )
        elif (
            source.shape != clean.shape
            or source.dtype != clean.dtype
            or source.device != clean.device
        ):
            raise ValueError("Source and clean target must share shape/dtype/device.")
        else:
            source = torch.where(valid, source, torch.zeros_like(source))

        if time is None:
            num_graphs = 1 if graph_index is None else int(graph_index.max().item()) + 1
            time = self.sample_time(
                num_graphs,
                reference=clean,
                generator=generator,
            )
        else:
            time = torch.as_tensor(
                time,
                dtype=clean.dtype,
                device=clean.device,
            ).reshape(-1)
            if time.numel() == 0 or not torch.isfinite(time).all():
                raise ValueError("Time must contain finite values.")
            if bool((time < self.config.time_min).any()) or bool(
                (time > self.config.time_max).any()
            ):
                raise ValueError("Training time lies outside the configured interval.")
        expanded_t = expand_per_graph(
            time,
            clean,
            graph_index=graph_index,
            entry_dim=entry_dim,
        )
        state = (1.0 - expanded_t) * source + expanded_t * clean
        state = torch.where(valid, state, torch.zeros_like(state))
        return EndpointFlowSample(
            source=source,
            clean_endpoint=clean,
            time=time,
            state=state,
            mask=valid,
        )

    def corrupt_joint(
        self,
        clean_node_target: Tensor,
        clean_edge_target: Tensor,
        *,
        node_graph_index: Tensor,
        edge_graph_index: Tensor,
        time: Tensor | None = None,
        node_source: Tensor | None = None,
        edge_source: Tensor | None = None,
        node_mask: Tensor | None = None,
        edge_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> JointEndpointFlowSample:
        """Corrupt node and edge states at the same sampled time per graph."""
        if (
            clean_node_target.dtype != clean_edge_target.dtype
            or clean_node_target.device != clean_edge_target.device
        ):
            raise ValueError("Node and edge targets must share dtype/device.")
        node_graph_index = _validate_graph_index(
            node_graph_index,
            entry_count=clean_node_target.shape[0],
            device=clean_node_target.device,
            name="node_graph_index",
        )
        edge_graph_index = _validate_graph_index(
            edge_graph_index,
            entry_count=clean_edge_target.shape[0],
            device=clean_edge_target.device,
            name="edge_graph_index",
        )
        node_graph_count = int(node_graph_index.max().item()) + 1
        edge_graph_count = int(edge_graph_index.max().item()) + 1
        if node_graph_count != edge_graph_count:
            raise ValueError(
                "Node and edge graph indices must describe the same graph batch."
            )
        if time is None:
            time = self.sample_time(
                node_graph_count,
                reference=clean_node_target,
                generator=generator,
            )
        node = self.corrupt(
            clean_node_target,
            time=time,
            source=node_source,
            mask=node_mask,
            graph_index=node_graph_index,
            generator=generator,
        )
        edge = self.corrupt(
            clean_edge_target,
            time=time,
            source=edge_source,
            mask=edge_mask,
            graph_index=edge_graph_index,
            generator=generator,
        )
        return JointEndpointFlowSample(node=node, edge=edge, time=node.time)

    @staticmethod
    def derived_velocity(
        clean_endpoint_prediction: Tensor,
        current_state: Tensor,
        time: Tensor,
        *,
        graph_index: Tensor | None = None,
        entry_dim: int = 0,
    ) -> Tensor:
        """Derive ``(endpoint_prediction - state) / (1 - t)`` for sampling."""
        if clean_endpoint_prediction.shape != current_state.shape:
            raise ValueError("Endpoint prediction and current state must share shape.")
        expanded_t = expand_per_graph(
            time.to(device=current_state.device, dtype=current_state.dtype),
            current_state,
            graph_index=graph_index,
            entry_dim=entry_dim,
        )
        denominator = 1.0 - expanded_t
        if bool((denominator <= 0).any()):
            raise ValueError("Endpoint-derived velocity requires t < 1.")
        return (clean_endpoint_prediction - current_state) / denominator
