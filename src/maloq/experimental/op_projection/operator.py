"""Differentiable callback binding for node-latent operator projection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


BackboneOutput = Mapping[str, torch.Tensor | None]
Projection = Callable[[BackboneOutput, Any, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class BoundOperatorCallback:
    """Bind one encoded geometry to a differentiable matrix-free projection.

    The bound callback is valid only while the backbone computation graph is
    alive. It should be created and consumed within one training step.
    """

    features: BackboneOutput
    batch: Any
    projection: Projection

    def __call__(self, probe: torch.Tensor) -> torch.Tensor:
        result = self.projection(self.features, self.batch, probe)
        if result.shape != probe.shape:
            raise ValueError(
                "operator projection must preserve the probe shape: "
                f"got {tuple(result.shape)} for {tuple(probe.shape)}"
            )
        return result


def bind_operator_callback(
    backbone_output: BackboneOutput,
    batch: Any,
    projection: Projection,
) -> BoundOperatorCallback:
    """Return ``probe -> projected_operator @ probe`` without a dense matrix.

    The projection receives node latents and graph geometry. It is responsible
    for streaming onsite and pair contributions; a persistent learned edge
    feature tensor is intentionally rejected by this experimental contract.
    """

    required = {"node_embeddings", "edge_index", "edge_distance", "edge_distance_vec"}
    missing = sorted(required.difference(backbone_output))
    if missing:
        raise KeyError(f"op_projection backbone output is missing: {missing}")
    if backbone_output.get("edge_embeddings") is not None:
        raise ValueError(
            "op_projection expects node-only learned features; pair terms must "
            "be produced lazily by the projection callback"
        )
    return BoundOperatorCallback(backbone_output, batch, projection)


__all__ = ["BoundOperatorCallback", "bind_operator_callback"]
