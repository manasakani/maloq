"""Random-probe utilities for matrix-free operator learning."""

from __future__ import annotations

import torch


def rademacher_probes(
    total_ao: int,
    num_probes: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return unnormalized Rademacher probes with shape ``[AO, probes]``."""
    if total_ao <= 0 or num_probes <= 0:
        raise ValueError("total_ao and num_probes must be positive")
    bits = torch.randint(
        0,
        2,
        (int(total_ao), int(num_probes)),
        device=device,
        generator=generator,
    )
    return bits.to(dtype=dtype).mul_(2).sub_(1)


def probe_matrix_mse(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
) -> torch.Tensor:
    """Unbiased matrix-element MSE estimate for Rademacher probes."""
    if predicted_action.shape != target_action.shape or predicted_action.ndim != 2:
        raise ValueError("operator actions must share shape [AO, probes]")
    total_ao, num_probes = predicted_action.shape
    return (predicted_action - target_action).square().sum() / (
        num_probes * total_ao**2
    )


def relative_action_error(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Squared relative Frobenius error on a fixed held-out probe set."""
    if predicted_action.shape != target_action.shape:
        raise ValueError("predicted and target actions must have the same shape")
    numerator = (predicted_action - target_action).square().sum()
    denominator = target_action.square().sum().clamp_min(eps)
    return numerator / denominator


__all__ = ["probe_matrix_mse", "rademacher_probes", "relative_action_error"]
