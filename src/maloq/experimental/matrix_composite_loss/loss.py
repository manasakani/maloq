"""Feature-local loss functions for the NablaDFT matrix-loss ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch


CompositeLossProfileId = Literal["rmse_mse_mae", "10x_rmse_mse_mae"]
LossCallable = Callable[[torch.Tensor, torch.Tensor, object | None], torch.Tensor]


def _rmse_mse_mae_padded_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    error = output - target
    mse = torch.mean(torch.square(error))
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(error))
    return float(scale) * (rmse + mse + mae)


def rmse_mse_mae_padded_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    req_irreps: object | None = None,
) -> torch.Tensor:
    """Return RMSE + MSE + MAE over the already padding-filtered entries."""
    del req_irreps
    return _rmse_mse_mae_padded_loss(output, target, scale=1.0)


def ten_x_rmse_mse_mae_padded_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    req_irreps: object | None = None,
) -> torch.Tensor:
    """Return 10 * (RMSE + MSE + MAE) over filtered entries."""
    del req_irreps
    return _rmse_mse_mae_padded_loss(output, target, scale=10.0)


@dataclass(frozen=True)
class CompositeLossProfile:
    id: CompositeLossProfileId
    formula: str
    scale: float
    loss: LossCallable


_PROFILES = {
    "rmse_mse_mae": CompositeLossProfile(
        id="rmse_mse_mae",
        formula="rmse+mse+mae",
        scale=1.0,
        loss=rmse_mse_mae_padded_loss,
    ),
    "10x_rmse_mse_mae": CompositeLossProfile(
        id="10x_rmse_mse_mae",
        formula="10*(rmse+mse+mae)",
        scale=10.0,
        loss=ten_x_rmse_mse_mae_padded_loss,
    ),
}


def get_composite_loss_profile(profile_id: str) -> CompositeLossProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown matrix composite-loss profile {profile_id!r}; "
            f"expected one of {sorted(_PROFILES)}."
        ) from error
