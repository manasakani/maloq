from __future__ import annotations

import torch

from maloq.experimental.matrix_composite_loss.loss import (
    get_composite_loss_profile,
    rmse_mse_mae_padded_loss,
    ten_x_rmse_mse_mae_padded_loss,
)


def test_rmse_mse_mae_matches_explicit_formula() -> None:
    output = torch.tensor([1.0, -1.0, 4.0], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    error = output - target
    expected = error.square().mean().sqrt() + error.square().mean() + error.abs().mean()

    actual = rmse_mse_mae_padded_loss(output, target)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float64


def test_ten_x_profile_scales_value_and_gradient() -> None:
    output = torch.tensor([0.25, -0.5, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 0.5, 1.0])
    base = rmse_mse_mae_padded_loss(output, target)
    base_gradient = torch.autograd.grad(base, output, retain_graph=True)[0]
    scaled = ten_x_rmse_mse_mae_padded_loss(output, target)
    scaled_gradient = torch.autograd.grad(scaled, output)[0]

    torch.testing.assert_close(scaled, 10.0 * base)
    torch.testing.assert_close(scaled_gradient, 10.0 * base_gradient)
    assert torch.isfinite(scaled_gradient).all()


def test_profiles_have_stable_ids_and_callable_names() -> None:
    base = get_composite_loss_profile("rmse_mse_mae")
    scaled = get_composite_loss_profile("10x_rmse_mse_mae")

    assert base.scale == 1.0
    assert base.loss.__name__ == "rmse_mse_mae_padded_loss"
    assert scaled.scale == 10.0
    assert scaled.loss.__name__ == "ten_x_rmse_mse_mae_padded_loss"
