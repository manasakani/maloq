import copy

import torch

from maloq.train_utils.optimizers import Muon, SOAP, zeropower_via_newton_schulz5


def test_newton_schulz_update_is_finite_and_shape_preserving():
    generator = torch.Generator().manual_seed(7)
    update = torch.randn(7, 3, generator=generator)

    orthogonalized = zeropower_via_newton_schulz5(update)

    assert orthogonalized.shape == update.shape
    assert torch.isfinite(orthogonalized).all()
    before_condition = torch.linalg.cond(update)
    after_condition = torch.linalg.cond(orthogonalized)
    assert after_condition < before_condition


def test_muon_updates_matrix_and_adamw_parameters_and_restores_state():
    generator = torch.Generator().manual_seed(11)
    matrix = torch.nn.Parameter(torch.randn(6, 4, generator=generator))
    bias = torch.nn.Parameter(torch.randn(6, generator=generator))
    optimizer = Muon(
        [
            {"params": [matrix], "use_muon": True, "lr": 2e-2},
            {"params": [bias], "use_muon": False, "lr": 1e-3},
        ]
    )
    matrix_before = matrix.detach().clone()
    bias_before = bias.detach().clone()

    for _ in range(2):
        loss = matrix.square().mean() + bias.square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert not torch.equal(matrix, matrix_before)
    assert not torch.equal(bias, bias_before)
    assert "momentum_buffer" in optimizer.state[matrix]
    assert "exp_avg_sq" in optimizer.state[bias]

    restored_matrix = torch.nn.Parameter(matrix.detach().clone())
    restored_bias = torch.nn.Parameter(bias.detach().clone())
    restored_optimizer = Muon(
        [
            {"params": [restored_matrix], "use_muon": True, "lr": 2e-2},
            {"params": [restored_bias], "use_muon": False, "lr": 1e-3},
        ]
    )
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    for params, opt in (
        ((matrix, bias), optimizer),
        ((restored_matrix, restored_bias), restored_optimizer),
    ):
        next_loss = params[0].square().mean() + params[1].square().mean()
        next_loss.backward()
        opt.step()
        opt.zero_grad()

    torch.testing.assert_close(matrix, restored_matrix)
    torch.testing.assert_close(bias, restored_bias)


def test_soap_initializes_then_updates_and_restores_state():
    generator = torch.Generator().manual_seed(17)
    parameter = torch.nn.Parameter(torch.randn(4, 3, generator=generator))
    optimizer = SOAP(
        [parameter],
        lr=1e-3,
        precondition_frequency=1,
        max_precond_dim=3,
    )
    initial = parameter.detach().clone()

    parameter.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.testing.assert_close(parameter, initial)
    assert optimizer.state[parameter]["GG"][0] is None
    assert optimizer.state[parameter]["GG"][1].shape == (3, 3)

    for _ in range(2):
        parameter.square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    assert not torch.equal(parameter, initial)
    assert torch.isfinite(parameter).all()

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored_optimizer = SOAP(
        [restored_parameter],
        lr=1e-3,
        precondition_frequency=1,
        max_precond_dim=3,
    )
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    for current, opt in (
        (parameter, optimizer),
        (restored_parameter, restored_optimizer),
    ):
        current.square().mean().backward()
        opt.step()
        opt.zero_grad()

    torch.testing.assert_close(parameter, restored_parameter)
