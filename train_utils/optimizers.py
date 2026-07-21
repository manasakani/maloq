"""Project-local implementations of SOAP and Muon optimizers.

The implementations follow the algorithms published by the authors:

* SOAP: https://arxiv.org/abs/2409.11321
* Muon: https://kellerjordan.github.io/posts/muon/

Muon is intended for hidden-layer matrix parameters. Parameters that are not
marked with ``use_muon=True`` are updated with AdamW in the same optimizer, so
one state dict is sufficient for checkpointing and restart.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


def zeropower_via_newton_schulz5(
    update: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximately orthogonalize a matrix update with quintic Newton-Schulz.

    Tensors with more than two dimensions are treated as a matrix whose first
    dimension is the row dimension, matching the Muon reference treatment of
    convolutional kernels.
    """

    if update.ndim < 2:
        raise ValueError("Muon requires parameters with at least two dimensions.")
    if steps < 1:
        raise ValueError("Newton-Schulz steps must be positive.")

    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1)
    rows, cols = matrix.shape
    compute_dtype = torch.bfloat16 if matrix.is_cuda else torch.float32
    x = matrix.to(dtype=compute_dtype)

    transposed = rows > cols
    if transposed:
        x = x.mT

    x = x / x.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x

    if transposed:
        x = x.mT

    # The aspect-ratio correction is part of the Muon reference update.
    x = x * math.sqrt(max(1.0, rows / cols))
    return x.reshape(original_shape).to(dtype=update.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for matrix parameters with an AdamW fallback parameter group.

    Each parameter group can set ``use_muon``. Muon groups must contain only
    tensors with at least two dimensions; auxiliary groups use AdamW and may
    contain parameters of any shape.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-10,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid Newton-Schulz step count: {ns_steps}")
        if any(not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid AdamW betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            use_muon=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients.")
            if parameter.ndim < 2:
                raise ValueError(
                    "A use_muon=True group contains a parameter with fewer than "
                    "two dimensions. Put scalar/vector parameters in an AdamW group."
                )

            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter.grad)
            buffer = state["momentum_buffer"]
            buffer.lerp_(parameter.grad, 1.0 - momentum)
            update = (
                parameter.grad.lerp(buffer, momentum)
                if group["nesterov"]
                else buffer
            )
            update = zeropower_via_newton_schulz5(
                update, steps=group["ns_steps"]
            )

            if group["weight_decay"]:
                parameter.mul_(1.0 - lr * group["weight_decay"])
            parameter.add_(update, alpha=-lr)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError(
                    "The Muon AdamW group does not support sparse gradients."
                )

            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter.grad)
                state["exp_avg_sq"] = torch.zeros_like(parameter.grad)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.lerp_(parameter.grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                parameter.grad, parameter.grad, value=1.0 - beta2
            )

            if group["weight_decay"]:
                parameter.mul_(1.0 - lr * group["weight_decay"])
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2))
            denominator.add_(group["eps"])
            parameter.addcdiv_(
                exp_avg, denominator, value=-(lr / bias_correction1)
            )


class SOAP(torch.optim.Optimizer):
    """Shampoo preconditioning combined with Adam in its eigenbasis.

    ``max_precond_dim`` is a memory guard: an axis larger than the limit uses
    an identity basis and does not allocate its quadratic covariance matrix.
    The first call initializes the preconditioner, as in the reference code,
    and parameter updates begin on the second call.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 256,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if any(not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid Adam betas: {betas}")
        if shampoo_beta >= 1.0:
            raise ValueError(f"Invalid Shampoo beta: {shampoo_beta}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if precondition_frequency < 1:
            raise ValueError("SOAP precondition frequency must be positive.")
        if max_precond_dim < 1:
            raise ValueError("SOAP max preconditioner dimension must be positive.")

        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            precondition_1d=precondition_1d,
            normalize_grads=normalize_grads,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients.")

                grad = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    self._initialize_preconditioner(grad, state, group)
                    self._update_preconditioner(grad, state, group)
                    # The reference algorithm avoids projecting with a basis
                    # estimated from the same gradient, so initialization does
                    # not update parameters.
                    continue

                projected_grad = self._project(grad, state["Q"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.lerp_(projected_grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    projected_grad, projected_grad, value=1.0 - beta2
                )
                denominator = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size *= math.sqrt(bias_correction2) / bias_correction1

                update = self._project_back(exp_avg / denominator, state["Q"])
                if group["normalize_grads"]:
                    rms = update.square().mean().sqrt().clamp_min(1e-30)
                    update = update / rms

                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-step_size)
                self._update_preconditioner(grad, state, group)

        return loss

    @staticmethod
    def _initialize_preconditioner(
        grad: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
    ) -> None:
        covariance_dtype = (
            torch.float32
            if grad.dtype in (torch.float16, torch.bfloat16)
            else grad.dtype
        )
        covariances: list[torch.Tensor | None] = []
        for axis_size in grad.shape:
            eligible = axis_size <= group["max_precond_dim"]
            if grad.ndim == 1 and not group["precondition_1d"]:
                eligible = False
            covariance = (
                torch.zeros(
                    axis_size,
                    axis_size,
                    device=grad.device,
                    dtype=covariance_dtype,
                )
                if eligible
                else None
            )
            covariances.append(covariance)

        state["GG"] = covariances
        state["Q"] = None
        state["shampoo_beta"] = (
            group["shampoo_beta"]
            if group["shampoo_beta"] >= 0.0
            else group["betas"][1]
        )

    @staticmethod
    def _axis_gram(grad: torch.Tensor, axis: int) -> torch.Tensor:
        if grad.ndim == 1:
            return torch.outer(grad, grad)
        contraction_axes = [index for index in range(grad.ndim) if index != axis]
        return torch.tensordot(
            grad,
            grad,
            dims=(contraction_axes, contraction_axes),
        )

    @staticmethod
    def _eigenbases(
        covariances: list[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        bases: list[torch.Tensor | None] = []
        for covariance in covariances:
            if covariance is None:
                bases.append(None)
                continue
            # eigh is more stable in float32/64 than in model low precision.
            symmetric = 0.5 * (covariance + covariance.mT)
            _, basis = torch.linalg.eigh(symmetric)
            bases.append(torch.flip(basis, dims=(1,)))
        return bases

    @staticmethod
    def _project(
        tensor: torch.Tensor,
        bases: list[torch.Tensor | None],
    ) -> torch.Tensor:
        result = tensor
        for basis in bases:
            if basis is None:
                result = result.movedim(0, -1)
            else:
                typed_basis = basis.to(dtype=result.dtype)
                result = torch.tensordot(result, typed_basis, dims=([0], [0]))
        return result

    @staticmethod
    def _project_back(
        tensor: torch.Tensor,
        bases: list[torch.Tensor | None],
    ) -> torch.Tensor:
        result = tensor
        for basis in bases:
            if basis is None:
                result = result.movedim(0, -1)
            else:
                typed_basis = basis.to(dtype=result.dtype)
                result = torch.tensordot(result, typed_basis, dims=([0], [1]))
        return result

    def _qr_bases(
        self,
        state: dict[str, Any],
    ) -> list[torch.Tensor | None]:
        bases: list[torch.Tensor | None] = []
        exp_avg_sq = state["exp_avg_sq"]
        for axis, (covariance, old_basis) in enumerate(
            zip(state["GG"], state["Q"])
        ):
            if covariance is None or old_basis is None:
                bases.append(None)
                continue
            work_basis = old_basis.to(dtype=covariance.dtype)
            estimated_eigenvalues = torch.diag(
                work_basis.mT @ covariance @ work_basis
            )
            order = torch.argsort(estimated_eigenvalues, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(axis, order)
            work_basis = work_basis[:, order]
            new_basis, _ = torch.linalg.qr(covariance @ work_basis)
            bases.append(new_basis.to(dtype=old_basis.dtype))
        state["exp_avg_sq"] = exp_avg_sq
        return bases

    def _update_preconditioner(
        self,
        grad: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
    ) -> None:
        if state["Q"] is not None:
            state["exp_avg"] = self._project_back(
                state["exp_avg"], state["Q"]
            )

        for axis, covariance in enumerate(state["GG"]):
            if covariance is None:
                continue
            gram = self._axis_gram(grad.to(dtype=covariance.dtype), axis)
            covariance.lerp_(gram, 1.0 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self._eigenbases(state["GG"])
        elif state["step"] > 0 and (
            state["step"] % group["precondition_frequency"] == 0
        ):
            state["Q"] = self._qr_bases(state)

        if state["step"] > 0:
            state["exp_avg"] = self._project(state["exp_avg"], state["Q"])
