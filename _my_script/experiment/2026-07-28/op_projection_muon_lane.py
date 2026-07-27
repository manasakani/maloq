"""Strict Muon optimizer lane for the NablaDFT operator-projection experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field

from maloq.experimental.op_projection.training import (
    OpProjectionOptimizationConfig,
    OpProjectionTrainingConfig,
)
from maloq.train_utils.optimizers import Muon


EXPECTED_MODEL_VARIANT = "nabladft-ntev2-op-projection-muon"
EXPECTED_WANDB_RUN_NAME = "NablaDFT | NTEV2-OpProjection | AdamW | RAW | V4 MUON"
MUON_GROUP_NAME = "matrix_muon"
AUXILIARY_GROUP_NAME = "auxiliary_adamw"


class OpProjectionMuonOptimizationConfig(OpProjectionOptimizationConfig):
    """Matched Muon recipe with AdamW reserved for non-matrix parameters."""

    optimizer_type: Literal["muon"] = "muon"
    muon_lr: float = Field(default=2.0e-2, gt=0.0)
    muon_momentum: float = Field(default=0.95, ge=0.0, lt=1.0)
    muon_nesterov: Literal[True] = True
    muon_ns_steps: int = Field(default=5, gt=0)


class OpProjectionMuonTrainingConfig(OpProjectionTrainingConfig):
    """Feature-local config that locks the V4 Muon comparison contract."""

    optimization: OpProjectionMuonOptimizationConfig

    def validate_contract(self) -> None:
        super().validate_contract()
        opt = self.optimization
        expected: dict[str, Any] = {
            "muon_lr": 2.0e-2,
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_ns_steps": 5,
            "lr_init": 5.0e-4,
            "weight_decay": 1.0e-4,
            "adamw_betas": (0.9, 0.95),
            "adamw_eps": 1.0e-10,
        }
        for name, wanted in expected.items():
            if getattr(opt, name) != wanted:
                raise ValueError(f"matched Muon recipe requires {name}={wanted!r}")
        if self.model.model_variant != EXPECTED_MODEL_VARIANT:
            raise ValueError(
                f"Muon lane requires model_variant={EXPECTED_MODEL_VARIANT!r}"
            )
        if self.tracking.wandb_run_name != EXPECTED_WANDB_RUN_NAME:
            raise ValueError(
                f"Muon lane requires wandb_run_name={EXPECTED_WANDB_RUN_NAME!r}"
            )

    def wandb_config(
        self,
        *,
        output_folder: str | Path,
        scope: Literal["smoke", "full"],
    ) -> dict[str, Any]:
        payload = super().wandb_config(
            output_folder=output_folder,
            scope=scope,
        )
        opt = self.optimization
        payload.update(
            {
                "optimizer_type": opt.optimizer_type,
                "optimizer_parameter_routing": (
                    "trainable-ndim-ge-2:muon;trainable-ndim-lt-2:adamw"
                ),
                "muon_lr": opt.muon_lr,
                "muon_momentum": opt.muon_momentum,
                "muon_nesterov": opt.muon_nesterov,
                "muon_ns_steps": opt.muon_ns_steps,
                "aux_adamw_lr": opt.lr_init,
                "aux_adamw_betas": list(opt.adamw_betas),
                "aux_adamw_eps": opt.adamw_eps,
            }
        )
        return payload


def _parameter_record(
    name: str,
    parameter: torch.nn.Parameter,
    *,
    group: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(parameter.shape),
        "ndim": parameter.ndim,
        "numel": parameter.numel(),
        "group": group,
    }


def build_muon_optimizer(
    model: torch.nn.Module,
    optimization: OpProjectionMuonOptimizationConfig,
) -> tuple[Muon, dict[str, Any]]:
    """Build the canonical shape-routed Muon/AdamW optimizer and manifest."""

    named = [
        (name, parameter)
        for name, parameter in model.named_parameters(remove_duplicate=True)
        if parameter.requires_grad
    ]
    matrix = [(name, parameter) for name, parameter in named if parameter.ndim >= 2]
    auxiliary = [(name, parameter) for name, parameter in named if parameter.ndim < 2]
    if not matrix or not auxiliary:
        raise ValueError("Muon and auxiliary AdamW groups must both be non-empty")

    all_ids = {id(parameter) for _, parameter in named}
    matrix_ids = {id(parameter) for _, parameter in matrix}
    auxiliary_ids = {id(parameter) for _, parameter in auxiliary}
    if len(all_ids) != len(named):
        raise RuntimeError(
            "duplicate trainable parameter escaped named-parameter dedup"
        )
    if matrix_ids & auxiliary_ids or matrix_ids | auxiliary_ids != all_ids:
        raise RuntimeError("optimizer parameter routing is not an exact partition")

    optimizer = Muon(
        [
            {
                "name": MUON_GROUP_NAME,
                "params": [parameter for _, parameter in matrix],
                "use_muon": True,
                "lr": optimization.muon_lr,
            },
            {
                "name": AUXILIARY_GROUP_NAME,
                "params": [parameter for _, parameter in auxiliary],
                "use_muon": False,
                "lr": optimization.lr_init,
                "betas": optimization.adamw_betas,
                "eps": optimization.adamw_eps,
            },
        ],
        lr=optimization.muon_lr,
        momentum=optimization.muon_momentum,
        nesterov=optimization.muon_nesterov,
        ns_steps=optimization.muon_ns_steps,
        weight_decay=optimization.weight_decay,
        betas=optimization.adamw_betas,
        eps=optimization.adamw_eps,
    )

    parameters = [
        *(
            _parameter_record(name, parameter, group=MUON_GROUP_NAME)
            for name, parameter in matrix
        ),
        *(
            _parameter_record(name, parameter, group=AUXILIARY_GROUP_NAME)
            for name, parameter in auxiliary
        ),
    ]
    routing_payload = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    provenance = {
        "optimizer_class": f"{Muon.__module__}.{Muon.__qualname__}",
        "routing_rule": (
            "all unique trainable parameters with ndim >= 2 use Muon; "
            "all remaining trainable parameters use auxiliary AdamW"
        ),
        "routing_sha256": hashlib.sha256(routing_payload).hexdigest(),
        "groups": {
            MUON_GROUP_NAME: {
                "tensor_count": len(matrix),
                "parameter_count": sum(parameter.numel() for _, parameter in matrix),
                "use_muon": True,
                "lr": optimization.muon_lr,
            },
            AUXILIARY_GROUP_NAME: {
                "tensor_count": len(auxiliary),
                "parameter_count": sum(parameter.numel() for _, parameter in auxiliary),
                "use_muon": False,
                "lr": optimization.lr_init,
            },
        },
        "hyperparameters": {
            "weight_decay": optimization.weight_decay,
            "muon_momentum": optimization.muon_momentum,
            "muon_nesterov": optimization.muon_nesterov,
            "muon_ns_steps": optimization.muon_ns_steps,
            "aux_adamw_betas": list(optimization.adamw_betas),
            "aux_adamw_eps": optimization.adamw_eps,
        },
        "parameters": parameters,
    }
    return optimizer, provenance


def optimizer_learning_rates(optimizer: Muon) -> dict[str, float]:
    """Return unambiguous primary and per-group learning-rate metrics."""

    rates = {
        str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups
    }
    if set(rates) != {MUON_GROUP_NAME, AUXILIARY_GROUP_NAME}:
        raise RuntimeError(f"unexpected optimizer parameter groups: {sorted(rates)}")
    return {
        "optimizer/learning_rate": rates[MUON_GROUP_NAME],
        "optimizer/muon_learning_rate": rates[MUON_GROUP_NAME],
        "optimizer/aux_adamw_learning_rate": rates[AUXILIARY_GROUP_NAME],
    }


def verify_muon_optimizer_state(optimizer: Muon) -> None:
    """Verify that both Muon and AdamW fallback states survived a step/reload."""

    groups = {str(group.get("name")): group for group in optimizer.param_groups}
    if set(groups) != {MUON_GROUP_NAME, AUXILIARY_GROUP_NAME}:
        raise RuntimeError("cannot verify unexpected optimizer parameter groups")
    if not any(
        "momentum_buffer" in optimizer.state.get(parameter, {})
        for parameter in groups[MUON_GROUP_NAME]["params"]
    ):
        raise RuntimeError("Muon optimizer state has no momentum buffer")
    if not any(
        "exp_avg" in optimizer.state.get(parameter, {})
        and "exp_avg_sq" in optimizer.state.get(parameter, {})
        for parameter in groups[AUXILIARY_GROUP_NAME]["params"]
    ):
        raise RuntimeError("auxiliary AdamW optimizer state has no moment buffers")
