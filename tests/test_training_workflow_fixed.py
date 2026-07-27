from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from maloq.train_utils.training_workflow_fixed import (
    CHECKPOINT_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    PREVIOUS_CHECKPOINT_NAME,
    TrainingWorkflowFixed,
    _atomic_checkpoint_write,
    load_training_checkpoint,
    resume_signature,
    signature_digest,
)

from maloq.train_utils.training_workflow_v2 import (
    TrainingWorkflowV2,
    TrainingWorkflowV2Fixed,
)


def test_v2_fixed_workflow_combines_v2_model_and_resume_contracts():
    assert issubclass(TrainingWorkflowV2Fixed, TrainingWorkflowV2)
    assert TrainingWorkflowV2Fixed.SUPPORTED_BACKBONE_TYPES == frozenset(
        {"esen", "maloq_nte_v2", "qhflow3"}
    )


def _checkpoint_payload(marker: int, world_size: int = 1) -> dict:
    signature = resume_signature({"dataset_name": "test"}, world_size)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "completed_epoch": marker,
        "optimizer_step": marker + 1,
        "optimizer_steps_per_epoch": 1,
        "world_size": world_size,
        "backbone_state_dict": {},
        "head_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "rng_states": {str(rank): {} for rank in range(world_size)},
        "history": {"node": [], "node_val": [], "total": []},
        "config_signature": signature,
        "config_signature_digest": signature_digest(signature),
    }


def test_atomic_checkpoint_rotates_and_falls_back(tmp_path: Path):
    _atomic_checkpoint_write(_checkpoint_payload(0), tmp_path)
    _atomic_checkpoint_write(_checkpoint_payload(1), tmp_path)

    assert (tmp_path / CHECKPOINT_NAME).is_file()
    assert (tmp_path / PREVIOUS_CHECKPOINT_NAME).is_file()

    current, current_path = load_training_checkpoint(tmp_path)
    assert current["completed_epoch"] == 1
    assert current_path.name == CHECKPOINT_NAME

    (tmp_path / CHECKPOINT_NAME).write_bytes(b"interrupted checkpoint")
    previous, previous_path = load_training_checkpoint(tmp_path)
    assert previous["completed_epoch"] == 0
    assert previous_path.name == PREVIOUS_CHECKPOINT_NAME


def test_fixed_resume_restores_optimizer_scheduler_and_epoch(tmp_path: Path):
    backbone = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(
        [*backbone.parameters(), *head.parameters()],
        lr=0.1,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / 10,
    )
    for _ in range(5):
        optimizer.zero_grad()
        (head(backbone(torch.ones(1, 2))).sum()).backward()
        optimizer.step()
        scheduler.step()

    checkpoint_config = {
        "dataset_name": "test",
        "num_epochs": 10,
        "output_folder": str(tmp_path / "continued"),
    }
    signature = resume_signature(checkpoint_config, 1)
    payload = {
        **_checkpoint_payload(4),
        "backbone_state_dict": backbone.state_dict(),
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_states": {
            "0": {
                "python": __import__("random").getstate(),
                "numpy": __import__("numpy").random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": None,
            }
        },
        "config_signature": signature,
        "config_signature_digest": signature_digest(signature),
    }
    checkpoint_dir = tmp_path / "checkpoint"
    _atomic_checkpoint_write(payload, checkpoint_dir)

    resumed_backbone = torch.nn.Linear(2, 2)
    resumed_head = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.Adam(
        [*resumed_backbone.parameters(), *resumed_head.parameters()],
        lr=0.1,
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer,
        lambda step: (step + 1) / 10,
    )
    workflow = object.__new__(TrainingWorkflowFixed)
    workflow.resume_source = checkpoint_dir
    workflow.config = {
        **checkpoint_config,
        "atom_scalar_embedding_mode": "element_charge_spin",
        "compute_uncoupled_loss": False,
        "compute_eigenvalues": True,
        "dataset_format": "auto",
        "omol_csh_metadata_policy": "preserve",
    }
    workflow.world_size = 1
    workflow.rank = 0
    workflow.device = torch.device("cpu")
    workflow.allow_config_mismatch = False
    workflow.loaded_checkpoint_path = None

    start_epoch, _ = workflow._load_resume_state(
        resumed_backbone,
        resumed_head,
        resumed_optimizer,
        resumed_scheduler,
    )

    assert start_epoch == 5
    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
    for expected, actual in zip(
        backbone.parameters(),
        resumed_backbone.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def _distributed_checkpoint_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    output_directory: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(100 + rank)
        backbone = torch.nn.Linear(2, 2)
        head = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(
            [*backbone.parameters(), *head.parameters()],
            lr=0.01,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 1.0,
        )
        workflow = object.__new__(TrainingWorkflowFixed)
        workflow.rank = rank
        workflow.world_size = world_size
        workflow.device = torch.device("cpu")
        workflow.config = {
            "dataset_name": "distributed-test",
            "num_epochs": 1,
            "output_folder": output_directory,
        }
        workflow.wandb_run = None
        workflow.loaded_checkpoint_path = None
        workflow._checkpoint_callback(
            epoch=0,
            backbone=backbone,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            history={
                "node": [1.0],
                "node_val": [2.0],
                "total": [3.0],
            },
            optimizer_steps_per_epoch=4,
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_checkpoint_collects_rank_rng_states(tmp_path: Path):
    rendezvous = tmp_path / "gloo-rendezvous"
    output_directory = tmp_path / "distributed-output"
    mp.spawn(
        _distributed_checkpoint_worker,
        args=(2, str(rendezvous), str(output_directory)),
        nprocs=2,
        join=True,
    )

    state, _ = load_training_checkpoint(output_directory)
    assert state["world_size"] == 2
    assert sorted(state["rng_states"]) == ["0", "1"]
    assert state["optimizer_step"] == 4
    assert os.path.isfile(output_directory / "head_training_loss.txt")
