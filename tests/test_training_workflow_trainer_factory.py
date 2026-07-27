from __future__ import annotations

from maloq.train_utils import splittrainer
from maloq.train_utils.training_workflow import TrainingWorkflow
from maloq.train_utils.training_workflow_fixed import TrainingWorkflowFixed


def test_canonical_trainer_factory_preserves_dependencies_and_config():
    workflow = object.__new__(TrainingWorkflow)
    workflow.config = {
        "run_name": "factory-contract",
        "save_frequency": 17,
    }
    workflow.wandb_run = object()
    backbone = object()
    head = object()
    head_irreps = object()

    trainer = workflow._build_trainer(
        backbone=backbone,
        head=head,
        head_irreps=head_irreps,
    )

    assert type(trainer) is splittrainer.SplitTrainer
    assert trainer.backbone is backbone
    assert trainer.head is head
    assert trainer.head_irreps is head_irreps
    assert trainer.save_frequency == 17
    assert trainer.wandb_run is workflow.wandb_run


class _TrainerSpy:
    def __init__(self):
        self.train_calls = []

    def train(self, *args, **kwargs):
        self.train_calls.append((args, kwargs))


class _FactoryOverrideWorkflow(TrainingWorkflowFixed):
    def _database(self):
        return self.database

    def prepare_loaders(self, database):
        assert database is self.database
        return (
            self.loader,
            self.val_loader,
            self.head_irreps,
            self.basis_transform,
            self.orbital_basis,
            self.ls_list,
        )

    def build_model(self, head_irreps, orbital_basis, ls_list):
        assert head_irreps is self.head_irreps
        assert orbital_basis is self.orbital_basis
        assert ls_list is self.ls_list
        return self.backbone, self.head, self.optimizer

    def _get_scheduler(self, optimizer, train_loader):
        assert optimizer is self.optimizer
        assert train_loader is self.loader
        return self.scheduler

    def _load_resume_state(self, backbone, head, optimizer, scheduler):
        assert backbone is self.backbone
        assert head is self.head
        assert optimizer is self.optimizer
        assert scheduler is self.scheduler
        return self.start_epoch, self.initial_history

    def _build_trainer(self, *, backbone, head, head_irreps):
        self.factory_call = {
            "backbone": backbone,
            "head": head,
            "head_irreps": head_irreps,
        }
        return self.trainer_spy

    def close(self):
        self.closed = True


def test_fixed_workflow_run_dispatches_through_overridden_trainer_factory():
    workflow = object.__new__(_FactoryOverrideWorkflow)
    workflow.config = {
        "train_or_eval": "train",
        "num_epochs": 5,
        "train_loss_fxn": object(),
        "loss_target": "fock_matrix",
        "output_folder": "unused",
        "train_backbone": True,
        "train_head": True,
    }
    workflow.device = object()
    workflow.stop_after_epoch = None
    workflow.database = object()
    workflow.loader = object()
    workflow.val_loader = object()
    workflow.head_irreps = object()
    workflow.basis_transform = object()
    workflow.orbital_basis = object()
    workflow.ls_list = object()
    workflow.backbone = object()
    workflow.head = object()
    workflow.optimizer = object()
    workflow.scheduler = object()
    workflow.start_epoch = 2
    workflow.initial_history = {"total": [1.0, 0.5]}
    workflow.trainer_spy = _TrainerSpy()
    workflow.factory_call = None
    workflow.closed = False

    workflow.run()

    assert workflow.factory_call == {
        "backbone": workflow.backbone,
        "head": workflow.head,
        "head_irreps": workflow.head_irreps,
    }
    assert len(workflow.trainer_spy.train_calls) == 1
    args, kwargs = workflow.trainer_spy.train_calls[0]
    assert args[:5] == (
        5,
        workflow.config["train_loss_fxn"],
        workflow.optimizer,
        workflow.scheduler,
        workflow.device,
    )
    assert kwargs["train_loader"] is workflow.loader
    assert kwargs["val_loader"] is workflow.val_loader
    assert kwargs["node_target_name"] == "node_y"
    assert kwargs["edge_target_name"] == "y"
    assert kwargs["start_epoch"] == workflow.start_epoch
    assert kwargs["initial_history"] is workflow.initial_history
    assert kwargs["checkpoint_callback"].__self__ is workflow
    assert workflow.closed
