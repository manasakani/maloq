from __future__ import annotations

import pytest
import torch
from e3nn import o3
from pydantic import ValidationError
from torch import nn

import maloq.experimental.flow_matching.workflow as flow_workflow
from maloq.experimental.flow_matching import (
    EndpointCorruptingLoader,
    EndpointEulerResult,
    EndpointFlowMaloqConfig,
    EndpointFlowTrainer,
    FlowMatchingWorkflow,
    FlowMatchingConfig,
    QHFlow2EndpointWorkflow,
)
from maloq.experimental.matrix_composite_loss import rmse_mse_mae_padded_loss
from maloq.helm.qhf_layer.so3 import SO3_Grid
from maloq.train_utils.loss import rmse_mse_padded_loss
from maloq.train_utils.splittrainer import SplitTrainer
from maloq.train_utils.training_workflow_v2 import TrainingWorkflowV2Fixed


class _TwoShellScalarBasis:
    def __init__(self) -> None:
        self.out_js_list = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.required_irreps_out = o3.Irreps("4x0e")

    @staticmethod
    def get_H(value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def get_net_out(value: torch.Tensor) -> torch.Tensor:
        return value


def _typed_config(
    backbone_type: str = "qhflow3", *, shift: bool = False
) -> EndpointFlowMaloqConfig:
    output_channels = None if backbone_type == "esen" else 64
    return EndpointFlowMaloqConfig.model_validate(
        {
            "model": {
                "backbone_type": backbone_type,
                "head_type": "maloq",
                "output_l_embedding_dim": output_channels,
                "num_edge_layers": 3,
                "mlp_type": "spectral" if backbone_type == "esen" else "grid",
            },
            "loss": {
                "loss_target": "fock_matrix",
                "scale_and_shift": shift,
                "scale_shift_mode": "shift_only",
                "scale_shift_path": "/tmp/shift.pt" if shift else None,
                "compute_uncoupled_loss": False,
                "delta_learning": False,
            },
            "splits": {"distribute_graphs": False, "dist_backend": "gloo"},
            "tracking": {
                "validation_matrix_metrics": True,
                "validation_matrix_metrics_frequency": 1,
            },
        }
    )


@pytest.mark.parametrize("backbone_type", ["esen", "maloq_nte_v2", "qhflow3"])
def test_typed_config_accepts_matched_backbones_and_shift(backbone_type: str) -> None:
    config = _typed_config(backbone_type, shift=True)
    assert config.model.backbone_type == backbone_type
    assert config.loss.scale_and_shift
    assert config.loss.scale_shift_mode == "shift_only"
    assert config.tracking.validation_matrix_metrics


def test_flow_matching_qhflow3_default_grid_is_ten_by_eleven() -> None:
    config = _typed_config("qhflow3")

    assert config.model.qhflow3_grid_resolution is None
    assert config.to_workflow_config()["qhflow3_grid_resolution"] is None
    assert FlowMatchingWorkflow.DEFAULTS["qhflow3_grid_resolution"] is None

    grid = SO3_Grid(4, 4, resolution=None)
    assert (grid.lat_resolution, grid.long_resolution) == (10, 11)


def test_typed_config_rejects_residual_mode_and_reduced_edges() -> None:
    payload = _typed_config().model_dump(mode="python")
    payload["loss"]["delta_learning"] = True
    with pytest.raises(ValidationError, match="residual parameterization"):
        EndpointFlowMaloqConfig.model_validate(payload)
    payload = _typed_config().model_dump(mode="python")
    payload["model"]["reduce_edge"] = True
    with pytest.raises(ValidationError, match="both directed edge blocks"):
        EndpointFlowMaloqConfig.model_validate(payload)


def test_typed_config_rejects_standardization_for_flow_coordinates() -> None:
    payload = _typed_config(shift=True).model_dump(mode="python")
    payload["loss"]["scale_shift_mode"] = "standardize"
    with pytest.raises(ValidationError, match="SHIFT-only"):
        EndpointFlowMaloqConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("train_loss", "expected_unit_error"),
    [
        (rmse_mse_padded_loss, 2.0),
        (rmse_mse_mae_padded_loss, 3.0),
    ],
)
def test_trainer_calls_canonical_super_loop_with_wrapped_loaders(
    monkeypatch: pytest.MonkeyPatch,
    train_loss,
    expected_unit_error: float,
) -> None:
    captured: dict[str, object] = {}

    def fake_train(self, *args, **kwargs):
        captured["self"] = self
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "canonical-loop"

    monkeypatch.setattr(SplitTrainer, "train", fake_train)
    trainer = EndpointFlowTrainer(
        backbone=nn.Identity(),
        head=nn.Identity(),
        head_irreps="irreps",
        config=FlowMatchingConfig(),
    )

    result = trainer.train(
        2,
        train_loss,
        object(),
        object(),
        "cpu",
        train_loader=[object()],
        val_loader=[object()],
        loss_target_string="fock_matrix",
        node_target_name="node_y",
        edge_target_name="y",
        basis_transform=_TwoShellScalarBasis(),
        validation_matrix_metrics=True,
    )

    assert result == "canonical-loop"
    args = captured["args"]
    kwargs = captured["kwargs"]
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[1] is train_loss
    assert args[1](torch.ones(2), torch.zeros(2), None).item() == pytest.approx(
        expected_unit_error
    )
    assert isinstance(kwargs["train_loader"], EndpointCorruptingLoader)
    assert isinstance(kwargs["val_loader"], EndpointCorruptingLoader)
    assert kwargs["validation_matrix_metrics"] is True


def test_workflow_is_fixed_inherited_path_and_validates_feature_profile() -> None:
    assert issubclass(FlowMatchingWorkflow, TrainingWorkflowV2Fixed)
    assert issubclass(QHFlow2EndpointWorkflow, TrainingWorkflowV2Fixed)
    assert "run" not in QHFlow2EndpointWorkflow.__dict__
    workflow = object.__new__(FlowMatchingWorkflow)
    workflow.config = _typed_config().to_workflow_config()

    workflow._validate_backbone_feature_config()

    assert workflow.flow_matching_config == FlowMatchingConfig()


def test_workflow_wraps_backbone_with_common_flow_conditioner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Backbone:
        pass

    class _Wrapper:
        def __init__(
            self,
            base,
            *,
            flow_irreps,
            embedding_lmax,
            embedding_channels,
        ) -> None:
            self.base = base
            captured["base"] = base
            captured["flow_irreps"] = flow_irreps
            captured["embedding_lmax"] = embedding_lmax
            captured["embedding_channels"] = embedding_channels

        def to(self, device):
            captured["device"] = device
            return self

    backbone = _Backbone()
    monkeypatch.setattr(
        TrainingWorkflowV2Fixed,
        "_build_backbone",
        lambda self, required_irreps: backbone,
    )
    monkeypatch.setattr(
        flow_workflow,
        "FlowConditionedBackbone",
        _Wrapper,
    )
    workflow = object.__new__(FlowMatchingWorkflow)
    workflow.config = {"backbone_type": "qhflow3", "output_l_embedding_dim": 64}
    workflow.device = torch.device("cpu")
    required_irreps = o3.Irreps("4x0e")

    result = workflow._build_backbone(required_irreps)

    assert isinstance(result, _Wrapper)
    assert result.base is backbone
    assert captured["flow_irreps"] == required_irreps
    assert captured["embedding_lmax"] == 0
    assert captured["embedding_channels"] == 64
    assert captured["device"] == torch.device("cpu")


def test_workflow_factory_builds_endpoint_trainer() -> None:
    workflow = object.__new__(FlowMatchingWorkflow)
    workflow.flow_matching_config = FlowMatchingConfig()
    workflow.config = {"seed": 44, "run_name": "test", "save_frequency": 2}
    workflow.rank = 1
    workflow.wandb_run = None

    trainer = workflow._build_trainer(
        backbone=nn.Identity(),
        head=nn.Identity(),
        head_irreps="irreps",
    )

    assert isinstance(trainer, EndpointFlowTrainer)
    assert trainer.flow_config == FlowMatchingConfig()
    assert trainer.validation_inference_seed == 44


def test_validation_matrix_metrics_use_deterministic_euler_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Batch:
        node_y = torch.zeros(1, 1)

    trainer = EndpointFlowTrainer(
        backbone=nn.Identity(),
        head=nn.Identity(),
        head_irreps="irreps",
        config=FlowMatchingConfig(),
        validation_inference_seed=44,
    )
    trainer._validation_inference_num_batches = 2
    sampled_seeds: list[int] = []
    canonical_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_sample_batch(
        batch,
        *,
        basis_transform,
        device,
        node_source=None,
        edge_source=None,
        generator=None,
    ) -> EndpointEulerResult:
        del batch, basis_transform, device, node_source, edge_source
        assert generator is not None
        sampled_seeds.append(generator.initial_seed())
        draw = torch.rand((), generator=generator)
        return EndpointEulerResult(
            node=torch.full((2, 2), draw.item()),
            edge=torch.full((2, 2), draw.item() + 1.0),
            times=torch.tensor([0.01, 1.0]),
        )

    def fake_canonical_metrics(
        self,
        batch,
        node_output,
        edge_output,
        node_target,
        edge_target,
        basis_transform,
    ):
        del self, batch, node_target, edge_target, basis_transform
        canonical_outputs.append((node_output.clone(), edge_output.clone()))
        return ("canonical", len(canonical_outputs))

    monkeypatch.setattr(trainer, "sample_batch", fake_sample_batch)
    monkeypatch.setattr(
        SplitTrainer,
        "compute_validation_matrix_error_sums",
        fake_canonical_metrics,
    )
    random_t_node = torch.full((2, 2), -100.0)
    random_t_edge = torch.full((2, 2), -200.0)
    node_target = torch.zeros(2, 2)
    edge_target = torch.zeros(2, 2)
    global_rng_before = torch.random.get_rng_state().clone()

    results = [
        trainer.compute_validation_matrix_error_sums(
            _Batch(),
            random_t_node,
            random_t_edge,
            node_target,
            edge_target,
            _TwoShellScalarBasis(),
        )
        for _ in range(3)
    ]

    torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)
    assert sampled_seeds == [44, 45, 44]
    assert results == [("canonical", 1), ("canonical", 2), ("canonical", 3)]
    torch.testing.assert_close(canonical_outputs[0][0], canonical_outputs[2][0])
    torch.testing.assert_close(canonical_outputs[0][1], canonical_outputs[2][1])
    assert not torch.equal(canonical_outputs[0][0], canonical_outputs[1][0])
    assert not torch.equal(canonical_outputs[0][0], random_t_node)
    assert not torch.equal(canonical_outputs[0][1], random_t_edge)


def test_validation_matrix_metric_variants_share_prior_noise_and_keep_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Batch:
        node_y = torch.zeros(1, 1)

    trainer = EndpointFlowTrainer(
        backbone=nn.Identity(),
        head=nn.Identity(),
        head_irreps="irreps",
        config=FlowMatchingConfig(num_ode_steps=3),
        validation_inference_seed=44,
    )
    samples: list[tuple[int | None, int, torch.Tensor]] = []

    def fake_sample_batch(
        batch,
        *,
        basis_transform,
        device,
        node_source=None,
        edge_source=None,
        generator=None,
        num_ode_steps=None,
    ) -> EndpointEulerResult:
        del batch, basis_transform, device, node_source, edge_source
        assert generator is not None
        draw = torch.rand(2, generator=generator)
        samples.append((num_ode_steps, generator.initial_seed(), draw.clone()))
        step_marker = 3.0 if num_ode_steps is None else float(num_ode_steps)
        return EndpointEulerResult(
            node=torch.tensor([[draw[0].item(), step_marker]]),
            edge=torch.tensor([[draw[1].item(), step_marker]]),
            times=torch.tensor([0.01, 1.0]),
        )

    def fake_canonical_metrics(
        self,
        batch,
        node_output,
        edge_output,
        node_target,
        edge_target,
        basis_transform,
    ):
        del self, batch, node_target, edge_target, basis_transform
        return (
            float(node_output[0, 1]),
            float(edge_output[0, 1]),
            *([1.0] * 10),
        )

    monkeypatch.setattr(trainer, "sample_batch", fake_sample_batch)
    monkeypatch.setattr(
        SplitTrainer,
        "compute_validation_matrix_error_sums",
        fake_canonical_metrics,
    )
    global_rng_before = torch.random.get_rng_state().clone()

    primary, variants = (
        trainer.compute_validation_matrix_error_sums_with_variants(
            _Batch(),
            torch.full((1, 2), -100.0),
            torch.full((1, 2), -200.0),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            _TwoShellScalarBasis(),
        )
    )

    torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)
    assert [sample[:2] for sample in samples] == [(None, 44), (1, 44)]
    torch.testing.assert_close(samples[0][2], samples[1][2])
    assert primary[:2] == (3.0, 3.0)
    assert variants["validation/flow_matching_configured_ode"] is primary
    assert variants["validation/flow_matching_one_shot"][:2] == (1.0, 1.0)
    assert trainer.flow_config.num_ode_steps == 3
    assert trainer._validation_inference_calls == 1


def test_sample_batch_connects_three_step_euler_to_backbone_and_head() -> None:
    class _Batch:
        def __init__(self) -> None:
            self.node_y = torch.zeros(1, 4, 4, dtype=torch.float64)
            self.y = torch.zeros(1, 4, 4, dtype=torch.float64)
            self.node_padding_mask = torch.ones((1, 4, 4), dtype=torch.bool)
            self.edge_padding_mask = torch.ones((1, 4, 4), dtype=torch.bool)
            self.batch = torch.tensor([0, 0, 1, 1])
            self.ptr = torch.tensor([0, 2, 4])
            self.edge_index = torch.tensor(
                [[0, 1, 2, 3], [1, 0, 3, 2]],
                dtype=torch.int32,
            )

        def to(self, device):
            for name, value in vars(self).items():
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.to(device))
            return self

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []

        def forward(self, batch):
            self.calls.append(
                (
                    batch.node_flow_t.clone(),
                    batch.init_ham_t.clone(),
                    batch.edge_flow_t.clone(),
                    batch.t.clone(),
                )
            )
            return batch

    class _Head(nn.Module):
        def __init__(self, node_endpoint, edge_endpoint) -> None:
            super().__init__()
            self.node_endpoint = node_endpoint
            self.edge_endpoint = edge_endpoint

        def forward(self, features, batch):
            assert features is batch
            return self.node_endpoint, self.edge_endpoint

    node_endpoint = torch.tensor(
        [
            [1.0, 2.0, 2.0, 4.0],
            [5.0, 6.0, 6.0, 8.0],
            [9.0, 10.0, 10.0, 12.0],
            [13.0, 14.0, 14.0, 16.0],
        ],
        dtype=torch.float64,
    )
    edge_endpoint = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 3.0, 2.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [5.0, 7.0, 6.0, 8.0],
        ],
        dtype=torch.float64,
    )

    backbone = _Backbone().train()
    head = _Head(node_endpoint, edge_endpoint).train()
    trainer = EndpointFlowTrainer(
        backbone=backbone,
        head=head,
        head_irreps="irreps",
        config=FlowMatchingConfig(num_ode_steps=3),
    )
    node_source = torch.zeros(4, 4, dtype=torch.float64)
    edge_source = torch.zeros(4, 4, dtype=torch.float64)

    batch = _Batch()
    assert batch.edge_index.dtype == torch.int32
    result = trainer.sample_batch(
        batch,
        basis_transform=_TwoShellScalarBasis(),
        device="cpu",
        node_source=node_source,
        edge_source=edge_source,
    )

    assert len(backbone.calls) == 3
    assert all(node.shape == torch.Size([4, 4]) for node, _, _, _ in backbone.calls)
    assert batch.edge_index.dtype == torch.long
    assert all(
        dense.shape == torch.Size([4, 2, 2]) for _, dense, _, _ in backbone.calls
    )
    assert all(edge.shape == torch.Size([4, 4]) for _, _, edge, _ in backbone.calls)
    assert all(time.shape == torch.Size([2]) for _, _, _, time in backbone.calls)
    first_node, _, first_edge, _ = backbone.calls[0]
    second_node, _, second_edge, _ = backbone.calls[1]
    assert torch.count_nonzero(first_node) == 0
    assert torch.count_nonzero(first_edge) == 0
    assert not torch.equal(second_node, first_node)
    assert not torch.equal(second_edge, first_edge)
    torch.testing.assert_close(result.node, node_endpoint)
    torch.testing.assert_close(result.edge, edge_endpoint)
    edge_dense = result.edge.reshape(4, 2, 2)
    torch.testing.assert_close(edge_dense[1], edge_dense[0].T)
    torch.testing.assert_close(edge_dense[3], edge_dense[2].T)
    assert backbone.training and head.training

    backbone.calls.clear()
    one_shot_result = trainer.sample_batch(
        batch,
        basis_transform=_TwoShellScalarBasis(),
        device="cpu",
        node_source=node_source,
        edge_source=edge_source,
        num_ode_steps=1,
    )
    assert len(backbone.calls) == 1
    assert one_shot_result.times.numel() == 2
    torch.testing.assert_close(one_shot_result.node, node_endpoint)
    torch.testing.assert_close(one_shot_result.edge, edge_endpoint)
    assert trainer.flow_config.num_ode_steps == 3
