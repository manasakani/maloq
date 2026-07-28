from __future__ import annotations

from maloq.experimental.matrix_composite_loss import workflow


def test_workflow_adapter_replaces_only_training_loss(monkeypatch) -> None:
    captured = {}

    class DummyWorkflow:
        def __init__(self, config):
            captured.update(config)

    monkeypatch.setattr(workflow, "TrainingWorkflowV2Fixed", DummyWorkflow)
    original_loss = object()
    original = {
        "train_loss_fxn": original_loss,
        "run_name": "matched-baseline",
        "num_epochs": 20,
    }

    result = workflow.build_matrix_composite_loss_workflow(
        original,
        profile_id="rmse_mse_mae",
    )

    assert isinstance(result, DummyWorkflow)
    assert original["train_loss_fxn"] is original_loss
    assert captured["train_loss_fxn"].__name__ == "rmse_mse_mae_padded_loss"
    assert captured["matrix_composite_loss_profile"] == "rmse_mse_mae"
    assert captured["matrix_composite_loss_scale"] == 1.0
    assert captured["run_name"] == "matched-baseline"
    assert captured["num_epochs"] == 20
