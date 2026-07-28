from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from maloq.experimental.flow_matching import (
    EndpointFlowMatcher,
    FlowMatchingConfig,
)
from maloq.train_utils.loss import rmse_mse_padded_loss


def test_config_is_strict_and_leaves_loss_to_canonical_maloq_config() -> None:
    config = FlowMatchingConfig()
    assert not hasattr(config, "endpoint_loss")
    assert not hasattr(config, "hamiltonian_weight")
    assert not hasattr(config, "time_scaled_loss")
    assert not hasattr(config, "pair_symmetry")
    with pytest.raises(ValidationError, match="Extra inputs"):
        FlowMatchingConfig.model_validate({"unknown": True})
    with pytest.raises(ValidationError, match="Extra inputs"):
        FlowMatchingConfig.model_validate({"endpoint_loss": "masked_frobenius_mse"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        FlowMatchingConfig.model_validate({"time_scaled_loss": False})
    with pytest.raises(ValidationError, match="strictly smaller"):
        FlowMatchingConfig(time_min=0.7, time_max=0.2)


def test_sampling_is_deterministic_and_targets_the_clean_endpoint() -> None:
    matcher = EndpointFlowMatcher(FlowMatchingConfig())
    target = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    graph_index = torch.tensor([0, 0, 1, 1])
    first = matcher.corrupt(
        target,
        graph_index=graph_index,
        generator=torch.Generator().manual_seed(17),
    )
    second = matcher.corrupt(
        target,
        graph_index=graph_index,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(first.time, second.time)
    assert torch.equal(first.source, second.source)
    assert torch.equal(first.state, second.state)
    assert torch.equal(first.clean_endpoint, target)
    assert first.time.shape == (2,)
    assert first.state.shape == target.shape
    assert first.state.dtype == target.dtype


def test_one_time_per_graph_broadcasts_over_node_entries() -> None:
    matcher = EndpointFlowMatcher(FlowMatchingConfig())
    target = torch.ones(4, 2)
    source = torch.zeros_like(target)
    sample = matcher.corrupt(
        target,
        time=torch.tensor([0.25, 0.75]),
        source=source,
        graph_index=torch.tensor([0, 0, 1, 1]),
    )

    assert torch.equal(
        sample.state,
        torch.tensor(
            [
                [0.25, 0.25],
                [0.25, 0.25],
                [0.75, 0.75],
                [0.75, 0.75],
            ]
        ),
    )


def test_prior_and_path_zero_padding_components() -> None:
    matcher = EndpointFlowMatcher(FlowMatchingConfig())
    target = torch.tensor([[1.0, 2.0], [99.0, 99.0]])
    sample = matcher.corrupt(
        target,
        time=torch.tensor([0.5]),
        mask=torch.tensor([True, False]),
        generator=torch.Generator().manual_seed(4),
    )

    assert torch.equal(sample.source[1], torch.zeros(2))
    assert torch.equal(sample.state[1], torch.zeros(2))
    assert torch.equal(sample.clean_endpoint[1], torch.zeros(2))


def test_path_and_frobenius_loss_commute_with_orthogonal_irrep_action() -> None:
    matcher = EndpointFlowMatcher(FlowMatchingConfig())
    endpoint = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], dtype=torch.float64)
    source = torch.tensor([[-0.3, 0.2, 1.5], [2.0, -1.0, 0.1]], dtype=torch.float64)
    prediction = endpoint + torch.tensor(
        [[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]], dtype=torch.float64
    )
    action = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    time = torch.tensor([0.37], dtype=torch.float64)

    base = matcher.corrupt(endpoint, time=time, source=source)
    rotated = matcher.corrupt(
        endpoint @ action.T,
        time=time,
        source=source @ action.T,
    )
    base_loss = rmse_mse_padded_loss(prediction, endpoint)
    rotated_loss = rmse_mse_padded_loss(prediction @ action.T, endpoint @ action.T)

    assert torch.allclose(rotated.state, base.state @ action.T)
    assert torch.allclose(rotated_loss, base_loss)


def test_joint_corruption_shares_one_time_across_node_and_edge_entries() -> None:
    matcher = EndpointFlowMatcher(FlowMatchingConfig())
    node_target = torch.ones(3, 2)
    edge_target = 2.0 * torch.ones(4, 2)
    sample = matcher.corrupt_joint(
        node_target,
        edge_target,
        node_source=torch.zeros_like(node_target),
        edge_source=torch.zeros_like(edge_target),
        node_graph_index=torch.tensor([0, 0, 1]),
        edge_graph_index=torch.tensor([0, 0, 1, 1]),
        time=torch.tensor([0.25, 0.75]),
    )

    assert torch.equal(sample.time, sample.node.time)
    assert torch.equal(sample.time, sample.edge.time)
    assert torch.equal(
        sample.node.state,
        torch.tensor([[0.25, 0.25], [0.25, 0.25], [0.75, 0.75]]),
    )
    assert torch.equal(
        sample.edge.state,
        torch.tensor([[0.5, 0.5], [0.5, 0.5], [1.5, 1.5], [1.5, 1.5]]),
    )


def test_endpoint_velocity_and_canonical_maloq_loss_are_exact() -> None:
    matcher = EndpointFlowMatcher(FlowMatchingConfig())
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    endpoint = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    velocity = matcher.derived_velocity(
        endpoint,
        state,
        torch.tensor([0.5]),
    )

    assert torch.equal(velocity, 2.0 * (endpoint - state))
    assert rmse_mse_padded_loss(endpoint, state).item() == pytest.approx(20.0)
