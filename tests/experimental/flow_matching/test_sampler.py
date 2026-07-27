from __future__ import annotations

import pytest
import torch

from maloq.experimental.flow_matching import (
    EndpointEulerSampler,
    EndpointPrediction,
    FlowMatchingConfig,
)


def test_three_step_endpoint_euler_reaches_node_and_edge_oracle_endpoints() -> None:
    sampler = EndpointEulerSampler(FlowMatchingConfig(num_ode_steps=3))
    node_source = torch.tensor([[-1.0, 2.0], [0.0, 1.0], [4.0, -2.0], [9.0, 9.0]])
    node_endpoint = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    edge_source = torch.tensor([[-3.0, 1.0], [2.0, 4.0], [7.0, -1.0]])
    edge_endpoint = torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])
    seen: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def oracle(
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        time: torch.Tensor,
    ) -> EndpointPrediction:
        seen.append((node_state.clone(), edge_state.clone(), time.clone()))
        return EndpointPrediction(node=node_endpoint, edge=edge_endpoint)

    result = sampler.sample(
        node_source,
        edge_source,
        node_graph_index=torch.tensor([0, 0, 1, 1]),
        edge_graph_index=torch.tensor([0, 0, 1]),
        predict_endpoint=oracle,
        node_mask=torch.tensor([True, True, True, False]),
        edge_mask=torch.tensor([True, False, True]),
    )

    assert result.times.shape == (4,)
    assert len(seen) == 3
    assert all(time.shape == (2,) for _, _, time in seen)
    assert not torch.equal(seen[1][1], seen[0][1])
    assert torch.allclose(result.node[:3], node_endpoint[:3])
    assert torch.equal(result.node[3], torch.zeros(2))
    assert torch.allclose(result.edge[[0, 2]], edge_endpoint[[0, 2]])
    assert torch.equal(result.edge[1], torch.zeros(2))


def test_joint_euler_sampler_commutes_with_orthogonal_irrep_action() -> None:
    sampler = EndpointEulerSampler(FlowMatchingConfig(num_ode_steps=3))
    node_source = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]])
    node_endpoint = torch.tensor([[0.0, 4.0, -1.0], [3.0, 2.0, 1.0]])
    edge_source = torch.tensor([[0.5, -3.0, 2.0], [4.0, 1.0, -2.0]])
    edge_endpoint = torch.tensor([[1.0, 0.0, 2.0], [-1.0, 3.0, 0.5]])
    action = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    graph_index = torch.tensor([0, 0])

    base = sampler.sample(
        node_source,
        edge_source,
        node_graph_index=graph_index,
        edge_graph_index=graph_index,
        predict_endpoint=lambda node, edge, time: EndpointPrediction(
            node=node_endpoint,
            edge=edge_endpoint,
        ),
    )
    rotated = sampler.sample(
        node_source @ action.T,
        edge_source @ action.T,
        node_graph_index=graph_index,
        edge_graph_index=graph_index,
        predict_endpoint=lambda node, edge, time: EndpointPrediction(
            node=node_endpoint @ action.T,
            edge=edge_endpoint @ action.T,
        ),
    )

    assert torch.allclose(rotated.node, base.node @ action.T)
    assert torch.allclose(rotated.edge, base.edge @ action.T)


def test_project_state_is_applied_to_node_and_edge_after_every_euler_step() -> None:
    num_steps = 4
    sampler = EndpointEulerSampler(FlowMatchingConfig(num_ode_steps=num_steps))
    node_source = torch.tensor([[1.0, -1.0], [2.0, 3.0]])
    edge_source = torch.tensor([[4.0, 0.5], [-2.0, 7.0]])
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def project_state(
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> EndpointPrediction:
        seen.append((node_state.clone(), edge_state.clone()))
        return EndpointPrediction(node=node_state + 1.0, edge=edge_state - 2.0)

    result = sampler.sample(
        node_source,
        edge_source,
        node_graph_index=torch.tensor([0, 0]),
        edge_graph_index=torch.tensor([0, 0]),
        predict_endpoint=lambda node, edge, time: EndpointPrediction(
            node=node,
            edge=edge,
        ),
        project_state=project_state,
    )

    assert len(seen) == num_steps
    for step, (node_state, edge_state) in enumerate(seen):
        torch.testing.assert_close(node_state, node_source + float(step))
        torch.testing.assert_close(edge_state, edge_source - 2.0 * float(step))
    torch.testing.assert_close(result.node, node_source + float(num_steps))
    torch.testing.assert_close(result.edge, edge_source - 2.0 * float(num_steps))


def test_joint_sampler_rejects_mismatched_graph_batches() -> None:
    sampler = EndpointEulerSampler(FlowMatchingConfig())
    with pytest.raises(ValueError, match="same graph batch"):
        sampler.sample(
            torch.zeros(2, 3),
            torch.zeros(3, 3),
            node_graph_index=torch.tensor([0, 0]),
            edge_graph_index=torch.tensor([0, 1, 1]),
            predict_endpoint=lambda node, edge, time: EndpointPrediction(
                node=node,
                edge=edge,
            ),
        )
