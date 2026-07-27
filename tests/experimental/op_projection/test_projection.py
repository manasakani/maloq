from __future__ import annotations

from types import SimpleNamespace

import torch
from e3nn.o3 import Irreps

from maloq.experimental.op_projection import (
    OpProjectionHead,
    PackedAOBlockMatvec,
    bind_operator_callback,
    probe_matrix_mse,
    rademacher_probes,
)


def _one_orbital_metadata():
    transform = SimpleNamespace(
        sort=None,
        in_slices=[0, 1],
        wms=[torch.ones(1, 1, 1)],
    )
    template = [[] for _ in range(100**2)]
    template[100 * 1 + 1] = [(slice(0, 1), slice(0, 1), slice(0, 1))]
    return transform, template, {1: [0]}


def test_packed_ao_matvec_matches_dense_orientation_and_backpropagates() -> None:
    transform, template, basis = _one_orbital_metadata()
    matvec = PackedAOBlockMatvec(
        basis_transformation=transform,
        orbital_template=template,
        orbital_basis=basis,
    )
    atomic_numbers = torch.tensor([1, 1])
    ao_ptr = matvec.make_ao_ptr(atomic_numbers)
    probe = torch.randn(2, 4)
    node_coupled = torch.tensor([[2.0], [3.0]], requires_grad=True)
    edge_coupled = torch.tensor([[5.0], [7.0]], requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 0]])

    actual = torch.zeros_like(probe)
    atoms = torch.arange(2)
    matvec.add_coupled_blocks(
        actual,
        node_coupled,
        atoms,
        atoms,
        atomic_numbers,
        ao_ptr,
        probe,
    )
    matvec.add_coupled_blocks(
        actual,
        edge_coupled,
        edge_index[0],
        edge_index[1],
        atomic_numbers,
        ao_ptr,
        probe,
    )

    dense = torch.tensor([[2.0, 5.0], [7.0, 3.0]])
    torch.testing.assert_close(actual, dense @ probe)
    actual.square().mean().backward()
    assert node_coupled.grad is not None
    assert edge_coupled.grad is not None
    assert torch.isfinite(node_coupled.grad).all()
    assert torch.isfinite(edge_coupled.grad).all()


def test_equivariant_projection_head_runs_as_chunked_callback() -> None:
    transform, template, basis = _one_orbital_metadata()
    head = OpProjectionHead(
        required_irreps=Irreps("1x0e"),
        ls_list=torch.tensor([0]),
        orbital_basis=basis,
        orbital_template=template,
        basis_transformation=transform,
        sphere_channels=2,
        hidden_channels=2,
        edge_channels=2,
        num_distance_basis=4,
        cutoff=4.0,
        pair_chunk_size=1,
    )
    node_embeddings = torch.randn(2, 1, 2, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    features = {
        "node_embeddings": node_embeddings,
        "edge_index": edge_index,
        "edge_distance": torch.tensor([1.0, 1.0]),
        "edge_distance_vec": torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        "partition": None,
    }
    batch = SimpleNamespace(atomic_numbers=torch.tensor([1, 1]))
    probe = rademacher_probes(2, 3, device="cpu", dtype=torch.float32)

    callback = bind_operator_callback(features, batch, head)
    predicted = callback(probe)
    target = torch.randn_like(predicted)
    loss = probe_matrix_mse(predicted, target)
    loss.backward()

    assert predicted.shape == probe.shape
    assert torch.isfinite(predicted).all()
    assert node_embeddings.grad is not None
    assert torch.isfinite(node_embeddings.grad).all()
    assert head.last_projection_stats == {
        "num_nodes": 2,
        "num_edges": 2,
        "total_ao": 2,
        "max_pair_chunk": 1,
    }
    trainable_grads = [
        parameter.grad
        for parameter in head.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_grads
    assert all(torch.isfinite(grad).all() for grad in trainable_grads)
