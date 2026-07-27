from __future__ import annotations

import torch
from e3nn import o3
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from maloq.helm.qhflow3 import QHFlow3Backbone


def _conditioned_batch(
    *,
    positions: torch.Tensor,
    overlap_blocks: list[torch.Tensor],
    flow_blocks: list[torch.Tensor],
    time: float,
) -> Batch:
    device = positions.device
    atomic_numbers = torch.full((2,), 6, dtype=torch.long, device=device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    displacement = positions[edge_index[1]] - positions[edge_index[0]]
    data = Data(
        pos=positions,
        z=atomic_numbers,
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        edge_attr=torch.cat(
            [displacement.norm(dim=-1, keepdim=True), displacement], dim=-1
        ),
        num_atoms_in_molecule=torch.tensor([2], device=device),
        charge=torch.zeros(1, dtype=torch.long, device=device),
        spin_multiplicity=torch.ones(1, dtype=torch.long, device=device),
    )
    batch = Batch.from_data_list([data]).to(device)
    batch.overlap_matrix = [torch.block_diag(*overlap_blocks).cpu().numpy()]
    batch.init_ham_t = torch.stack(flow_blocks).to(device)
    batch.t = torch.tensor([time], dtype=positions.dtype, device=device)
    return batch


def test_qhflow3_flow_state_is_equivariant_for_general_rotation() -> None:
    """Rotate geometry and AO flow state through the real QHFlow3 backbone."""
    device = torch.device("cpu")
    torch.manual_seed(44)
    model = (
        QHFlow3Backbone(
            sh_lmax=2,
            hidden_size=8,
            bottle_hidden_size=4,
            num_gnn_layers=1,
            num_ham_gnn_layers=1,
            max_radius=12.0,
            radius_embed_dim=8,
            escn_edge_channels=8,
            escn_num_distance_basis=8,
            esen_max_radius=15.0,
            basis="def2-svp",
            default_hamiltonian_input="init_ham",
            grid_resolution=48,
            grid_ffn_chunk_size=None,
        )
        .to(device)
        .eval()
    )
    assert model.default_hamiltonian_input == "init_ham"

    positions = torch.tensor(
        [[0.13, -0.21, 0.08], [0.74, 0.35, -0.42]],
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator().manual_seed(145)
    overlap_blocks = []
    flow_blocks = []
    for _ in range(2):
        raw_overlap = torch.randn(14, 14, generator=generator)
        overlap_blocks.append(raw_overlap @ raw_overlap.T / 14 + 0.25 * torch.eye(14))
        raw_flow = torch.randn(14, 14, generator=generator)
        flow_blocks.append(0.5 * (raw_flow + raw_flow.T))

    cartesian_rotation = o3.angles_to_matrix(
        torch.tensor(0.37),
        torch.tensor(1.11),
        torch.tensor(-0.62),
    )
    xyz_to_yzx = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=cartesian_rotation.dtype,
    )
    internal_rotation = xyz_to_yzx @ cartesian_rotation @ xyz_to_yzx.T
    ao_rotation = Irreps("3x0e + 2x1e + 1x2e").D_from_matrix(internal_rotation)
    rotated_overlap = [ao_rotation @ block @ ao_rotation.T for block in overlap_blocks]
    rotated_flow = [ao_rotation @ block @ ao_rotation.T for block in flow_blocks]

    with torch.no_grad():
        reference = model(
            _conditioned_batch(
                positions=positions,
                overlap_blocks=overlap_blocks,
                flow_blocks=flow_blocks,
                time=0.37,
            )
        )
        zero_state = model(
            _conditioned_batch(
                positions=positions,
                overlap_blocks=overlap_blocks,
                flow_blocks=[torch.zeros_like(block) for block in flow_blocks],
                time=0.37,
            )
        )
        changed_time = model(
            _conditioned_batch(
                positions=positions,
                overlap_blocks=overlap_blocks,
                flow_blocks=flow_blocks,
                time=0.73,
            )
        )
        observed = model(
            _conditioned_batch(
                positions=positions @ cartesian_rotation.T,
                overlap_blocks=rotated_overlap,
                flow_blocks=rotated_flow,
                time=0.37,
            )
        )

    state_effect = max(
        torch.max(torch.abs(reference[name] - zero_state[name])).item()
        for name in ("node_embeddings", "edge_embeddings")
    )
    time_effect = max(
        torch.max(torch.abs(reference[name] - changed_time[name])).item()
        for name in ("node_embeddings", "edge_embeddings")
    )
    assert state_effect > 1.0e-6
    assert time_effect > 1.0e-6

    for embedding_name in ("node_embeddings", "edge_embeddings"):
        for degree in range(3):
            component_slice = slice(degree**2, (degree + 1) ** 2)
            degree_rotation = o3.Irrep(degree, 1).D_from_matrix(internal_rotation)
            expected = torch.einsum(
                "ij,njc->nic",
                degree_rotation,
                reference[embedding_name][:, component_slice, :],
            )
            torch.testing.assert_close(
                observed[embedding_name][:, component_slice, :],
                expected,
                atol=1.0e-4,
                rtol=1.0e-4,
            )
