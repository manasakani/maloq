from __future__ import annotations

import pytest
import torch
from e3nn import o3

from maloq.experimental.flow_matching import (
    CoupledAOCodec,
    CoupledNodeStateDecoder,
    EndpointCorruptingLoader,
    EndpointFlowMatcher,
    FlowMatchingConfig,
)
from maloq.fock_utils.utils_tensor_decomp import e3TensorDecomp


class _TwoShellScalarBasis:
    def __init__(self) -> None:
        self.out_js_list = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.required_irreps_out = o3.Irreps("4x0e")
        self.last_input: torch.Tensor | None = None

    def get_H(self, net_out: torch.Tensor) -> torch.Tensor:
        self.last_input = net_out
        return net_out

    def get_net_out(self, hamiltonian: torch.Tensor) -> torch.Tensor:
        return hamiltonian


class _Batch:
    def __init__(self) -> None:
        self.node_y = torch.arange(1, 17, dtype=torch.float64).reshape(1, 4, 4)
        self.y = torch.arange(21, 37, dtype=torch.float64).reshape(1, 4, 4)
        self.node_padding_mask = torch.ones((1, 4, 4), dtype=torch.bool)
        self.edge_padding_mask = torch.ones((1, 4, 4), dtype=torch.bool)
        self.batch = torch.tensor([0, 0, 1, 1])
        self.ptr = torch.tensor([0, 2, 4])
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32)

    def to(self, device: torch.device | str):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


def test_decoder_masks_coupled_state_and_reshapes_square_ao_blocks() -> None:
    basis = _TwoShellScalarBasis()
    decoder = CoupledNodeStateDecoder(basis)
    state = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    decoded = decoder(state, mask=torch.tensor([True, False, True]))

    assert decoded.shape == (3, 2, 2)
    assert decoded.dtype == state.dtype
    assert decoded.device == state.device
    assert basis.last_input is not None
    assert torch.equal(basis.last_input[1], torch.zeros(4, dtype=state.dtype))


def test_decoder_rejects_wrong_packed_width() -> None:
    class _BadBasis(_TwoShellScalarBasis):
        def get_H(self, net_out: torch.Tensor) -> torch.Tensor:
            return net_out[:, :3]

    with pytest.raises(ValueError, match="Packed AO width must be 4"):
        CoupledNodeStateDecoder(_BadBasis())(torch.ones(2, 4))


def test_real_wigner_decoder_commutes_with_rotation() -> None:
    basis = e3TensorDecomp(
        net_irreps_out=None,
        out_js_list=[(1, 1)],
        default_dtype_torch=torch.float64,
    )
    decoder = CoupledNodeStateDecoder(basis)
    matrix = torch.tensor(
        [
            [0.7, -0.2, 0.4],
            [0.3, 1.1, -0.5],
            [-0.6, 0.8, 0.2],
        ],
        dtype=torch.float64,
    )
    coupled = basis.get_net_out(matrix.reshape(1, -1))
    rotation = o3.angles_to_matrix(
        torch.tensor(0.37, dtype=torch.float64),
        torch.tensor(1.11, dtype=torch.float64),
        torch.tensor(-0.62, dtype=torch.float64),
    )
    orbital_rotation = o3.Irrep(1, 1).D_from_matrix(rotation)
    coupled_rotation = basis.required_irreps_out.D_from_matrix(rotation)

    decoded = decoder(coupled)
    decoded_rotated = decoder(coupled @ coupled_rotation.T)
    expected_rotated = orbital_rotation @ matrix @ orbital_rotation.T

    torch.testing.assert_close(decoded[0], matrix, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        decoded_rotated[0],
        expected_rotated,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_loader_corrupts_node_and_edge_states_with_shared_graph_time() -> None:
    basis = _TwoShellScalarBasis()
    original = _Batch()
    wrapped = EndpointCorruptingLoader(
        [original],
        matcher=EndpointFlowMatcher(FlowMatchingConfig()),
        basis_transform=basis,
        device="cpu",
    )

    assert original.edge_index.dtype == torch.int32
    batch = next(iter(wrapped))
    codec = CoupledAOCodec(basis)
    edge_state_dense = codec.decode(batch.edge_flow_t)
    edge_endpoint_dense = codec.decode(batch.y[0])

    assert len(wrapped) == 1
    assert batch.t.shape == (2,)
    assert bool((batch.t >= 0.01).all())
    assert batch.edge_index.dtype == torch.long
    assert bool((batch.t <= 0.99).all())
    assert batch.node_flow_t.shape == (4, 4)
    assert batch.init_ham_t.shape == (4, 2, 2)
    assert batch.edge_flow_t.shape == (4, 4)
    torch.testing.assert_close(codec.decode(batch.node_flow_t), batch.init_ham_t)
    assert not torch.allclose(
        batch.edge_flow_t,
        batch.y[0],
    )
    torch.testing.assert_close(
        batch.init_ham_t,
        batch.init_ham_t.transpose(-1, -2),
    )
    for first, reverse in ((0, 1), (2, 3)):
        torch.testing.assert_close(
            edge_state_dense[reverse],
            edge_state_dense[first].T,
        )
        torch.testing.assert_close(
            edge_endpoint_dense[reverse],
            edge_endpoint_dense[first].T,
        )


def test_loader_seed_is_reproducible() -> None:
    def first_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(7)
        wrapped = EndpointCorruptingLoader(
            [_Batch()],
            matcher=EndpointFlowMatcher(FlowMatchingConfig()),
            basis_transform=_TwoShellScalarBasis(),
            device="cpu",
        )
        batch = next(iter(wrapped))
        return batch.t, batch.init_ham_t, batch.edge_flow_t

    first_t, first_node_state, first_edge_state = first_state()
    second_t, second_node_state, second_edge_state = first_state()
    assert torch.equal(first_t, second_t)
    assert torch.equal(first_node_state, second_node_state)
    assert torch.equal(first_edge_state, second_edge_state)
