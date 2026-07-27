"""AO codecs and symmetry projection for coupled Hamiltonian flow states."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor

from maloq.helm.qhflow3 import _transpose_indices_from_edge_index

from .objective import broadcast_mask


class CoupledBasisTransform(Protocol):
    """Transform contract supplied by the canonical Fock target builder."""

    out_js_list: Sequence[tuple[int, int]]
    required_irreps_out: Any

    def get_H(self, net_out: Tensor) -> Tensor: ...

    def get_net_out(self, hamiltonian: Tensor) -> Tensor: ...


def _shell_width(angular_momentum: int) -> int:
    angular_momentum = int(angular_momentum) % 10
    if angular_momentum < 0:
        raise ValueError("Shell angular momenta must be non-negative.")
    return 2 * angular_momentum + 1


class CoupledAOCodec:
    """Convert coupled targets to and from the padded dense AO block.

    ``e3TensorDecomp.get_H`` returns a concatenation of flattened shell-pair
    blocks. That packed order is not row-major dense AO order when more than
    one shell is present, so a direct square reshape is incorrect.
    """

    def __init__(self, basis_transform: CoupledBasisTransform):
        for method in ("get_H", "get_net_out"):
            if not callable(getattr(basis_transform, method, None)):
                raise TypeError(f"basis_transform must provide {method}().")
        out_js_list = getattr(basis_transform, "out_js_list", None)
        if not isinstance(out_js_list, Sequence) or not out_js_list:
            raise TypeError("basis_transform must provide a non-empty out_js_list.")

        pair_count = len(out_js_list)
        shell_count = isqrt(pair_count)
        if shell_count * shell_count != pair_count:
            raise ValueError(
                "out_js_list must contain a full Cartesian product of shell pairs."
            )
        pairs = tuple((int(l1), int(l2)) for l1, l2 in out_js_list)
        row_shells = tuple(
            pairs[index * shell_count][0] for index in range(shell_count)
        )
        col_shells = tuple(pairs[index][1] for index in range(shell_count))
        if tuple(shell % 10 for shell in row_shells) != tuple(
            shell % 10 for shell in col_shells
        ):
            raise ValueError("AO row and column shell layouts must agree.")
        expected_pairs = tuple(
            (row_shells[row], row_shells[col])
            for row in range(shell_count)
            for col in range(shell_count)
        )
        if tuple((l1 % 10, l2 % 10) for l1, l2 in pairs) != tuple(
            (l1 % 10, l2 % 10) for l1, l2 in expected_pairs
        ):
            raise ValueError(
                "out_js_list is not in canonical row-major shell-pair order."
            )

        self.basis_transform = basis_transform
        self.shells = tuple(shell % 10 for shell in row_shells)
        self.shell_widths = tuple(_shell_width(shell) for shell in self.shells)
        offsets = [0]
        for width in self.shell_widths:
            offsets.append(offsets[-1] + width)
        self.shell_offsets = tuple(offsets)
        self.ao_dim = offsets[-1]
        self.packed_dim = sum(
            row_width * col_width
            for row_width in self.shell_widths
            for col_width in self.shell_widths
        )
        if self.packed_dim != self.ao_dim * self.ao_dim:
            raise RuntimeError(
                "Packed shell-pair dimensions do not cover dense AO space."
            )

        irreps = getattr(basis_transform, "required_irreps_out", None)
        coupled_dim = getattr(irreps, "dim", None)
        self.coupled_dim = None if coupled_dim is None else int(coupled_dim)

    @staticmethod
    def _validate_float_2d(value: Tensor, *, name: str) -> None:
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a 2D tensor.")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point.")

    def packed_to_dense(self, packed: Tensor) -> Tensor:
        self._validate_float_2d(packed, name="packed AO blocks")
        if packed.shape[-1] != self.packed_dim:
            raise ValueError(
                f"Packed AO width must be {self.packed_dim}, got {packed.shape[-1]}."
            )
        dense = packed.new_zeros(packed.shape[0], self.ao_dim, self.ao_dim)
        packed_start = 0
        for row in range(len(self.shells)):
            row_start, row_stop = self.shell_offsets[row : row + 2]
            for col in range(len(self.shells)):
                col_start, col_stop = self.shell_offsets[col : col + 2]
                width = (row_stop - row_start) * (col_stop - col_start)
                dense[:, row_start:row_stop, col_start:col_stop] = packed[
                    :, packed_start : packed_start + width
                ].reshape(
                    packed.shape[0],
                    row_stop - row_start,
                    col_stop - col_start,
                )
                packed_start += width
        return dense

    def dense_to_packed(self, dense: Tensor) -> Tensor:
        if (
            not isinstance(dense, Tensor)
            or dense.ndim != 3
            or dense.shape[-2:] != (self.ao_dim, self.ao_dim)
        ):
            raise ValueError(
                "Dense AO blocks must have shape "
                f"[entries, {self.ao_dim}, {self.ao_dim}]."
            )
        if not dense.is_floating_point():
            raise TypeError("Dense AO blocks must be floating point.")
        blocks = []
        for row in range(len(self.shells)):
            row_start, row_stop = self.shell_offsets[row : row + 2]
            for col in range(len(self.shells)):
                col_start, col_stop = self.shell_offsets[col : col + 2]
                blocks.append(
                    dense[:, row_start:row_stop, col_start:col_stop].flatten(1)
                )
        return torch.cat(blocks, dim=-1)

    def decode(self, coupled: Tensor) -> Tensor:
        self._validate_float_2d(coupled, name="coupled AO coefficients")
        if self.coupled_dim is not None and coupled.shape[-1] != self.coupled_dim:
            raise ValueError(
                f"Coupled AO width must be {self.coupled_dim}, got {coupled.shape[-1]}."
            )
        packed = self.basis_transform.get_H(coupled)
        if (
            not isinstance(packed, Tensor)
            or packed.device != coupled.device
            or packed.dtype != coupled.dtype
        ):
            raise ValueError(
                "basis_transform.get_H() must preserve tensor dtype and device."
            )
        return self.packed_to_dense(packed)

    def encode(self, dense: Tensor) -> Tensor:
        packed = self.dense_to_packed(dense)
        coupled = self.basis_transform.get_net_out(packed)
        if (
            not isinstance(coupled, Tensor)
            or coupled.ndim != 2
            or coupled.shape[0] != dense.shape[0]
            or coupled.device != dense.device
            or coupled.dtype != dense.dtype
        ):
            raise ValueError(
                "basis_transform.get_net_out() must return a 2D tensor while "
                "preserving entry count, dtype, and device."
            )
        if self.coupled_dim is not None and coupled.shape[-1] != self.coupled_dim:
            raise ValueError(
                "basis_transform.get_net_out() returned an unexpected width."
            )
        return coupled


class CoupledNodeStateDecoder:
    """Map padded coupled-irrep node states to dense square AO blocks."""

    def __init__(self, basis_transform: CoupledBasisTransform):
        self.codec = CoupledAOCodec(basis_transform)
        self.basis_transform = basis_transform

    def __call__(self, state: Tensor, *, mask: Tensor | None = None) -> Tensor:
        if state.ndim != 2 or not state.is_floating_point():
            raise ValueError(
                "Closed-shell coupled node state must be a 2D float tensor."
            )
        valid = broadcast_mask(mask, state, entry_dim=0)
        sanitized = torch.where(valid, state, torch.zeros_like(state))
        return self.codec.decode(sanitized)


@dataclass(frozen=True)
class ProjectedHamiltonianState:
    """Node and directed-edge states after Hermitian projection."""

    node: Tensor
    edge: Tensor


class HamiltonianSymmetryProjector:
    """Project coupled node/edge states onto real-Hermitian AO constraints."""

    def __init__(self, basis_transform: CoupledBasisTransform):
        self.codec = CoupledAOCodec(basis_transform)

    @staticmethod
    def _validate_state_pair(node: Tensor, edge: Tensor) -> None:
        if (
            not isinstance(node, Tensor)
            or not isinstance(edge, Tensor)
            or node.ndim != 2
            or edge.ndim != 2
        ):
            raise ValueError("Node and edge states must both be 2D tensors.")
        if not node.is_floating_point() or not edge.is_floating_point():
            raise TypeError("Node and edge states must be floating point.")
        if node.shape[-1] != edge.shape[-1]:
            raise ValueError("Node and edge states must share a coupled width.")
        if node.dtype != edge.dtype or node.device != edge.device:
            raise ValueError("Node and edge states must share dtype and device.")

    def __call__(
        self,
        node: Tensor,
        edge: Tensor,
        *,
        edge_index: Tensor,
        node_mask: Tensor | None = None,
        edge_mask: Tensor | None = None,
    ) -> ProjectedHamiltonianState:
        self._validate_state_pair(node, edge)
        node_valid = broadcast_mask(node_mask, node, entry_dim=0)
        edge_valid = broadcast_mask(edge_mask, edge, entry_dim=0)
        if (
            not isinstance(edge_index, Tensor)
            or edge_index.ndim != 2
            or edge_index.shape != (2, edge.shape[0])
        ):
            raise ValueError("edge_index must have shape [2, edge_count].")
        if edge_index.device != edge.device:
            raise ValueError("edge_index and edge state must share a device.")
        if edge_index.numel() and edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long.")

        num_nodes = node.shape[0]
        if edge_index.numel() and (
            int(edge_index.min().item()) < 0
            or int(edge_index.max().item()) >= num_nodes
        ):
            raise ValueError("edge_index contains an out-of-range node index.")
        if edge_index.numel() and bool((edge_index[0] == edge_index[1]).any().item()):
            raise ValueError("Hamiltonian edge states must not include self edges.")

        reverse = _transpose_indices_from_edge_index(
            edge_index,
            num_nodes=num_nodes,
        )
        expected = torch.arange(edge.shape[0], device=edge.device)
        if not torch.equal(reverse.index_select(0, reverse), expected):
            raise ValueError("Reverse-directed edge mapping must be an involution.")

        node_dense = self.codec.decode(
            torch.where(node_valid, node, torch.zeros_like(node))
        )
        edge_dense = self.codec.decode(
            torch.where(edge_valid, edge, torch.zeros_like(edge))
        )
        node_dense = 0.5 * (node_dense + node_dense.transpose(-1, -2))
        reverse_dense = edge_dense.index_select(0, reverse).transpose(-1, -2)
        edge_dense = 0.5 * (edge_dense + reverse_dense)

        projected_node = self.codec.encode(node_dense)
        projected_edge = self.codec.encode(edge_dense)
        projected_node = torch.where(
            node_valid, projected_node, torch.zeros_like(projected_node)
        )
        projected_edge = torch.where(
            edge_valid, projected_edge, torch.zeros_like(projected_edge)
        )
        return ProjectedHamiltonianState(
            node=projected_node,
            edge=projected_edge,
        )
