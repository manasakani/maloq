"""Full-output Hermitian projector used as a correctness oracle."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from maloq.helm.common.irreps_utils import (
    get_all_indices_dict,
    get_all_len,
    get_product_ls,
)


def _shell_list(ls_list: Sequence[int] | Tensor) -> tuple[int, ...]:
    if isinstance(ls_list, Tensor):
        if ls_list.ndim != 1:
            raise ValueError("ls_list must be one-dimensional.")
        values = ls_list.detach().cpu().tolist()
    else:
        values = list(ls_list)
    shells = tuple(int(value) for value in values)
    if not shells or any(shell < 0 for shell in shells):
        raise ValueError("ls_list must contain non-negative angular momenta.")
    return shells


def _coupled_transpose_map(
    shells: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    """Return the exact AO-block transpose permutation and CG exchange sign."""
    starts = get_all_indices_dict(shells)
    coupled_dim = get_all_len(shells)
    indices = torch.empty(coupled_dim, dtype=torch.long)
    signs = torch.empty(coupled_dim, dtype=torch.float32)

    for row, left_l in enumerate(shells):
        for col, right_l in enumerate(shells):
            for degree in get_product_ls(left_l, right_l):
                destination = starts[(row, col, degree)]
                source = starts[(col, row, degree)]
                width = 2 * degree + 1
                indices[destination : destination + width] = torch.arange(
                    source,
                    source + width,
                    dtype=torch.long,
                )
                exchange_sign = -1.0 if (left_l + right_l + degree) % 2 else 1.0
                signs[destination : destination + width] = exchange_sign

    expected = torch.arange(coupled_dim, dtype=torch.long)
    if not torch.equal(indices.index_select(0, indices), expected):
        raise RuntimeError("Coupled transpose permutation must be an involution.")
    if not torch.equal(
        signs * signs.index_select(0, indices),
        torch.ones_like(signs),
    ):
        raise RuntimeError("Coupled transpose signs must square to identity.")
    return indices, signs


class CoupledTranspose(nn.Module):
    """Apply AO-block transpose directly to full coupled-irrep coefficients."""

    def __init__(
        self,
        ls_list: Sequence[int] | Tensor,
        *,
        coupled_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.shells = _shell_list(ls_list)
        indices, signs = _coupled_transpose_map(self.shells)
        if coupled_dim is not None and int(coupled_dim) != indices.numel():
            raise ValueError(
                "Coupled width does not match ls_list: "
                f"{int(coupled_dim)} != {indices.numel()}."
            )
        self.coupled_dim = int(indices.numel())
        self.register_buffer("indices", indices)
        self.register_buffer("signs", signs)

    def forward(self, values: Tensor) -> Tensor:
        if not isinstance(values, Tensor) or values.ndim < 2:
            raise ValueError("Coupled values must have at least two dimensions.")
        if values.shape[-1] != self.coupled_dim:
            raise ValueError(
                f"Expected coupled width {self.coupled_dim}, "
                f"got {values.shape[-1]}."
            )
        if not values.is_floating_point():
            raise TypeError("Coupled values must be floating point.")
        return values.index_select(-1, self.indices) * self.signs.to(
            dtype=values.dtype
        )


def reverse_edge_indices(
    edge_index: Tensor,
    *,
    num_nodes: int,
) -> Tensor:
    """Map each directed edge to its unique reverse-directed partner."""
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError("edge_index must have shape [2, edge_count].")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long.")
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")

    edge_count = int(edge_index.shape[1])
    if edge_count == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    if num_nodes == 0:
        raise ValueError("A non-empty edge_index requires at least one node.")
    if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes:
        raise ValueError("edge_index contains an out-of-range node index.")
    if bool((edge_index[0] == edge_index[1]).any().item()):
        raise ValueError("Hamiltonian pair features must not contain self edges.")

    source, target = edge_index
    edge_hash = source * num_nodes + target
    reverse_hash = target * num_nodes + source
    order = torch.argsort(edge_hash)
    sorted_hash = edge_hash.index_select(0, order)
    positions = torch.searchsorted(sorted_hash, reverse_hash)
    safe_positions = positions.clamp(max=edge_count - 1)
    found = (positions < edge_count) & (
        sorted_hash.index_select(0, safe_positions) == reverse_hash
    )
    if not bool(found.all().item()):
        raise ValueError(
            "Every directed Hamiltonian edge must have a reverse partner."
        )

    reverse = order.index_select(0, safe_positions)
    expected = torch.arange(edge_count, device=edge_index.device)
    if not torch.equal(reverse.index_select(0, reverse), expected):
        raise ValueError("Reverse-edge pairing must be unique and involutive.")
    return reverse


class CoupledHamiltonianProjector(nn.Module):
    """Reference projector; the training head uses reduced irreps instead."""

    def __init__(
        self,
        ls_list: Sequence[int] | Tensor,
        *,
        coupled_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.transpose = CoupledTranspose(ls_list, coupled_dim=coupled_dim)

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        *,
        edge_index: Tensor,
        reverse: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if node.ndim < 2 or edge.ndim != node.ndim:
            raise ValueError("Node and edge outputs must have matching rank >= 2.")
        if node.shape[:-2] != edge.shape[:-2]:
            raise ValueError("Node and edge outputs must share leading dimensions.")
        if node.shape[-1] != edge.shape[-1]:
            raise ValueError("Node and edge outputs must share coupled width.")
        if node.dtype != edge.dtype or node.device != edge.device:
            raise ValueError("Node and edge outputs must share dtype and device.")
        if edge_index.device != edge.device:
            raise ValueError("edge_index and edge output must share a device.")
        if edge_index.shape != (2, edge.shape[-2]):
            raise ValueError(
                "edge_index must have shape [2, edge_output.shape[-2]]."
            )

        if reverse is None:
            reverse = reverse_edge_indices(
                edge_index,
                num_nodes=int(node.shape[-2]),
            )
        else:
            if (
                reverse.dtype != torch.long
                or reverse.device != edge.device
                or reverse.shape != (edge.shape[-2],)
            ):
                raise ValueError(
                    "reverse must be a device-local long tensor of edge count."
                )
            expected = torch.arange(edge.shape[-2], device=edge.device)
            if not torch.equal(reverse.index_select(0, reverse), expected):
                raise ValueError("reverse must be an involution.")

        node_transpose = self.transpose(node)
        reverse_edge = edge.index_select(edge.ndim - 2, reverse)
        edge_transpose = self.transpose(reverse_edge)
        return (
            0.5 * (node + node_transpose),
            0.5 * (edge + edge_transpose),
        )


__all__ = [
    "CoupledHamiltonianProjector",
    "CoupledTranspose",
    "reverse_edge_indices",
]
