# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Equivariant, matrix-free AO operator projection from node latents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear

from ...helm.common.rotation import eulers_to_wigner, init_edge_rot_euler_angles
from ...helm.common.so3 import CoefficientMapping, SO3_Grid
from ...helm.nn.radial import GaussianSmearing
from .block import PairProjectionBlock
from .operator import bind_operator_callback


def _slice_bounds(value: slice) -> tuple[int, int]:
    if value.start is None or value.stop is None or value.step not in (None, 1):
        raise ValueError(f"orbital template requires bounded unit slices, got {value}")
    return int(value.start), int(value.stop)


class CoupledToPackedAO(nn.Module):
    """Device-aware copy of MALOQ's coupled-to-uncoupled transformation."""

    def __init__(self, basis_transformation: Any) -> None:
        super().__init__()
        if getattr(basis_transformation, "sort", None) is not None:
            raise ValueError(
                "op_projection currently requires an unsorted AO transform"
            )
        self.in_slices = tuple(int(value) for value in basis_transformation.in_slices)
        if len(self.in_slices) != len(basis_transformation.wms) + 1:
            raise ValueError("basis transform slices and Wigner blocks disagree")
        packed_dim = 0
        for index, wigner_block in enumerate(basis_transformation.wms):
            block = wigner_block.detach().clone()
            self.register_buffer(f"wigner_block_{index}", block)
            packed_dim += int(block.shape[0] * block.shape[1])
        self.packed_dim = packed_dim

    @property
    def coupled_dim(self) -> int:
        return self.in_slices[-1]

    def forward(self, coupled: torch.Tensor) -> torch.Tensor:
        if coupled.ndim != 2 or coupled.shape[1] != self.coupled_dim:
            raise ValueError(
                "coupled AO coefficients must have shape "
                f"[blocks, {self.coupled_dim}], got {tuple(coupled.shape)}"
            )
        packed = []
        for index, (start, stop) in enumerate(
            zip(self.in_slices[:-1], self.in_slices[1:], strict=True)
        ):
            wigner_block = getattr(self, f"wigner_block_{index}")
            block = torch.einsum(
                "abk,nk->nab",
                wigner_block.to(dtype=coupled.dtype),
                coupled[:, start:stop],
            )
            packed.append(block.flatten(1))
        return torch.cat(packed, dim=1)


class PackedAOBlockMatvec(nn.Module):
    """Apply packed atom-pair AO blocks directly to probe vectors.

    No molecule-scale dense matrix is constructed. ``edge_index[0]`` is the
    AO row atom and ``edge_index[1]`` is the AO column atom, matching MALOQ's
    matrix-to-label convention.
    """

    def __init__(
        self,
        *,
        basis_transformation: Any,
        orbital_template: list,
        orbital_basis: dict[int, list[int]],
        max_num_elements: int = 100,
    ) -> None:
        super().__init__()
        self.max_num_elements = int(max_num_elements)
        self.decode = CoupledToPackedAO(basis_transformation)

        normalized_template = []
        for entries in orbital_template:
            normalized_entries = []
            for row_slice, col_slice, output_slice in entries:
                row_start, row_stop = _slice_bounds(row_slice)
                col_start, col_stop = _slice_bounds(col_slice)
                output_start, output_stop = _slice_bounds(output_slice)
                if (row_stop - row_start) * (col_stop - col_start) != (
                    output_stop - output_start
                ):
                    raise ValueError("orbital template block dimensions disagree")
                normalized_entries.append(
                    (
                        row_start,
                        row_stop,
                        col_start,
                        col_stop,
                        output_start,
                        output_stop,
                    )
                )
            normalized_template.append(tuple(normalized_entries))
        expected_templates = self.max_num_elements**2
        if len(normalized_template) != expected_templates:
            raise ValueError(
                f"expected {expected_templates} element-pair templates, "
                f"got {len(normalized_template)}"
            )
        self.orbital_template = tuple(normalized_template)

        ao_counts = torch.zeros(self.max_num_elements, dtype=torch.long)
        for atomic_number, orbitals in orbital_basis.items():
            ao_counts[int(atomic_number)] = sum(
                2 * (int(degree) % 10) + 1 for degree in orbitals
            )
        self.register_buffer("ao_counts", ao_counts, persistent=True)
        gather_keys = []
        for key, entries in enumerate(self.orbital_template):
            if not entries:
                continue
            row_atomic_number, col_atomic_number = divmod(key, self.max_num_elements)
            row_width = int(ao_counts[row_atomic_number].item())
            col_width = int(ao_counts[col_atomic_number].item())
            if row_width <= 0 or col_width <= 0:
                continue
            gather = torch.full((row_width * col_width,), -1, dtype=torch.long)
            for (
                row_start,
                row_stop,
                col_start,
                col_stop,
                output_start,
                output_stop,
            ) in entries:
                packed_indices = torch.arange(output_start, output_stop).reshape(
                    row_stop - row_start, col_stop - col_start
                )
                local_rows = torch.arange(row_start, row_stop)[:, None]
                local_cols = torch.arange(col_start, col_stop)[None, :]
                dense_indices = local_rows * col_width + local_cols
                gather[dense_indices.flatten()] = packed_indices.flatten()
            if torch.any(gather < 0):
                raise ValueError(f"orbital template {key} does not cover its AO block")
            self.register_buffer(f"packed_gather_{key}", gather, persistent=False)
            gather_keys.append(key)
        self.gather_keys = frozenset(gather_keys)

    def make_ao_ptr(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        atomic_numbers = atomic_numbers.to(
            device=self.ao_counts.device, dtype=torch.long
        )
        if atomic_numbers.ndim != 1:
            raise ValueError("atomic_numbers must be one-dimensional")
        if atomic_numbers.numel() and (
            atomic_numbers.min() < 0 or atomic_numbers.max() >= self.max_num_elements
        ):
            raise ValueError("atomic number is outside the configured basis table")
        counts = self.ao_counts.index_select(0, atomic_numbers)
        if torch.any(counts <= 0):
            unsupported = torch.unique(atomic_numbers[counts <= 0]).tolist()
            raise ValueError(f"orbital basis is missing elements {unsupported}")
        return torch.cat((counts.new_zeros(1), counts.cumsum(0)))

    def add_packed_blocks(
        self,
        output: torch.Tensor,
        packed: torch.Tensor,
        row_atoms: torch.Tensor,
        col_atoms: torch.Tensor,
        atomic_numbers: torch.Tensor,
        ao_ptr: torch.Tensor,
        probe: torch.Tensor,
    ) -> torch.Tensor:
        if packed.ndim != 2 or packed.shape[0] != row_atoms.numel():
            raise ValueError("packed blocks and atom-index arrays disagree")
        if packed.shape[1] != self.decode.packed_dim:
            raise ValueError(
                f"expected packed width {self.decode.packed_dim}, got {packed.shape[1]}"
            )
        if probe.ndim != 2 or output.shape != probe.shape:
            raise ValueError("probe and output must share shape [total_ao, probes]")

        row_atoms = row_atoms.to(device=packed.device, dtype=torch.long)
        col_atoms = col_atoms.to(device=packed.device, dtype=torch.long)
        atomic_numbers = atomic_numbers.to(device=packed.device, dtype=torch.long)
        ao_ptr = ao_ptr.to(device=packed.device, dtype=torch.long)
        keys = self.max_num_elements * atomic_numbers.index_select(
            0, row_atoms
        ) + atomic_numbers.index_select(0, col_atoms)
        num_probes = probe.shape[1]
        for key_tensor in torch.unique(keys):
            key = int(key_tensor.item())
            if key not in self.gather_keys:
                continue
            block_ids = torch.nonzero(keys == key_tensor, as_tuple=False).flatten()
            rows = row_atoms.index_select(0, block_ids)
            cols = col_atoms.index_select(0, block_ids)
            row_atomic_number, col_atomic_number = divmod(key, self.max_num_elements)
            row_width = int(self.ao_counts[row_atomic_number].item())
            col_width = int(self.ao_counts[col_atomic_number].item())
            gather = getattr(self, f"packed_gather_{key}")
            blocks = (
                packed.index_select(0, block_ids)
                .index_select(1, gather)
                .reshape(-1, row_width, col_width)
            )
            local_cols = torch.arange(col_width, device=packed.device)
            probe_indices = ao_ptr.index_select(0, cols)[:, None] + local_cols
            contributions = torch.bmm(blocks, probe[probe_indices])
            local_rows = torch.arange(row_width, device=packed.device)
            output_indices = ao_ptr.index_select(0, rows)[:, None] + local_rows
            output.index_add_(
                0,
                output_indices.flatten(),
                contributions.reshape(-1, num_probes),
            )
        return output

    def add_coupled_blocks(
        self,
        output: torch.Tensor,
        coupled: torch.Tensor,
        row_atoms: torch.Tensor,
        col_atoms: torch.Tensor,
        atomic_numbers: torch.Tensor,
        ao_ptr: torch.Tensor,
        probe: torch.Tensor,
    ) -> torch.Tensor:
        return self.add_packed_blocks(
            output,
            self.decode(coupled),
            row_atoms,
            col_atoms,
            atomic_numbers,
            ao_ptr,
            probe,
        )


class _NativeScalarGate(nn.Module):
    """Invariant scalar conditioning for native spherical tensors."""

    def __init__(self, channels: int, lmax: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.scalar = nn.Linear(self.channels, self.channels)
        self.gates = nn.Linear(self.channels, self.channels * self.lmax)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scalar = features[:, 0, :]
        blocks = [self.scalar(scalar)[:, None, :]]
        if self.lmax:
            gates = torch.sigmoid(self.gates(scalar)).reshape(
                features.shape[0], self.lmax, self.channels
            )
            for degree in range(1, self.lmax + 1):
                blocks.append(
                    features[:, degree**2 : (degree + 1) ** 2, :]
                    * gates[:, degree - 1, None, :]
                )
        return torch.cat(blocks, dim=1)


class EquivariantFockProjection(nn.Module):
    """Map native SO(3) latents to coupled AO coefficients.

    The scalar gates are rotation invariant and the final e3nn maps preserve
    every output irrep. Separate node and pair maps let the two block classes
    learn independently without importing the canonical training workflow.
    """

    def __init__(
        self,
        *,
        sphere_channels: int,
        lmax: int,
        required_irreps: Irreps,
    ) -> None:
        super().__init__()
        self.sphere_channels = int(sphere_channels)
        self.lmax = int(lmax)
        self.irreps_in = Irreps(
            [(self.sphere_channels, (degree, 1)) for degree in range(self.lmax + 1)]
        )
        self.required_irreps = Irreps(required_irreps)
        self.node_gate = _NativeScalarGate(self.sphere_channels, self.lmax)
        self.edge_gate = _NativeScalarGate(self.sphere_channels, self.lmax)
        self.node_linear = Linear(self.irreps_in, self.required_irreps, biases=True)
        self.edge_linear = Linear(self.irreps_in, self.required_irreps, biases=True)

    def _stack_native(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                features[:, degree**2 : (degree + 1) ** 2, :]
                .transpose(1, 2)
                .reshape(features.shape[0], -1)
                for degree in range(self.lmax + 1)
            ],
            dim=1,
        )

    def project_nodes(self, features: torch.Tensor) -> torch.Tensor:
        return self.node_linear(self._stack_native(self.node_gate(features)))

    def project_edges(self, features: torch.Tensor) -> torch.Tensor:
        return self.edge_linear(self._stack_native(self.edge_gate(features)))


class OpProjectionHead(nn.Module):
    """Generate AO blocks lazily and stream their action into probe vectors."""

    def __init__(
        self,
        *,
        required_irreps: Irreps | str,
        ls_list,
        orbital_basis: dict[int, list[int]],
        orbital_template: list,
        basis_transformation: Any,
        sphere_channels: int,
        lmax: int | None = None,
        mmax: int | None = None,
        hidden_channels: int | None = None,
        edge_channels: int | None = None,
        num_distance_basis: int = 32,
        cutoff: float = 10.0,
        gaussian_width: float = 1.0,
        grid_resolution: int | None = None,
        norm_type: str = "rms_norm_sh",
        pair_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        del ls_list
        required_irreps = Irreps(required_irreps)
        self.lmax = required_irreps.lmax if lmax is None else int(lmax)
        if self.lmax != required_irreps.lmax:
            raise ValueError(
                "operator head lmax must match the required AO irreps: "
                f"{self.lmax} != {required_irreps.lmax}"
            )
        self.mmax = self.lmax if mmax is None else int(mmax)
        self.sphere_channels = int(sphere_channels)
        self.hidden_channels = int(hidden_channels or sphere_channels)
        self.edge_channels = int(edge_channels or sphere_channels)
        self.num_distance_basis = int(num_distance_basis)
        self.cutoff = float(cutoff)
        self.gaussian_width = float(gaussian_width)
        self.pair_chunk_size = int(pair_chunk_size)
        if self.pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")

        jd_list = torch.load(Path(__file__).resolve().parents[2] / "helm" / "Jd.pt")
        for degree in range(self.lmax + 1):
            self.register_buffer(f"Jd_{degree}", jd_list[degree])
        self.mapping_reduced = CoefficientMapping(self.lmax, self.mmax)
        self.so3_grid = nn.ModuleDict(
            {
                "lmax_lmax": SO3_Grid(
                    self.lmax,
                    self.lmax,
                    resolution=grid_resolution,
                    rescale=True,
                ),
                "lmax_mmax": SO3_Grid(
                    self.lmax,
                    self.mmax,
                    resolution=grid_resolution,
                    rescale=True,
                ),
            }
        )
        self.distance_expansion = GaussianSmearing(
            0.0,
            self.cutoff,
            self.num_distance_basis,
            self.gaussian_width,
        )
        edge_channels_list = [
            self.num_distance_basis,
            self.edge_channels,
            self.edge_channels,
        ]
        self.pair_encoder = PairProjectionBlock(
            sphere_channels=self.sphere_channels,
            hidden_channels=self.hidden_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mapping=self.mapping_reduced,
            grids=self.so3_grid,
            edge_channels_list=edge_channels_list,
            cutoff=self.cutoff,
            norm_type=norm_type,
        )

        self.fock_projection = EquivariantFockProjection(
            required_irreps=required_irreps,
            lmax=self.lmax,
            sphere_channels=self.sphere_channels,
        )
        self.block_matvec = PackedAOBlockMatvec(
            basis_transformation=basis_transformation,
            orbital_template=orbital_template,
            orbital_basis=orbital_basis,
        )
        self.last_projection_stats: dict[str, int] = {}

    def _get_wigner(
        self, edge_distance_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        jd_buffers = [
            getattr(self, f"Jd_{degree}").to(dtype=edge_distance_vec.dtype)
            for degree in range(self.lmax + 1)
        ]
        euler_angles = init_edge_rot_euler_angles(edge_distance_vec)
        wigner = eulers_to_wigner(euler_angles, 0, self.lmax, jd_buffers)
        return wigner, wigner.transpose(1, 2).contiguous()

    def _project_nodes(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fock_projection.project_nodes(embeddings)

    def _project_edges(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fock_projection.project_edges(embeddings)

    def forward(self, features, batch, probe: torch.Tensor) -> torch.Tensor:
        if features.get("edge_embeddings") is not None:
            raise ValueError("op_projection head requires node-only backbone output")
        nodes = features["node_embeddings"]
        expected_components = (self.lmax + 1) ** 2
        if nodes.ndim != 3 or nodes.shape[1:] != (
            expected_components,
            self.sphere_channels,
        ):
            raise ValueError(
                "node embeddings must have shape "
                f"[nodes, {expected_components}, {self.sphere_channels}], "
                f"got {tuple(nodes.shape)}"
            )

        squeeze_probe = probe.ndim == 1
        if squeeze_probe:
            probe = probe[:, None]
        if probe.ndim != 2:
            raise ValueError("probe must have shape [total_ao] or [total_ao, probes]")

        atomic_numbers = batch.atomic_numbers.to(device=nodes.device, dtype=torch.long)
        ao_ptr = self.block_matvec.make_ao_ptr(atomic_numbers)
        if probe.shape[0] != int(ao_ptr[-1].item()):
            raise ValueError(
                f"probe has {probe.shape[0]} AO rows, expected {int(ao_ptr[-1].item())}"
            )
        probe = probe.to(device=nodes.device, dtype=nodes.dtype)
        output = torch.zeros_like(probe)

        node_atoms = torch.arange(nodes.shape[0], device=nodes.device)
        self.block_matvec.add_coupled_blocks(
            output,
            self._project_nodes(nodes),
            node_atoms,
            node_atoms,
            atomic_numbers,
            ao_ptr,
            probe,
        )

        edge_index = features["edge_index"].to(device=nodes.device, dtype=torch.long)
        edge_distance = features["edge_distance"].to(device=nodes.device)
        edge_distance_vec = features["edge_distance_vec"].to(device=nodes.device)
        normalized_nodes = self.pair_encoder.norm_1(nodes)
        num_edges = edge_index.shape[1]
        max_chunk = 0
        for start in range(0, num_edges, self.pair_chunk_size):
            stop = min(start + self.pair_chunk_size, num_edges)
            edge_chunk = edge_index[:, start:stop]
            distance_chunk = edge_distance[start:stop]
            vector_chunk = edge_distance_vec[start:stop]
            wigner, wigner_inv = self._get_wigner(vector_chunk)
            pair_embeddings = self.pair_encoder.from_normalized_nodes(
                normalized_nodes,
                self.distance_expansion(distance_chunk),
                distance_chunk,
                edge_chunk,
                wigner,
                wigner_inv,
            )
            self.block_matvec.add_coupled_blocks(
                output,
                self._project_edges(pair_embeddings),
                edge_chunk[0],
                edge_chunk[1],
                atomic_numbers,
                ao_ptr,
                probe,
            )
            max_chunk = max(max_chunk, stop - start)

        self.last_projection_stats = {
            "num_nodes": int(nodes.shape[0]),
            "num_edges": int(num_edges),
            "total_ao": int(ao_ptr[-1].item()),
            "max_pair_chunk": int(max_chunk),
        }
        return output[:, 0] if squeeze_probe else output


class OpProjectionModel(nn.Module):
    """Own the backbone/head parameters while invoking an ephemeral callback."""

    def __init__(self, backbone: nn.Module, operator_head: OpProjectionHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.operator_head = operator_head

    def encode(self, batch):
        return self.backbone(batch)

    def forward(self, batch, probe: torch.Tensor) -> torch.Tensor:
        features = self.encode(batch)
        callback = bind_operator_callback(features, batch, self.operator_head)
        return callback(probe)


__all__ = [
    "CoupledToPackedAO",
    "EquivariantFockProjection",
    "OpProjectionHead",
    "OpProjectionModel",
    "PackedAOBlockMatvec",
]
