"""Optional matrix conditioning for the eSEN/MALOQ-NTE input scalar."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from .qhf_layer.embedding import ChgSpinEmbedding
from .qhflow3_clean import (
    ParamContraction,
    _orbital_masks_for_basis,
    get_time_embedding,
)


class NTEMatrixConditioning(nn.Module):
    """Build an invariant per-atom scalar before NTE edge-degree embedding.

    ``overlap`` adds only an overlap-derived scalar to NTE's existing
    atom/charge/spin scalar. ``qhflow3_exact`` replaces that scalar with the
    active QHFlow3 zero-H/S/time/atom mixing path while leaving all subsequent
    NTE node and edge blocks unchanged.
    """

    MODES = {"overlap", "qhflow3_exact"}

    def __init__(
        self,
        *,
        mode: Literal["overlap", "qhflow3_exact"],
        basis: str,
        hidden_size: int,
        delta_learning: bool = False,
        delta_target: Literal["fock_matrix", "density_matrix"] = "fock_matrix",
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported NTE matrix conditioning mode: {mode!r}.")
        if delta_target not in {"fock_matrix", "density_matrix"}:
            raise ValueError(
                "delta_target must be 'fock_matrix' or 'density_matrix'."
            )

        self.mode = mode
        self.basis = basis
        self.hidden_size = int(hidden_size)
        self.delta_learning = bool(delta_learning)
        self.delta_target = delta_target
        self._orbital_masks, self.matrix_dim = _orbital_masks_for_basis(basis)
        basis_irreps = {
            "def2-svp": "3x0e + 2x1e + 1x2e",
            "def2-svp-nabla": "5x0e + 4x1e + 3x2e",
        }[basis]
        scalar_irreps = f"{self.hidden_size}x0e"

        self.overlap_contraction = ParamContraction(
            basis_irreps,
            basis_irreps,
            scalar_irreps,
        )
        self.overlap_embedding = nn.Linear(self.hidden_size, self.hidden_size)

        if self.mode == "qhflow3_exact":
            self.primary_contraction = ParamContraction(
                basis_irreps,
                basis_irreps,
                scalar_irreps,
            )
            self.primary_embedding = nn.Linear(
                self.hidden_size,
                self.hidden_size,
            )
            if self.delta_learning:
                self.auxiliary_contraction = ParamContraction(
                    basis_irreps,
                    basis_irreps,
                    scalar_irreps,
                )
                self.auxiliary_embedding = nn.Linear(
                    self.hidden_size,
                    self.hidden_size,
                )
            matrix_mix_count = 5 if self.delta_learning else 4
            self.mix_matrix = nn.Linear(
                matrix_mix_count * self.hidden_size,
                self.hidden_size,
            )
            self.charge_embedding = ChgSpinEmbedding(
                "pos_emb",
                "charge",
                self.hidden_size,
                grad=False,
            )
            self.spin_embedding = ChgSpinEmbedding(
                "pos_emb",
                "spin",
                self.hidden_size,
                grad=False,
            )
            self.mix_csd = nn.Linear(
                2 * self.hidden_size,
                self.hidden_size,
            )

    def _matrix_blocks(
        self,
        batch: Any,
        attribute: str,
        matrix_label: str,
    ) -> torch.Tensor:
        matrices = getattr(batch, attribute, None)
        if matrices is None:
            raise ValueError(
                f"NTE {self.mode} conditioning requires {attribute}."
            )
        if not isinstance(matrices, (list, tuple)):
            matrices = [matrices]
        if not hasattr(batch, "ptr"):
            raise ValueError("NTE matrix conditioning requires PyG batch pointers.")

        ptr = batch.ptr.detach().cpu().tolist()
        if len(matrices) != len(ptr) - 1:
            raise ValueError(
                f"Expected {len(ptr) - 1} {matrix_label} matrices, "
                f"got {len(matrices)}."
            )

        all_blocks = []
        for graph_index, matrix in enumerate(matrices):
            atoms = batch.atomic_numbers[ptr[graph_index] : ptr[graph_index + 1]]
            atoms_cpu = [int(value) for value in atoms.detach().cpu().tolist()]
            masks = []
            for atomic_number in atoms_cpu:
                if atomic_number not in self._orbital_masks:
                    raise ValueError(
                        f"NTE conditioning basis {self.basis!r} does not support "
                        f"Z={atomic_number}."
                    )
                masks.append(self._orbital_masks[atomic_number])

            sizes = [int(mask.numel()) for mask in masks]
            offsets = np.cumsum([0, *sizes]).tolist()
            matrix_tensor = torch.as_tensor(
                matrix,
                device=batch.pos.device,
                dtype=batch.pos.dtype,
            )
            expected = offsets[-1]
            if tuple(matrix_tensor.shape) != (expected, expected):
                raise ValueError(
                    f"{matrix_label} shape {tuple(matrix_tensor.shape)} does not "
                    f"match the {expected} compact orbitals."
                )

            blocks = matrix_tensor.new_zeros(
                len(atoms_cpu),
                self.matrix_dim,
                self.matrix_dim,
            )
            for atom_index, mask_cpu in enumerate(masks):
                mask = mask_cpu.to(device=matrix_tensor.device)
                start, stop = offsets[atom_index], offsets[atom_index + 1]
                blocks[atom_index][mask[:, None], mask[None, :]] = matrix_tensor[
                    start:stop,
                    start:stop,
                ]
            all_blocks.append(blocks)
        return torch.cat(all_blocks, dim=0)

    def _primary_and_auxiliary_blocks(
        self,
        batch: Any,
        overlap_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.delta_learning:
            return torch.zeros_like(overlap_blocks), None
        if self.delta_target == "density_matrix":
            primary_attribute = "initial_density_matrix"
            auxiliary_attribute = "initial_hamiltonian"
        else:
            primary_attribute = "initial_hamiltonian"
            auxiliary_attribute = "initial_density_matrix"
        return (
            self._matrix_blocks(batch, primary_attribute, "primary initial matrix"),
            self._matrix_blocks(
                batch,
                auxiliary_attribute,
                "auxiliary initial matrix",
            ),
        )

    def system_embedding(
        self,
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return QHFlow3's zero-charge/spin scalar for per-block injection."""
        if self.mode != "qhflow3_exact":
            raise ValueError(
                "Per-block system embedding requires mode='qhflow3_exact'."
            )
        zeros = torch.zeros(num_nodes, device=device, dtype=torch.long)
        charge_spin = torch.cat(
            (
                self.charge_embedding(zeros),
                self.spin_embedding(zeros),
            ),
            dim=-1,
        )
        return torch.nn.functional.silu(self.mix_csd(charge_spin)).to(dtype)

    def forward(
        self,
        batch: Any,
        *,
        atom_embedding: torch.Tensor,
        base_scalar: torch.Tensor,
        molecule_indices: torch.Tensor,
    ) -> torch.Tensor:
        overlap_blocks = self._matrix_blocks(
            batch,
            "overlap_matrix",
            "overlap",
        )
        overlap_feature = self.overlap_embedding(
            self.overlap_contraction(overlap_blocks)
        )
        if self.mode == "overlap":
            return base_scalar + overlap_feature

        primary_blocks, auxiliary_blocks = self._primary_and_auxiliary_blocks(
            batch,
            overlap_blocks,
        )
        primary_feature = self.primary_embedding(
            self.primary_contraction(primary_blocks)
        )
        graph_count = int(batch.ptr.shape[0] - 1)
        time_feature = get_time_embedding(
            torch.ones(
                graph_count,
                device=base_scalar.device,
                dtype=base_scalar.dtype,
            ),
            self.hidden_size,
        ).to(base_scalar.dtype)
        time_feature = time_feature.index_select(0, molecule_indices)

        mixed_inputs = [atom_embedding, primary_feature]
        conditioned_scalar = atom_embedding + time_feature + primary_feature
        if auxiliary_blocks is not None:
            auxiliary_feature = self.auxiliary_embedding(
                self.auxiliary_contraction(auxiliary_blocks)
            )
            mixed_inputs.append(auxiliary_feature)
            conditioned_scalar = conditioned_scalar + auxiliary_feature
        mixed_inputs.extend((overlap_feature, time_feature))
        conditioned_scalar = conditioned_scalar + overlap_feature
        conditioned_scalar = conditioned_scalar + self.mix_matrix(
            torch.cat(mixed_inputs, dim=-1)
        )

        return conditioned_scalar + self.system_embedding(
            atom_embedding.shape[0],
            device=atom_embedding.device,
            dtype=atom_embedding.dtype,
        )
