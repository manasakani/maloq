"""Collation utilities for DFT data with variable-size matrices.

PyTorch DataLoader에서 다양한 크기의 분자 데이터를 batching하기 위한 도구.
GNN-style (concatenate + batch_idx)과 matrix padding을 모두 지원.
On-the-fly graph construction (radius_graph) 및 PBC 지원.

Usage:
    from collate import DFTCollator, BucketBatchSampler, build_dataloader

    # MLIP 학습 (가장 빠름)
    loader = build_dataloader("data.lmdb", preset="mlip", batch_size=64)

    # MLIP + on-the-fly graph + PyG 필드명
    loader = build_dataloader("data.lmdb", preset="mlip_pyg", batch_size=32)

    # Reference 구분
    loader = build_dataloader("data.lmdb", preset="mlip",
                              rename={"energy": "energy_ref", "forces": "forces_ref"})

    # Manual setup
    from collate import GraphOptions
    collator = DFTCollator(
        targets=["energy", "forces"],
        graph=GraphOptions(cutoff=5.0, max_neighbors=32),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.utils.data import ConcatDataset, DataLoader
except ImportError:
    torch = None
    Tensor = None

from .molecule import Molecule, _unpack_matrix_from_dict


# ── Graph Options ────────────────────────────────────────────────

@dataclass
class GraphOptions:
    """On-the-fly graph construction settings.

    Args:
        cutoff: radius cutoff in Angstrom.
        max_neighbors: max edges per atom.
        backend: graph construction backend.
            "pyg" — torch_geometric radius_graph (non-PBC only, fast)
            "ase" — ASE neighbor_list (PBC-aware, CPU, per-sample)
    """
    cutoff: float = 5.0
    max_neighbors: int = 32
    backend: str = "pyg"


# ── Helpers ──────────────────────────────────────────────────────

def _as_numpy(val, dtype_str: str = "int32") -> np.ndarray:
    """bytes or ndarray → ndarray (avoids redundant frombuffer)."""
    if isinstance(val, np.ndarray):
        return val
    if torch is not None and isinstance(val, Tensor):
        return val.numpy()
    return np.frombuffer(val, dtype=np.dtype(dtype_str)).copy()


def _decode_array(d: dict, name: str, num_atoms: int) -> np.ndarray | None:
    """Decode an array field from raw or pre-decoded dict."""
    val = d.get(name)
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val
    if torch is not None and isinstance(val, Tensor):
        return val.numpy()
    # Raw bytes
    dtype = d.get(f"{name}_dtype", "float64")
    shape = d.get(f"{name}_shape", (num_atoms, 3))
    return np.frombuffer(val, dtype=np.dtype(dtype)).reshape(shape)


def _unpack_symmetric(d: dict, prefix: str) -> np.ndarray | None:
    """Extract a (possibly packed) symmetric matrix from a Molecule dict."""
    if prefix in d and isinstance(d[prefix], np.ndarray):
        return d[prefix]
    if torch is not None and prefix in d and isinstance(d[prefix], Tensor):
        return d[prefix].numpy()
    is_packed = d.get("_packed", False)
    return _unpack_matrix_from_dict(d, prefix, is_packed)


def _get_nao_from_dict(d: dict) -> int | None:
    """Extract nao from a sample dict."""
    if "nao" in d:
        return int(d["nao"])
    for prefix in ("density_matrix", "hamiltonian", "overlap"):
        # packed format: {prefix}_nao
        nao = d.get(f"{prefix}_nao")
        if nao is not None:
            return int(nao)
        # unpacked format: {prefix}_shape → (nao, nao)
        shape = d.get(f"{prefix}_shape")
        if shape is not None:
            return int(shape[0])
        # pre-decoded ndarray
        val = d.get(prefix)
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            return int(val.shape[-1])
    return None


# ── Graph Construction ───────────────────────────────────────────

def _build_graph_pyg(
    pos: "Tensor",
    batch: "Tensor",
    cutoff: float,
    max_neighbors: int,
) -> "Tensor":
    """Non-PBC radius graph via torch_geometric."""
    from torch_geometric.nn import radius_graph
    return radius_graph(
        pos, r=cutoff, batch=batch,
        max_num_neighbors=max_neighbors,
    )


def _build_graph_ase_pbc(
    positions_list: list[np.ndarray],
    cells_list: list[np.ndarray],
    pbc_list: list[np.ndarray],
    cutoff: float,
    max_neighbors: int,
) -> tuple["Tensor", "Tensor"]:
    """PBC-aware graph via ASE neighbor_list (per-sample, CPU).

    Returns:
        edge_index: (2, E) int64
        cell_offsets: (E, 3) int64 — shift vectors
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    all_i, all_j, all_S = [], [], []
    offset = 0

    for pos, cell, pbc in zip(positions_list, cells_list, pbc_list):
        atoms = Atoms(positions=pos, cell=cell, pbc=pbc)
        i, j, S = neighbor_list("ijS", atoms, cutoff, self_interaction=False)

        # Enforce max_neighbors per atom
        if max_neighbors is not None and max_neighbors > 0:
            from ase.neighborlist import neighbor_list as _nl
            # Compute distances for sorting
            D = pos[j] - pos[i] + S @ cell
            dists = np.linalg.norm(D, axis=1)
            # Sort by distance per atom, keep top max_neighbors
            n_atoms = len(pos)
            mask = np.ones(len(i), dtype=bool)
            for a in range(n_atoms):
                atom_mask = (i == a)
                if atom_mask.sum() > max_neighbors:
                    atom_dists = dists[atom_mask]
                    atom_indices = np.where(atom_mask)[0]
                    cutoff_idx = np.argsort(atom_dists)[max_neighbors:]
                    mask[atom_indices[cutoff_idx]] = False
            i, j, S = i[mask], j[mask], S[mask]

        all_i.append(i + offset)
        all_j.append(j + offset)
        all_S.append(S)
        offset += len(pos)

    edge_index = torch.tensor(
        np.stack([np.concatenate(all_i), np.concatenate(all_j)]),
        dtype=torch.long,
    )
    cell_offsets = torch.tensor(
        np.concatenate(all_S).astype(np.int64),
        dtype=torch.long,
    )
    return edge_index, cell_offsets


# ── DFTCollator ──────────────────────────────────────────────────

@dataclass
class BlockOptions:
    """On-the-fly orbital block decomposition settings.

    행렬을 per-atom diagonal/off-diagonal blocks로 분해.
    QHFlow2 등 GNN 모델에서 node/edge feature로 사용.

    Args:
        pad_to: block padding size. None이면 basis의 max_block_size 사용.
        align: padding 정렬. "shell" (default) or "contiguous".
        matrix_fields: block 분해할 행렬 필드. 기본: ["hamiltonian", "overlap"]
    """
    pad_to: Optional[int] = None
    align: str = "shell"
    matrix_fields: list[str] = field(default_factory=lambda: ["hamiltonian", "overlap"])


@dataclass
class DFTCollator:
    """Collate variable-size DFT data for PyTorch DataLoader.

    Handles:
    - Atom-level data: concatenate + batch_idx (GNN-style)
    - Scalar targets: stack into tensors
    - Matrix targets: pad to max_nao in batch + return mask
    - On-the-fly graph construction (edge_index)
    - On-the-fly block decomposition (OrbitalMatrix → OrbitalBlocks)
    - Periodic boundary conditions (cell, pbc)
    - Field renaming (predicted/reference 구분)

    Args:
        targets: field names to include.
        pad_matrices: pad variable-size matrices to uniform shape.
        max_nao: skip samples exceeding this nao.
        dtype: torch dtype for floating point tensors.
        convention: target convention for matrices ("e3nn", "pyscf", etc.).
        graph: on-the-fly graph construction options.
        blocks: on-the-fly block decomposition options.
            설정하면 pad_matrices 대신 block 형태로 출력.
        rename: output key 이름 변경.
    """
    targets: list[str] = field(default_factory=lambda: ["energy", "forces"])
    pad_matrices: bool = True
    max_nao: Optional[int] = None
    dtype: str = "float32"
    convention: Optional[str] = None
    graph: Optional[GraphOptions] = None
    blocks: Optional[BlockOptions] = None
    rename: Optional[dict[str, str]] = None

    def __call__(self, batch: list) -> dict:
        """Collate a list of Molecule objects or dicts into batched tensors."""
        if torch is None:
            raise ImportError("PyTorch required for DFTCollator")

        tdtype = getattr(torch, self.dtype)

        # Normalize inputs to dicts
        dicts = []
        for item in batch:
            if isinstance(item, Molecule):
                dicts.append(item.to_dict(packed=False))
            elif isinstance(item, dict):
                dicts.append(item)
            else:
                raise TypeError(f"Expected Molecule or dict, got {type(item)}")

        if self.max_nao is not None:
            dicts = [d for d in dicts if (_get_nao_from_dict(d) or 0) <= self.max_nao]

        if not dicts:
            return {}

        result = {}
        batch_size = len(dicts)

        # ── Atom-level data (GNN-style: concatenate + batch_idx) ──

        all_Z = []
        all_pos = []
        all_batch = []
        num_atoms_list = []

        for i, d in enumerate(dicts):
            Z = _as_numpy(d["atomic_numbers"], "int32")
            n = len(Z)
            num_atoms_list.append(n)

            pos = _decode_array(d, "positions", n)
            if pos is None:
                pos_shape = d.get("positions_shape", (n, 3))
                pos = np.frombuffer(d["positions"], dtype=np.float64).reshape(pos_shape)

            all_Z.append(Z)
            all_pos.append(pos)
            all_batch.append(np.full(n, i, dtype=np.int64))

        result["atomic_numbers"] = torch.from_numpy(np.concatenate(all_Z))
        result["positions"] = torch.from_numpy(
            np.concatenate(all_pos).astype(np.float64)
        ).to(tdtype)
        result["batch"] = torch.from_numpy(np.concatenate(all_batch))
        result["num_atoms"] = torch.tensor(num_atoms_list, dtype=torch.long)

        # ── Cell / PBC ──

        has_cell = any(_decode_array(d, "cell", 0) is not None for d in dicts)
        if has_cell:
            cells, pbcs = [], []
            for d in dicts:
                c = _decode_array(d, "cell", 0)
                p = _decode_array(d, "pbc", 0)
                cells.append(c if c is not None else np.zeros((3, 3), dtype=np.float64))
                pbcs.append(p if p is not None else np.array([False, False, False]))
            result["cell"] = torch.from_numpy(np.stack(cells).astype(np.float64)).to(tdtype)
            result["pbc"] = torch.from_numpy(np.stack(pbcs).astype(bool))

        # ── On-the-fly graph construction ──

        if self.graph is not None:
            def _has_pbc(d):
                p = _decode_array(d, "pbc", 0)
                return p is not None and np.any(p)
            is_periodic = has_cell and any(_has_pbc(d) for d in dicts)

            if is_periodic and self.graph.backend == "ase":
                edge_index, cell_offsets = _build_graph_ase_pbc(
                    all_pos, [c.numpy() if isinstance(c, Tensor) else c
                              for c in (cells if has_cell else [])],
                    [p if isinstance(p, np.ndarray) else np.array([False]*3)
                     for p in (pbcs if has_cell else [])],
                    self.graph.cutoff, self.graph.max_neighbors,
                )
                result["edge_index"] = edge_index
                result["cell_offsets"] = cell_offsets
            elif is_periodic:
                # Default PBC backend: ASE (always available, safe)
                edge_index, cell_offsets = _build_graph_ase_pbc(
                    all_pos, cells, pbcs,
                    self.graph.cutoff, self.graph.max_neighbors,
                )
                result["edge_index"] = edge_index
                result["cell_offsets"] = cell_offsets
            else:
                # Non-PBC: use torch_geometric radius_graph
                result["edge_index"] = _build_graph_pyg(
                    result["positions"], result["batch"],
                    self.graph.cutoff, self.graph.max_neighbors,
                )

        # ── Scalar targets ──

        if "energy" in self.targets:
            energies = [d.get("energy", 0.0) for d in dicts]
            result["energy"] = torch.tensor(energies, dtype=tdtype)

        if "stress" in self.targets:
            all_stress = []
            for d in dicts:
                s = _decode_array(d, "stress", 0)
                all_stress.append(s if s is not None else np.zeros(6, dtype=np.float64))
            result["stress"] = torch.from_numpy(
                np.stack(all_stress).astype(np.float64)
            ).to(tdtype)

        # ── Force targets (variable-length, concatenated like positions) ──

        if "forces" in self.targets:
            all_forces = []
            for i, d in enumerate(dicts):
                f = _decode_array(d, "forces", num_atoms_list[i])
                if f is not None:
                    all_forces.append(f)
            if all_forces:
                result["forces"] = torch.from_numpy(
                    np.concatenate(all_forces).astype(np.float64)
                ).to(tdtype)

        # ── Matrix targets ──

        _KNOWN_MATRIX_TARGETS = {
            "hamiltonian", "overlap", "density_matrix",
            "initial_hamiltonian", "initial_density_matrix",
        }
        matrix_targets = [t for t in self.targets if t in _KNOWN_MATRIX_TARGETS]

        # ── Block decomposition mode ──

        if matrix_targets and self.blocks is not None:
            self._collate_blocks(result, dicts, matrix_targets, all_Z, tdtype)

        # ── Padded matrix mode (default) ──

        elif matrix_targets and self.pad_matrices:
            nao_list = []
            for d in dicts:
                nao = _get_nao_from_dict(d)
                nao_list.append(nao if nao is not None else 0)
            max_nao = max(nao_list) if nao_list else 0

            result["nao"] = torch.tensor(nao_list, dtype=torch.long)

            mask = torch.zeros(batch_size, max_nao, max_nao, dtype=torch.bool)
            for i, nao in enumerate(nao_list):
                if nao > 0:
                    mask[i, :nao, :nao] = True
            result["matrix_mask"] = mask

            for target in matrix_targets:
                matrices = []
                for d in dicts:
                    M = _unpack_symmetric(d, target)
                    matrices.append(M)

                if all(m is None for m in matrices):
                    continue

                if self.convention is not None:
                    matrices = self._apply_convention(matrices, dicts)

                has_spin = any(m is not None and m.ndim == 3 for m in matrices)

                if has_spin:
                    padded = torch.zeros(batch_size, 2, max_nao, max_nao, dtype=tdtype)
                    for i, M in enumerate(matrices):
                        if M is not None:
                            nao = M.shape[-1]
                            padded[i, :, :nao, :nao] = torch.from_numpy(
                                np.asarray(M)
                            ).to(tdtype)
                else:
                    padded = torch.zeros(batch_size, max_nao, max_nao, dtype=tdtype)
                    for i, M in enumerate(matrices):
                        if M is not None:
                            nao = M.shape[-1]
                            padded[i, :nao, :nao] = torch.from_numpy(
                                np.asarray(M)
                            ).to(tdtype)

                result[target] = padded

        # ── Convention tracking ──

        if matrix_targets:
            if self.convention is not None:
                result["convention"] = self.convention
            else:
                # Infer from first sample's basis_info
                for d in dicts:
                    bi = d.get("_basis_info")
                    if bi is not None:
                        conv = bi.get("convention") if isinstance(bi, dict) else getattr(bi, "convention", None)
                        if conv is not None:
                            result["convention"] = conv
                            break

        # ── Source tracking (MixDataset) ──

        if any("_source" in d for d in dicts):
            result["_source"] = [d.get("_source", "") for d in dicts]
            result["_source_idx"] = torch.tensor(
                [d.get("_source_idx", -1) for d in dicts], dtype=torch.long
            )

        # ── Rename ──

        if self.rename:
            for old_key, new_key in self.rename.items():
                if old_key in result:
                    result[new_key] = result.pop(old_key)

        return result

    def _apply_convention(
        self, matrices: list, dicts: list[dict]
    ) -> list:
        """Apply convention reordering to matrices."""
        from dft_dataset.conventions import build_reorder_indices, reorder_matrix
        from dft_dataset.molecule import BasisInfo

        result = []
        for M, d in zip(matrices, dicts):
            if M is None:
                result.append(None)
                continue
            bi_raw = d.get("_basis_info")
            if bi_raw is None:
                result.append(M)
                continue
            if isinstance(bi_raw, dict):
                bi = BasisInfo.from_dict(bi_raw)
            else:
                bi = bi_raw
            if bi.convention == self.convention:
                result.append(M)
                continue
            Z = _as_numpy(d["atomic_numbers"], "int32")
            reorder_idx = build_reorder_indices(
                Z, bi.atom_to_shells, src=bi.convention, dst=self.convention
            )
            if M.ndim == 3:
                result.append(np.stack([
                    reorder_matrix(M[0], reorder_idx),
                    reorder_matrix(M[1], reorder_idx),
                ]))
            else:
                result.append(reorder_matrix(M, reorder_idx))
        return result

    def _collate_blocks(
        self,
        result: dict,
        dicts: list[dict],
        matrix_targets: list[str],
        all_Z: list[np.ndarray],
        tdtype,
    ) -> None:
        """행렬을 OrbitalBlocks로 분해하여 GNN-style로 배칭."""
        from dft_dataset.orbital_matrix import OrbitalMatrix
        from dft_dataset.molecule import BasisInfo

        block_fields = self.blocks.matrix_fields
        targets_to_block = [t for t in matrix_targets if t in block_fields]

        if not targets_to_block:
            return

        # Resolve BasisInfo for each sample
        basis_infos = []
        for d in dicts:
            bi_raw = d.get("_basis_info")
            if bi_raw is None:
                raise ValueError("Block decomposition requires _basis_info in data")
            bi = BasisInfo.from_dict(bi_raw) if isinstance(bi_raw, dict) else bi_raw
            basis_infos.append(bi)

        # Convention target
        conv_target = self.convention or basis_infos[0].convention

        # Decompose each sample, concatenate GNN-style
        all_diag = {t: [] for t in targets_to_block}
        all_offdiag = {t: [] for t in targets_to_block}
        all_diag_mask = []
        all_offdiag_mask = []
        all_block_edge_index = []
        atom_offset = 0
        pad_to = self.blocks.pad_to

        for i, d in enumerate(dicts):
            Z = all_Z[i]
            bi = basis_infos[i]

            for target in targets_to_block:
                M = _unpack_symmetric(d, target)
                if M is None:
                    continue

                om = OrbitalMatrix(data=M, atomic_numbers=Z, basis_info=bi)
                if bi.convention != conv_target:
                    om = om.in_convention(conv_target)

                actual_pad = pad_to if pad_to is not None else om.basis_info.max_block_size
                blocks = om.to_blocks(pad_to=actual_pad, align=self.blocks.align)

                all_diag[target].append(blocks.diagonal)
                all_offdiag[target].append(blocks.off_diagonal)

                # Masks: only compute once (same for all matrix fields of same sample)
                if target == targets_to_block[0]:
                    all_diag_mask.append(blocks.diagonal_mask)
                    all_offdiag_mask.append(blocks.off_diagonal_mask)
                    # Offset edge_index for GNN batching
                    ei = blocks.edge_index.copy()
                    ei += atom_offset
                    all_block_edge_index.append(ei)

            atom_offset += len(Z)

        # Stack / concatenate
        for target in targets_to_block:
            if all_diag[target]:
                result[f"diagonal_{target}"] = torch.from_numpy(
                    np.concatenate(all_diag[target])
                ).to(tdtype)
                result[f"non_diagonal_{target}"] = torch.from_numpy(
                    np.concatenate(all_offdiag[target])
                ).to(tdtype)

        if all_diag_mask:
            result["diagonal_mask"] = torch.from_numpy(
                np.concatenate(all_diag_mask)
            ).to(tdtype)
            result["non_diagonal_mask"] = torch.from_numpy(
                np.concatenate(all_offdiag_mask)
            ).to(tdtype)
            result["block_edge_index"] = torch.from_numpy(
                np.concatenate(all_block_edge_index, axis=1)
            ).long()

        # Pad size metadata
        if all_diag[targets_to_block[0]]:
            result["block_pad_size"] = all_diag[targets_to_block[0]][0].shape[-1]


# ── MixDataset ───────────────────────────────────────────────────


class MixDataset:
    """여러 LMDB 데이터셋을 비율 기반으로 혼합.

    각 데이터셋은 독립적인 fields/preset을 가질 수 있음.
    sampling 비율로 epoch당 각 데이터셋에서 몇 개를 뽑을지 결정.

    Args:
        datasets: LMDBDataset 리스트.
        weights: 데이터셋별 sampling 비율. None이면 크기 비례.
            예: [0.5, 0.5] → 절반씩, [0.8, 0.2] → 8:2
        labels: 데이터셋별 이름 (batch에 source 추적용).

    Usage:
        from dft_dataset.lmdb_dataset import LMDBDataset

        mix = MixDataset(
            datasets=[
                LMDBDataset("qh9.lmdb", fields={"atomic_numbers", "positions", ...}),
                LMDBDataset("omol25.lmdb", fields={"atomic_numbers", "positions", ...}),
            ],
            weights=[0.5, 0.5],
            labels=["qh9", "omol25"],
        )
        sampler = MixSampler(mix, total_samples=10000)
        loader = DataLoader(mix, sampler=sampler, batch_size=32, collate_fn=collator)
    """

    def __init__(
        self,
        datasets: list,
        weights: list[float] | None = None,
        labels: list[str] | None = None,
    ):
        self.datasets = datasets
        self.sizes = [len(ds) for ds in datasets]
        self.cumulative = np.cumsum(self.sizes)
        self._total = int(self.cumulative[-1])

        if weights is None:
            self.weights = [s / self._total for s in self.sizes]
        else:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]

        self.labels = labels or [str(i) for i in range(len(datasets))]

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self._total
        ds_idx = int(np.searchsorted(self.cumulative, idx, side="right"))
        local_idx = idx if ds_idx == 0 else idx - int(self.cumulative[ds_idx - 1])
        sample = self.datasets[ds_idx][local_idx]

        if isinstance(sample, dict):
            sample["_source"] = self.labels[ds_idx]
            sample["_source_idx"] = ds_idx
        return sample

    def get_nao_list(self) -> np.ndarray:
        """전체 데이터셋의 nao 리스트 (BucketBatchSampler용)."""
        parts = []
        for ds in self.datasets:
            if hasattr(ds, "get_nao_list"):
                parts.append(ds.get_nao_list())
            else:
                parts.append(np.zeros(len(ds), dtype=np.int32))
        return np.concatenate(parts)


class MixSampler:
    """MixDataset용 비율 기반 sampler.

    각 epoch에서 weights 비율로 각 데이터셋에서 샘플링.
    작은 데이터셋은 oversample (with replacement),
    큰 데이터셋은 subsample 가능.

    Args:
        mix_dataset: MixDataset 인스턴스.
        total_samples: epoch당 총 샘플 수. None이면 전체 크기.
        shuffle: 셔플 여부.
        seed: 랜덤 시드.
    """

    def __init__(
        self,
        mix_dataset: MixDataset,
        total_samples: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.mix = mix_dataset
        self.total_samples = total_samples or len(mix_dataset)
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        indices = []
        cumulative = [0] + list(self.mix.cumulative)

        for ds_idx, (size, weight) in enumerate(
            zip(self.mix.sizes, self.mix.weights)
        ):
            n_samples = int(self.total_samples * weight)
            if n_samples <= size:
                ds_indices = self.rng.choice(size, n_samples, replace=False)
            else:
                ds_indices = self.rng.choice(size, n_samples, replace=True)
            global_indices = ds_indices + cumulative[ds_idx]
            indices.append(global_indices)

        all_indices = np.concatenate(indices)
        if self.shuffle:
            self.rng.shuffle(all_indices)

        yield from all_indices.tolist()

    def __len__(self) -> int:
        return self.total_samples


# ── BucketBatchSampler ───────────────────────────────────────────

class BucketBatchSampler:
    """Batch sampler that groups samples by nao to minimize padding waste.

    Args:
        nao_list: nao per sample index (list or array of ints)
        batch_size: target batch size
        boundaries: bucket boundaries. Default: geometric from 20 to 4000.
        shuffle: whether to shuffle within buckets each epoch
        drop_last: drop incomplete last batch in each bucket
    """

    def __init__(
        self,
        nao_list: list[int] | np.ndarray,
        batch_size: int = 32,
        boundaries: list[int] | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.nao_list = np.asarray(nao_list)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)

        if boundaries is None:
            boundaries = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10000]
        self.boundaries = sorted(boundaries)

        self.buckets: dict[int, list[int]] = {}
        for idx, nao in enumerate(self.nao_list):
            bucket = 0
            for b in self.boundaries:
                if nao <= b:
                    break
                bucket = b
            if bucket not in self.buckets:
                self.buckets[bucket] = []
            self.buckets[bucket].append(idx)

    def __iter__(self):
        batches = []
        for bucket_key in sorted(self.buckets.keys()):
            indices = self.buckets[bucket_key].copy()
            if self.shuffle:
                self.rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        yield from batches

    def __len__(self):
        total = 0
        for bucket_key in self.buckets:
            n = len(self.buckets[bucket_key])
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


# ── Presets & Factory ────────────────────────────────────────────

PRESETS = {
    "mlip": {
        "fields": {"atomic_numbers", "positions", "energy", "forces",
                    "num_atoms", "cell", "pbc", "stress"},
        "targets": ["energy", "forces"],
    },
    "mlip_pyg": {
        "fields": {"atomic_numbers", "positions", "energy", "forces",
                    "num_atoms", "cell", "pbc", "stress"},
        "targets": ["energy", "forces"],
        "graph": GraphOptions(cutoff=5.0, max_neighbors=32),
        "rename": {"atomic_numbers": "z", "positions": "pos"},
    },
    "hamiltonian": {
        "fields": {
            "atomic_numbers", "positions", "energy", "forces",
            "hamiltonian", "overlap", "num_atoms", "charge", "spin",
            "unrestricted", "_basis_info", "_packed", "cell", "pbc",
        },
        "targets": ["energy", "forces", "hamiltonian", "overlap"],
    },
    "density": {
        "fields": {
            "atomic_numbers", "positions", "energy", "forces",
            "density_matrix", "overlap", "num_atoms", "charge", "spin",
            "unrestricted", "_basis_info", "_packed", "cell", "pbc",
        },
        "targets": ["energy", "forces", "density_matrix", "overlap"],
    },
    "hamiltonian_blocks": {
        "fields": {
            "atomic_numbers", "positions", "energy", "forces",
            "hamiltonian", "overlap", "num_atoms", "charge", "spin",
            "unrestricted", "_basis_info", "_packed", "cell", "pbc",
        },
        "targets": ["energy", "forces", "hamiltonian", "overlap"],
        "blocks": BlockOptions(matrix_fields=["hamiltonian", "overlap"]),
    },
    "full": {
        "fields": None,
        "targets": ["energy", "forces"],
    },
}


def build_dataloader(
    lmdb_path: str | list[str],
    preset: str = "mlip",
    fields: set[str] | None = ...,  # sentinel: use preset default
    targets: list[str] | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    bucket_batching: bool | None = None,
    shuffle: bool = True,
    max_nao: int | None = None,
    convention: str | None = None,
    graph: GraphOptions | dict | None = ...,  # sentinel: use preset default
    blocks: BlockOptions | dict | None = ...,  # sentinel: use preset default
    rename: dict[str, str] | None = ...,  # sentinel: use preset default
    rank: int | None = None,  # DDP rank (None이면 single-GPU)
    world_size: int | None = None,  # DDP world size
    dtype: str = "float32",
    **dataloader_kwargs,
) -> DataLoader:
    """Build a DataLoader for DFT data with optimized loading.

    Args:
        lmdb_path: LMDB 파일 경로. list면 ConcatDataset으로 결합.
        preset: 사전 정의된 필드/타겟/graph/rename 조합.
            "mlip" — energy + forces (가장 빠름)
            "mlip_pyg" — energy + forces + edge_index + PyG 필드명
            "hamiltonian" — H, S 행렬 포함
            "density" — density matrix + S 포함
            "full" — 모든 필드, Molecule 반환
        fields: preset 대신 직접 지정할 필드 집합.
        targets: collator에 전달할 타겟 리스트.
        graph: on-the-fly graph construction. dict면 GraphOptions(**dict)로 변환.
            None이면 edge_index 안 만듦. ...이면 preset 기본값 사용.
        rename: output key 이름 변경. 예: {"energy": "energy_ref"}.
            ...이면 preset 기본값 사용.
        batch_size, num_workers, pin_memory, persistent_workers: DataLoader 설정.
        bucket_batching: nao 기준 버킷 배칭 (None이면 matrix target시 자동).
        shuffle, max_nao, convention, dtype: collator 설정.
    """
    if torch is None:
        raise ImportError("PyTorch required for build_dataloader")

    from dft_dataset.lmdb_dataset import LMDBDataset

    # Resolve preset
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS)}")
    preset_cfg = PRESETS[preset]

    resolved_fields = preset_cfg["fields"] if fields is ... else fields
    resolved_targets = targets if targets is not None else preset_cfg["targets"]

    # Resolve graph options
    if graph is ...:
        resolved_graph = preset_cfg.get("graph")
    elif isinstance(graph, dict):
        resolved_graph = GraphOptions(**graph)
    else:
        resolved_graph = graph

    # Resolve blocks
    if blocks is ...:
        resolved_blocks = preset_cfg.get("blocks")
    elif isinstance(blocks, dict):
        resolved_blocks = BlockOptions(**blocks)
    else:
        resolved_blocks = blocks

    # Resolve rename
    if rename is ...:
        resolved_rename = preset_cfg.get("rename")
    else:
        resolved_rename = rename

    # Auto-detect settings
    _MATRIX_TARGET_NAMES = {
        "hamiltonian", "overlap", "density_matrix",
        "initial_hamiltonian", "initial_density_matrix",
    }
    has_matrix_targets = any(t in _MATRIX_TARGET_NAMES for t in resolved_targets)
    if bucket_batching is None:
        bucket_batching = has_matrix_targets and resolved_blocks is None
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # Build dataset(s)
    if isinstance(lmdb_path, (list, tuple)):
        datasets = [
            LMDBDataset(p, fields=resolved_fields, to_torch=False, dtype=dtype)
            for p in lmdb_path
        ]
        dataset = ConcatDataset(datasets)
    else:
        dataset = LMDBDataset(
            lmdb_path, fields=resolved_fields, to_torch=False, dtype=dtype
        )

    # Build collator
    collator = DFTCollator(
        targets=resolved_targets,
        pad_matrices=has_matrix_targets and resolved_blocks is None,
        max_nao=max_nao,
        dtype=dtype,
        convention=convention,
        graph=resolved_graph,
        blocks=resolved_blocks,
        rename=resolved_rename,
    )

    # Build sampler
    batch_sampler = None
    sampler = None
    dl_shuffle = shuffle

    # DDP support
    is_ddp = rank is not None and world_size is not None
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=shuffle, drop_last=True,
        )
        dl_shuffle = False  # sampler handles shuffling

    if bucket_batching:
        if isinstance(dataset, ConcatDataset):
            nao_lists = [ds.get_nao_list() for ds in datasets]
            nao_list = np.concatenate(nao_lists)
        else:
            nao_list = dataset.get_nao_list()

        batch_sampler = BucketBatchSampler(
            nao_list, batch_size=batch_size, shuffle=shuffle,
        )
        dl_shuffle = False
        batch_size = 1

    # Build DataLoader
    dl_kwargs = dict(
        dataset=dataset,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if batch_sampler is not None:
        dl_kwargs["batch_sampler"] = batch_sampler
    elif sampler is not None:
        dl_kwargs["sampler"] = sampler
        dl_kwargs["batch_size"] = batch_size
    else:
        dl_kwargs["batch_size"] = batch_size
        dl_kwargs["shuffle"] = dl_shuffle

    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers

    dl_kwargs.update(dataloader_kwargs)

    return DataLoader(**dl_kwargs)


def build_mix_dataloader(
    sources: list[dict],
    preset: str = "mlip",
    total_samples: int | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    convention: str | None = None,
    graph: GraphOptions | dict | None = ...,
    blocks: BlockOptions | dict | None = ...,
    rename: dict[str, str] | None = ...,
    dtype: str = "float32",
    **dataloader_kwargs,
) -> DataLoader:
    """여러 LMDB를 비율 기반으로 혼합하는 DataLoader.

    Args:
        sources: 데이터셋 설정 리스트. 각 항목은 dict:
            {
                "path": "train.lmdb",            # 필수
                "weight": 0.5,                    # 선택 (기본: 크기 비례)
                "label": "qh9",                   # 선택 (기본: 인덱스)
                "fields": {...},                  # 선택 (기본: preset 사용)
                "preset": "hamiltonian",          # 선택 (기본: 상위 preset)
            }
        preset: 공통 preset (source별로 override 가능).
        total_samples: epoch당 총 샘플 수. None이면 전체 합산.
        batch_size, num_workers, ...: DataLoader 설정.

    Usage:
        loader = build_mix_dataloader([
            {"path": "qh9.lmdb", "weight": 0.5, "label": "qh9"},
            {"path": "omol25.lmdb", "weight": 0.3, "label": "omol25"},
            {"path": "nabladft.lmdb", "weight": 0.2, "label": "nabladft"},
        ], preset="mlip", total_samples=100_000, batch_size=64)

        for batch in loader:
            print(batch["_source"])  # ["qh9", "omol25", "qh9", ...]
    """
    if torch is None:
        raise ImportError("PyTorch required for build_mix_dataloader")

    from dft_dataset.lmdb_dataset import LMDBDataset

    # Resolve per-source datasets
    datasets = []
    weights = []
    labels = []

    for src in sources:
        src_preset = src.get("preset", preset)
        if src_preset not in PRESETS:
            raise ValueError(f"Unknown preset '{src_preset}'")
        pcfg = PRESETS[src_preset]

        src_fields = src.get("fields", pcfg["fields"])
        ds = LMDBDataset(
            src["path"], fields=src_fields, to_torch=False, dtype=dtype,
        )
        datasets.append(ds)
        weights.append(src.get("weight", None))
        labels.append(src.get("label", src["path"]))

    # None weights → size-proportional
    if all(w is None for w in weights):
        weights_resolved = None
    else:
        total = sum(len(ds) for ds in datasets)
        weights_resolved = [
            w if w is not None else len(ds) / total
            for w, ds in zip(weights, datasets)
        ]

    mix = MixDataset(datasets, weights=weights_resolved, labels=labels)
    sampler = MixSampler(mix, total_samples=total_samples, shuffle=True)

    # Resolve collator settings from preset
    pcfg = PRESETS[preset]
    resolved_targets = pcfg["targets"]

    if graph is ...:
        resolved_graph = pcfg.get("graph")
    elif isinstance(graph, dict):
        resolved_graph = GraphOptions(**graph)
    else:
        resolved_graph = graph

    if blocks is ...:
        resolved_blocks = pcfg.get("blocks")
    elif isinstance(blocks, dict):
        resolved_blocks = BlockOptions(**blocks)
    else:
        resolved_blocks = blocks

    if rename is ...:
        resolved_rename = pcfg.get("rename")
    else:
        resolved_rename = rename

    has_matrix = any(
        t in ("hamiltonian", "overlap", "density_matrix")
        for t in resolved_targets
    )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    collator = DFTCollator(
        targets=resolved_targets,
        pad_matrices=has_matrix and resolved_blocks is None,
        dtype=dtype,
        convention=convention,
        graph=resolved_graph,
        blocks=resolved_blocks,
        rename=resolved_rename,
    )

    dl_kwargs = dict(
        dataset=mix,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
    dl_kwargs.update(dataloader_kwargs)

    return DataLoader(**dl_kwargs)


# ── PyG adapters ─────────────────────────────────────────────────

def to_pyg_data(sample: dict):
    """sample dict → PyG Data 객체 변환.

    Args:
        sample: decoded sample dict (from decode_fields or collator output)

    Returns:
        torch_geometric.data.Data object
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "torch_geometric required for to_pyg_data. "
            "Install with: pip install torch_geometric"
        )

    if torch is None:
        raise ImportError("PyTorch required for to_pyg_data")

    data = Data()
    for k, v in sample.items():
        if isinstance(v, (torch.Tensor, int, float, str, bool)):
            setattr(data, k, v)
        elif isinstance(v, np.ndarray):
            setattr(data, k, torch.from_numpy(v))
    return data


def to_pyg_batch(batch_dict: dict):
    """DFTCollator 출력 dict → PyG Batch 객체 변환.

    MLIP 모델에 직접 입력 가능한 PyG Batch를 생성.
    batch_dict에 edge_index가 있으면 포함, 없으면 생략.

    Args:
        batch_dict: DFTCollator.__call__() 결과

    Returns:
        torch_geometric.data.Batch object
    """
    try:
        from torch_geometric.data import Batch, Data
    except ImportError:
        raise ImportError("torch_geometric required for to_pyg_batch")

    if torch is None:
        raise ImportError("PyTorch required for to_pyg_batch")

    data = Data()
    for k, v in batch_dict.items():
        if isinstance(v, (torch.Tensor, int, float, str, bool)):
            setattr(data, k, v)

    # PyG Batch에서 필요한 ptr 생성
    if hasattr(data, "num_atoms"):
        num_atoms = data.num_atoms
        data.ptr = torch.cat([
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(num_atoms, dim=0),
        ])
        data.num_graphs = len(num_atoms)

    # Batch 클래스로 변환 (이미 배치된 상태)
    batch = Batch()
    for k, v in data:
        setattr(batch, k, v)
    batch._num_graphs = getattr(data, "num_graphs", 1)

    return batch
