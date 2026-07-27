"""Convention-aware orbital matrix container and block decomposition.

OrbitalMatrix — 대칭 행렬(H, S, D 등)을 convention과 함께 관리.
OrbitalBlocks — 원자별 diagonal/off-diagonal block 분해.

이 모듈의 핵심 가치:
  1) 행렬이 어떤 convention에 있는지 항상 추적 → 실수 방지
  2) convention 변환 + block 분해를 통합된 API로 제공
  3) QHFlow2 등 GNN 모델의 전처리 파이프라인에서 재사용

사용 예시::

    H = OrbitalMatrix.from_molecule(mol, "hamiltonian")  # pyscf
    H_e3nn = H.in_convention("e3nn")                      # 변환
    blocks = H_e3nn.to_blocks(pad_to=14)                  # block 분해
    H_reconstructed = blocks.to_dense()                    # 복원
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .conventions import (
    KNOWN_CONVENTIONS,
    build_reorder_indices,
    reorder_matrix,
)
from .molecule import BasisInfo


# ── OrbitalBlocks ────────────────────────────────────────────────


@dataclass
class OrbitalBlocks:
    """Per-atom block decomposition of an orbital matrix.

    Full (nao, nao) 행렬을 원자 쌍별로 분해한 결과를 저장.
    GNN 모델에서 node/edge feature로 사용.

    Attributes:
        diagonal: (n_atoms, pad, pad) — 같은 원자 블록 (e.g., C-C)
        off_diagonal: (n_edges, pad, pad) — 다른 원자 쌍 블록 (e.g., C-H)
        diagonal_mask: (n_atoms, pad, pad) — diagonal 유효 위치 (bool)
        off_diagonal_mask: (n_edges, pad, pad) — off-diagonal 유효 위치 (bool)
        edge_index: (2, n_edges) — [dst, src] graph connectivity
        atomic_numbers: (n_atoms,) — 원자 번호
        basis_info: 기반 basis set 정보
        pad_size: 패딩된 블록 크기 (e.g., 14 for def2-svp)
    """

    diagonal: np.ndarray
    off_diagonal: np.ndarray
    diagonal_mask: np.ndarray
    off_diagonal_mask: np.ndarray
    edge_index: np.ndarray
    atomic_numbers: np.ndarray
    basis_info: BasisInfo
    pad_size: int

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index.size > 0 else 0

    def to_dense(self) -> np.ndarray:
        """Block 분해를 원래 (nao, nao) 행렬로 복원.

        mask를 사용하여 유효한 위치의 값만 추출하므로,
        shell-aligned이든 contiguous이든 정확하게 복원됩니다.

        Returns:
            np.ndarray: (nao, nao) matrix
        """
        nao = self.basis_info.total_nao(self.atomic_numbers)
        slices = self.basis_info.orbital_slices(self.atomic_numbers)
        n_atoms = self.n_atoms

        M = np.zeros((nao, nao), dtype=self.diagonal.dtype)

        # Diagonal blocks: mask에서 유효 위치를 찾아 복원
        for i in range(n_atoms):
            sl = slices[i]
            # mask의 유효 row/col 인덱스 (padded block 내 위치)
            mask_2d = self.diagonal_mask[i]
            row_valid = np.where(mask_2d.any(axis=1))[0]
            col_valid = np.where(mask_2d.any(axis=0))[0]
            values = self.diagonal[i][np.ix_(row_valid, col_valid)]
            M[sl, sl] = values

        # Off-diagonal blocks
        for e in range(self.n_edges):
            dst_idx = int(self.edge_index[0, e])
            src_idx = int(self.edge_index[1, e])
            sl_dst = slices[dst_idx]
            sl_src = slices[src_idx]
            mask_2d = self.off_diagonal_mask[e]
            row_valid = np.where(mask_2d.any(axis=1))[0]
            col_valid = np.where(mask_2d.any(axis=0))[0]
            values = self.off_diagonal[e][np.ix_(row_valid, col_valid)]
            M[sl_dst, sl_src] = values

        return M

    def to_orbital_matrix(self) -> "OrbitalMatrix":
        """복원된 full matrix를 OrbitalMatrix로 반환."""
        return OrbitalMatrix(
            data=self.to_dense(),
            atomic_numbers=self.atomic_numbers,
            basis_info=self.basis_info,
        )

    def to_dict(self) -> dict:
        """Serialize to dict (for LMDB storage)."""
        return {
            "diagonal": self.diagonal,
            "off_diagonal": self.off_diagonal,
            "diagonal_mask": self.diagonal_mask,
            "off_diagonal_mask": self.off_diagonal_mask,
            "edge_index": self.edge_index,
            "atomic_numbers": self.atomic_numbers,
            "basis_info": self.basis_info.to_dict(),
            "pad_size": self.pad_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrbitalBlocks":
        """Deserialize from dict."""
        return cls(
            diagonal=d["diagonal"],
            off_diagonal=d["off_diagonal"],
            diagonal_mask=d["diagonal_mask"],
            off_diagonal_mask=d["off_diagonal_mask"],
            edge_index=d["edge_index"],
            atomic_numbers=d["atomic_numbers"],
            basis_info=BasisInfo.from_dict(d["basis_info"]),
            pad_size=d["pad_size"],
        )


# ── OrbitalMatrix ───────────────────────────────────────────────


@dataclass
class OrbitalMatrix:
    """Convention-aware symmetric matrix in an orbital basis.

    H, S, D 등 orbital-indexed 대칭 행렬을 convention 정보와 함께 관리.
    convention 변환, block 분해, 직렬화를 단일 인터페이스로 제공.

    Attributes:
        data: (nao, nao) symmetric matrix (float64)
        atomic_numbers: (n_atoms,) array of atomic numbers
        basis_info: BasisInfo with convention tracking
    """

    data: np.ndarray
    atomic_numbers: np.ndarray
    basis_info: BasisInfo

    def __post_init__(self):
        if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]:
            raise ValueError(
                f"data must be a square 2D matrix, got shape {self.data.shape}"
            )
        expected_nao = self.basis_info.total_nao(self.atomic_numbers)
        if self.data.shape[0] != expected_nao:
            raise ValueError(
                f"Matrix size {self.data.shape[0]} != expected nao {expected_nao} "
                f"from basis_info"
            )

    @property
    def convention(self) -> str:
        """현재 m-ordering convention."""
        return self.basis_info.convention

    @property
    def nao(self) -> int:
        return self.data.shape[0]

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    # ── Convention 변환 ──────────────────────────────────────────

    def in_convention(self, target: str) -> "OrbitalMatrix":
        """목표 convention으로 변환된 새 OrbitalMatrix 반환.

        이미 같은 convention이면 self 반환 (복사 없음).

        Args:
            target: "pyscf", "e3nn", "orca"

        Returns:
            새 OrbitalMatrix (원본 변경 없음)
        """
        if target not in KNOWN_CONVENTIONS:
            raise ValueError(
                f"Unknown convention '{target}'. Known: {KNOWN_CONVENTIONS}"
            )
        if self.convention == target:
            return self

        idx = build_reorder_indices(
            self.atomic_numbers,
            self.basis_info.atom_to_shells,
            self.convention, target,
            self.basis_info.angular_type,
        )
        new_data = reorder_matrix(self.data, idx)
        new_basis = BasisInfo(
            name=self.basis_info.name,
            angular_type=self.basis_info.angular_type,
            convention=target,
            atom_to_shells=self.basis_info.atom_to_shells,
            atom_to_nao=self.basis_info.atom_to_nao,
        )
        return OrbitalMatrix(
            data=new_data,
            atomic_numbers=self.atomic_numbers,
            basis_info=new_basis,
        )

    # ── Block 분해 ───────────────────────────────────────────────

    def to_blocks(
        self,
        pad_to: Optional[int] = None,
        align: str = "shell",
    ) -> OrbitalBlocks:
        """Per-atom diagonal + off-diagonal blocks로 분해.

        각 원자 쌍의 orbital 상호작용 블록을 추출하고 pad_to 크기로 패딩.
        GNN에서 node/edge feature로 사용.

        Args:
            pad_to: 블록을 패딩할 크기. None이면 basis의 max_block_size 사용.
            align: 패딩 정렬 방식.
                "shell" (default) — shell 구조에 맞춰 정렬. 같은 shell type이
                    모든 원자에서 같은 위치에 오도록 배치.
                    e.g., H(ssp)는 max atom(sssppd) 기준으로 [0,1,3,4,5]에 배치.
                "contiguous" — 첫 위치부터 빈틈없이 채움.
                    e.g., H(ssp)는 [0,1,2,3,4]에 배치.

        Returns:
            OrbitalBlocks
        """
        if pad_to is None:
            pad_to = self.basis_info.max_block_size

        orbital_indices = self._build_orbital_indices(pad_to, align)
        slices = self.basis_info.orbital_slices(self.atomic_numbers)
        n_atoms = self.n_atoms
        dtype = self.data.dtype

        diagonal_list = []
        off_diagonal_list = []
        diagonal_mask_list = []
        off_diagonal_mask_list = []
        edge_indices = []

        for src_idx in range(n_atoms):
            sl_src = slices[src_idx]
            src_oidx = orbital_indices[int(self.atomic_numbers[src_idx])]

            for dst_idx in range(n_atoms):
                sl_dst = slices[dst_idx]
                dst_oidx = orbital_indices[int(self.atomic_numbers[dst_idx])]

                # Extract submatrix from full matrix
                block_raw = self.data[sl_dst, sl_src]

                # Place into padded block using orbital indices
                block = np.zeros((pad_to, pad_to), dtype=dtype)
                mask = np.zeros((pad_to, pad_to), dtype=dtype)
                block[np.ix_(dst_oidx, src_oidx)] = block_raw
                mask[np.ix_(dst_oidx, src_oidx)] = 1.0

                if src_idx == dst_idx:
                    diagonal_list.append(block)
                    diagonal_mask_list.append(mask)
                else:
                    off_diagonal_list.append(block)
                    off_diagonal_mask_list.append(mask)
                    edge_indices.append([dst_idx, src_idx])

        diagonal = np.stack(diagonal_list) if diagonal_list else np.empty(
            (0, pad_to, pad_to), dtype=dtype
        )
        off_diagonal = np.stack(off_diagonal_list) if off_diagonal_list else np.empty(
            (0, pad_to, pad_to), dtype=dtype
        )
        diagonal_mask = np.stack(diagonal_mask_list) if diagonal_mask_list else np.empty(
            (0, pad_to, pad_to), dtype=dtype
        )
        off_diagonal_mask = np.stack(off_diagonal_mask_list) if off_diagonal_mask_list else np.empty(
            (0, pad_to, pad_to), dtype=dtype
        )
        edge_index = (
            np.array(edge_indices, dtype=np.int64).T
            if edge_indices
            else np.empty((2, 0), dtype=np.int64)
        )

        return OrbitalBlocks(
            diagonal=diagonal,
            off_diagonal=off_diagonal,
            diagonal_mask=diagonal_mask,
            off_diagonal_mask=off_diagonal_mask,
            edge_index=edge_index,
            atomic_numbers=self.atomic_numbers,
            basis_info=self.basis_info,
            pad_size=pad_to,
        )

    def _build_orbital_indices(
        self, pad_to: int, align: str,
    ) -> dict[int, np.ndarray]:
        """각 원자 타입별 padded block 내 orbital 위치를 계산.

        Args:
            pad_to: 패딩 크기
            align: "shell" or "contiguous"

        Returns:
            {atomic_number: np.ndarray of indices into (pad_to,) block}
        """
        from conventions import SHELL_TO_L, shell_size

        unique_z = sorted(set(int(z) for z in self.atomic_numbers))
        result = {}

        if align == "contiguous":
            for z in unique_z:
                n = self.basis_info.atom_to_nao[z]
                result[z] = np.arange(n, dtype=np.int64)
            return result

        # align == "shell": shell 타입 기준 정렬
        # 1) max atom의 shell 구조로 reference layout 생성
        max_z = max(unique_z, key=lambda z: self.basis_info.atom_to_nao[z])
        ref_shells = self.basis_info.atom_to_shells[max_z]

        # Reference layout: 각 shell의 시작 위치
        ref_shell_starts = {}  # {shell_label: [start_positions]}
        offset = 0
        for label in ref_shells:
            ref_shell_starts.setdefault(label, [])
            ref_shell_starts[label].append(offset)
            offset += shell_size(label, self.basis_info.angular_type)

        # 2) 각 원자 타입의 shell을 reference에 매핑
        for z in unique_z:
            atom_shells = self.basis_info.atom_to_shells[z]
            indices = []
            # shell label별 사용 카운터
            shell_counters = {label: 0 for label in ref_shell_starts}

            for label in atom_shells:
                l_val = SHELL_TO_L[label]
                n = shell_size(label, self.basis_info.angular_type)
                counter = shell_counters[label]
                if counter < len(ref_shell_starts[label]):
                    start = ref_shell_starts[label][counter]
                    indices.extend(range(start, start + n))
                    shell_counters[label] = counter + 1
                else:
                    # Fallback: shell이 reference보다 많으면 끝에 추가
                    start = max(indices) + 1 if indices else 0
                    indices.extend(range(start, start + n))

            result[z] = np.array(indices, dtype=np.int64)

        return result

    # ── Atom blocks (simple slicing) ─────────────────────────────

    def atom_block(self, i: int, j: Optional[int] = None) -> np.ndarray:
        """원자 i (또는 i-j 쌍)의 orbital block 추출 (패딩 없이).

        Args:
            i: 원자 인덱스
            j: 두 번째 원자 인덱스. None이면 diagonal block.

        Returns:
            (nao_i, nao_j) submatrix
        """
        slices = self.basis_info.orbital_slices(self.atomic_numbers)
        if j is None:
            j = i
        return self.data[slices[i], slices[j]]

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def from_molecule(
        cls,
        mol: "Molecule",
        field: str = "hamiltonian",
    ) -> "OrbitalMatrix":
        """Molecule에서 특정 행렬 필드를 OrbitalMatrix로 추출.

        Args:
            mol: Molecule instance (must have basis_info)
            field: "hamiltonian", "overlap", or "density_matrix"

        Returns:
            OrbitalMatrix with convention from mol.basis_info
        """
        mat = getattr(mol, field, None)
        if mat is None:
            raise ValueError(f"Molecule has no '{field}' field")
        if mol.basis_info is None:
            raise ValueError("Molecule must have basis_info for OrbitalMatrix")
        if mat.ndim == 3:
            raise ValueError(
                f"Unrestricted (2, nao, nao) matrices are not yet supported. "
                f"Got shape {mat.shape} for '{field}'."
            )
        return cls(
            data=mat,
            atomic_numbers=mol.atomic_numbers,
            basis_info=mol.basis_info,
        )

    @classmethod
    def from_dense(
        cls,
        matrix: np.ndarray,
        atomic_numbers: np.ndarray,
        basis_name: str = "def2-svp",
        convention: str = "pyscf",
    ) -> "OrbitalMatrix":
        """Dense matrix + 기본 정보로 OrbitalMatrix 생성.

        Args:
            matrix: (nao, nao) symmetric matrix
            atomic_numbers: (n_atoms,) array
            basis_name: basis set name (KNOWN_BASIS에 있어야 함)
            convention: current convention of the matrix
        """
        basis_info = BasisInfo.from_name(basis_name, convention=convention)
        return cls(
            data=matrix,
            atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
            basis_info=basis_info,
        )

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to dict."""
        from .molecule import pack_upper_triangle

        packed, nao = pack_upper_triangle(self.data)
        return {
            "packed": packed,
            "nao": nao,
            "dtype": str(self.data.dtype),
            "atomic_numbers": self.atomic_numbers,
            "basis_info": self.basis_info.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrbitalMatrix":
        """Deserialize from dict."""
        from .molecule import unpack_upper_triangle

        packed = d["packed"]
        nao = d["nao"]
        dtype = np.dtype(d.get("dtype", "float64"))
        matrix = unpack_upper_triangle(packed.astype(dtype), nao)
        return cls(
            data=matrix,
            atomic_numbers=d["atomic_numbers"],
            basis_info=BasisInfo.from_dict(d["basis_info"]),
        )

    def __repr__(self) -> str:
        return (
            f"OrbitalMatrix(nao={self.nao}, n_atoms={self.n_atoms}, "
            f"convention='{self.convention}', basis='{self.basis_info.name}')"
        )
