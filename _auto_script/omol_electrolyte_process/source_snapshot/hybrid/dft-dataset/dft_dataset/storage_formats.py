"""Pluggable storage formats for LMDB serialization.

직렬화 로직을 교체 가능하게 추상화. 새 포맷 추가 시:
1. StorageFormat을 구현하는 클래스 작성
2. FORMAT_REGISTRY에 등록
3. LMDBDataset.write(..., format="new_format") 으로 사용

현재 포맷:
  - "pickle" (v1): 전체 dict를 pickle blob으로 저장. 간단하고 범용적.
  - "split_key" (v2): 스칼라는 pickle, 대용량 array는 별도 key로 저장.
      필요한 필드만 LMDB에서 읽을 수 있어 I/O 최소화.

Usage:
    from storage_formats import get_format, PickleFormat

    fmt = get_format("pickle")
    encoded = fmt.encode_sample(mol.to_dict(packed=True))
    decoded = fmt.decode_sample(encoded, txn, idx)  # full dict
    partial = fmt.decode_fields(encoded, txn, idx, fields={"energy", "forces"})
"""

from __future__ import annotations

import pickle
from typing import Protocol, runtime_checkable

import numpy as np


# ── Protocol ─────────────────────────────────────────────────────

@runtime_checkable
class StorageFormat(Protocol):
    """LMDB 직렬화 포맷 인터페이스.

    새 포맷을 추가하려면 이 프로토콜을 구현하고
    FORMAT_REGISTRY에 등록하면 됩니다.
    """

    name: str

    def encode_sample(self, d: dict, idx: int, txn) -> None:
        """Sample dict를 LMDB transaction에 기록.

        Args:
            d: Molecule.to_dict() 결과.
            idx: sample index (0-based).
            txn: LMDB write transaction.
        """
        ...

    def decode_sample(self, idx: int, txn) -> dict:
        """LMDB에서 전체 sample dict를 복원.

        Args:
            idx: sample index.
            txn: LMDB read transaction.

        Returns:
            Molecule.to_dict() 형식의 dict.
        """
        ...

    def decode_fields(
        self, idx: int, txn, fields: set[str] | None = None
    ) -> dict:
        """LMDB에서 필요한 필드만 선택적으로 복원.

        Args:
            idx: sample index.
            txn: LMDB read transaction.
            fields: 필요한 필드 집합. None이면 전체.

        Returns:
            요청한 필드만 포함된 raw dict (bytes 값 그대로).
        """
        ...

    def read_nao(self, idx: int, txn) -> int | None:
        """nao 메타데이터만 빠르게 읽기 (BucketBatchSampler용).

        Returns:
            nao 값, 또는 행렬이 없으면 None.
        """
        ...


# ── Pickle Format (v1) ──────────────────────────────────────────

class PickleFormat:
    """기존 포맷: 전체 dict를 하나의 pickle blob으로 저장.

    장점: 단순, 범용, 기존 LMDB 호환.
    단점: pickle.loads()가 전체 dict를 복원 — 필드 선택적 I/O 불가.
    """

    name = "pickle"

    def encode_sample(self, d: dict, idx: int, txn) -> None:
        key = idx.to_bytes(4, "big")
        txn.put(key, pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL))

    def decode_sample(self, idx: int, txn) -> dict:
        key = idx.to_bytes(4, "big")
        raw = txn.get(key)
        if raw is None:
            raise KeyError(f"key {idx} not found in LMDB")
        return pickle.loads(raw)

    def decode_fields(
        self, idx: int, txn, fields: set[str] | None = None
    ) -> dict:
        # pickle은 전체를 deserialize할 수밖에 없음
        return self.decode_sample(idx, txn)

    def read_nao(self, idx: int, txn) -> int | None:
        d = self.decode_sample(idx, txn)
        for prefix in _MATRIX_NAMES:
            nao = d.get(f"{prefix}_nao")
            if nao is not None:
                return int(nao)
        return None


# ── Split-Key Format (v2) ────────────────────────────────────────

# 스칼라/메타데이터 필드 (pickle blob에 함께 저장)
_SCALAR_KEYS = {
    "num_atoms", "mol_id", "formula", "smiles", "charge", "spin",
    "unrestricted", "energy_unit", "length_unit", "energy", "homo",
    "lumo", "zpve", "U0", "U", "H", "G", "Cv", "polarizability",
    "xc", "source", "_packed", "_basis_info",
}

# 대용량 array 필드 (별도 key로 저장)
_ARRAY_KEYS = {
    "atomic_numbers", "positions", "forces", "stress", "dipole",
    "orbital_energies", "orbital_coeffs", "cell", "pbc",
    "mulliken_charges", "lowdin_charges", "nbo_charges",
    "mulliken_spins", "lowdin_spins", "nbo_spins",
}

# 행렬 필드 (packed 여부에 따라 key 이름이 달라짐)
_MATRIX_NAMES = {
    "hamiltonian", "overlap", "density_matrix",
    "initial_hamiltonian", "initial_density_matrix",
}


def _field_key(idx: int, field: str) -> bytes:
    """sample idx + field name → LMDB key.

    Format: 4-byte idx + b":" + field name (UTF-8).
    예: idx=42, field="forces" → b'\\x00\\x00\\x00*:forces'
    """
    return idx.to_bytes(4, "big") + b":" + field.encode("ascii")


class SplitKeyFormat:
    """필드별 분리 저장: 스칼라는 pickle, array/행렬은 별도 key.

    LMDB key 구조:
        {idx}           → pickle({스칼라 + 메타데이터 + shape/dtype 정보})
        {idx}:positions  → raw bytes (np.ndarray.tobytes())
        {idx}:forces     → raw bytes
        {idx}:hamiltonian_packed → raw bytes (또는 _packed_a, _packed_b)
        ...

    장점: 필요한 필드만 LMDB에서 읽어서 pickle.loads() 오버헤드 최소화.
    단점: key 수 증가 (샘플당 ~5-10개), 쓰기 약간 느림.
    """

    name = "split_key"

    def encode_sample(self, d: dict, idx: int, txn) -> None:
        main_key = idx.to_bytes(4, "big")
        meta = {}  # 스칼라 + shape/dtype 정보

        for k, v in d.items():
            if k in _SCALAR_KEYS:
                meta[k] = v
            elif isinstance(v, bytes) and not k.startswith("_"):
                # array/matrix raw bytes → 별도 key
                txn.put(_field_key(idx, k), v)
            else:
                # shape, dtype 메타데이터 등 나머지
                meta[k] = v

        txn.put(main_key, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

    def decode_sample(self, idx: int, txn) -> dict:
        main_key = idx.to_bytes(4, "big")
        raw = txn.get(main_key)
        if raw is None:
            raise KeyError(f"key {idx} not found in LMDB")
        d = pickle.loads(raw)

        # array/matrix 필드 복원
        self._load_array_keys(idx, txn, d, fields=None)
        return d

    def decode_fields(
        self, idx: int, txn, fields: set[str] | None = None
    ) -> dict:
        main_key = idx.to_bytes(4, "big")
        raw = txn.get(main_key)
        if raw is None:
            raise KeyError(f"key {idx} not found in LMDB")
        d = pickle.loads(raw)  # 스칼라 + 메타만 (작아서 빠름)

        # 요청된 필드의 array만 로드
        self._load_array_keys(idx, txn, d, fields=fields)
        return d

    def _load_array_keys(
        self, idx: int, txn, d: dict, fields: set[str] | None
    ) -> None:
        """필요한 array 필드만 LMDB에서 읽어 dict에 추가."""
        is_packed = d.get("_packed", False)

        # 어떤 key를 읽어야 하는지 결정
        candidate_keys = set()
        for name in _ARRAY_KEYS:
            if fields is None or name in fields:
                candidate_keys.add(name)

        for name in _MATRIX_NAMES:
            if fields is not None and name not in fields:
                continue
            if is_packed:
                nao_key = f"{name}_nao"
                if nao_key in d:
                    if d.get(f"{name}_unrestricted"):
                        candidate_keys.add(f"{name}_packed_a")
                        candidate_keys.add(f"{name}_packed_b")
                    else:
                        candidate_keys.add(f"{name}_packed")
            else:
                candidate_keys.add(name)

        # LMDB에서 실제 읽기
        for key_name in candidate_keys:
            fk = _field_key(idx, key_name)
            raw = txn.get(fk)
            if raw is not None:
                d[key_name] = raw

    def read_nao(self, idx: int, txn) -> int | None:
        main_key = idx.to_bytes(4, "big")
        raw = txn.get(main_key)
        if raw is None:
            return None
        d = pickle.loads(raw)  # 스칼라만 — 매우 빠름
        for prefix in _MATRIX_NAMES:
            nao = d.get(f"{prefix}_nao")
            if nao is not None:
                return int(nao)
        return None


# ── Registry ─────────────────────────────────────────────────────

FORMAT_REGISTRY: dict[str, StorageFormat] = {
    "pickle": PickleFormat(),
    "split_key": SplitKeyFormat(),
}


def get_format(name: str) -> StorageFormat:
    """이름으로 StorageFormat 인스턴스 반환."""
    if name not in FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown storage format '{name}'. "
            f"Available: {list(FORMAT_REGISTRY)}"
        )
    return FORMAT_REGISTRY[name]


def detect_format(env) -> StorageFormat:
    """LMDB에서 저장 포맷을 자동 감지.

    __format__ key가 있으면 해당 포맷, 없으면 pickle (v1 호환).
    """
    with env.begin() as txn:
        raw = txn.get(b"__format__")
        if raw is None:
            return FORMAT_REGISTRY["pickle"]
        name = raw.decode("ascii")
        return get_format(name)
