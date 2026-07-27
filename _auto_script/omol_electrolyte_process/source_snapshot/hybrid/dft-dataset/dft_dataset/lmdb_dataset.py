"""LMDB-backed dataset for DFT Molecule data.

LMDB 파일에 Molecule dict를 저장/로드. 직렬화 포맷은 교체 가능:
  - "pickle" (기본): 전체 dict를 하나의 pickle blob으로. 기존 LMDB 호환.
  - "split_key": 스칼라는 pickle, 대용량 array는 별도 LMDB key로.
      필요한 필드만 읽어서 I/O 최소화 (행렬 큰 데이터에 유리).

Note: lmdb >= 2.1.1 필수. 2.1.0에서 MDB_CORRUPTED 버그 있음.

Usage:
    # Write (기본 pickle 포맷)
    LMDBDataset.write(molecules, "output.lmdb")

    # Write (split_key 포맷 — 대규모 DFT 데이터)
    LMDBDataset.write(molecules, "output.lmdb", format="split_key")

    # Read — 포맷 자동 감지
    ds = LMDBDataset("output.lmdb")
    mol = ds[0]          # → Molecule

    # Read (fast path)
    ds = LMDBDataset("output.lmdb", fields={"atomic_numbers", "positions", "energy", "forces"})
    sample = ds[0]       # → dict
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence

import lmdb
import numpy as np

from .molecule import Molecule, decode_fields
from .storage_formats import StorageFormat, detect_format, get_format


class LMDBDataset:
    """LMDB-backed dataset of Molecule objects.

    Storage format:
        key   = 4-byte big-endian integer index
        value = format-dependent (see storage_formats.py)
        Special keys: b"__len__", b"__schema__", b"__format__"

    Args:
        path: LMDB 파일 경로.
        readonly: True면 읽기 전용 (기본값).
        fields: decode할 필드 집합. None이면 Molecule 반환 (기존 동작).
            예: {"atomic_numbers", "positions", "energy", "forces"}
        to_torch: True면 numpy 대신 torch.Tensor 반환 (fields 모드에서만).
        dtype: float tensor dtype (fields + to_torch 모드에서만).
    """

    def __init__(
        self,
        path: str,
        readonly: bool = True,
        fields: Sequence[str] | set[str] | None = None,
        to_torch: bool = False,
        dtype: str = "float32",
    ):
        self.path = str(path)
        self._readonly = readonly
        self.fields = set(fields) if fields is not None else None
        self.to_torch = to_torch
        self.dtype = dtype
        self._env = None  # lazy init for multi-worker safety
        self._fmt: StorageFormat | None = None
        self._len = self._read_len()

    def _get_env(self) -> lmdb.Environment:
        """Lazy LMDB env — worker fork 시 각 프로세스가 독립 핸들 생성."""
        if self._env is None:
            self._env = lmdb.open(
                self.path,
                readonly=self._readonly,
                lock=False,
                readahead=False,
                max_readers=1024,
                meminit=False,
            )
            self._fmt = detect_format(self._env)
        return self._env

    def _get_fmt(self) -> StorageFormat:
        """현재 LMDB의 StorageFormat (lazy detect)."""
        if self._fmt is None:
            self._get_env()
        return self._fmt

    def _read_len(self) -> int:
        """Read dataset length (one-time, own env handle)."""
        env = lmdb.open(
            self.path, readonly=True, lock=False,
            readahead=False, max_readers=1024, meminit=False,
        )
        with env.begin() as txn:
            raw = txn.get(b"__len__")
            length = int.from_bytes(raw, "big") if raw else 0
        env.close()
        return length

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        """샘플 반환. fields 설정에 따라 Molecule 또는 decoded dict."""
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(f"index {idx} out of range [0, {self._len})")

        env = self._get_env()
        fmt = self._get_fmt()

        with env.begin() as txn:
            if self.fields is not None:
                d = fmt.decode_fields(idx, txn, self.fields)
                return decode_fields(d, self.fields, self.to_torch, self.dtype)
            else:
                d = fmt.decode_sample(idx, txn)
                return Molecule.from_dict(d)

    def get_dict(self, idx: int) -> dict:
        """Raw dict without any decoding (가장 빠름)."""
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(f"index {idx} out of range [0, {self._len})")
        env = self._get_env()
        with env.begin() as txn:
            return self._get_fmt().decode_sample(idx, txn)

    def get_molecule(self, idx: int) -> Molecule:
        """항상 Molecule 반환 (fields 설정 무시)."""
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(f"index {idx} out of range [0, {self._len})")
        env = self._get_env()
        with env.begin() as txn:
            d = self._get_fmt().decode_sample(idx, txn)
        return Molecule.from_dict(d)

    def get_nao_list(self) -> np.ndarray:
        """모든 샘플의 nao를 추출 (BucketBatchSampler용, 1회성 스캔).

        split_key 포맷에서는 스칼라만 읽으므로 pickle 포맷보다 빠름.
        """
        nao_list = np.zeros(self._len, dtype=np.int32)
        env = self._get_env()
        fmt = self._get_fmt()
        with env.begin() as txn:
            for idx in range(self._len):
                nao = fmt.read_nao(idx, txn)
                if nao is not None:
                    nao_list[idx] = nao
        return nao_list

    @property
    def format_name(self) -> str:
        """현재 LMDB의 저장 포맷 이름."""
        return self._get_fmt().name

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._fmt = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Schema ─────────────────────────────────────────────────

    @property
    def schema(self) -> dict | None:
        """Dataset-level schema metadata (version 2+).

        Returns None if no schema stored (version 1 datasets).
        """
        env = self._get_env()
        with env.begin() as txn:
            raw = txn.get(b"__schema__")
            return pickle.loads(raw) if raw else None

    @property
    def targets(self) -> list[str]:
        """Available training targets in this dataset.

        Checks schema first; falls back to inspecting first sample.
        """
        schema = self.schema
        if schema and "targets" in schema:
            return [k for k, v in schema["targets"].items() if v]
        # Fallback: inspect first sample
        if self._len == 0:
            return []
        d = self.get_dict(0)
        targets = []
        if "energy" in d:
            targets.append("energy")
        if "forces" in d or "forces_shape" in d:
            targets.append("forces")
        if any(k.startswith("density_matrix") for k in d):
            targets.append("density_matrix")
        if any(k.startswith("hamiltonian") for k in d):
            targets.append("hamiltonian")
        if any(k.startswith("overlap") for k in d):
            targets.append("overlap")
        return targets

    # ── Writer ──────────────────────────────────────────────────

    @staticmethod
    def write(
        molecules,
        path: str,
        packed: bool = True,
        batch_size: int = 500,
        map_size_per_sample_mb: float = 2.0,
        schema: dict | None = None,
        format: str = "pickle",
    ) -> int:
        """Molecule 리스트를 LMDB로 저장.

        Args:
            molecules: iterable of Molecule (or list)
            path: 출력 LMDB 경로
            packed: 대칭 행렬 upper-triangle 압축
            batch_size: 트랜잭션당 쓰기 개수
            map_size_per_sample_mb: 샘플당 예상 크기 (MB)
            schema: dataset-level metadata dict (optional, stored as __schema__)
            format: 저장 포맷 ("pickle" or "split_key")

        Returns:
            저장된 샘플 수
        """
        fmt = get_format(format)
        molecules = list(molecules)
        n = len(molecules)
        map_size = max(int(n * map_size_per_sample_mb * 1024 * 1024), 1 << 30)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(path, map_size=map_size, readonly=False, lock=True)

        count = 0
        for start in range(0, n, batch_size):
            batch = molecules[start : start + batch_size]
            with env.begin(write=True) as txn:
                for i, mol in enumerate(batch):
                    idx = start + i
                    d = mol.to_dict(packed=packed)
                    fmt.encode_sample(d, idx, txn)
                    count += 1

        # Write length + schema + format marker
        with env.begin(write=True) as txn:
            txn.put(b"__len__", count.to_bytes(4, "big"))
            txn.put(b"__format__", format.encode("ascii"))
            if schema is not None:
                txn.put(
                    b"__schema__",
                    pickle.dumps(schema, protocol=pickle.HIGHEST_PROTOCOL),
                )

        env.close()
        return count

    @staticmethod
    def write_iter(
        mol_iter,
        path: str,
        total: Optional[int] = None,
        packed: bool = True,
        batch_size: int = 500,
        map_size: int = 1 << 32,  # 4 GB default
        schema: dict | None = None,
        format: str = "pickle",
    ) -> int:
        """Iterator/generator에서 LMDB로 스트리밍 저장 (메모리 효율).

        Args:
            mol_iter: Molecule iterator
            path: 출력 LMDB 경로
            total: 예상 총 개수 (map_size 계산용, None이면 map_size 직접 사용)
            packed: 대칭 행렬 upper-triangle 압축
            batch_size: 트랜잭션당 쓰기 개수
            map_size: LMDB map size in bytes
            schema: dataset-level metadata dict (optional, stored as __schema__)
            format: 저장 포맷 ("pickle" or "split_key")

        Returns:
            저장된 샘플 수
        """
        fmt = get_format(format)
        if total is not None:
            map_size = max(total * 2 * 1024 * 1024, 1 << 30)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(path, map_size=map_size, readonly=False, lock=True)

        count = 0
        buf_dicts = []

        for mol in mol_iter:
            d = mol.to_dict(packed=packed)
            buf_dicts.append(d)
            count += 1

            if len(buf_dicts) >= batch_size:
                with env.begin(write=True) as txn:
                    for i, dd in enumerate(buf_dicts):
                        fmt.encode_sample(dd, count - len(buf_dicts) + i, txn)
                buf_dicts.clear()

        # Flush remaining
        if buf_dicts:
            with env.begin(write=True) as txn:
                for i, dd in enumerate(buf_dicts):
                    fmt.encode_sample(dd, count - len(buf_dicts) + i, txn)

        # Write length + schema + format marker
        with env.begin(write=True) as txn:
            txn.put(b"__len__", count.to_bytes(4, "big"))
            txn.put(b"__format__", format.encode("ascii"))
            if schema is not None:
                txn.put(
                    b"__schema__",
                    pickle.dumps(schema, protocol=pickle.HIGHEST_PROTOCOL),
                )

        env.close()
        return count
