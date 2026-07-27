"""Model prediction results with metadata.

평가 결과를 저장/로드하고, reference 데이터와 비교 분석하는 도구.
Molecule(reference)과 분리된 독립 구조 — Molecule을 변경하지 않음.
MLIP(energy/forces)와 DFT(Hamiltonian/density matrix) 모두 지원.

Usage:
    # MLIP eval
    collector = PredictionCollector(metadata={"model": "nequip", ...})
    for batch in loader:
        collector.collect(model(batch), batch)
    result = collector.finalize()
    result.save("predictions/nequip.npz")

    # Hamiltonian eval
    collector = PredictionCollector(
        metadata={"model": "qhflow2"},
        matrix_keys=["hamiltonian"],  # 행렬 예측도 수집
    )
    for batch in loader:
        collector.collect(model(batch), batch)
    result = collector.finalize()
    result.save("predictions/qhflow2.npz")  # 행렬은 packed으로 저장

    # Reference 없이 분석
    result = PredictionResult.load("predictions/qhflow2.npz")
    print(result.describe())

    # Reference와 비교
    report = result.compare(dataset)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def _pack_upper_tri(M: np.ndarray) -> np.ndarray:
    """Symmetric matrix → packed 1D (upper triangle)."""
    nao = M.shape[-1]
    idx = np.triu_indices(nao)
    if M.ndim == 3:  # (2, nao, nao) unrestricted
        return np.stack([M[0][idx], M[1][idx]])
    return M[idx]


def _unpack_upper_tri(packed: np.ndarray, nao: int) -> np.ndarray:
    """Packed 1D → symmetric matrix."""
    idx = np.triu_indices(nao)
    if packed.ndim == 2:  # (2, tri_len) unrestricted
        M = np.zeros((2, nao, nao), dtype=packed.dtype)
        M[0][idx] = packed[0]
        M[0][idx[1], idx[0]] = packed[0]
        M[1][idx] = packed[1]
        M[1][idx[1], idx[0]] = packed[1]
        return M
    M = np.zeros((nao, nao), dtype=packed.dtype)
    M[idx] = packed
    M[idx[1], idx[0]] = packed
    return M


# ── PredictionResult ─────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Model prediction results for a set of molecules.

    Per-sample 예측값 + geometry + 행렬 + 공유 metadata.

    Args:
        energy: (N,) 예측 에너지.
        forces: list of (n_atoms, 3) 예측 포스.
        stress: (N, 6) 예측 stress (Voigt).
        matrices: {"hamiltonian": [...], "density_matrix": [...], ...}
            각 value는 list of (nao, nao) or (2, nao, nao).
        atomic_numbers: list of (n_atoms,) 원자 번호.
        positions: list of (n_atoms, 3) 좌표.
        indices: (N,) 원본 dataset에서의 샘플 인덱스.
        metadata: 모델/실험 정보.
    """

    energy: Optional[np.ndarray] = None
    forces: Optional[list[np.ndarray]] = None
    stress: Optional[np.ndarray] = None
    matrices: Optional[dict[str, list[np.ndarray]]] = None
    atomic_numbers: Optional[list[np.ndarray]] = None
    positions: Optional[list[np.ndarray]] = None
    indices: Optional[np.ndarray] = None
    convention: Optional[str] = None  # 행렬의 m-ordering convention ("pyscf", "e3nn", ...)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        if self.energy is not None:
            return len(self.energy)
        if self.forces is not None:
            return len(self.forces)
        if self.matrices:
            first = next(iter(self.matrices.values()))
            return len(first)
        if self.indices is not None:
            return len(self.indices)
        return 0

    # ── Save / Load ──────────────────────────────────────────────

    def save(self, path: str, pack_matrices: bool = True) -> None:
        """예측 결과를 .npz 파일로 저장.

        Args:
            path: 출력 경로 (.npz).
            pack_matrices: True면 대칭 행렬을 upper triangle로 압축 저장.
                nao=100 기준 ~2x 용량 절감.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        n = len(self)
        save_dict = {"_n_samples": np.array(n)}

        if self.energy is not None:
            save_dict["energy"] = np.asarray(self.energy, dtype=np.float64)

        if self.forces is not None:
            for i, f in enumerate(self.forces):
                save_dict[f"forces_{i}"] = np.asarray(f, dtype=np.float64)

        if self.stress is not None:
            save_dict["stress"] = np.asarray(self.stress, dtype=np.float64)

        # Matrices (packed upper triangle)
        if self.matrices:
            mat_names = sorted(self.matrices.keys())
            save_dict["_matrix_names"] = np.array(json.dumps(mat_names))
            for name in mat_names:
                mat_list = self.matrices[name]
                for i, M in enumerate(mat_list):
                    M = np.asarray(M, dtype=np.float64)
                    nao = M.shape[-1]
                    save_dict[f"{name}_nao_{i}"] = np.array(nao)
                    if pack_matrices:
                        save_dict[f"{name}_packed_{i}"] = _pack_upper_tri(M)
                    else:
                        save_dict[f"{name}_{i}"] = M

        if self.atomic_numbers is not None:
            for i, z in enumerate(self.atomic_numbers):
                save_dict[f"Z_{i}"] = np.asarray(z, dtype=np.int32)

        if self.positions is not None:
            for i, p in enumerate(self.positions):
                save_dict[f"pos_{i}"] = np.asarray(p, dtype=np.float64)

        if self.indices is not None:
            save_dict["indices"] = np.asarray(self.indices, dtype=np.int64)

        if self.convention is not None:
            save_dict["_convention"] = np.array(self.convention)

        meta = {**self.metadata}
        if "date" not in meta:
            meta["date"] = datetime.now().isoformat(timespec="seconds")
        save_dict["metadata_json"] = np.array(json.dumps(meta, ensure_ascii=False))

        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> "PredictionResult":
        """저장된 .npz에서 로드."""
        raw = np.load(path, allow_pickle=False)

        n = int(raw["_n_samples"]) if "_n_samples" in raw else 0
        if n == 0 and "forces_n_samples" in raw:
            n = int(raw["forces_n_samples"])

        energy = raw["energy"] if "energy" in raw else None
        stress = raw["stress"] if "stress" in raw else None
        indices = raw["indices"] if "indices" in raw else None

        forces = None
        if "forces_0" in raw:
            forces = [raw[f"forces_{i}"] for i in range(n)]

        atomic_numbers = None
        if "Z_0" in raw:
            atomic_numbers = [raw[f"Z_{i}"] for i in range(n)]

        positions = None
        if "pos_0" in raw:
            positions = [raw[f"pos_{i}"] for i in range(n)]

        # Matrices
        matrices = None
        if "_matrix_names" in raw:
            mat_names = json.loads(str(raw["_matrix_names"]))
            matrices = {}
            for name in mat_names:
                mat_list = []
                for i in range(n):
                    nao = int(raw[f"{name}_nao_{i}"])
                    packed_key = f"{name}_packed_{i}"
                    full_key = f"{name}_{i}"
                    if packed_key in raw:
                        mat_list.append(_unpack_upper_tri(raw[packed_key], nao))
                    elif full_key in raw:
                        mat_list.append(raw[full_key])
                matrices[name] = mat_list

        convention = str(raw["_convention"]) if "_convention" in raw else None

        metadata = {}
        if "metadata_json" in raw:
            metadata = json.loads(str(raw["metadata_json"]))

        return cls(
            energy=energy,
            forces=forces,
            stress=stress,
            matrices=matrices,
            atomic_numbers=atomic_numbers,
            positions=positions,
            indices=indices,
            convention=convention,
            metadata=metadata,
        )

    # ── Describe (reference 없이) ────────────────────────────────

    def describe(self) -> str:
        """prediction 자체의 통계 요약 (reference 불필요)."""
        lines = [f"PredictionResult (n={len(self)})"]

        if "model" in self.metadata:
            lines.append(f"  model: {self.metadata['model']}")
        if "date" in self.metadata:
            lines.append(f"  date: {self.metadata['date']}")
        if "dataset" in self.metadata:
            lines.append(f"  dataset: {self.metadata['dataset']}")

        if self.energy is not None:
            e = self.energy
            lines.append(
                f"  energy: mean={e.mean():.4f}  std={e.std():.4f}"
                f"  range=[{e.min():.4f}, {e.max():.4f}]"
            )

        if self.forces is not None:
            f_norms = np.array([np.linalg.norm(f, axis=1).mean() for f in self.forces])
            lines.append(
                f"  forces: mean_norm={f_norms.mean():.4f}"
                f"  max_norm={f_norms.max():.4f}"
            )

        if self.convention is not None:
            lines.append(f"  convention: {self.convention}")

        if self.matrices:
            for name, mat_list in self.matrices.items():
                nao_vals = [m.shape[-1] for m in mat_list]
                sizes = f"nao={min(nao_vals)}-{max(nao_vals)}" if min(nao_vals) != max(nao_vals) else f"nao={nao_vals[0]}"
                spin = "unrestricted" if mat_list[0].ndim == 3 else "restricted"
                lines.append(f"  {name}: {sizes}, {spin}, {len(mat_list)} samples")

        if self.atomic_numbers is not None:
            n_atoms = [len(z) for z in self.atomic_numbers]
            lines.append(f"  atoms: {min(n_atoms)}-{max(n_atoms)} atoms/mol")

        return "\n".join(lines)

    # ── Compare (reference 필요) ─────────────────────────────────

    def compare(
        self,
        dataset=None,
        ref_energy: np.ndarray | None = None,
        ref_forces: list[np.ndarray] | None = None,
        ref_matrices: dict[str, list[np.ndarray]] | None = None,
        energy_unit: str = "eV",
        force_unit: str = "meV/A",
    ) -> "ComparisonReport":
        """Reference와 비교하여 오차 통계 생성.

        Args:
            dataset: LMDBDataset 또는 Molecule 리스트 (자동으로 reference 추출).
            ref_energy: (N,) reference 에너지 (dataset 대신 직접 전달).
            ref_forces: list of (n_atoms, 3) reference 포스.
            ref_matrices: {"hamiltonian": [...], ...} reference 행렬.
            energy_unit: 에너지 오차 단위.
            force_unit: 포스 오차 단위.
        """
        n = len(self)

        if dataset is not None and ref_energy is None and ref_forces is None:
            ref_energy, ref_forces, ref_matrices_auto = self._load_reference(dataset)
            if ref_matrices is None:
                ref_matrices = ref_matrices_auto

        report = ComparisonReport(metadata=self.metadata, n_samples=n)

        # Energy errors
        if self.energy is not None and ref_energy is not None:
            pred_e = np.asarray(self.energy, dtype=np.float64)
            ref_e = np.asarray(ref_energy, dtype=np.float64)
            k = min(len(pred_e), len(ref_e))
            err = pred_e[:k] - ref_e[:k]
            report.energy_mae = float(np.abs(err).mean())
            report.energy_rmse = float(np.sqrt((err ** 2).mean()))
            report.energy_max_error = float(np.abs(err).max())
            report.energy_errors = err
            report.energy_unit = energy_unit

        # Force errors
        if self.forces is not None and ref_forces is not None:
            all_err = []
            per_sample_mae = []
            for pred_f, ref_f in zip(self.forces, ref_forces):
                err = np.asarray(pred_f) - np.asarray(ref_f)
                all_err.append(err)
                per_sample_mae.append(np.abs(err).mean())

            all_err_flat = np.concatenate([e.ravel() for e in all_err])
            scale = 1000.0 if force_unit == "meV/A" else 1.0

            report.force_mae = float(np.abs(all_err_flat).mean() * scale)
            report.force_rmse = float(np.sqrt((all_err_flat ** 2).mean()) * scale)
            report.force_max_error = float(np.abs(all_err_flat).max() * scale)
            report.force_per_sample_mae = np.array(per_sample_mae) * scale
            report.force_unit = force_unit

        # Matrix errors
        if self.matrices and ref_matrices:
            report.matrix_errors = {}
            for name in self.matrices:
                if name not in ref_matrices:
                    continue
                pred_list = self.matrices[name]
                ref_list = ref_matrices[name]
                all_err = []
                for pred_m, ref_m in zip(pred_list, ref_list):
                    err = np.asarray(pred_m) - np.asarray(ref_m)
                    all_err.append(np.abs(err).mean())
                all_err = np.array(all_err)
                report.matrix_errors[name] = {
                    "mae": float(all_err.mean()),
                    "rmse": float(np.sqrt((all_err ** 2).mean())),
                    "max": float(all_err.max()),
                }

        return report

    def _load_reference(self, dataset) -> tuple:
        """dataset에서 reference energy/forces/matrices 추출."""
        from dft_dataset.molecule import Molecule

        n = len(self)
        indices = self.indices if self.indices is not None else np.arange(n)

        ref_energies, ref_forces = [], []
        ref_matrices = {}
        matrix_names = list(self.matrices.keys()) if self.matrices else []

        for idx in indices:
            idx = int(idx)
            if isinstance(dataset, (list, tuple)):
                mol = dataset[idx]
            else:
                mol = dataset.get_molecule(idx) if hasattr(dataset, "get_molecule") else dataset[idx]

            if not isinstance(mol, Molecule):
                raise TypeError(f"Expected Molecule, got {type(mol)}")

            if mol.energy is not None:
                ref_energies.append(mol.energy)
            if mol.forces is not None:
                ref_forces.append(mol.forces)

            for name in matrix_names:
                val = getattr(mol, name, None)
                if val is not None:
                    ref_matrices.setdefault(name, []).append(val)

        ref_e = np.array(ref_energies, dtype=np.float64) if ref_energies else None
        ref_f = ref_forces if ref_forces else None
        ref_m = ref_matrices if ref_matrices else None
        return ref_e, ref_f, ref_m

    def summary(self) -> str:
        parts = [f"PredictionResult(n={len(self)})"]
        if self.energy is not None:
            parts.append("E")
        if self.forces is not None:
            parts.append("F")
        if self.stress is not None:
            parts.append("S")
        if self.matrices:
            parts.append("mat:" + ",".join(sorted(self.matrices.keys())))
        if self.positions is not None:
            parts.append("geom")
        if "model" in self.metadata:
            parts.append(f"model={self.metadata['model']}")
        if "date" in self.metadata:
            parts.append(f"date={self.metadata['date']}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return self.summary()


# ── PredictionCollector ──────────────────────────────────────────

class PredictionCollector:
    """Eval loop에서 batch 단위로 prediction을 수집.

    Usage:
        # MLIP
        collector = PredictionCollector(metadata={"model": "nequip"})

        # Hamiltonian prediction
        collector = PredictionCollector(
            metadata={"model": "qhflow2"},
            matrix_keys=["hamiltonian"],
        )

        for batch in loader:
            out = model(batch)
            collector.collect(out, batch)
        result = collector.finalize()
    """

    def __init__(
        self,
        metadata: dict | None = None,
        store_geometry: bool = False,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        matrix_keys: list[str] | None = None,
        flush_every: int | None = None,
        flush_dir: str | None = None,
    ):
        """
        Args:
            metadata: 모델/실험 정보.
            store_geometry: True면 positions, atomic_numbers도 저장.
            energy_key: model output에서 energy key.
            forces_key: model output에서 forces key.
            stress_key: model output에서 stress key.
            matrix_keys: model output에서 수집할 행렬 key 목록.
            flush_every: N 샘플마다 중간 결과를 디스크에 flush (메모리 절약).
                None이면 전부 메모리에 유지 (기본 동작).
            flush_dir: flush할 디렉토리 경로. flush_every 설정 시 필수.
        """
        self.metadata = metadata or {}
        self.store_geometry = store_geometry
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.matrix_keys = matrix_keys or []

        self._energies: list[np.ndarray] = []
        self._forces: list[np.ndarray] = []
        self._stress: list[np.ndarray] = []
        self._matrices: dict[str, list[np.ndarray]] = {k: [] for k in self.matrix_keys}
        self._atomic_numbers: list[np.ndarray] = []
        self._positions: list[np.ndarray] = []
        self._has_energy = False
        self._has_forces = False
        self._has_stress = False
        self._convention: str | None = None
        self._flush_every = flush_every
        self._flush_dir = flush_dir
        self._flush_count = 0
        self._flushed_parts: list[str] = []
        self._n_collected = 0

    def collect(self, output: dict, batch: dict) -> None:
        """한 batch의 모델 출력을 수집.

        Args:
            output: model forward 결과.
                MLIP: {"energy": (B,), "forces": (N_total, 3)}
                DFT:  {"hamiltonian": (B, nao, nao), ...}
            batch: DataLoader batch. num_atoms 또는 batch index 필요.
                행렬은 nao 또는 matrix_mask로 분리.
        """
        num_atoms = _get(batch, "num_atoms")
        batch_idx = _get(batch, "batch")

        if num_atoms is not None:
            num_atoms = _to_numpy(num_atoms)
        elif batch_idx is not None:
            batch_idx_np = _to_numpy(batch_idx)
            num_atoms = np.bincount(batch_idx_np.astype(np.int64))

        if num_atoms is None:
            raise ValueError("batch must contain 'num_atoms' or 'batch'")

        B = len(num_atoms)
        splits = np.cumsum(num_atoms)[:-1]

        # Energy
        pred_energy = _get(output, self.energy_key)
        if pred_energy is not None:
            self._has_energy = True
            self._energies.append(_to_numpy(pred_energy).ravel()[:B])

        # Forces (per-atom → split by molecule)
        pred_forces = _get(output, self.forces_key)
        if pred_forces is not None:
            self._has_forces = True
            per_mol = np.split(_to_numpy(pred_forces), splits)
            self._forces.extend(per_mol)

        # Stress
        pred_stress = _get(output, self.stress_key)
        if pred_stress is not None:
            self._has_stress = True
            self._stress.append(_to_numpy(pred_stress)[:B])

        # Matrices (padded batch → unpad per sample)
        nao_tensor = _get(batch, "nao")
        for key in self.matrix_keys:
            pred_mat = _get(output, key)
            if pred_mat is None:
                continue
            pred_mat = _to_numpy(pred_mat)

            if nao_tensor is not None:
                # Padded: (B, [2,] max_nao, max_nao) → unpad each sample
                nao_list = _to_numpy(nao_tensor)
                for i in range(B):
                    nao = int(nao_list[i])
                    if pred_mat.ndim == 4:  # (B, 2, max_nao, max_nao) unrestricted
                        self._matrices[key].append(pred_mat[i, :, :nao, :nao].copy())
                    else:  # (B, max_nao, max_nao)
                        self._matrices[key].append(pred_mat[i, :nao, :nao].copy())
            else:
                # No padding info → store as-is (assumes uniform size)
                for i in range(B):
                    self._matrices[key].append(pred_mat[i].copy())

        # Convention (from batch, set by DFTCollator)
        if self._convention is None:
            conv = _get(batch, "convention")
            if conv is not None:
                self._convention = str(conv)

        # Geometry
        if self.store_geometry:
            Z = _get(batch, "atomic_numbers", "z")
            pos = _get(batch, "positions", "pos")
            if Z is not None:
                self._atomic_numbers.extend(np.split(_to_numpy(Z), splits))
            if pos is not None:
                self._positions.extend(np.split(_to_numpy(pos), splits))

        # Auto-flush (streaming evaluation)
        self._n_collected += B
        if self._flush_every and self._n_collected >= self._flush_every:
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        """현재 메모리의 수집 데이터를 디스크에 중간 저장 후 메모리 해제."""
        if self._flush_dir is None:
            raise ValueError("flush_dir must be set for streaming evaluation")

        from pathlib import Path
        Path(self._flush_dir).mkdir(parents=True, exist_ok=True)

        part = self._build_result()
        part_path = str(Path(self._flush_dir) / f"part_{self._flush_count:04d}.npz")
        part.save(part_path)
        self._flushed_parts.append(part_path)
        self._flush_count += 1

        # Clear memory
        self._energies.clear()
        self._forces.clear()
        self._stress.clear()
        for k in self._matrices:
            self._matrices[k].clear()
        self._atomic_numbers.clear()
        self._positions.clear()
        self._n_collected = 0

    def _build_result(self) -> PredictionResult:
        """현재 메모리의 데이터로 PredictionResult 생성 (flush/finalize 공용)."""
        matrices = None
        if any(len(v) > 0 for v in self._matrices.values()):
            matrices = {k: v for k, v in self._matrices.items() if v}

        return PredictionResult(
            energy=np.concatenate(self._energies) if self._has_energy and self._energies else None,
            forces=self._forces.copy() if self._has_forces and self._forces else None,
            stress=np.concatenate(self._stress) if self._has_stress and self._stress else None,
            matrices=matrices,
            atomic_numbers=self._atomic_numbers.copy() if self._atomic_numbers else None,
            positions=self._positions.copy() if self._positions else None,
            convention=self._convention,
            metadata=self.metadata,
        )

    def finalize(self) -> PredictionResult:
        """수집된 데이터를 PredictionResult로 변환.

        streaming 모드(flush_every)에서는 디스크의 part 파일들을 합쳐서 반환.
        """
        # Flush remaining
        if self._flushed_parts and (self._energies or self._forces or
                                     any(self._matrices.values())):
            self._flush_to_disk()

        # Non-streaming: build from memory
        if not self._flushed_parts:
            return self._build_result()

        # Streaming: merge all parts
        return self._merge_parts()

    def _merge_parts(self) -> PredictionResult:
        """디스크에 flush된 part 파일들을 하나로 합침."""
        parts = [PredictionResult.load(p) for p in self._flushed_parts]

        energy = None
        if any(p.energy is not None for p in parts):
            energy = np.concatenate([p.energy for p in parts if p.energy is not None])

        forces = None
        if any(p.forces is not None for p in parts):
            forces = []
            for p in parts:
                if p.forces:
                    forces.extend(p.forces)

        stress = None
        if any(p.stress is not None for p in parts):
            stress = np.concatenate([p.stress for p in parts if p.stress is not None])

        matrices = None
        if any(p.matrices is not None for p in parts):
            matrices = {}
            for p in parts:
                if p.matrices:
                    for k, v in p.matrices.items():
                        matrices.setdefault(k, []).extend(v)

        atomic_numbers = None
        if any(p.atomic_numbers is not None for p in parts):
            atomic_numbers = []
            for p in parts:
                if p.atomic_numbers:
                    atomic_numbers.extend(p.atomic_numbers)

        positions = None
        if any(p.positions is not None for p in parts):
            positions = []
            for p in parts:
                if p.positions:
                    positions.extend(p.positions)

        return PredictionResult(
            energy=energy,
            forces=forces,
            stress=stress,
            matrices=matrices,
            atomic_numbers=atomic_numbers,
            positions=positions,
            convention=self._convention,
            metadata=self.metadata,
        )

    def __len__(self) -> int:
        if self._energies:
            return sum(len(e) for e in self._energies)
        if self._forces:
            return len(self._forces)
        for v in self._matrices.values():
            if v:
                return len(v)
        return 0


# ── Helpers ──────────────────────────────────────────────────────

def _get(obj, key, fallback_key=None):
    """dict or PyG Batch에서 값 추출."""
    if isinstance(obj, dict):
        v = obj.get(key)
        if v is None and fallback_key:
            v = obj.get(fallback_key)
        return v
    v = getattr(obj, key, None)
    if v is None and fallback_key:
        v = getattr(obj, fallback_key, None)
    return v


def _to_numpy(val) -> np.ndarray:
    """torch.Tensor or ndarray → ndarray."""
    if isinstance(val, np.ndarray):
        return val
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy()
    return np.asarray(val)


# ── ComparisonReport ─────────────────────────────────────────────

@dataclass
class ComparisonReport:
    """Reference vs prediction 비교 결과."""

    n_samples: int = 0
    metadata: dict = field(default_factory=dict)

    # Energy
    energy_mae: Optional[float] = None
    energy_rmse: Optional[float] = None
    energy_max_error: Optional[float] = None
    energy_errors: Optional[np.ndarray] = None
    energy_unit: str = "eV"

    # Forces
    force_mae: Optional[float] = None
    force_rmse: Optional[float] = None
    force_max_error: Optional[float] = None
    force_per_sample_mae: Optional[np.ndarray] = None
    force_unit: str = "meV/A"

    # Matrices
    matrix_errors: Optional[dict[str, dict]] = None

    def __repr__(self) -> str:
        lines = [f"ComparisonReport (n={self.n_samples})"]
        if "model" in self.metadata:
            lines.append(f"  model: {self.metadata['model']}")
        if self.energy_mae is not None:
            lines.append(
                f"  energy: MAE={self.energy_mae:.4f} {self.energy_unit}"
                f"  RMSE={self.energy_rmse:.4f}  max={self.energy_max_error:.4f}"
            )
        if self.force_mae is not None:
            lines.append(
                f"  forces: MAE={self.force_mae:.1f} {self.force_unit}"
                f"  RMSE={self.force_rmse:.1f}  max={self.force_max_error:.1f}"
            )
        if self.matrix_errors:
            for name, errs in self.matrix_errors.items():
                lines.append(
                    f"  {name}: MAE={errs['mae']:.6f}"
                    f"  RMSE={errs['rmse']:.6f}  max={errs['max']:.6f}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serializable dict로 변환."""
        d = {
            "n_samples": self.n_samples,
            "metadata": self.metadata,
        }
        if self.energy_mae is not None:
            d["energy"] = {
                "mae": self.energy_mae,
                "rmse": self.energy_rmse,
                "max_error": self.energy_max_error,
                "unit": self.energy_unit,
            }
        if self.force_mae is not None:
            d["forces"] = {
                "mae": self.force_mae,
                "rmse": self.force_rmse,
                "max_error": self.force_max_error,
                "unit": self.force_unit,
            }
        if self.matrix_errors:
            d["matrices"] = self.matrix_errors
        return d
