"""Spherical harmonics m-ordering conventions and conversion.

DFT 코드(PySCF)와 ML 프레임워크(e3nn)는 구면 조화 함수의 m 순서가 다르다.
Hamiltonian 등 orbital-indexed 행렬을 다룰 때 convention 변환이 필수.

Conventions:
    pyscf   — PySCF/libcint: l=1은 [+1,-1,0](px,py,pz), l≥2는 [-l,...,+l]
    e3nn    — e3nn: 표준 ascending [-l, ..., -1, 0, +1, ..., +l]
    orca    — ORCA: [0, +1, -1, +2, -2, ..., +l, -l] (모든 l)

참고: Claude Tips/docs/projects/pyscf_e3nn_convention.md
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass
from functools import lru_cache


# ── m-ordering definitions ───────────────────────────────────────────

def _pyscf_m_order(l: int) -> list[int]:
    """PySCF/libcint의 m 순서.

    l=0: [0]
    l=1: [+1, -1, 0]  (px, py, pz — libcint 비표준)
    l≥2: [-l, ..., -1, 0, +1, ..., +l]  (ascending, e3nn과 동일)
         d: dxy, dyz, dz², dxz, dx²-y²
    """
    if l == 0:
        return [0]
    if l == 1:
        return [1, -1, 0]
    # l >= 2: ascending (same as e3nn)
    return list(range(-l, l + 1))


def _e3nn_m_order(l: int) -> list[int]:
    """e3nn 표준 m 순서: [-l, -l+1, ..., -1, 0, +1, ..., +l]."""
    return list(range(-l, l + 1))


def _orca_m_order(l: int) -> list[int]:
    """ORCA m 순서: [0, +1, -1, +2, -2, ..., +l, -l].

    ORCA는 모든 angular momentum에서 동일한 패턴을 사용.
    l=0: [0]
    l=1: [0, +1, -1]  (pz, px, py)
    l=2: [0, +1, -1, +2, -2]  (dz², dxz, dyz, dx²-y², dxy)
    """
    if l == 0:
        return [0]
    order = [0]
    for m in range(1, l + 1):
        order.extend([+m, -m])
    return order


_CONVENTION_REGISTRY = {
    "pyscf": _pyscf_m_order,
    "e3nn": _e3nn_m_order,
    "orca": _orca_m_order,
}

KNOWN_CONVENTIONS = list(_CONVENTION_REGISTRY.keys())


def get_m_order(convention: str, l: int) -> list[int]:
    """Get m-ordering for angular momentum l in given convention."""
    if convention not in _CONVENTION_REGISTRY:
        raise ValueError(f"Unknown convention '{convention}'. Known: {KNOWN_CONVENTIONS}")
    return _CONVENTION_REGISTRY[convention](l)


# ── Shell definitions ────────────────────────────────────────────────

# shell label → angular momentum quantum number
SHELL_TO_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def shell_size(label: str, angular_type: str = "spherical") -> int:
    """Shell label → number of basis functions.

    Spherical: 2l+1 (s=1, p=3, d=5, f=7)
    Cartesian: (l+1)(l+2)/2 (s=1, p=3, d=6, f=10)
    """
    l = SHELL_TO_L[label]
    if angular_type == "spherical":
        return 2 * l + 1
    return (l + 1) * (l + 2) // 2


# ── Reorder index computation ────────────────────────────────────────

@lru_cache(maxsize=64)
def shell_reorder(l: int, src: str, dst: str) -> tuple[int, ...]:
    """Single shell의 src → dst reorder indices.

    Returns:
        permutation p such that dst_ordered[i] = src_ordered[p[i]]
    """
    if src == dst:
        return tuple(range(2 * l + 1))

    src_m = get_m_order(src, l)
    dst_m = get_m_order(dst, l)
    src_pos = {m: idx for idx, m in enumerate(src_m)}
    return tuple(src_pos[m] for m in dst_m)


def _build_atom_reorder_idx(
    shells: str,
    src: str,
    dst: str,
    angular_type: str = "spherical",
) -> np.ndarray:
    """단일 원자의 shell string에 대한 reorder index.

    Args:
        shells: e.g. "sssppd" (3 s-shells, 2 p-shells, 1 d-shell)
        src: source convention (e.g. "pyscf")
        dst: target convention (e.g. "e3nn")

    Returns:
        np.ndarray of indices, shape (nao_atom,)
    """
    indices = []
    offset = 0
    for label in shells:
        l = SHELL_TO_L[label]
        n = shell_size(label, angular_type)
        perm = shell_reorder(l, src, dst)
        indices.extend(offset + p for p in perm)
        offset += n
    return np.array(indices, dtype=np.int64)


def build_reorder_indices(
    atomic_numbers: np.ndarray,
    atom_to_shells: dict[int, str],
    src: str,
    dst: str,
    angular_type: str = "spherical",
) -> np.ndarray:
    """분자 전체의 orbital reorder index (H, S, D 행렬 변환용).

    Args:
        atomic_numbers: (N,) array of atomic numbers
        atom_to_shells: {Z: shell_string}, e.g. {1: "ssp", 8: "sssppd"}
        src: source convention
        dst: target convention

    Returns:
        np.ndarray of indices, shape (nao_total,)
        Usage: H_dst = H_src[np.ix_(idx, idx)]
    """
    if src == dst:
        nao = sum(
            sum(shell_size(c, angular_type) for c in atom_to_shells[int(z)])
            for z in atomic_numbers
        )
        return np.arange(nao, dtype=np.int64)

    all_indices = []
    offset = 0
    for z in atomic_numbers:
        shell_str = atom_to_shells[int(z)]
        atom_idx = _build_atom_reorder_idx(shell_str, src, dst, angular_type)
        all_indices.extend(offset + atom_idx)
        offset += len(atom_idx)
    return np.array(all_indices, dtype=np.int64)


def reorder_matrix(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Apply reorder to both axes of a square matrix."""
    return M[np.ix_(idx, idx)]


def reorder_rows(C: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Apply reorder to row axis (AO basis axis) of coefficient matrix."""
    return C[idx]


def signed_reorder_matrix(M: np.ndarray, idx: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply a signed AO permutation to both axes of a square matrix.

    This returns ``G * M[idx, idx] * G`` where ``G = diag(signs)``.  Signed
    permutations are orthogonal, so transforming both density and overlap with
    the same ``idx``/``signs`` preserves ``Tr(D S)``.
    """
    idx = np.asarray(idx, dtype=np.int64)
    signs = np.asarray(signs, dtype=np.asarray(M).dtype)
    if signs.shape != (len(idx),):
        raise ValueError(f"signs shape {signs.shape} does not match idx length {len(idx)}")
    out = reorder_matrix(M, idx)
    return (signs[:, None] * out) * signs[None, :]


def invert_signed_reorder_matrix(
    M: np.ndarray,
    idx: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    """Invert :func:`signed_reorder_matrix` for a square matrix."""
    idx = np.asarray(idx, dtype=np.int64)
    signs = np.asarray(signs, dtype=np.asarray(M).dtype)
    if signs.shape != (len(idx),):
        raise ValueError(f"signs shape {signs.shape} does not match idx length {len(idx)}")
    unflipped = (signs[:, None] * M) * signs[None, :]
    return reorder_matrix(unflipped, invert_reorder_indices(idx))


# ── Full AO-layout conversion ────────────────────────────────────────

@dataclass(frozen=True)
class AOShellSpec:
    """One contracted AO shell in a concrete AO layout.

    `shell_id` is a label such as `6s` or `4d`, paired with the atom index.
    It lets us separate shell order from per-shell angular/m ordering.
    """

    atom_index: int
    shell_id: str
    l: int
    size: int


_AO_LABEL_RE = re.compile(r"^\s*(?P<atom>\d+)\s+\S+\s+(?P<orb>\S+)\s*$")
_ORBITAL_RE = re.compile(r"^(?P<n>\d+)(?P<label>[spdfgh])")


def invert_reorder_indices(idx: np.ndarray) -> np.ndarray:
    """Return inverse permutation for reorder indices.

    If `dst = src[idx]`, then `src = dst[invert_reorder_indices(idx)]`.
    """
    idx = np.asarray(idx, dtype=np.int64)
    inv = np.empty_like(idx)
    inv[idx] = np.arange(len(idx), dtype=np.int64)
    return inv


def _ao_label_to_shell_key(label: str) -> tuple[int, str, int]:
    """Parse a PySCF/ORCA-Molden AO label into `(atom_index, shell_id, l)`."""
    match = _AO_LABEL_RE.match(label)
    if match is None:
        raise ValueError(f"Cannot parse AO label: {label!r}")

    atom_index = int(match.group("atom"))
    orbital = match.group("orb")
    orbital_match = _ORBITAL_RE.match(orbital)
    if orbital_match is None:
        raise ValueError(f"Cannot parse orbital label: {orbital!r} from {label!r}")

    shell_label = orbital_match.group("label")
    if shell_label not in SHELL_TO_L:
        raise ValueError(f"Unsupported shell label {shell_label!r} in {label!r}")
    shell_id = f"{orbital_match.group('n')}{shell_label}"
    return atom_index, shell_id, SHELL_TO_L[shell_label]


def build_ao_shell_specs_from_labels(
    ao_labels: list[str],
    angular_type: str = "spherical",
) -> list[AOShellSpec]:
    """Build concrete shell specs from AO labels.

    PySCF `mol.ao_labels()` and `pyscf.tools.molden.load(...).ao_labels()`
    both expose labels rich enough to identify shell order. ORCA raw density
    (`.densities` / OMol `density_mat.npz`) uses ORCA's shell order, while
    PySCF uses its own shell order. These specs let us map one to the other.
    """
    specs: list[AOShellSpec] = []
    current_key: tuple[int, str, int] | None = None
    current_size = 0

    def flush() -> None:
        nonlocal current_key, current_size
        if current_key is None:
            return
        atom_index, shell_id, l = current_key
        expected = shell_size("spdfg"[l], angular_type)
        if current_size != expected:
            raise ValueError(
                f"AO labels for atom {atom_index} shell {shell_id} have size "
                f"{current_size}, expected {expected}"
            )
        specs.append(AOShellSpec(atom_index, shell_id, l, current_size))
        current_key = None
        current_size = 0

    for label in ao_labels:
        key = _ao_label_to_shell_key(label)
        if current_key is None:
            current_key = key
            current_size = 1
        elif key == current_key:
            current_size += 1
        else:
            flush()
            current_key = key
            current_size = 1
    flush()
    return specs


def build_ao_shell_specs_from_mol(mol, angular_type: str = "spherical") -> list[AOShellSpec]:
    """Build shell specs from a PySCF Mole-like object."""
    return build_ao_shell_specs_from_labels(mol.ao_labels(), angular_type=angular_type)


def build_layout_reorder_indices(
    src_shells: list[AOShellSpec],
    dst_shells: list[AOShellSpec],
    src_convention: str,
    dst_convention: str,
) -> np.ndarray:
    """Build reorder indices between concrete AO layouts.

    This generalizes `build_reorder_indices`: it handles both shell-order
    changes and per-shell angular/m-order changes.

    Returns:
        `idx` such that `M_dst = M_src[np.ix_(idx, idx)]`.
    """
    src_offsets: list[int] = []
    offset = 0
    src_lookup: dict[tuple[int, str], int] = {}
    for i, shell in enumerate(src_shells):
        key = (shell.atom_index, shell.shell_id)
        if key in src_lookup:
            raise ValueError(f"Duplicate source shell key: {key}")
        src_lookup[key] = i
        src_offsets.append(offset)
        offset += shell.size

    indices: list[int] = []
    for dst_shell in dst_shells:
        key = (dst_shell.atom_index, dst_shell.shell_id)
        if key not in src_lookup:
            raise ValueError(f"Destination shell {key} not found in source layout")
        src_shell = src_shells[src_lookup[key]]
        if src_shell.l != dst_shell.l or src_shell.size != dst_shell.size:
            raise ValueError(
                f"Shell mismatch for {key}: source l/size={src_shell.l}/{src_shell.size}, "
                f"destination l/size={dst_shell.l}/{dst_shell.size}"
            )
        perm = shell_reorder(src_shell.l, src_convention, dst_convention)
        src_offset = src_offsets[src_lookup[key]]
        indices.extend(src_offset + p for p in perm)
    return np.asarray(indices, dtype=np.int64)


def build_orca_to_pyscf_indices(orca_molden_mol, pyscf_mol) -> np.ndarray:
    """Return indices for ORCA raw density order -> PySCF AO order.

    `orca_molden_mol` should come from `pyscf.tools.molden.load` on an
    `orca_2mkl -molden` file for the same basis/atom ordering. The Molden AO
    labels provide ORCA's shell order; the raw ORCA density still uses ORCA's
    internal angular order inside each shell.
    """
    return build_layout_reorder_indices(
        build_ao_shell_specs_from_mol(orca_molden_mol),
        build_ao_shell_specs_from_mol(pyscf_mol),
        src_convention="orca",
        dst_convention="pyscf",
    )


def build_pyscf_to_orca_indices(orca_molden_mol, pyscf_mol) -> np.ndarray:
    """Return indices for PySCF AO order -> ORCA raw density order."""
    return invert_reorder_indices(build_orca_to_pyscf_indices(orca_molden_mol, pyscf_mol))


def build_orca_to_e3nn_indices(orca_molden_mol, pyscf_mol) -> np.ndarray:
    """Return indices for ORCA raw density order -> e3nn AO order.

    e3nn uses the same shell order as PySCF here, but ascending m-order within
    every shell.
    """
    return build_layout_reorder_indices(
        build_ao_shell_specs_from_mol(orca_molden_mol),
        build_ao_shell_specs_from_mol(pyscf_mol),
        src_convention="orca",
        dst_convention="e3nn",
    )


def build_e3nn_to_orca_indices(orca_molden_mol, pyscf_mol) -> np.ndarray:
    """Return indices for e3nn AO order -> ORCA raw density order."""
    return invert_reorder_indices(build_orca_to_e3nn_indices(orca_molden_mol, pyscf_mol))


# ── Cached ORCA shell-order conversion ───────────────────────────────

# ORCA shell order is basis/element specific and geometry independent.  OMol
# stores raw ORCA `.densities` matrices, but producing a Molden file for every
# molecule just to recover shell order is too expensive.  This cache was
# calibrated with ORCA 6.1.1 `orca_2mkl -molden` for the OMol lanes.
ORCA_SHELL_ORDER_CACHE: dict[str, dict[int, list[str]]] = {
    "def2-tzvpd": {
        1: ["1s", "2s", "3s", "2p", "3p"],
        3: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p"],
        4: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "3d", "5p"],
        5: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "3d", "4d", "4f", "6s", "5d"],
        6: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "3d", "4d", "4f", "6s", "5d"],
        7: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "3d", "4d", "4f", "6s", "5d"],
        8: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "3d", "4d", "4f", "6s", "5p", "5d"],
        9: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "3d", "4d", "4f", "6s", "5p", "5d"],
        11: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "6p"],
        12: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "6p"],
        13: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "4f", "6s", "5d"],
        14: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "4f", "6s", "5d"],
        15: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "4f", "6s", "5d"],
        16: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "4f", "6s", "7p", "5d"],
        17: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "4f", "6s", "7p", "5d"],
        19: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d"],
        20: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d"],
        21: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        22: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        23: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        24: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        25: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        26: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        27: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        28: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        29: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6d", "4f", "6p"],
        30: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "3d", "4d", "5d", "5p", "6p", "6d", "4f", "7p"],
        31: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "6d", "4f", "7s", "7d"],
        33: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "6d", "4f", "7s", "7d"],
        35: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "6d", "4f", "7s", "7p", "7d"],
        37: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d"],
        38: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d"],
        39: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        40: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        41: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        44: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        45: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        46: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        47: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        48: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        49: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "4f", "5f", "7s", "6d"],
        50: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "4f", "5f", "7s", "6d"],
        51: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "4f", "5f", "7s", "6d"],
        53: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "4f", "5f", "7s", "7p", "6d"],
        55: ["1s", "2s", "3s", "4s", "5s", "2p", "3p", "4p", "5p", "3d", "4d", "5d"],
        56: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f"],
        57: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f", "6p"],
        72: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f", "6p"],
        73: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f", "6p"],
        77: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f", "6p"],
        78: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f", "6p"],
        79: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "3d", "4d", "5d", "4f", "6p"],
        80: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "4f", "7p"],
        81: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "4f", "5f", "7s", "6d"],
        82: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "4f", "5f", "7s", "6d"],
        83: ["1s", "2s", "3s", "4s", "5s", "6s", "2p", "3p", "4p", "5p", "6p", "3d", "4d", "5d", "4f", "5f", "7s", "6d"],
    },
}


def _normalize_basis_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _nuclear_charges(mol) -> list[int]:
    """Return true nuclear charges, including cores represented by an ECP."""
    effective = [int(z) for z in mol.atom_charges()]
    atom_nelec_core = getattr(mol, "atom_nelec_core", None)
    if atom_nelec_core is None:
        return effective
    return [
        z + int(atom_nelec_core(atom_index))
        for atom_index, z in enumerate(effective)
    ]


def build_orca_shell_order_cache_from_mol(orca_molden_mol) -> dict[int, list[str]]:
    """Extract `{Z: [shell_id, ...]}` from an ORCA Molden-loaded molecule.

    The result is useful for promoting a one-time ORCA/Molden calibration into
    a reusable basis/element cache.
    """
    charges = _nuclear_charges(orca_molden_mol)
    by_atom: dict[tuple[int, int], list[str]] = {}
    for shell in build_ao_shell_specs_from_mol(orca_molden_mol):
        z = charges[shell.atom_index]
        by_atom.setdefault((shell.atom_index, z), []).append(shell.shell_id)

    by_z: dict[int, list[str]] = {}
    for (_atom, z), order in by_atom.items():
        if z in by_z and by_z[z] != order:
            raise ValueError(f"Inconsistent ORCA shell order for Z={z}: {by_z[z]} vs {order}")
        by_z[z] = order
    return by_z


def build_orca_shell_specs_from_cache(
    pyscf_mol,
    basis_name: str,
    shell_order_cache: dict[str, dict[int, list[str]]] | None = None,
) -> list[AOShellSpec]:
    """Build ORCA raw-layout shell specs from a PySCF molecule and cache.

    The PySCF molecule supplies atom order, shell identities, angular momenta,
    and shell sizes.  The cache supplies only ORCA's per-element shell order.
    """
    cache = shell_order_cache or ORCA_SHELL_ORDER_CACHE
    basis_key = _normalize_basis_name(basis_name)
    if basis_key not in cache:
        raise KeyError(f"No ORCA shell-order cache for basis {basis_name!r}")

    pyscf_shells = build_ao_shell_specs_from_mol(pyscf_mol)
    by_atom: dict[int, dict[str, AOShellSpec]] = {}
    for shell in pyscf_shells:
        by_atom.setdefault(shell.atom_index, {})[shell.shell_id] = shell

    charges = _nuclear_charges(pyscf_mol)
    out: list[AOShellSpec] = []
    for atom_index, z in enumerate(charges):
        if z not in cache[basis_key]:
            raise KeyError(f"No ORCA shell-order cache for basis {basis_name!r}, Z={z}")
        atom_shells = by_atom.get(atom_index, {})
        for shell_id in cache[basis_key][z]:
            if shell_id not in atom_shells:
                raise ValueError(
                    f"Cached ORCA shell {shell_id!r} for Z={z} is not present "
                    f"in PySCF atom {atom_index} shells {sorted(atom_shells)}"
                )
            out.append(atom_shells[shell_id])

        extra = set(atom_shells) - set(cache[basis_key][z])
        if extra:
            raise ValueError(
                f"PySCF atom {atom_index} Z={z} has shells absent from ORCA cache: {sorted(extra)}"
            )
    return out


def build_cached_orca_to_pyscf_indices(
    pyscf_mol,
    basis_name: str,
    shell_order_cache: dict[str, dict[int, list[str]]] | None = None,
) -> np.ndarray:
    """Return ORCA raw density order -> PySCF AO order using shell-order cache."""
    return build_layout_reorder_indices(
        build_orca_shell_specs_from_cache(pyscf_mol, basis_name, shell_order_cache),
        build_ao_shell_specs_from_mol(pyscf_mol),
        src_convention="orca",
        dst_convention="pyscf",
    )


def build_cached_orca_to_e3nn_indices(
    pyscf_mol,
    basis_name: str,
    shell_order_cache: dict[str, dict[int, list[str]]] | None = None,
) -> np.ndarray:
    """Return ORCA raw density order -> e3nn AO order using shell-order cache."""
    return build_layout_reorder_indices(
        build_orca_shell_specs_from_cache(pyscf_mol, basis_name, shell_order_cache),
        build_ao_shell_specs_from_mol(pyscf_mol),
        src_convention="orca",
        dst_convention="e3nn",
    )


# ── ORCA JSON signed conversion ──────────────────────────────────────

def build_orca_json_signs_for_layout(
    dst_shells: list[AOShellSpec],
    dst_convention: str,
) -> np.ndarray:
    """Return destination-order signs for ORCA ``orca_2json`` AO matrices.

    ORCA 6.1.1 ``orca_2json`` exports overlap/density matrices that follow the
    cached ORCA shell and m ordering, but H2O/def2-TZVPD calibration showed an
    additional sign flip for f-shell ``m = +/-3`` components before they match
    PySCF/libcint overlap matrices.  The returned vector lives in the
    destination layout, i.e. after applying the ORCA-to-destination permutation.

    This is intentionally named ``orca_json`` rather than ``orca`` because the
    existing OMol raw ``.densities``/``.npz`` path must be revalidated before
    using the same sign correction there.
    """
    signs: list[float] = []
    for shell in dst_shells:
        for m in get_m_order(dst_convention, shell.l):
            sign = -1.0 if shell.l == 3 and abs(m) == 3 else 1.0
            signs.append(sign)
    return np.asarray(signs, dtype=np.float64)


def _raw_density_shell_source(
    *,
    atom_z: int,
    basis_key: str,
    dst_shell_id: str,
) -> tuple[str, float]:
    """Return source shell/sign for PySCF -> ORCA raw-density layout.

    Most ORCA raw-density shells are the same contracted functions as PySCF,
    up to shell/m ordering.  Be/def2-TZVPD is a calibrated exception: ORCA's
    raw p radial shells match PySCF after swapping the first two p shells and
    flipping the destination 2p shell sign.
    """
    if basis_key == "def2-tzvpd" and atom_z == 4:
        if dst_shell_id == "2p":
            return "3p", -1.0
        if dst_shell_id == "3p":
            return "2p", 1.0
    return dst_shell_id, 1.0


def build_orca_raw_density_layout_transform(
    pyscf_mol,
    basis_name: str,
    src_convention: str = "pyscf",
    dst_convention: str = "e3nn",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(idx, signs)`` for PySCF-like matrices -> raw-density layout.

    The destination uses the same atom/shell slots as the e3nn/PySCF layout
    consumed by ml_dft, but with the signed contracted-shell convention used by
    OMol's ORCA raw ``.densities`` and ``orca.S.tmp`` matrices.
    """
    shells = build_ao_shell_specs_from_mol(pyscf_mol)
    source_offsets: dict[tuple[int, str], int] = {}
    offset = 0
    for shell in shells:
        source_offsets[(shell.atom_index, shell.shell_id)] = offset
        offset += shell.size

    charges = _nuclear_charges(pyscf_mol)
    basis_key = _normalize_basis_name(basis_name)
    json_like_signs = build_orca_json_signs_for_layout(shells, dst_convention)

    indices: list[int] = []
    signs: list[float] = []
    dst_offset = 0
    for shell in shells:
        atom_z = charges[shell.atom_index]
        source_shell_id, shell_sign = _raw_density_shell_source(
            atom_z=atom_z,
            basis_key=basis_key,
            dst_shell_id=shell.shell_id,
        )
        key = (shell.atom_index, source_shell_id)
        if key not in source_offsets:
            raise ValueError(
                f"Raw-density source shell {source_shell_id!r} for atom "
                f"{shell.atom_index} Z={atom_z} is absent from PySCF shells"
            )
        src_offset = source_offsets[key]
        perm = shell_reorder(shell.l, src_convention, dst_convention)
        indices.extend(src_offset + p for p in perm)
        signs.extend(
            shell_sign * float(json_like_signs[dst_offset + i])
            for i in range(shell.size)
        )
        dst_offset += shell.size

    return np.asarray(indices, dtype=np.int64), np.asarray(signs, dtype=np.float64)


def build_cached_orca_json_to_layout_transform(
    pyscf_mol,
    basis_name: str,
    dst_convention: str = "pyscf",
    shell_order_cache: dict[str, dict[int, list[str]]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(idx, signs)`` for ORCA JSON matrices -> destination layout.

    The transform is applied with :func:`signed_reorder_matrix`.
    """
    src_shells = build_orca_shell_specs_from_cache(
        pyscf_mol,
        basis_name,
        shell_order_cache,
    )
    dst_shells = build_ao_shell_specs_from_mol(pyscf_mol)
    idx = build_layout_reorder_indices(
        src_shells,
        dst_shells,
        src_convention="orca",
        dst_convention=dst_convention,
    )
    signs = build_orca_json_signs_for_layout(dst_shells, dst_convention)
    return idx, signs


def orca_json_matrix_to_layout(
    M: np.ndarray,
    pyscf_mol,
    basis_name: str,
    dst_convention: str = "pyscf",
    shell_order_cache: dict[str, dict[int, list[str]]] | None = None,
) -> np.ndarray:
    """Convert an ORCA ``orca_2json`` AO matrix to PySCF/e3nn-like layout."""
    idx, signs = build_cached_orca_json_to_layout_transform(
        pyscf_mol,
        basis_name,
        dst_convention=dst_convention,
        shell_order_cache=shell_order_cache,
    )
    return signed_reorder_matrix(M, idx, signs)


def layout_matrix_to_orca_json(
    M: np.ndarray,
    pyscf_mol,
    basis_name: str,
    src_convention: str = "pyscf",
    shell_order_cache: dict[str, dict[int, list[str]]] | None = None,
) -> np.ndarray:
    """Convert a PySCF/e3nn-like AO matrix to ORCA ``orca_2json`` style."""
    idx, signs = build_cached_orca_json_to_layout_transform(
        pyscf_mol,
        basis_name,
        dst_convention=src_convention,
        shell_order_cache=shell_order_cache,
    )
    return invert_signed_reorder_matrix(M, idx, signs)


def pyscf_overlap_to_orca_json(
    pyscf_mol,
    basis_name: str,
    shell_order_cache: dict[str, dict[int, list[str]]] | None = None,
) -> np.ndarray:
    """Build an ORCA-JSON-style overlap matrix from a PySCF molecule."""
    overlap = pyscf_mol.intor("int1e_ovlp")
    return layout_matrix_to_orca_json(
        overlap,
        pyscf_mol,
        basis_name,
        src_convention="pyscf",
        shell_order_cache=shell_order_cache,
    )


def layout_matrix_to_orca_raw_density_layout(
    M: np.ndarray,
    pyscf_mol,
    basis_name: str,
    src_convention: str = "pyscf",
    dst_convention: str = "e3nn",
) -> np.ndarray:
    """Convert a PySCF/e3nn-like AO matrix to the OMol ORCA raw-density sign.

    This is deliberately different from :func:`layout_matrix_to_orca_json`.
    When an overlap or SAD matrix is generated from PySCF integrals for an
    existing OMol raw density matrix, apply this function so ``Tr(D S)`` is
    computed in the same signed contracted-AO basis as the density.  This path
    includes raw-density calibrations that are not part of the ORCA JSON API,
    such as the Be/def2-TZVPD p-shell signed swap.

    The returned matrix is in ``dst_convention`` shell/m ordering with signs
    matching the ORCA raw-density convention.
    """
    idx, signs = build_orca_raw_density_layout_transform(
        pyscf_mol,
        basis_name,
        src_convention=src_convention,
        dst_convention=dst_convention,
    )
    return signed_reorder_matrix(M, idx, signs)


def pyscf_overlap_to_orca_raw_density_layout(
    pyscf_mol,
    basis_name: str,
    dst_convention: str = "e3nn",
) -> np.ndarray:
    """Build PySCF overlap in the signed layout used by OMol ORCA raw density."""
    overlap = pyscf_mol.intor("int1e_ovlp")
    return layout_matrix_to_orca_raw_density_layout(
        overlap,
        pyscf_mol,
        basis_name,
        src_convention="pyscf",
        dst_convention=dst_convention,
    )
