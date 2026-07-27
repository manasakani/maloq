"""Matrix visualization utilities — orbital labels, distance decay, block norms.

Ported from dft-viz/components/matrix_viz.py, adapted to use Molecule objects.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from molecule import Molecule

from .structure import ELEMENT_MAP, BOHR2ANG

# def2-SVP orbital info (shell, orbital_name) per atomic number
_ORBITAL_INFO = {
    1: [
        ("s", "1s"), ("s", "2s"),
        ("p", "2p\u2093"), ("p", "2p\u1D67"), ("p", "2pz"),
    ],
    6: [
        ("s", "1s"), ("s", "2s"), ("s", "3s"),
        ("p", "2p\u2093"), ("p", "2p\u1D67"), ("p", "2pz"),
        ("p", "3p\u2093"), ("p", "3p\u1D67"), ("p", "3pz"),
        ("d", "3dxy"), ("d", "3dxz"), ("d", "3dyz"), ("d", "3dx\u00B2"), ("d", "3dz\u00B2"),
    ],
}
for _z in (7, 8, 9):
    _ORBITAL_INFO[_z] = _ORBITAL_INFO[6]

_ORB_SIZES = {1: 5, 6: 14, 7: 14, 8: 14, 9: 14}


def get_orbital_labels(atoms: np.ndarray) -> list[str]:
    """Generate flat orbital labels like 'O(0)-1s'."""
    labels = []
    for i, z in enumerate(atoms):
        sym = ELEMENT_MAP.get(int(z), f"X{int(z)}")
        info = _ORBITAL_INFO.get(int(z), [("?", f"\u03C6{j}") for j in range(14)])
        for _, orb in info:
            labels.append(f"{sym}({i})-{orb}")
    return labels


def get_atom_labels(atoms: np.ndarray) -> list[str]:
    """Generate per-orbital atom labels like 'O(0)'."""
    labels = []
    for i, z in enumerate(atoms):
        sym = ELEMENT_MAP.get(int(z), f"X{int(z)}")
        n = _ORB_SIZES.get(int(z), 14)
        labels.extend([f"{sym}({i})"] * n)
    return labels


def get_block_boundaries(atoms: np.ndarray) -> list[dict]:
    """Get atom and shell boundary positions for drawing lines on heatmaps."""
    lines = []
    cur = 0
    for z in atoms:
        info = _ORBITAL_INFO.get(int(z), [("?", f"o{j}") for j in range(14)])
        prev_shell = None
        for shell, _ in info:
            cur += 1
            if prev_shell is not None and shell != prev_shell:
                lines.append({"pos": cur - 1, "kind": "shell"})
            prev_shell = shell
        lines.append({"pos": cur, "kind": "atom"})
    return lines


def _build_orb2atom_map(atoms: np.ndarray) -> list[int]:
    """Map each orbital index to its parent atom index."""
    mapping = []
    for i, z in enumerate(atoms):
        n = _ORB_SIZES.get(int(z), 14)
        mapping.extend([i] * n)
    return mapping


def _build_orb_shell_map(atoms: np.ndarray) -> list[str]:
    """Map each orbital index to its shell type (s/p/d)."""
    shells = []
    for z in atoms:
        info = _ORBITAL_INFO.get(int(z), [("?", f"o{j}") for j in range(14)])
        for shell, _ in info:
            shells.append(shell)
    return shells


def compute_distance_decay(
    matrix: np.ndarray,
    atoms: np.ndarray,
    positions: np.ndarray,
) -> dict:
    """Compute |H[mu][nu]| vs interatomic distance.

    Args:
        matrix: (nao, nao) Hamiltonian or overlap matrix.
        atoms: (N,) atomic numbers.
        positions: (N, 3) positions in Angstrom.

    Returns:
        dict with keys: distance, abs_value, pair_type, atom_pair
    """
    n = matrix.shape[0]
    orb2atom = _build_orb2atom_map(atoms)
    orb_shells = _build_orb_shell_map(atoms)

    distances, abs_values, pair_types, atom_pairs = [], [], [], []

    for mu in range(n):
        for nu in range(mu, n):
            ai, aj = orb2atom[mu], orb2atom[nu]
            d = float(np.linalg.norm(positions[ai] - positions[aj]))
            v = abs(float(matrix[mu, nu]))

            s1, s2 = orb_shells[mu], orb_shells[nu]
            pt = f"{min(s1, s2)}-{max(s1, s2)}"
            sym_i = ELEMENT_MAP.get(int(atoms[ai]), "X")
            sym_j = ELEMENT_MAP.get(int(atoms[aj]), "X")
            ap = f"{min(sym_i, sym_j)}-{max(sym_i, sym_j)}"

            distances.append(d)
            abs_values.append(v)
            pair_types.append(pt)
            atom_pairs.append(ap)

    return {
        "distance": distances,
        "abs_value": abs_values,
        "pair_type": pair_types,
        "atom_pair": atom_pairs,
    }


def compute_block_norms(
    matrix: np.ndarray,
    atoms: np.ndarray,
) -> np.ndarray:
    """Compute Frobenius norm per atom-pair block.

    Returns:
        (n_atoms, n_atoms) array of block norms.
    """
    n_atoms = len(atoms)
    block_starts = []
    cur = 0
    for z in atoms:
        block_starts.append(cur)
        cur += _ORB_SIZES.get(int(z), 14)
    block_starts.append(cur)

    norms = np.zeros((n_atoms, n_atoms))
    for ai in range(n_atoms):
        for aj in range(n_atoms):
            block = matrix[
                block_starts[ai] : block_starts[ai + 1],
                block_starts[aj] : block_starts[aj + 1],
            ]
            norms[ai, aj] = np.linalg.norm(block, "fro")
    return norms
