"""Molecular structure utilities — XYZ, cube file generation."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from molecule import Molecule

ELEMENT_MAP = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 35: "Br", 53: "I",
}

BOHR2ANG = 0.5291772105638411
ANG2BOHR = 1.0 / BOHR2ANG


def to_xyz(mol: "Molecule") -> str:
    """Generate XYZ format string. Positions in Angstrom."""
    n = len(mol.atomic_numbers)
    lines = [str(n), f"{mol.formula or 'Molecule'}"]
    for z, pos in zip(mol.atomic_numbers, mol.positions):
        sym = ELEMENT_MAP.get(int(z), f"X{int(z)}")
        lines.append(f"{sym:2s} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}")
    return "\n".join(lines) + "\n"


def to_cube(
    mol: "Molecule",
    field_3d: np.ndarray,
    grid_meta: dict,
    comment: str = "DFT Dataset",
) -> str:
    """Generate Gaussian cube format string.

    Args:
        mol: Molecule with atomic_numbers and positions (Angstrom).
        field_3d: (nx, ny, nz) volumetric data.
        grid_meta: dict with keys boxorig, box (3x3), nx, ny, nz.
        comment: header comment.
    """
    atoms = mol.atomic_numbers
    pos_bohr = mol.positions * ANG2BOHR
    orig = np.asarray(grid_meta["boxorig"])
    box = np.asarray(grid_meta["box"])
    nx, ny, nz = grid_meta["nx"], grid_meta["ny"], grid_meta["nz"]

    lines = [comment, "Cube file from DFT Dataset"]
    lines.append(f"{len(atoms):5d} {orig[0]:12.6f} {orig[1]:12.6f} {orig[2]:12.6f}")

    # Step vectors
    for i, n_i in enumerate([nx, ny, nz]):
        step = box[i] / n_i
        lines.append(f"{n_i:5d} {step[0]:12.6f} {step[1]:12.6f} {step[2]:12.6f}")

    # Atoms
    for z, p in zip(atoms, pos_bohr):
        lines.append(f"{int(z):5d} {float(z):12.6f} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}")

    # Volumetric data
    flat = field_3d.flatten()
    for i in range(0, len(flat), 6):
        chunk = flat[i : i + 6]
        lines.append(" ".join(f"{v:13.5e}" for v in chunk))

    return "\n".join(lines) + "\n"
