"""DFT matrix analysis and visualization utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .solvers import solve_gen_eigh


# ── Checks ───────────────────────────────────────────────────────────

def check_symmetry(mol: "Molecule", tol: float = 1e-10) -> dict:
    """Check symmetry of H, S, D matrices."""
    results = {}
    for name in ("hamiltonian", "overlap", "density_matrix"):
        M = getattr(mol, name)
        if M is None:
            continue
        diff = float(np.max(np.abs(M - M.T)))
        results[name] = {"max_asymmetry": diff, "symmetric": diff < tol}
    return results


def check_trace(mol: "Molecule") -> dict:
    """Check Tr(D @ S) = n_electrons."""
    results = {"n_electrons": mol.num_electrons}
    if mol.density_matrix is not None and mol.overlap is not None:
        results["Tr(DS)"] = float(np.trace(mol.density_matrix @ mol.overlap))
    return results


def check_idempotency(mol: "Molecule", tol: float = 1e-6) -> float:
    """Check D @ S @ D ≈ D (for normalized D/2, closed-shell)."""
    if mol.density_matrix is None or mol.overlap is None:
        return float("nan")
    D_half = mol.density_matrix / 2.0
    residual = D_half @ mol.overlap @ D_half - D_half
    return float(np.max(np.abs(residual)))


def band_energy(mol: "Molecule") -> Optional[float]:
    """Tr(H @ D) — one-electron band energy in Hartree."""
    if mol.hamiltonian is None or mol.density_matrix is None:
        return None
    return float(np.trace(mol.hamiltonian @ mol.density_matrix))


def summary(mol: "Molecule") -> str:
    """Detailed multi-line summary (print-ready)."""
    lines = [
        f"{'='*60}",
        f"Molecule Summary",
        f"{'='*60}",
        f"  ID          : {mol.mol_id or '?'}",
        f"  Formula     : {mol.formula or '?'}",
        f"  Atoms       : {mol.num_atoms} atoms, Z = {mol.atomic_numbers.tolist()}",
        f"  Electrons   : {mol.num_electrons}",
    ]
    if mol.basis_info is not None:
        lines.append(f"  Basis       : {mol.basis_info.name} ({mol.nao} AOs, {mol.basis_info.convention})")
    if mol.xc is not None:
        lines.append(f"  XC          : {mol.xc}")
    if mol.hamiltonian is not None:
        H = mol.hamiltonian
        lines.append(f"  H range     : [{H.min():.6f}, {H.max():.6f}] Ha")
    if mol.overlap is not None:
        S = mol.overlap
        lines.append(f"  S range     : [{S.min():.6f}, {S.max():.6f}]")
        lines.append(f"  S cond num  : {np.linalg.cond(S):.2e}")
    if mol.density_matrix is not None and mol.overlap is not None:
        tr = check_trace(mol)
        lines.append(f"  Tr(DS)      : {tr.get('Tr(DS)', '?'):.6f} (expect {mol.num_electrons})")
    if mol.energy is not None:
        lines.append(f"  Energy      : {mol.energy:.8f} Ha")
    if mol.homo is not None and mol.lumo is not None:
        lines.append(f"  HOMO        : {mol.homo:.6f} Ha ({mol.homo * 27.211386:.4f} eV)")
        lines.append(f"  LUMO        : {mol.lumo:.6f} Ha ({mol.lumo * 27.211386:.4f} eV)")
        lines.append(f"  Gap         : {mol.homo_lumo_gap:.4f} eV")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


# ── Visualization ────────────────────────────────────────────────────

def plot_matrix(mol: "Molecule", which: str = "hamiltonian", ax=None, cmap: str = "RdBu_r", **kwargs):
    """Plot H, S, or D as heatmap."""
    import matplotlib.pyplot as plt

    M = getattr(mol, which)
    if M is None:
        raise ValueError(f"'{which}' is not available")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))

    vmax = np.max(np.abs(M))
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, **kwargs)
    ax.set_title(f"{which} ({M.shape[0]}×{M.shape[1]})")
    ax.set_xlabel("AO index")
    ax.set_ylabel("AO index")
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax


def plot_orbital_energies(mol: "Molecule", ax=None, n_show: int = 30):
    """Plot orbital energy level diagram."""
    import matplotlib.pyplot as plt

    if mol.orbital_energies is None:
        if mol.hamiltonian is None or mol.overlap is None:
            raise ValueError("orbital_energies, or (hamiltonian + overlap) required")
        energies, _ = solve_gen_eigh(mol.hamiltonian, mol.overlap)
    else:
        energies = mol.orbital_energies

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 6))

    n_occ = mol.num_electrons // 2
    energies_ev = energies[:n_show] * 27.211386

    colors = ["tab:blue" if i < n_occ else "tab:red" for i in range(min(n_show, len(energies_ev)))]
    for e, c in zip(energies_ev, colors):
        ax.hlines(e, 0.3, 0.7, colors=c, linewidths=2)

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Orbital Energies (blue=occ, red=virt)")
    return ax


def plot_sparsity(mol: "Molecule", which: str = "hamiltonian", threshold: float = 1e-8, ax=None):
    """Plot sparsity pattern."""
    import matplotlib.pyplot as plt

    M = getattr(mol, which)
    if M is None:
        raise ValueError(f"'{which}' is not available")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    mask = np.abs(M) > threshold
    ax.spy(mask, markersize=1)
    nonzero = np.sum(mask)
    total = M.size
    ax.set_title(f"{which} sparsity: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
    return ax
