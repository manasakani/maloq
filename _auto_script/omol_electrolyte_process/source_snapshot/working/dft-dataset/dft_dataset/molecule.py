"""Unified Molecule dataclass for MLIP and DFT datasets.

하나의 데이터 구조로 MLIP 수준(에너지/포스)과 DFT 수준(H/S/D/전하) 데이터를 표현.
필드는 계층으로 구분:
  1) Core: 모든 분자에 반드시 있는 필드 (atomic_numbers, positions)
  2) Properties: 에너지, 포스, 전하 등 (optional)
  3) Electronic: H, S, D 행렬, orbital 정보 (optional)
  4) Metadata: 단위, basis, xc, 출처 (optional)

단위 체계:
  energy_unit = "Ha" (Hartree) 또는 "eV"
  length_unit = "Ang" (Angstrom) 또는 "Bohr"
  to_units() 메서드로 변환 가능.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import json

import numpy as np

from .conventions import (
    KNOWN_CONVENTIONS,
    build_reorder_indices,
    reorder_matrix,
    reorder_rows,
)


# ── Unit constants ───────────────────────────────────────────────────

HA_TO_EV = 27.211386245988
EV_TO_HA = 1.0 / HA_TO_EV
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


# ── Packing utilities ────────────────────────────────────────────────

def pack_upper_triangle(M: np.ndarray) -> tuple[np.ndarray, int]:
    """Symmetric matrix → packed 1D (upper triangle, diagonal 포함).

    Returns:
        (packed, nao) where nao is the matrix dimension.
    """
    assert M.ndim == 2 and M.shape[0] == M.shape[1]
    nao = M.shape[0]
    return M[np.triu_indices(nao)], nao


def unpack_upper_triangle(packed: np.ndarray, nao: int) -> np.ndarray:
    """Packed 1D → symmetric matrix."""
    M = np.zeros((nao, nao), dtype=packed.dtype)
    iu = np.triu_indices(nao)
    M[iu] = packed
    M[iu[1], iu[0]] = packed  # mirror
    return M


# ── Basis info ───────────────────────────────────────────────────────

# 주요 basis set의 원자별 shell/orbital 정보 (spherical)
KNOWN_BASIS = {
    "def2-svp": {
        "angular_type": "spherical",
        "shells": {
            1:  ("ssp",      5),    # 2s + 1p = 2×1 + 1×3
            6:  ("sssppd",  14),    # 3s + 2p + 1d = 3 + 6 + 5
            7:  ("sssppd",  14),
            8:  ("sssppd",  14),
            9:  ("sssppd",  14),
            16: ("sssspppd", 18),   # 4s + 3p + 1d = 4 + 9 + 5
            17: ("sssspppd", 18),
            35: ("sssssppppddd", 32),  # 5s + 4p + 3d = 5 + 12 + 15
        },
    },
    "def2-tzvp": {
        "angular_type": "spherical",
        "shells": {
            1:  ("sssp",       9),
            6:  ("sssspppdd", 31),
            7:  ("sssspppdd", 31),
            8:  ("sssspppdd", 31),
            9:  ("sssspppdd", 31),
        },
    },
}


@dataclass
class BasisInfo:
    """Basis set 정보 — shell 구성, orbital 순서, m-ordering convention.

    Attributes:
        name: basis set 이름 (e.g. "def2-svp")
        angular_type: "spherical" (5d) or "cartesian" (6d)
        convention: m-ordering convention (e.g. "pyscf", "e3nn")
        atom_to_shells: {atomic_number: shell_string}
            shell 문자열은 각 shell을 한 글자로: s(1), p(3), d(5/6), f(7/10)
            e.g. "sssppd" = 3 s-shells + 2 p-shells + 1 d-shell
        atom_to_nao: {atomic_number: n_orbitals}
            각 원자의 basis function 개수
    """
    name: str                                   # e.g. "def2-svp"
    angular_type: str = "spherical"             # "spherical" or "cartesian"
    convention: str = "pyscf"                   # m-ordering: "pyscf" or "e3nn"
    atom_to_shells: dict[int, str] = field(default_factory=dict)
    atom_to_nao: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_name(cls, name: str, convention: str = "pyscf") -> "BasisInfo":
        """KNOWN_BASIS에서 lookup."""
        key = name.lower().replace("_", "-")
        if key not in KNOWN_BASIS:
            raise ValueError(
                f"Unknown basis '{name}'. Known: {list(KNOWN_BASIS.keys())}. "
                f"Use BasisInfo() constructor for custom basis."
            )
        info = KNOWN_BASIS[key]
        return cls(
            name=key,
            angular_type=info["angular_type"],
            convention=convention,
            atom_to_shells={z: v[0] for z, v in info["shells"].items()},
            atom_to_nao={z: v[1] for z, v in info["shells"].items()},
        )

    def total_nao(self, atomic_numbers: np.ndarray) -> int:
        """분자의 총 basis function 개수."""
        return sum(self.atom_to_nao[int(z)] for z in atomic_numbers)

    def orbital_slices(self, atomic_numbers: np.ndarray) -> list[slice]:
        """각 원자의 orbital index 범위 (H, S 행렬에서의 block 위치)."""
        slices = []
        offset = 0
        for z in atomic_numbers:
            n = self.atom_to_nao[int(z)]
            slices.append(slice(offset, offset + n))
            offset += n
        return slices

    @property
    def max_block_size(self) -> int:
        return max(self.atom_to_nao.values()) if self.atom_to_nao else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "angular_type": self.angular_type,
            "convention": self.convention,
            "atom_to_shells": {str(k): v for k, v in self.atom_to_shells.items()},
            "atom_to_nao": {str(k): v for k, v in self.atom_to_nao.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BasisInfo":
        return cls(
            name=d["name"],
            angular_type=d.get("angular_type", "spherical"),
            convention=d.get("convention", "pyscf"),
            atom_to_shells={int(k): v for k, v in d.get("atom_to_shells", {}).items()},
            atom_to_nao={int(k): v for k, v in d.get("atom_to_nao", {}).items()},
        )

    # Per-element cache: {(basis_name, Z): (shells_str, nao)}
    _ELEMENT_CACHE: ClassVar[dict[tuple[str, int], tuple[str, int]]] = {}

    @classmethod
    def from_pyscf_basis(
        cls,
        basis_name: str,
        atomic_numbers: np.ndarray,
        convention: str = "pyscf",
    ) -> "BasisInfo":
        """PySCF에서 basis 정보를 동적으로 추출.

        원소별 결과를 캐싱하므로, 같은 원소/basis 조합은 한 번만 계산.
        PySCF가 import 가능해야 함.
        """
        from pyscf import gto

        L_TO_SHELL = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}
        unique_z = sorted(set(int(z) for z in atomic_numbers))

        atom_to_shells = {}
        atom_to_nao = {}

        for z in unique_z:
            cache_key = (basis_name, z)
            if cache_key in cls._ELEMENT_CACHE:
                shells_str, nao = cls._ELEMENT_CACHE[cache_key]
            else:
                # 더미 원자 하나로 PySCF Mole 생성
                sym = gto.elements.ELEMENTS[z]
                mol = gto.M(atom=f"{sym} 0 0 0", basis=basis_name,
                            unit="Angstrom", verbose=0, spin=z % 2)
                # Shell 구조 추출
                shells = []
                for i in range(mol.nbas):
                    l = mol.bas_angular(i)
                    shells.append(L_TO_SHELL.get(l, f"l{l}"))
                shells_str = "".join(shells)
                nao = mol.nao
                cls._ELEMENT_CACHE[cache_key] = (shells_str, nao)

            atom_to_shells[z] = shells_str
            atom_to_nao[z] = nao

        return cls(
            name=basis_name,
            angular_type="spherical",
            convention=convention,
            atom_to_shells=atom_to_shells,
            atom_to_nao=atom_to_nao,
        )


# ── Molecule ─────────────────────────────────────────────────────────

@dataclass
class Molecule:
    """Single molecule with geometry and optional properties.

    MLIP 수준(에너지/포스만)부터 DFT 수준(H/S/D 행렬)까지 단일 구조로 표현.
    energy_unit/length_unit으로 단위를 추적하고, to_units()로 변환.
    """

    # ── Core (필수) ──────────────────────────────────────────────
    atomic_numbers: np.ndarray          # (N,)  int, 원자 번호
    positions: np.ndarray               # (N,3) float

    # ── Identifiers (optional) ───────────────────────────────────
    mol_id: Optional[str] = None        # 고유 ID (e.g. "qm9_000001")
    formula: Optional[str] = None       # 분자식 (e.g. "C2H6O")
    smiles: Optional[str] = None        # SMILES 문자열

    # ── System info ──────────────────────────────────────────────
    charge: int = 0                     # 총 전하
    spin: int = 0                       # 2S (spin multiplicity - 1)
    unrestricted: bool = False          # UKS 여부 (True면 행렬이 (2, nao, nao))
    num_ecp_electrons: int = 0          # ECP로 제거된 core electron 수

    # ── Units ────────────────────────────────────────────────────
    energy_unit: str = "Ha"             # "Ha" or "eV"
    length_unit: str = "Ang"            # "Ang" or "Bohr"

    # ── PBC ──────────────────────────────────────────────────────
    cell: Optional[np.ndarray] = None   # (3,3) lattice vectors
    pbc: Optional[np.ndarray] = None    # (3,) bool

    # ── Properties ───────────────────────────────────────────────
    energy: Optional[float] = None      # total energy
    forces: Optional[np.ndarray] = None # (N,3)
    stress: Optional[np.ndarray] = None # (6,) Voigt order
    dipole: Optional[np.ndarray] = None # (3,), Debye

    # Orbital energies
    homo: Optional[float] = None        # HOMO energy
    lumo: Optional[float] = None        # LUMO energy

    # Thermodynamic (QM9-style)
    zpve: Optional[float] = None
    U0: Optional[float] = None
    U: Optional[float] = None
    H: Optional[float] = None
    G: Optional[float] = None
    Cv: Optional[float] = None          # cal/(mol·K), 단위 변환 대상 아님

    # Electronic
    polarizability: Optional[float] = None  # isotropic, Bohr³

    # ── Partial charges ──────────────────────────────────────────
    mulliken_charges: Optional[np.ndarray] = None   # (N,)
    lowdin_charges: Optional[np.ndarray] = None     # (N,)
    nbo_charges: Optional[np.ndarray] = None        # (N,)
    mulliken_spins: Optional[np.ndarray] = None     # (N,)
    lowdin_spins: Optional[np.ndarray] = None       # (N,)
    nbo_spins: Optional[np.ndarray] = None          # (N,)

    # ── Electronic structure (행렬) ──────────────────────────────
    hamiltonian: Optional[np.ndarray] = None    # (nao, nao), Fock matrix
    overlap: Optional[np.ndarray] = None        # (nao, nao), dimensionless
    density_matrix: Optional[np.ndarray] = None # (nao, nao), dimensionless
    initial_hamiltonian: Optional[np.ndarray] = None  # (nao, nao), H_core = T + V_ne
    initial_density_matrix: Optional[np.ndarray] = None  # (nao, nao), P from H_core
    orbital_energies: Optional[np.ndarray] = None   # (nao,)
    orbital_coeffs: Optional[np.ndarray] = None     # (nao, nao)

    # ── Metadata ─────────────────────────────────────────────────
    basis_info: Optional[BasisInfo] = None  # basis set + convention
    xc: Optional[str] = None                # e.g. "b3lyp"
    source: Optional[str] = None            # 출처: "omol25", "nabladft", "qh9", "pyscf"

    # ── Computed properties ──────────────────────────────────────

    @property
    def basis(self) -> Optional[str]:
        """Basis set name (shortcut)."""
        return self.basis_info.name if self.basis_info else None

    @property
    def convention(self) -> Optional[str]:
        """Current m-ordering convention (shortcut)."""
        return self.basis_info.convention if self.basis_info else None

    @property
    def nao(self) -> Optional[int]:
        """Total number of atomic orbitals."""
        if self.hamiltonian is not None:
            return self.hamiltonian.shape[-1]  # (nao,nao) or (2,nao,nao)
        if self.basis_info is not None:
            return self.basis_info.total_nao(self.atomic_numbers)
        return None

    @property
    def num_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def elements(self) -> list[int]:
        """Sorted unique atomic numbers."""
        return sorted(set(self.atomic_numbers.tolist()))

    @property
    def homo_lumo_gap(self) -> Optional[float]:
        """HOMO-LUMO gap in eV."""
        if self.homo is not None and self.lumo is not None:
            gap = self.lumo - self.homo
            if self.energy_unit == "Ha":
                gap *= HA_TO_EV
            return gap
        return None

    @property
    def is_periodic(self) -> bool:
        """True if any PBC dimension is True."""
        return self.pbc is not None and np.any(self.pbc)

    @property
    def num_electrons(self) -> int:
        return int(self.atomic_numbers.sum()) - self.charge - self.num_ecp_electrons

    # ── Unit conversion ────────────────────────────────────────

    def to_units(self, energy_unit: str = "Ha", length_unit: str = "Ang") -> "Molecule":
        """Return a new Molecule with converted units.

        Args:
            energy_unit: "Ha" or "eV"
            length_unit: "Ang" or "Bohr"

        Returns:
            New Molecule. Returns self if already in target units.
        """
        if energy_unit == self.energy_unit and length_unit == self.length_unit:
            return self

        # Energy scale factor
        e_scale = 1.0
        if self.energy_unit == "Ha" and energy_unit == "eV":
            e_scale = HA_TO_EV
        elif self.energy_unit == "eV" and energy_unit == "Ha":
            e_scale = EV_TO_HA

        # Length scale factor
        l_scale = 1.0
        if self.length_unit == "Ang" and length_unit == "Bohr":
            l_scale = ANG_TO_BOHR
        elif self.length_unit == "Bohr" and length_unit == "Ang":
            l_scale = BOHR_TO_ANG

        # Force = energy / length
        fl_scale = e_scale / l_scale if l_scale != 0 else 1.0

        def _s(val, scale):
            return val * scale if val is not None else None

        def _a(arr, scale):
            return arr * scale if arr is not None else None

        import copy
        mol = copy.copy(self)
        mol.energy_unit = energy_unit
        mol.length_unit = length_unit

        # Positions, cell
        mol.positions = self.positions * l_scale
        mol.cell = _a(self.cell, l_scale)

        # Energy-like scalars
        mol.energy = _s(self.energy, e_scale)
        mol.homo = _s(self.homo, e_scale)
        mol.lumo = _s(self.lumo, e_scale)
        mol.zpve = _s(self.zpve, e_scale)
        mol.U0 = _s(self.U0, e_scale)
        mol.U = _s(self.U, e_scale)
        mol.H = _s(self.H, e_scale)
        mol.G = _s(self.G, e_scale)
        # Cv is cal/(mol·K), not converted

        # Force-like arrays (energy/length)
        mol.forces = _a(self.forces, fl_scale)
        mol.stress = _a(self.stress, e_scale)  # stress ~ energy/volume, scale by energy

        # Electronic matrices (energy unit)
        mol.hamiltonian = _a(self.hamiltonian, e_scale)
        mol.orbital_energies = _a(self.orbital_energies, e_scale)
        # overlap, density_matrix: dimensionless, no conversion

        return mol

    # ── Constructors ────────────────────────────────────────────

    @classmethod
    def from_pyscf(
        cls,
        mf,
        mol_id: Optional[str] = None,
        convention: str = "pyscf",
    ) -> "Molecule":
        """PySCF SCF 객체에서 생성 (RKS/UKS 자동 감지).

        Args:
            mf: PySCF SCF object (e.g. dft.RKS or dft.UKS after .kernel())
            mol_id: optional molecule ID
            convention: m-ordering convention (기본 "pyscf")

        UKS의 경우:
            - hamiltonian: (2, nao, nao) [alpha, beta] Fock matrices
            - density_matrix: (2, nao, nao) [alpha, beta]
            - orbital_energies: (2, nao)
            - orbital_coeffs: (2, nao, nao)
            - overlap: (nao, nao) — alpha/beta 동일

        Example:
            from pyscf import gto, dft
            mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                        basis='def2-svp')
            mf = dft.RKS(mol, xc='b3lyp').run()
            molecule = Molecule.from_pyscf(mf)

            # UKS
            mf_u = dft.UKS(mol, xc='b3lyp').run()
            molecule_u = Molecule.from_pyscf(mf_u)  # unrestricted=True 자동
        """
        pyscf_mol = mf.mol
        Z = np.array(pyscf_mol.atom_charges(), dtype=np.int32)
        pos = pyscf_mol.atom_coords(unit="ANG").astype(np.float64)

        basis_name = pyscf_mol.basis if isinstance(pyscf_mol.basis, str) else "custom"
        try:
            basis_info = BasisInfo.from_name(basis_name, convention=convention)
        except ValueError:
            basis_info = BasisInfo(name=basis_name, convention=convention)

        # UKS 감지: mo_energy가 tuple/list of 2 arrays이면 unrestricted
        is_uks = (
            isinstance(mf.mo_energy, (tuple, list))
            or (isinstance(mf.mo_energy, np.ndarray) and mf.mo_energy.ndim == 2)
        )

        if is_uks:
            # UKS: alpha/beta 분리
            mo_e = np.stack(mf.mo_energy)       # (2, nao)
            mo_c = np.stack(mf.mo_coeff)        # (2, nao, nao)
            fock = np.stack(mf.get_fock()) if mf.converged else np.stack([mf.get_hcore()] * 2)
            dm = np.stack(mf.make_rdm1())       # (2, nao, nao)

            # HOMO/LUMO from alpha channel (mo_occ로 정확히 계산)
            mo_occ = np.stack(mf.mo_occ)        # (2, nao)
            alpha_occ = np.where(mo_occ[0] > 0)[0]
            beta_occ = np.where(mo_occ[1] > 0)[0]
            homo_a = float(mo_e[0, alpha_occ[-1]]) if len(alpha_occ) > 0 else None
            homo_b = float(mo_e[1, beta_occ[-1]]) if len(beta_occ) > 0 else None
            # HOMO = max of alpha/beta HOMO
            homo_e = max(h for h in [homo_a, homo_b] if h is not None) if (homo_a is not None or homo_b is not None) else None
            # LUMO from alpha
            alpha_virt = np.where(mo_occ[0] == 0)[0]
            lumo_e = float(mo_e[0, alpha_virt[0]]) if len(alpha_virt) > 0 else None
        else:
            # RKS
            mo_e = mf.mo_energy
            mo_c = mf.mo_coeff
            fock = mf.get_fock() if mf.converged else mf.get_hcore()
            dm = mf.make_rdm1()

            n_occ = pyscf_mol.nelectron // 2
            homo_e = float(mf.mo_energy[n_occ - 1]) if mf.mo_energy is not None else None
            lumo_e = float(mf.mo_energy[n_occ]) if mf.mo_energy is not None else None

        return cls(
            atomic_numbers=Z,
            positions=pos,
            mol_id=mol_id,
            charge=pyscf_mol.charge,
            spin=pyscf_mol.spin,
            unrestricted=is_uks,
            energy=float(mf.e_tot) if mf.converged else None,
            hamiltonian=fock,
            overlap=pyscf_mol.intor_symmetric("int1e_ovlp"),
            density_matrix=dm,
            orbital_energies=mo_e,
            orbital_coeffs=mo_c,
            homo=homo_e,
            lumo=lumo_e,
            basis_info=basis_info,
            xc=mf.xc if hasattr(mf, "xc") else None,
            source="pyscf",
        )

    # ── Computation ──────────────────────────────────────────────

    def solve_eigenproblem(self, method: str = "eigh") -> tuple[np.ndarray, np.ndarray]:
        """HC = SCε를 풀고 orbital_energies, orbital_coeffs, homo, lumo를 설정.

        UKS (unrestricted=True)인 경우 alpha/beta 채널 각각에 대해 풀고,
        결과는 (2, nao) / (2, nao, nao) 형태로 저장.

        Returns:
            (energies, coefficients)
        """
        from solvers import solve_gen_eigh

        if self.hamiltonian is None or self.overlap is None:
            raise ValueError("hamiltonian과 overlap이 모두 필요합니다")

        if self.unrestricted:
            # alpha/beta 각각 풀기
            e_a, c_a = solve_gen_eigh(self.hamiltonian[0], self.overlap, method=method)
            e_b, c_b = solve_gen_eigh(self.hamiltonian[1], self.overlap, method=method)
            energies = np.stack([e_a, e_b])     # (2, nao)
            coeffs = np.stack([c_a, c_b])       # (2, nao, nao)

            # HOMO from alpha (assuming aufbau)
            n_alpha = (self.num_electrons + self.spin) // 2
            n_beta = (self.num_electrons - self.spin) // 2
            homo_a = float(e_a[n_alpha - 1]) if n_alpha > 0 else None
            homo_b = float(e_b[n_beta - 1]) if n_beta > 0 else None
            self.homo = max(h for h in [homo_a, homo_b] if h is not None) if (homo_a or homo_b) else None
            self.lumo = float(e_a[n_alpha]) if n_alpha < len(e_a) else None
        else:
            energies, coeffs = solve_gen_eigh(self.hamiltonian, self.overlap, method=method)
            n_occ = self.num_electrons // 2
            if n_occ > 0 and len(energies) > n_occ:
                self.homo = float(energies[n_occ - 1])
                self.lumo = float(energies[n_occ])

        self.orbital_energies = energies
        self.orbital_coeffs = coeffs
        return energies, coeffs

    def compute_density_matrix(self, occupation: float = 2.0) -> np.ndarray:
        """MO coefficients에서 density matrix 계산.

        UKS인 경우 occupation=1.0으로 alpha/beta 각각 계산.
        orbital_coeffs가 없으면 자동으로 solve_eigenproblem() 호출.
        """
        from solvers import build_density_matrix

        if self.orbital_coeffs is None:
            self.solve_eigenproblem()

        if self.unrestricted:
            n_alpha = (self.num_electrons + self.spin) // 2
            n_beta = (self.num_electrons - self.spin) // 2
            dm_a = build_density_matrix(self.orbital_coeffs[0], n_alpha, occupation=1.0)
            dm_b = build_density_matrix(self.orbital_coeffs[1], n_beta, occupation=1.0)
            self.density_matrix = np.stack([dm_a, dm_b])
        else:
            self.density_matrix = build_density_matrix(
                self.orbital_coeffs, self.num_electrons, occupation
            )
        return self.density_matrix

    # ── Convention conversion ───────────────────────────────────

    def to_convention(self, target: str) -> "Molecule":
        """행렬의 m-ordering convention을 변환한 새 Molecule 반환.

        H, S, D 행렬과 orbital_coeffs를 target convention으로 reorder.
        orbital_energies, energy 등 invariant 값은 그대로 유지.

        Args:
            target: 목표 convention ("pyscf" or "e3nn")

        Returns:
            새 Molecule (원본은 변경하지 않음). 이미 같은 convention이면 self 반환.
        """
        if self.basis_info is None:
            raise ValueError("basis_info가 없으면 convention 변환 불가")
        if target not in KNOWN_CONVENTIONS:
            raise ValueError(f"Unknown convention '{target}'. Known: {KNOWN_CONVENTIONS}")

        src = self.basis_info.convention
        if src == target:
            return self

        idx = build_reorder_indices(
            self.atomic_numbers,
            self.basis_info.atom_to_shells,
            src, target,
            self.basis_info.angular_type,
        )

        def _reorder_mat(mat):
            if mat is None:
                return None
            if mat.ndim == 3:  # (2, nao, nao) unrestricted
                return np.stack([reorder_matrix(mat[s], idx) for s in range(2)])
            return reorder_matrix(mat, idx)

        def _reorder_coeffs(coeffs):
            if coeffs is None:
                return None
            if coeffs.ndim == 3:  # (2, nao, nao) unrestricted
                return np.stack([reorder_rows(coeffs[s], idx) for s in range(2)])
            return reorder_rows(coeffs, idx)

        new_H = _reorder_mat(self.hamiltonian)
        new_S = reorder_matrix(self.overlap, idx) if self.overlap is not None else None
        new_D = _reorder_mat(self.density_matrix)
        new_H_init = _reorder_mat(self.initial_hamiltonian)
        new_D_init = _reorder_mat(self.initial_density_matrix)
        new_C = _reorder_coeffs(self.orbital_coeffs)

        new_basis = BasisInfo(
            name=self.basis_info.name,
            angular_type=self.basis_info.angular_type,
            convention=target,
            atom_to_shells=self.basis_info.atom_to_shells,
            atom_to_nao=self.basis_info.atom_to_nao,
        )

        return Molecule(
            atomic_numbers=self.atomic_numbers,
            positions=self.positions,
            mol_id=self.mol_id,
            formula=self.formula,
            smiles=self.smiles,
            charge=self.charge,
            spin=self.spin,
            unrestricted=self.unrestricted,
            energy_unit=self.energy_unit,
            length_unit=self.length_unit,
            cell=self.cell,
            pbc=self.pbc,
            energy=self.energy,
            forces=self.forces,
            stress=self.stress,
            dipole=self.dipole,
            homo=self.homo,
            lumo=self.lumo,
            zpve=self.zpve,
            U0=self.U0,
            U=self.U,
            H=self.H,
            G=self.G,
            Cv=self.Cv,
            polarizability=self.polarizability,
            mulliken_charges=self.mulliken_charges,
            lowdin_charges=self.lowdin_charges,
            nbo_charges=self.nbo_charges,
            mulliken_spins=self.mulliken_spins,
            lowdin_spins=self.lowdin_spins,
            nbo_spins=self.nbo_spins,
            hamiltonian=new_H,
            overlap=new_S,
            density_matrix=new_D,
            initial_hamiltonian=new_H_init,
            initial_density_matrix=new_D_init,
            orbital_energies=self.orbital_energies,
            orbital_coeffs=new_C,
            basis_info=new_basis,
            xc=self.xc,
            source=self.source,
        )

    # ── I/O ──────────────────────────────────────────────────────

    _MATRIX_FIELDS = (
        "hamiltonian", "overlap", "density_matrix",
        "initial_hamiltonian", "initial_density_matrix",
    )

    def save_npz(
        self,
        path: str,
        packed: bool = False,
        convention: Optional[str] = None,
    ) -> None:
        """Save to .npz file.

        Args:
            path: 저장 경로
            packed: True면 대칭 행렬을 upper triangle로 압축
            convention: 저장 시 변환할 m-ordering convention.
                None이면 현재 convention 그대로 저장.
                e.g. convention="e3nn" → pyscf 데이터를 e3nn 순서로 변환 후 저장
        """
        mol = self if convention is None else self.to_convention(convention)

        data: dict[str, np.ndarray] = {}

        # Core
        data["atomic_numbers"] = mol.atomic_numbers
        data["positions"] = mol.positions

        # Storage format flag
        data["_packed"] = np.array(packed)

        # Scalars → 0-d array
        scalar_fields = [
            "mol_id", "formula", "smiles", "charge", "spin", "unrestricted",
            "energy_unit", "length_unit",
            "energy", "homo", "lumo", "zpve", "U0", "U", "H", "G", "Cv",
            "polarizability", "xc", "source",
        ]
        for name in scalar_fields:
            val = getattr(mol, name)
            if val is not None:
                data[name] = np.array(val)

        # BasisInfo → JSON string
        if mol.basis_info is not None:
            data["_basis_info_json"] = np.array(json.dumps(mol.basis_info.to_dict()))

        # Symmetric matrices — optionally packed
        for name in self._MATRIX_FIELDS:
            mat = getattr(mol, name)
            if mat is None:
                continue
            if packed:
                if mat.ndim == 3:  # (2, nao, nao) unrestricted
                    tri_a, nao = pack_upper_triangle(mat[0])
                    tri_b, _ = pack_upper_triangle(mat[1])
                    data[f"{name}_packed_a"] = tri_a
                    data[f"{name}_packed_b"] = tri_b
                    data[f"{name}_nao"] = np.array(nao)
                    data[f"{name}_unrestricted"] = np.array(True)
                else:
                    tri, nao = pack_upper_triangle(mat)
                    data[f"{name}_packed"] = tri
                    data[f"{name}_nao"] = np.array(nao)
            else:
                data[name] = mat

        # Non-symmetric arrays
        _npz_array_fields = (
            "forces", "stress", "dipole", "orbital_energies", "orbital_coeffs",
            "cell", "pbc",
            "mulliken_charges", "lowdin_charges", "nbo_charges",
            "mulliken_spins", "lowdin_spins", "nbo_spins",
        )
        for name in _npz_array_fields:
            val = getattr(mol, name)
            if val is not None:
                data[name] = val

        np.savez_compressed(path, **data)

    @classmethod
    def load_npz(cls, path: str) -> "Molecule":
        """Load from .npz file. Packed/unpacked, convention 자동 감지."""
        raw = np.load(path, allow_pickle=True)

        def _scalar(name, default=None):
            if name in raw:
                val = raw[name].item()
                return val if val is not None else default
            return default

        def _array(name):
            return raw[name] if name in raw else None

        is_packed = _scalar("_packed", False)

        # Symmetric matrices
        matrices = {}
        for name in cls._MATRIX_FIELDS:
            if is_packed and f"{name}_packed_a" in raw:
                # Unrestricted: alpha + beta packed separately
                nao = int(raw[f"{name}_nao"].item())
                mat_a = unpack_upper_triangle(raw[f"{name}_packed_a"], nao)
                mat_b = unpack_upper_triangle(raw[f"{name}_packed_b"], nao)
                matrices[name] = np.stack([mat_a, mat_b])
            elif is_packed and f"{name}_packed" in raw:
                tri = raw[f"{name}_packed"]
                nao = int(raw[f"{name}_nao"].item())
                matrices[name] = unpack_upper_triangle(tri, nao)
            else:
                matrices[name] = _array(name)

        # BasisInfo
        basis_info = None
        if "_basis_info_json" in raw:
            basis_info = BasisInfo.from_dict(json.loads(str(raw["_basis_info_json"])))

        return cls(
            atomic_numbers=raw["atomic_numbers"],
            positions=raw["positions"],
            mol_id=_scalar("mol_id"),
            formula=_scalar("formula"),
            smiles=_scalar("smiles"),
            charge=_scalar("charge", 0),
            spin=_scalar("spin", 0),
            unrestricted=bool(_scalar("unrestricted", False)),
            energy_unit=_scalar("energy_unit", "Ha"),
            length_unit=_scalar("length_unit", "Ang"),
            cell=_array("cell"),
            pbc=_array("pbc"),
            energy=_scalar("energy"),
            forces=_array("forces"),
            stress=_array("stress"),
            dipole=_array("dipole"),
            homo=_scalar("homo"),
            lumo=_scalar("lumo"),
            zpve=_scalar("zpve"),
            U0=_scalar("U0"),
            U=_scalar("U"),
            H=_scalar("H"),
            G=_scalar("G"),
            Cv=_scalar("Cv"),
            polarizability=_scalar("polarizability"),
            mulliken_charges=_array("mulliken_charges"),
            lowdin_charges=_array("lowdin_charges"),
            nbo_charges=_array("nbo_charges"),
            mulliken_spins=_array("mulliken_spins"),
            lowdin_spins=_array("lowdin_spins"),
            nbo_spins=_array("nbo_spins"),
            hamiltonian=matrices["hamiltonian"],
            overlap=matrices["overlap"],
            density_matrix=matrices["density_matrix"],
            initial_hamiltonian=matrices["initial_hamiltonian"],
            initial_density_matrix=matrices["initial_density_matrix"],
            orbital_energies=_array("orbital_energies"),
            orbital_coeffs=_array("orbital_coeffs"),
            basis_info=basis_info,
            xc=_scalar("xc"),
            source=_scalar("source"),
        )

    # ── Dict serialization (LMDB 호환) ──────────────────────────

    _SCALAR_FIELDS = (
        "mol_id", "formula", "smiles", "charge", "spin", "unrestricted",
        "num_ecp_electrons",
        "energy_unit", "length_unit",
        "energy", "homo", "lumo", "zpve", "U0", "U", "H", "G", "Cv",
        "polarizability", "xc", "source",
    )
    _ARRAY_FIELDS = (
        "forces", "stress", "dipole", "orbital_energies", "orbital_coeffs",
        "cell", "pbc",
        "mulliken_charges", "lowdin_charges", "nbo_charges",
        "mulliken_spins", "lowdin_spins", "nbo_spins",
    )

    def to_dict(self, packed: bool = True) -> dict:
        """Pickle-friendly dict로 변환 (LMDB 저장용).

        Args:
            packed: True면 대칭 행렬을 upper triangle로 압축

        Returns:
            dict — pickle.dumps()로 직렬화 가능한 dict.
            행렬은 bytes로 변환, 스칼라/메타데이터 그대로 보존.
        """
        d: dict = {}

        # Core arrays → bytes
        d["atomic_numbers"] = self.atomic_numbers.astype(np.int32).tobytes()
        d["positions"] = self.positions.astype(np.float64).tobytes()
        d["num_atoms"] = self.num_atoms

        # Scalars
        for name in self._SCALAR_FIELDS:
            val = getattr(self, name)
            if val is not None:
                d[name] = val

        # Symmetric matrices
        for name in self._MATRIX_FIELDS:
            mat = getattr(self, name)
            if mat is None:
                continue
            if packed:
                if mat.ndim == 3:  # (2, nao, nao) unrestricted
                    tri_a, nao = pack_upper_triangle(mat[0])
                    tri_b, _ = pack_upper_triangle(mat[1])
                    d[f"{name}_packed_a"] = tri_a.tobytes()
                    d[f"{name}_packed_b"] = tri_b.tobytes()
                    d[f"{name}_nao"] = nao
                    d[f"{name}_dtype"] = str(tri_a.dtype)
                    d[f"{name}_unrestricted"] = True
                else:
                    tri, nao = pack_upper_triangle(mat)
                    d[f"{name}_packed"] = tri.tobytes()
                    d[f"{name}_nao"] = nao
                    d[f"{name}_dtype"] = str(tri.dtype)
            else:
                d[name] = mat.tobytes()
                d[f"{name}_shape"] = mat.shape
                d[f"{name}_dtype"] = str(mat.dtype)
        d["_packed"] = packed

        # Non-symmetric arrays → bytes
        for name in self._ARRAY_FIELDS:
            val = getattr(self, name)
            if val is not None:
                d[name] = val.tobytes()
                d[f"{name}_shape"] = val.shape
                d[f"{name}_dtype"] = str(val.dtype)

        # BasisInfo
        if self.basis_info is not None:
            d["_basis_info"] = self.basis_info.to_dict()

        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Molecule":
        """to_dict()으로 생성된 dict에서 복원."""
        num_atoms = d["num_atoms"]
        atomic_numbers = np.frombuffer(d["atomic_numbers"], dtype=np.int32).copy()
        positions = np.frombuffer(d["positions"], dtype=np.float64).reshape(num_atoms, 3).copy()

        is_packed = d.get("_packed", False)

        # Symmetric matrices
        matrices = {}
        for name in cls._MATRIX_FIELDS:
            if is_packed and f"{name}_packed_a" in d:
                # Unrestricted: alpha + beta packed separately
                nao = d[f"{name}_nao"]
                dtype = np.dtype(d.get(f"{name}_dtype", "float32"))
                tri_a = np.frombuffer(d[f"{name}_packed_a"], dtype=dtype).copy()
                tri_b = np.frombuffer(d[f"{name}_packed_b"], dtype=dtype).copy()
                mat_a = unpack_upper_triangle(tri_a, nao)
                mat_b = unpack_upper_triangle(tri_b, nao)
                matrices[name] = np.stack([mat_a, mat_b])
            elif is_packed and f"{name}_packed" in d:
                nao = d[f"{name}_nao"]
                dtype = np.dtype(d.get(f"{name}_dtype", "float32"))
                tri = np.frombuffer(d[f"{name}_packed"], dtype=dtype).copy()
                matrices[name] = unpack_upper_triangle(tri, nao)
            elif name in d:
                dtype = np.dtype(d.get(f"{name}_dtype", "float64"))
                shape = d[f"{name}_shape"]
                matrices[name] = np.frombuffer(d[name], dtype=dtype).reshape(shape).copy()
            else:
                matrices[name] = None

        # Non-symmetric arrays
        def _load_array(name):
            if name not in d:
                return None
            dtype = np.dtype(d.get(f"{name}_dtype", "float64"))
            shape = d[f"{name}_shape"]
            return np.frombuffer(d[name], dtype=dtype).reshape(shape).copy()

        # Scalars
        def _scalar(name, default=None):
            return d.get(name, default)

        # BasisInfo
        basis_info = None
        if "_basis_info" in d:
            basis_info = BasisInfo.from_dict(d["_basis_info"])

        return cls(
            atomic_numbers=atomic_numbers,
            positions=positions,
            mol_id=_scalar("mol_id"),
            formula=_scalar("formula"),
            smiles=_scalar("smiles"),
            charge=_scalar("charge", 0),
            spin=_scalar("spin", 0),
            unrestricted=bool(_scalar("unrestricted", False)),
            num_ecp_electrons=int(_scalar("num_ecp_electrons", 0)),
            energy_unit=_scalar("energy_unit", "Ha"),
            length_unit=_scalar("length_unit", "Ang"),
            cell=_load_array("cell"),
            pbc=_load_array("pbc"),
            energy=_scalar("energy"),
            forces=_load_array("forces"),
            stress=_load_array("stress"),
            dipole=_load_array("dipole"),
            homo=_scalar("homo"),
            lumo=_scalar("lumo"),
            zpve=_scalar("zpve"),
            U0=_scalar("U0"),
            U=_scalar("U"),
            H=_scalar("H"),
            G=_scalar("G"),
            Cv=_scalar("Cv"),
            polarizability=_scalar("polarizability"),
            mulliken_charges=_load_array("mulliken_charges"),
            lowdin_charges=_load_array("lowdin_charges"),
            nbo_charges=_load_array("nbo_charges"),
            mulliken_spins=_load_array("mulliken_spins"),
            lowdin_spins=_load_array("lowdin_spins"),
            nbo_spins=_load_array("nbo_spins"),
            hamiltonian=matrices["hamiltonian"],
            overlap=matrices["overlap"],
            density_matrix=matrices["density_matrix"],
            initial_hamiltonian=matrices["initial_hamiltonian"],
            initial_density_matrix=matrices["initial_density_matrix"],
            orbital_energies=_load_array("orbital_energies"),
            orbital_coeffs=_load_array("orbital_coeffs"),
            basis_info=basis_info,
            xc=_scalar("xc"),
            source=_scalar("source"),
        )

    # ── Dataset converters ─────────────────────────────────────

    @classmethod
    def from_omol25_row(cls, row) -> "Molecule":
        """OMol25 parquet row (pd.Series 또는 dict)에서 생성.

        Units: eV / Angstrom (OMol25 parquet 원본 단위).
        property_metadata JSON에서 전하, HOMO/LUMO, partial charges, xc 등 추출.

        OMol25 property_metadata 필드:
            charge, spin, homo_energy, homo_lumo_gap, n_basis,
            mulliken/lowdin/nbo_charges, mulliken/lowdin/nbo_spins,
            basis_set, composition, source, nl_energy, num_electrons
        """
        positions = np.stack(row["positions"]).astype(np.float64)
        atomic_numbers = np.asarray(row["atomic_numbers"], dtype=np.int32)
        forces = np.stack(row["atomic_forces"]).astype(np.float64)

        # Parse property_metadata
        meta_raw = row.get("property_metadata")
        meta = {}
        if meta_raw is not None:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw

        charge = int(meta.get("charge", 0))
        multiplicity = int(meta.get("spin", row.get("multiplicity", 1)))
        spin = multiplicity - 1
        unrestricted = bool(meta.get("unrestricted", False))
        num_ecp_electrons = int(meta.get("num_ecp_electrons", 0))

        # HOMO/LUMO (stored in Ha in metadata, convert to eV)
        homo, lumo = None, None
        if "homo_energy" in meta and meta["homo_energy"]:
            homo = float(meta["homo_energy"][0]) * HA_TO_EV
        if "homo_lumo_gap" in meta and meta["homo_lumo_gap"] and homo is not None:
            gap_ev = float(meta["homo_lumo_gap"][0]) * HA_TO_EV
            lumo = homo + gap_ev

        def _charge_array(key):
            v = meta.get(key)
            if v is not None:
                return np.asarray(v, dtype=np.float64)
            return None

        # Cell/PBC
        cell = np.stack(row["cell"]).astype(np.float64) if "cell" in row else None
        pbc = np.asarray(row["pbc"], dtype=bool) if "pbc" in row else None
        if cell is not None and np.allclose(cell, 0):
            cell = None

        # mol_id from configuration_id or property_id
        mol_id = row.get("configuration_id") or row.get("property_id")

        # XC functional from method column
        xc = row.get("method")  # e.g. "ωB97M-V"

        return cls(
            atomic_numbers=atomic_numbers,
            positions=positions,
            mol_id=mol_id,
            formula=row.get("chemical_formula_hill"),
            charge=charge,
            spin=spin,
            unrestricted=unrestricted,
            num_ecp_electrons=num_ecp_electrons,
            energy_unit="eV",
            length_unit="Ang",
            cell=cell,
            pbc=pbc,
            energy=float(row["energy"]),
            forces=forces,
            homo=homo,
            lumo=lumo,
            mulliken_charges=_charge_array("mulliken_charges"),
            lowdin_charges=_charge_array("lowdin_charges"),
            nbo_charges=_charge_array("nbo_charges"),
            mulliken_spins=_charge_array("mulliken_spins"),
            lowdin_spins=_charge_array("lowdin_spins"),
            nbo_spins=_charge_array("nbo_spins"),
            xc=xc,
            source="omol25",
        )

    @classmethod
    def from_nabladft_row(
        cls, Z, R, E, F, H_mat=None, S_mat=None,
        basis: str = "def2-svp", xc: str = "wb97x-d",
    ) -> "Molecule":
        """nablaDFT SQLite row에서 생성.

        Args:
            Z: atomic numbers blob → (N,) int32
            R: positions blob → (N,3) float32 in **Bohr**
            E: total energy (Hartree)
            F: forces blob → (N,3) float32 in Ha/Bohr
            H_mat: Hamiltonian matrix (nao,nao) or None
            S_mat: Overlap matrix (nao,nao) or None
            basis: basis set name
            xc: XC functional
        """
        atomic_numbers = np.frombuffer(Z, dtype=np.int32).copy()
        n_atoms = len(atomic_numbers)
        positions_bohr = np.frombuffer(R, dtype=np.float32).reshape(n_atoms, 3).copy()
        positions = positions_bohr.astype(np.float64) * BOHR_TO_ANG

        forces = None
        if F is not None:
            forces = np.frombuffer(F, dtype=np.float32).reshape(n_atoms, 3).copy().astype(np.float64)

        hamiltonian = None
        overlap = None
        basis_info = None
        if H_mat is not None or S_mat is not None:
            try:
                basis_info = BasisInfo.from_name(basis, convention="pyscf")
            except ValueError:
                basis_info = BasisInfo(name=basis, convention="pyscf")
            if basis_info:
                nao = basis_info.total_nao(atomic_numbers)
                if H_mat is not None:
                    hamiltonian = np.frombuffer(H_mat, dtype=np.float32).reshape(nao, nao).copy().astype(np.float64)
                if S_mat is not None:
                    overlap = np.frombuffer(S_mat, dtype=np.float32).reshape(nao, nao).copy().astype(np.float64)

        return cls(
            atomic_numbers=atomic_numbers,
            positions=positions,
            charge=0,
            spin=0,
            energy_unit="Ha",
            length_unit="Ang",
            energy=float(E),
            forces=forces,
            hamiltonian=hamiltonian,
            overlap=overlap,
            basis_info=basis_info,
            xc=xc,
            source="nabladft",
        )

    # ── PyG conversion ────────────────────────────────────────

    def to_pyg_data(self, cutoff: float = 5.0, max_neighbors: int = 32):
        """PyG Data 객체로 변환 (MLIP 학습/추론용).

        내부적으로 eV/Ang 단위로 변환 후 PyG Data 생성.
        torch, torch_geometric이 필요 (lazy import).

        Returns:
            torch_geometric.data.Data with pos, z, edge_index, energy, forces.
        """
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import radius_graph

        mol = self.to_units("eV", "Ang")

        pos = torch.tensor(mol.positions, dtype=torch.float32)
        z = torch.tensor(mol.atomic_numbers, dtype=torch.long)
        edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=max_neighbors)

        data = Data(pos=pos, z=z, edge_index=edge_index, num_nodes=len(z))

        if mol.energy is not None:
            data.energy = torch.tensor(mol.energy, dtype=torch.float64)
        if mol.forces is not None:
            data.forces = torch.tensor(mol.forces, dtype=torch.float32)
        if mol.cell is not None:
            data.cell = torch.tensor(mol.cell, dtype=torch.float32).unsqueeze(0)
        if mol.stress is not None:
            data.stress = torch.tensor(mol.stress, dtype=torch.float32)

        return data

    @classmethod
    def from_pyg_data(cls, data, energy_unit: str = "eV", length_unit: str = "Ang") -> "Molecule":
        """PyG Data → Molecule."""
        return cls(
            atomic_numbers=data.z.numpy(),
            positions=data.pos.numpy(),
            energy_unit=energy_unit,
            length_unit=length_unit,
            energy=data.energy.item() if hasattr(data, "energy") and data.energy is not None else None,
            forces=data.forces.numpy() if hasattr(data, "forces") and data.forces is not None else None,
            cell=data.cell.squeeze(0).numpy() if hasattr(data, "cell") and data.cell is not None else None,
        )

    # ── ASE Atoms conversion ─────────────────────────────────

    def to_atoms(self, units: str = "eV/Ang") -> "ase.Atoms":
        """Molecule → ASE Atoms.

        MLIPCalculator, MD, optimization 등 ASE 파이프라인에서 사용.
        charge/spin/source 등 메타데이터는 atoms.info에 저장.

        Args:
            units: "eV/Ang" (기본, MLIP 표준) or "Ha/Bohr" (DFT 표준).
                   "as_is" → 현재 단위 그대로.

        Returns:
            ase.Atoms (calc 미설정 상태).
        """
        from ase import Atoms

        if units == "as_is":
            mol = self
        elif units == "eV/Ang":
            mol = self.to_units("eV", "Ang")
        elif units == "Ha/Bohr":
            mol = self.to_units("Ha", "Bohr")
        else:
            raise ValueError(f"Unknown units: {units}. Use 'eV/Ang', 'Ha/Bohr', or 'as_is'.")

        atoms = Atoms(
            numbers=mol.atomic_numbers,
            positions=mol.positions,
            cell=mol.cell if mol.cell is not None else None,
            pbc=mol.pbc if mol.pbc is not None else False,
        )

        # Store metadata in atoms.info
        if mol.charge != 0:
            atoms.info["charge"] = mol.charge
        if mol.spin != 0:
            atoms.info["spin"] = mol.spin
        if mol.mol_id is not None:
            atoms.info["mol_id"] = mol.mol_id
        if mol.formula is not None:
            atoms.info["formula"] = mol.formula
        if mol.smiles is not None:
            atoms.info["smiles"] = mol.smiles
        if mol.source is not None:
            atoms.info["source"] = mol.source
        atoms.info["energy_unit"] = mol.energy_unit
        atoms.info["length_unit"] = mol.length_unit

        # Store reference results in arrays (SinglePointCalculator-compatible)
        if mol.energy is not None or mol.forces is not None or mol.stress is not None:
            from ase.calculators.singlepoint import SinglePointCalculator
            spc = SinglePointCalculator(
                atoms,
                energy=mol.energy,
                forces=mol.forces,
                stress=mol.stress,
            )
            atoms.calc = spc

        return atoms

    @classmethod
    def from_atoms(
        cls,
        atoms: "ase.Atoms",
        energy_unit: str = "eV",
        length_unit: str = "Ang",
        source: Optional[str] = None,
    ) -> "Molecule":
        """ASE Atoms → Molecule.

        atoms.calc이 있으면 에너지/포스/스트레스를 추출.
        atoms.info에서 charge/spin/mol_id 등 메타데이터 복원.

        Args:
            atoms: ASE Atoms object.
            energy_unit: 좌표/에너지의 현재 단위.
            length_unit: 좌표의 현재 단위.
            source: 데이터 출처 (atoms.info["source"]보다 우선).
        """
        energy, forces, stress = None, None, None
        if atoms.calc is not None:
            try:
                energy = atoms.get_potential_energy()
            except Exception:
                pass
            try:
                forces = atoms.get_forces()
            except Exception:
                pass
            try:
                stress = atoms.get_stress(voigt=True)
            except Exception:
                pass

        cell = atoms.cell.array if np.any(atoms.pbc) else None
        pbc = np.array(atoms.pbc, dtype=bool) if np.any(atoms.pbc) else None

        info = atoms.info
        return cls(
            atomic_numbers=np.array(atoms.get_atomic_numbers(), dtype=np.int32),
            positions=atoms.get_positions().astype(np.float64),
            mol_id=info.get("mol_id"),
            formula=info.get("formula"),
            smiles=info.get("smiles"),
            charge=info.get("charge", 0),
            spin=info.get("spin", 0),
            energy_unit=info.get("energy_unit", energy_unit),
            length_unit=info.get("length_unit", length_unit),
            cell=cell,
            pbc=pbc,
            energy=energy,
            forces=forces,
            stress=stress,
            source=source or info.get("source"),
        )

    # ── Unit validation ───────────────────────────────────────

    def warn_if_atomic_units(self) -> None:
        """MLIP 학습에 사용하기 전 단위 확인.

        Hartree/Bohr 단위인 경우 경고. MLIP 모델은 보통 eV/Å를 기대.
        """
        msgs = []
        if self.energy_unit == "Ha":
            msgs.append(
                f"energy_unit='Ha' (Hartree). "
                f"MLIP 모델은 보통 eV를 사용합니다. "
                f".to_units('eV', 'Ang')로 변환을 권장합니다."
            )
        if self.length_unit == "Bohr":
            msgs.append(
                f"length_unit='Bohr'. "
                f"MLIP 모델은 보통 Angstrom을 사용합니다. "
                f".to_units('eV', 'Ang')로 변환을 권장합니다."
            )
        for msg in msgs:
            warnings.warn(msg, stacklevel=2)

    # ── Display ───────────────────────────────────────────────

    def summary(self) -> str:
        """One-line summary."""
        parts = [
            self.mol_id or "?",
            self.formula or f"Z={self.elements}",
            f"{self.num_atoms} atoms",
        ]
        if self.energy is not None:
            parts.append(f"E={self.energy:.6f} {self.energy_unit}")
        if self.homo_lumo_gap is not None:
            parts.append(f"gap={self.homo_lumo_gap:.2f} eV")
        if self.forces is not None:
            parts.append("F")
        if self.mulliken_charges is not None:
            parts.append("Q")
        if self.unrestricted:
            parts.append("UKS")
        if self.hamiltonian is not None:
            shape_str = f"{self.hamiltonian.shape[-1]}"
            if self.hamiltonian.ndim == 3:
                shape_str = f"2×{shape_str}"
            parts.append(f"H[{shape_str}]")
        if self.basis_info is not None:
            parts.append(f"{self.basis_info.name}({self.basis_info.convention})")
        if self.is_periodic:
            parts.append("[periodic]")
        if self.source:
            parts.append(f"src={self.source}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"Molecule({self.summary()})"


# ── Fast field-selective decoding ────────────────────────────────────

# Field type registry (mirrors Molecule class field categories)
_SCALAR_FIELDS = set(Molecule._SCALAR_FIELDS)
_ARRAY_FIELDS = set(Molecule._ARRAY_FIELDS)
_MATRIX_FIELDS = set(Molecule._MATRIX_FIELDS)
_CORE_FIELDS = {"atomic_numbers", "positions", "num_atoms"}
_META_FIELDS = {"_packed", "_basis_info", "unrestricted"}


def decode_fields(
    d: dict,
    fields: set[str] | None = None,
    to_torch: bool = False,
    dtype: str = "float32",
) -> dict:
    """Selectively decode fields from a Molecule.to_dict() dict.

    Molecule.from_dict()의 lightweight 대안. Molecule 객체를 만들지 않고
    필요한 필드만 bytes → numpy (또는 torch) 변환.

    Args:
        d: Molecule.to_dict()로 생성된 raw dict (LMDBDataset.get_dict() 결과).
        fields: decode할 필드 이름 집합. None이면 모든 필드 decode.
            예: {"atomic_numbers", "positions", "energy", "forces"}
        to_torch: True면 numpy 대신 torch.Tensor로 반환.
        dtype: float 배열의 출력 dtype (e.g., "float32", "float64").

    Returns:
        decoded dict. key는 요청한 필드명, value는 numpy array / torch tensor / scalar.
    """
    if to_torch:
        import torch
        tdtype = getattr(torch, dtype)

    want_all = fields is None
    want = fields if fields is not None else None

    def _want(name: str) -> bool:
        return want_all or name in want

    is_packed = d.get("_packed", False)
    num_atoms = d["num_atoms"]
    result = {}

    # ── num_atoms (always useful) ──
    if _want("num_atoms"):
        result["num_atoms"] = num_atoms

    # ── Core arrays ──
    if _want("atomic_numbers"):
        Z = np.frombuffer(d["atomic_numbers"], dtype=np.int32).copy()
        if to_torch:
            import torch
            result["atomic_numbers"] = torch.from_numpy(Z)
        else:
            result["atomic_numbers"] = Z

    if _want("positions"):
        pos = np.frombuffer(d["positions"], dtype=np.float64).reshape(num_atoms, 3).copy()
        if to_torch:
            result["positions"] = torch.from_numpy(pos).to(tdtype)
        else:
            result["positions"] = pos

    # ── Scalar fields ──
    for name in _SCALAR_FIELDS:
        if _want(name) and name in d:
            result[name] = d[name]

    # ── Non-symmetric array fields ──
    for name in _ARRAY_FIELDS:
        if not _want(name) or name not in d:
            continue
        arr_dtype = np.dtype(d.get(f"{name}_dtype", "float64"))
        shape = d[f"{name}_shape"]
        arr = np.frombuffer(d[name], dtype=arr_dtype).reshape(shape).copy()
        if to_torch:
            t = torch.from_numpy(arr)
            if t.is_floating_point():
                t = t.to(tdtype)
            result[name] = t
        else:
            result[name] = arr

    # ── Symmetric matrix fields (only unpack if requested) ──
    for name in _MATRIX_FIELDS:
        if not _want(name):
            continue
        M = _unpack_matrix_from_dict(d, name, is_packed)
        if M is None:
            continue
        if to_torch:
            result[name] = torch.from_numpy(M).to(tdtype)
        else:
            result[name] = M

    # ── Metadata ──
    if _want("_packed"):
        result["_packed"] = is_packed
    if _want("unrestricted"):
        result["unrestricted"] = d.get("unrestricted", False)
    if _want("_basis_info") and "_basis_info" in d:
        result["_basis_info"] = d["_basis_info"]  # keep as raw dict, not BasisInfo

    # nao metadata (useful for collator)
    for name in _MATRIX_FIELDS:
        nao_key = f"{name}_nao"
        if nao_key in d and (_want(name) or _want("nao")):
            result["nao"] = d[nao_key]
            break

    return result


def _unpack_matrix_from_dict(
    d: dict, name: str, is_packed: bool
) -> np.ndarray | None:
    """Unpack a single symmetric matrix field from a raw Molecule dict."""
    if is_packed and f"{name}_packed_a" in d:
        # Unrestricted: alpha + beta packed separately
        nao = d[f"{name}_nao"]
        mat_dtype = np.dtype(d.get(f"{name}_dtype", "float64"))
        tri_a = np.frombuffer(d[f"{name}_packed_a"], dtype=mat_dtype).copy()
        tri_b = np.frombuffer(d[f"{name}_packed_b"], dtype=mat_dtype).copy()
        return np.stack([
            unpack_upper_triangle(tri_a, nao),
            unpack_upper_triangle(tri_b, nao),
        ])
    elif is_packed and f"{name}_packed" in d:
        nao = d[f"{name}_nao"]
        mat_dtype = np.dtype(d.get(f"{name}_dtype", "float64"))
        tri = np.frombuffer(d[f"{name}_packed"], dtype=mat_dtype).copy()
        return unpack_upper_triangle(tri, nao)
    elif name in d:
        mat_dtype = np.dtype(d.get(f"{name}_dtype", "float64"))
        shape = d[f"{name}_shape"]
        return np.frombuffer(d[name], dtype=mat_dtype).reshape(shape).copy()
    return None
