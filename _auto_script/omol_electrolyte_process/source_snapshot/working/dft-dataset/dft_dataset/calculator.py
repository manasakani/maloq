"""Generic ASE Calculator & MD interface for ML potentials.

Framework-agnostic: works with PyTorch, JAX, PySCF, or any callable that
maps ASE Atoms → {"energy": float, "forces": ndarray(N,3)}.

Core idea: the Calculator takes a `predict_fn` — a plain callable:
    predict_fn(atoms: Atoms) -> {"energy": float, "forces": (N,3), ["stress": (6,)]}

Factory classmethods build predict_fn from various backends:
    MLIPCalculator.from_torch(model, ...)       # PyTorch nn.Module (dict input)
    MLIPCalculator.from_pyg(model, ...)         # PyG-based models
    MLIPCalculator.from_jax(apply_fn, params, ...)  # JAX
    MLIPCalculator.from_pyscf(mf_factory, ...)  # PySCF
    MLIPCalculator.from_ase(calculator)         # Wrap existing ASE calculator
    MLIPCalculator(predict_fn)                  # Any callable

Compatible with ASE optimizers, MD, OMol25 evaluation, etc.

Usage:
    from calculator import MLIPCalculator, MolecularDynamics

    # PyTorch (dict-based)
    calc = MLIPCalculator.from_torch(model, device="cuda")

    # PyG
    calc = MLIPCalculator.from_pyg(model, device="cuda", cutoff=5.0)

    # JAX
    calc = MLIPCalculator.from_jax(apply_fn, params, atom_to_input_fn)

    # PySCF
    calc = MLIPCalculator.from_pyscf(lambda atoms: build_mf(atoms))

    # Any callable
    calc = MLIPCalculator(lambda atoms: {"energy": 0.0, "forces": np.zeros((len(atoms),3))})

    # Single point
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()

    # MD
    md = MolecularDynamics(atoms, calc, ensemble="nvt", temperature=300, timestep=1.0)
    md.run(steps=10000)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

if TYPE_CHECKING:
    from ase import Atoms
    from pathlib import Path


# ---------------------------------------------------------------------------
# Predict function protocol
# ---------------------------------------------------------------------------

class PredictFn(Protocol):
    """Protocol for predict functions.

    Any callable with this signature works as a predict_fn.
    """
    def __call__(self, atoms: Atoms) -> dict[str, Any]:
        """Returns dict with "energy" (float) and "forces" (N,3 ndarray).
        Optionally "stress" (6, Voigt)."""
        ...


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class MLIPCalculator(Calculator):
    """Framework-agnostic ASE Calculator.

    Takes a predict_fn: Atoms → {"energy", "forces", ["stress"]}.
    Use classmethods (from_torch, from_pyg, from_jax, from_pyscf)
    for convenient construction from specific backends.
    """

    implemented_properties = ["energy", "forces", "free_energy"]
    default_parameters: dict = {}

    def __init__(
        self,
        predict_fn: Callable[[Atoms], dict[str, Any]],
        has_stress: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            predict_fn: Callable(Atoms) → {"energy": float, "forces": (N,3), ["stress": (6,)]}
            has_stress: If True, adds "stress" to implemented_properties for NPT.
        """
        super().__init__(**kwargs)
        self.predict_fn = predict_fn
        if has_stress:
            self.implemented_properties = [
                "energy", "forces", "free_energy", "stress",
            ]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        result = self.predict_fn(atoms)

        self.results["energy"] = float(result["energy"])
        self.results["free_energy"] = float(result["energy"])
        self.results["forces"] = np.asarray(result["forces"])

        if "stress" in result and result["stress"] is not None:
            stress = np.asarray(result["stress"])
            if stress.shape == (3, 3):
                from ase.stress import full_3x3_to_voigt_6_stress
                stress = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress

    # --- Factory: PyTorch (dict input) ---

    @classmethod
    def from_torch(
        cls,
        model,
        device: str = "cpu",
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str | None = None,
    ) -> MLIPCalculator:
        """Wrap a PyTorch model that takes a dict batch.

        Model interface:
            model({"atomic_numbers": (N,), "pos": (N,3), "batch": (N,), ...})
            → {"energy": (B,), "forces": (N,3), ["stress": (B,3,3)]}
        """
        import torch

        device = torch.device(device)
        model = model.to(device).eval()

        @torch.inference_mode(False)
        @torch.enable_grad()
        def predict_fn(atoms: Atoms) -> dict:
            pos = torch.tensor(
                atoms.get_positions(), dtype=torch.float32, device=device,
            )
            pos.requires_grad_(True)
            batch_data = {
                "atomic_numbers": torch.tensor(
                    atoms.get_atomic_numbers(), dtype=torch.long, device=device,
                ),
                "pos": pos,
                "batch": torch.zeros(len(atoms), dtype=torch.long, device=device),
            }
            if "charge" in atoms.info:
                batch_data["charge"] = torch.tensor(
                    [atoms.info["charge"]], dtype=torch.long, device=device,
                )
            if "spin" in atoms.info:
                batch_data["spin"] = torch.tensor(
                    [atoms.info["spin"]], dtype=torch.long, device=device,
                )
            out = model(batch_data)
            result = {
                "energy": out[energy_key].detach().cpu().item(),
                "forces": out[forces_key].detach().cpu().numpy(),
            }
            if stress_key and stress_key in out:
                result["stress"] = out[stress_key].detach().cpu().numpy()
            return result

        return cls(predict_fn, has_stress=stress_key is not None)

    # --- Factory: PyG ---

    @classmethod
    def from_pyg(
        cls,
        model,
        device: str = "cpu",
        cutoff: float = 5.0,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str | None = None,
    ) -> MLIPCalculator:
        """Wrap a PyG-based model.

        Model interface:
            model(Data(z, pos, batch, edge_index))
            → {"energy": (B,), "forces": (N,3)}
        """
        import torch

        device = torch.device(device)
        model = model.to(device).eval()

        @torch.inference_mode(False)
        @torch.enable_grad()
        def predict_fn(atoms: Atoms) -> dict:
            from torch_geometric.data import Data
            from torch_geometric.nn import radius_graph

            pos = torch.tensor(
                atoms.get_positions(), dtype=torch.float32, device=device,
            )
            pos.requires_grad_(True)
            z = torch.tensor(
                atoms.get_atomic_numbers(), dtype=torch.long, device=device,
            )
            batch = torch.zeros(len(atoms), dtype=torch.long, device=device)
            edge_index = radius_graph(pos, r=cutoff, batch=batch)
            data = Data(z=z, pos=pos, batch=batch, edge_index=edge_index)

            if "charge" in atoms.info:
                data.charge = torch.tensor(
                    [atoms.info["charge"]], dtype=torch.long, device=device,
                )
            if "spin" in atoms.info:
                data.spin = torch.tensor(
                    [atoms.info["spin"]], dtype=torch.long, device=device,
                )

            out = model(data)
            result = {
                "energy": out[energy_key].detach().cpu().item(),
                "forces": out[forces_key].detach().cpu().numpy(),
            }
            if stress_key and stress_key in out:
                result["stress"] = out[stress_key].detach().cpu().numpy()
            return result

        return cls(predict_fn, has_stress=stress_key is not None)

    # --- Factory: JAX ---

    @classmethod
    def from_jax(
        cls,
        apply_fn: Callable,
        params: Any,
        atoms_to_input: Callable[[Atoms], Any],
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str | None = None,
    ) -> MLIPCalculator:
        """Wrap a JAX model.

        Args:
            apply_fn: JAX function(params, input) → output dict.
            params: Model parameters (pytree).
            atoms_to_input: Callable(Atoms) → JAX-compatible input.
            energy_key: Key for energy in output dict.
            forces_key: Key for forces in output dict.
        """
        import jax.numpy as jnp

        def predict_fn(atoms: Atoms) -> dict:
            inp = atoms_to_input(atoms)
            out = apply_fn(params, inp)
            result = {
                "energy": float(out[energy_key]),
                "forces": np.asarray(out[forces_key]),
            }
            if stress_key and stress_key in out:
                result["stress"] = np.asarray(out[stress_key])
            return result

        return cls(predict_fn, has_stress=stress_key is not None)

    # --- Factory: PySCF ---

    @classmethod
    def from_pyscf(
        cls,
        mf_factory: Callable[[Atoms], Any],
    ) -> MLIPCalculator:
        """Wrap a PySCF mean-field calculation.

        Args:
            mf_factory: Callable(Atoms) → converged pyscf.scf object.
                        Must handle mol construction, basis, xc, and SCF convergence.

        Example:
            def build_mf(atoms):
                from pyscf import gto, dft
                mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(),
                            atoms.get_positions())], basis='def2-svp', unit='Ang')
                mf = dft.RKS(mol, xc='pbe0').run()
                return mf

            calc = MLIPCalculator.from_pyscf(build_mf)
        """
        from ase.units import Hartree, Bohr

        def predict_fn(atoms: Atoms) -> dict:
            mf = mf_factory(atoms)
            energy_hartree = mf.e_tot
            grad_hartree_bohr = mf.nuc_grad_method().kernel()
            return {
                "energy": energy_hartree * Hartree,
                "forces": -grad_hartree_bohr * Hartree / Bohr,
            }

        return cls(predict_fn)

    # --- Factory: wrap existing ASE calculator ---

    @classmethod
    def from_ase(cls, calculator: Calculator) -> MLIPCalculator:
        """Wrap an existing ASE calculator (e.g., EMT, GPAW, etc.)."""

        def predict_fn(atoms: Atoms) -> dict:
            atoms_copy = atoms.copy()
            atoms_copy.calc = calculator
            result = {
                "energy": atoms_copy.get_potential_energy(),
                "forces": atoms_copy.get_forces(),
            }
            try:
                result["stress"] = atoms_copy.get_stress()
            except Exception:
                pass
            return result

        return cls(predict_fn)

    # --- Factory: Lightning checkpoint ---

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_class: type,
        device: str = "cpu",
        adapter: str = "torch",
        cutoff: float = 5.0,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str | None = None,
        **model_kwargs,
    ) -> MLIPCalculator:
        """Load from a PyTorch Lightning checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file.
            model_class: LightningModule class.
            device: Target device.
            adapter: "torch" (dict input) or "pyg" (PyG Data input).
            cutoff: Radial cutoff for pyg adapter.
        """
        model = model_class.load_from_checkpoint(
            checkpoint_path, map_location=device, **model_kwargs,
        )
        factory = cls.from_pyg if adapter == "pyg" else cls.from_torch
        return factory(
            model, device=device, cutoff=cutoff,
            energy_key=energy_key, forces_key=forces_key, stress_key=stress_key,
        )


# ---------------------------------------------------------------------------
# MLDFT Calculator
# ---------------------------------------------------------------------------


class MLDFTCalculator(Calculator):
    """ASE Calculator for ML models that predict DFT quantities.

    Unlike MLIPCalculator (stateless, E/F only), MLDFTCalculator:
    - Uses the Molecule dataclass as its core abstraction
    - Delegates eigensolve & density to pipeline.properties_from_hamiltonian
    - Stores all electronic structure intermediates for analysis

    Pipeline:
        ASE Atoms → Molecule → PySCF mol → (S, H_init)
        → predict_hamiltonian(mol, S, H_init) → H_pred
        → inject_hamiltonian → (orbital_energies, density_matrix, homo, lumo)
        → E = mf.energy_tot(D), F = -∇E

    Usage:
        def predict_H(pyscf_mol, S, H_init):
            return my_model(pyscf_mol, S, H_init)  # (nao, nao) ndarray

        calc = MLDFTCalculator(predict_H, basis='def2-svp', xc='pbe')
        atoms.calc = calc
        atoms.get_potential_energy()

        # Access the Molecule with all intermediates
        calc.molecule.hamiltonian       # predicted H
        calc.molecule.overlap           # S
        calc.molecule.orbital_energies  # eigenvalues
        calc.molecule.density_matrix    # D
        calc.molecule.energy            # E (Ha)
        calc.homo, calc.lumo            # frontier orbitals (Ha)
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        predict_hamiltonian: Callable,
        basis: str = "def2-svp",
        xc: str = "pbe",
        density_fit: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            predict_hamiltonian: Callable(pyscf_mol, S, H_init) → H_pred (nao, nao).
                pyscf_mol: pyscf.gto.Mole object.
                S: overlap matrix (nao, nao) ndarray.
                H_init: initial Fock matrix (nao, nao) ndarray.
                Returns: predicted Hamiltonian (nao, nao) numpy array.
            basis: Gaussian basis set.
            xc: Exchange-correlation functional.
            density_fit: Use density fitting for PySCF.
        """
        super().__init__(**kwargs)
        self.predict_hamiltonian = predict_hamiltonian
        self.basis = basis
        self.xc = xc
        self.density_fit = density_fit

        # Populated after calculate()
        self.molecule = None   # Molecule with all electronic structure
        self._pyscf_mol = None
        self._mf = None
        self._props: dict | None = None

    @property
    def homo(self) -> float | None:
        return self._props["homo"] if self._props else None

    @property
    def lumo(self) -> float | None:
        return self._props["lumo"] if self._props else None

    def _build_pyscf(self, atoms):
        """Build PySCF mol + mean-field from ASE Atoms."""
        from pyscf import gto, dft

        mol = gto.M(
            atom=[(s, p) for s, p in zip(
                atoms.get_chemical_symbols(), atoms.get_positions()
            )],
            basis=self.basis, unit="Ang", verbose=0,
        )
        mf = dft.RKS(mol)
        if self.density_fit:
            mf = mf.density_fit()
        mf.xc = self.xc
        return mol, mf

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ) -> None:
        from ase.units import Hartree, Bohr
        from .solvers import solve_eigenproblem, build_density_matrix
        from .molecule import Molecule

        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        # 1. Build PySCF infrastructure
        pyscf_mol, mf = self._build_pyscf(atoms)
        self._pyscf_mol = pyscf_mol
        self._mf = mf

        # 2. Overlap & initial Hamiltonian
        S = pyscf_mol.intor("int1e_ovlp")
        init_dm = mf.init_guess_by_minao()
        H_init = mf.get_fock(dm=init_dm)

        # 3. Predict Hamiltonian
        H_pred = self.predict_hamiltonian(pyscf_mol, S, H_init)

        # 4. Eigensolve (HC = SCε) & density matrix via solvers
        n_electrons = pyscf_mol.nelectron
        mo_energy, mo_coeff = solve_eigenproblem(H_pred, S)
        D = build_density_matrix(mo_coeff, n_electrons)

        n_occ = n_electrons // 2
        homo = float(mo_energy[n_occ - 1]) if n_occ > 0 else None
        lumo = float(mo_energy[n_occ]) if n_occ < len(mo_energy) else None

        # 5. Energy via PySCF (XC functional evaluation on predicted density)
        energy_ha = mf.energy_tot(D)

        # 6. Store as Molecule
        molecule = Molecule(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            charge=0,
            spin=0,
            energy_unit="Ha",
            length_unit="Ang",
        )
        molecule.overlap = S
        molecule.hamiltonian = H_pred
        molecule.orbital_energies = mo_energy
        molecule.orbital_coeffs = mo_coeff
        molecule.density_matrix = D
        molecule.energy = energy_ha
        molecule.homo = homo
        molecule.lumo = lumo
        molecule.xc = self.xc

        self.molecule = molecule
        self._props = {"homo": homo, "lumo": lumo}

        self.results["energy"] = energy_ha * Hartree
        self.results["free_energy"] = energy_ha * Hartree

        # 7. Forces via PySCF analytical gradient
        if "forces" in properties:
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
            grad = mf.nuc_grad_method()
            grad.base.auxbasis_response = True
            forces = -grad.kernel(
                mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
            )
            self.results["forces"] = forces * Hartree / Bohr

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_class: type,
        predict_fn_builder: Callable,
        basis: str = "def2-svp",
        xc: str = "pbe",
        device: str = "cpu",
        **kwargs,
    ) -> MLDFTCalculator:
        """Load from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file.
            model_class: LightningModule class.
            predict_fn_builder: Callable(model) → predict_hamiltonian function.
                Takes the loaded model, returns callable(pyscf_mol, S, H_init) → H_pred.
            basis: Gaussian basis set.
            xc: Exchange-correlation functional.
            device: Target device.

        Example:
            def build_predictor(model):
                def predict_H(pyscf_mol, S, H_init):
                    # model-specific preprocessing ...
                    return model.predict(S, H_init)
                return predict_H

            calc = MLDFTCalculator.from_checkpoint(
                "best.ckpt", MyLitModule, build_predictor,
                basis="def2-svp", xc="pbe", device="cuda",
            )
        """
        model = model_class.load_from_checkpoint(
            checkpoint_path, map_location=device,
        )
        model.eval()
        predict_fn = predict_fn_builder(model)
        return cls(predict_fn, basis=basis, xc=xc, **kwargs)


# ---------------------------------------------------------------------------
# Geometry Optimization
# ---------------------------------------------------------------------------


def optimize(
    atoms: Atoms,
    calculator: Calculator,
    fmax: float = 0.01,
    max_steps: int = 500,
    optimizer: str = "LBFGS",
    trajectory: str | None = None,
    logfile: str | None = "-",
) -> Atoms:
    """Geometry optimization.

    Args:
        atoms: Initial structure (modified in-place).
        calculator: Any ASE Calculator.
        fmax: Force convergence threshold (eV/Å).
        max_steps: Maximum optimization steps.
        optimizer: "LBFGS", "BFGS", or "FIRE".
        trajectory: Save trajectory to this path (.traj), or None.
        logfile: Log output. "-" = stdout, None = silent.

    Returns:
        Optimized Atoms (same reference as input).
    """
    from ase.optimize import BFGS, FIRE, LBFGS

    OPT_MAP = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}
    if optimizer not in OPT_MAP:
        raise ValueError(f"Unknown optimizer: {optimizer}. Choose from {list(OPT_MAP)}")

    atoms.calc = calculator
    opt = OPT_MAP[optimizer](atoms, trajectory=trajectory, logfile=logfile)
    opt.run(fmax=fmax, steps=max_steps)

    atoms.info["opt_converged"] = opt.converged()
    atoms.info["opt_nsteps"] = opt.nsteps
    return atoms


# ---------------------------------------------------------------------------
# Molecular Dynamics
# ---------------------------------------------------------------------------


class MolecularDynamics:
    """Convenience wrapper around ASE MD integrators.

    Supports NVE, NVT (Langevin), NVT (Berendsen), and NPT (Berendsen).

    Usage:
        md = MolecularDynamics(
            atoms, calc,
            ensemble="nvt", temperature=300, timestep=1.0,
            friction=0.01, trajectory="md.traj", loginterval=10,
        )
        md.run(steps=10000)
    """

    ENSEMBLES = ["nve", "nvt", "nvt_berendsen", "npt_berendsen"]

    def __init__(
        self,
        atoms: Atoms,
        calculator: Calculator,
        ensemble: str = "nvt",
        temperature: float = 300.0,
        timestep: float = 1.0,
        friction: float = 0.01,
        pressure: float = 1.01325,
        taut: float | None = None,
        taup: float | None = None,
        compressibility: float = 4.57e-5,
        trajectory: str | None = None,
        logfile: str | None = "-",
        loginterval: int = 10,
        append_trajectory: bool = False,
    ) -> None:
        """
        Args:
            atoms: ASE Atoms (velocities initialized if absent).
            calculator: Any ASE Calculator.
            ensemble: "nve", "nvt" (Langevin), "nvt_berendsen", "npt_berendsen".
            temperature: Target temperature (K).
            timestep: Integration timestep (fs).
            friction: Langevin friction (1/fs), nvt only.
            pressure: Target pressure (bar), npt only. Default = 1 atm.
            taut: Thermostat time constant (fs). Default = 100 * timestep.
            taup: Barostat time constant (fs). Default = 1000 * timestep.
            compressibility: Isothermal compressibility (1/bar). Default = water@300K.
            trajectory: Save trajectory (.traj), or None.
            logfile: Log file. "-" = stdout, None = silent.
            loginterval: Write/log interval in steps.
            append_trajectory: Append to existing trajectory.
        """
        from ase import units

        self.atoms = atoms
        self.atoms.calc = calculator
        self.ensemble = ensemble.lower()
        self.temperature = temperature
        self.timestep_fs = timestep
        self.dt = timestep * units.fs

        if taut is None:
            taut = 100 * timestep
        if taup is None:
            taup = 1000 * timestep

        # Initialize velocities if not set
        if atoms.get_velocities() is None or np.allclose(atoms.get_velocities(), 0.0):
            from ase.md.velocitydistribution import (
                MaxwellBoltzmannDistribution,
                Stationary,
                ZeroRotation,
            )
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
            Stationary(atoms)
            if not np.any(atoms.pbc):
                ZeroRotation(atoms)

        # Build integrator
        if self.ensemble == "nve":
            from ase.md.verlet import VelocityVerlet
            self.dyn = VelocityVerlet(atoms, timestep=self.dt, logfile=logfile)

        elif self.ensemble == "nvt":
            from ase.md.langevin import Langevin
            self.dyn = Langevin(
                atoms, timestep=self.dt, temperature_K=temperature,
                friction=friction / units.fs, logfile=logfile,
            )

        elif self.ensemble == "nvt_berendsen":
            from ase.md import NVTBerendsen
            self.dyn = NVTBerendsen(
                atoms, timestep=self.dt, temperature_K=temperature,
                taut=taut * units.fs, logfile=logfile,
            )

        elif self.ensemble == "npt_berendsen":
            from ase.md import NPTBerendsen
            self.dyn = NPTBerendsen(
                atoms, timestep=self.dt, temperature_K=temperature,
                pressure_au=pressure * units.bar,
                taut=taut * units.fs, taup=taup * units.fs,
                compressibility_au=compressibility / units.bar, logfile=logfile,
            )

        else:
            raise ValueError(
                f"Unknown ensemble: {ensemble}. Choose from {self.ENSEMBLES}"
            )

        # Trajectory writer
        self._traj = None
        if trajectory is not None:
            from ase.io import Trajectory
            self._traj = Trajectory(
                trajectory, "a" if append_trajectory else "w", atoms,
            )
            self.dyn.attach(self._traj.write, interval=loginterval)

        self._observers: list = []
        self.loginterval = loginterval

    def run(self, steps: int) -> None:
        """Run MD for the given number of steps."""
        self.dyn.run(steps)

    def attach(self, callback, interval: int | None = None) -> None:
        """Attach a callback called every `interval` steps."""
        if interval is None:
            interval = self.loginterval
        self.dyn.attach(callback, interval=interval)
        self._observers.append(callback)

    def get_temperature(self) -> float:
        return self.atoms.get_temperature()

    def get_kinetic_energy(self) -> float:
        return self.atoms.get_kinetic_energy()

    def get_potential_energy(self) -> float:
        return self.atoms.get_potential_energy()

    def get_total_energy(self) -> float:
        return self.get_kinetic_energy() + self.get_potential_energy()

    def close(self) -> None:
        if self._traj is not None:
            self._traj.close()

    def __del__(self):
        self.close()


class MDLogger:
    """Records MD observables to a dict for analysis.

    Usage:
        logger = MDLogger(md)
        md.attach(logger, interval=10)
        md.run(10000)
        df = logger.to_dataframe()
    """

    def __init__(self, md: MolecularDynamics) -> None:
        self.md = md
        self.data: dict[str, list] = {
            "step": [], "time_fs": [], "temperature": [],
            "kinetic_energy": [], "potential_energy": [], "total_energy": [],
        }
        self._step = 0

    def __call__(self) -> None:
        self._step += self.md.loginterval
        ke = self.md.get_kinetic_energy()
        pe = self.md.get_potential_energy()
        self.data["step"].append(self._step)
        self.data["time_fs"].append(self._step * self.md.timestep_fs)
        self.data["temperature"].append(self.md.get_temperature())
        self.data["kinetic_energy"].append(ke)
        self.data["potential_energy"].append(pe)
        self.data["total_energy"].append(ke + pe)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.data)


# ---------------------------------------------------------------------------
# OMol25 Evaluation
# ---------------------------------------------------------------------------


class OMol25Evaluator:
    """OMol25 leaderboard evaluation helper.

    Works with any ASE Calculator (MLIPCalculator, FAIRChemCalculator, etc.).
    """

    CHEMISTRY_TASKS = [
        "conformers", "protonation", "ieea", "spin_gap",
        "ligand_pocket", "ligand_strain", "distance_scaling",
    ]

    def __init__(self, calculator: Calculator) -> None:
        self.calculator = calculator

    def generate_s2ef_submission(
        self,
        atoms_list: list[Atoms],
        output_path: str,
        id_key: str = "source",
    ) -> None:
        """Generate .npz for S2EF leaderboard submission."""
        ids, energies, all_forces, natoms = [], [], [], []
        for atoms in atoms_list:
            atoms.calc = self.calculator
            energies.append(atoms.get_potential_energy())
            all_forces.append(atoms.get_forces())
            ids.append(atoms.info.get(id_key, ""))
            natoms.append(len(atoms))
        np.savez_compressed(
            output_path,
            ids=np.array(ids), energy=np.array(energies),
            forces=np.concatenate(all_forces, axis=0), natoms=np.array(natoms),
        )

    def run_chemistry_task(
        self, task_name: str, input_path: str, output_path: str,
    ) -> dict:
        """Run one of the 7 chemistry evaluation tasks.

        Requires: pip install fairchem-data-omol>=0.1.1
        """
        import gzip, json, pickle

        if task_name not in self.CHEMISTRY_TASKS:
            raise ValueError(
                f"Unknown task: {task_name}. Choose from {self.CHEMISTRY_TASKS}"
            )
        from fairchem.core.components.calculate.recipes import omol
        recipe_fn = getattr(omol, task_name)
        with open(input_path, "rb") as f:
            input_data = pickle.load(f)
        results = recipe_fn(input_data, self.calculator)
        if output_path.endswith(".gz"):
            with gzip.open(output_path, "wb") as f:
                f.write(json.dumps(results, default=str).encode("utf-8"))
        else:
            with open(output_path, "w") as f:
                json.dump(results, f, default=str)
        return results

    def run_all_chemistry_tasks(
        self, input_dir: str, output_dir: str,
    ) -> dict[str, dict]:
        """Run all 7 tasks. Expects {task}_inputs.pkl in input_dir."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        for task in self.CHEMISTRY_TASKS:
            input_path = os.path.join(input_dir, f"{task}_inputs.pkl")
            output_path = os.path.join(output_dir, f"{task}_results.json.gz")
            if not os.path.exists(input_path):
                print(f"[skip] {task}: {input_path} not found")
                continue
            print(f"[run]  {task}...")
            all_results[task] = self.run_chemistry_task(task, input_path, output_path)
            print(f"[done] {task} → {output_path}")
        return all_results
