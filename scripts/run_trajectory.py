from ase import units, Atoms
from ase.io import Trajectory, write
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import torch
import matplotlib.pyplot as plt

from torch_geometric.data import Data as gnnData
from fock_utils import basis_sets
from fock_utils.get_energy_from_fock import get_density_from_fock
from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

import run_HELM

# Checkpoints:
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
HELM_backbone = './ICLR_2025/nablaDFT_final_Hamiltonian_models/outputs_nablaDFT_medium/backbone.pt'
HELM_fock_head = './ICLR_2025/nablaDFT_final_Hamiltonian_models/outputs_nablaDFT_medium/head.pt'
orbital_basis = basis_sets.orbital_basis_def2_svp_nabla

dtype = torch.float64
torch.set_default_dtype(dtype)

# -------- Get atomic structure ----------------------------
database = HamiltonianDatabase("/checkpoint/ocp/manasakani/fock_datasets/nabla2_DFT/test_10k_conformers.db")
atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = database[1]
atoms = Atoms(symbols=atomic_numbers, positions=positions)
name = str(atoms.symbols)
dataset_name = "nablaDFT"
# atoms = molecule("H2O")

# -------- Run Molecular Dynamics with UMA/OMol ------------
calc = FAIRChemCalculator(predictor, task_name="omol")
atoms.calc = calc
dyn = Langevin(
                atoms,
                timestep=0.1 * units.fs,
                temperature_K=1500,
                friction=0.001 / units.fs,
            )
trajectory = Trajectory(name+".traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)
dyn.run(steps=300)


# -------- Compute electron densities with HELM ------------
node_reference_file = './fock_datasets/element_scale_shifts_' + dataset_name + '.pt'
HELM = run_HELM.load_models(HELM_backbone,
                            HELM_fock_head,
                            orbital_basis,
                            dataset_name=dataset_name,
                            node_ref_file=node_reference_file)

traj = Trajectory(name+".traj", "r")
# write(".xyz", traj)
for i, atoms in enumerate(traj):

    # 2. Make graph
    atom_graph = run_HELM.make_graph(atoms)

    # 3. Get Fock matrix
    fock_matrix = run_HELM.run_HELM_fock(atom_graph, HELM)

    # 4. Compute electron density from fock
    label = name+f"_{i:04d}"
    electron_density = get_density_from_fock(atom_graph, fock_matrix, HELM, label=label)
