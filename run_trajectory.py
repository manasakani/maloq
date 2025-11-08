from ase import units
from ase.io import Trajectory, write
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import torch
import matplotlib.pyplot as plt

from torch_geometric.data import Data as gnnData
from dataset_utils import get_loader
from fock_utils import fock_targets, utils_tensor_decomp, matrix2labels_kernels
from fock_utils.get_energy_from_fock import get_density_from_fock
import run_HELM

# Checkpoints:
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
HELM_backbone = './ICLR_2025/MD17_final_Hamiltonian_models/outputs_QM7_water/backbone.pt'
HELM_fock_head = './ICLR_2025/MD17_final_Hamiltonian_models/outputs_QM7_water/head.pt'
orbital_basis = get_loader.orbital_basis_def2_svp_QM7
node_reference_file = None

dtype = torch.float64
torch.set_default_dtype(dtype)

# -------- Run Molecular Dynamics with UMA/OMol ------------
calc = FAIRChemCalculator(predictor, task_name="omol")
atoms = molecule("H2O")
atoms.calc = calc
dyn = Langevin(
                atoms,
                timestep=0.1 * units.fs,
                temperature_K=800,
                friction=0.001 / units.fs,
            )
trajectory = Trajectory("water.traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)
dyn.run(steps=500)

# -------- Compute electron densities with HELM ------------
HELM = run_HELM.load_models(HELM_backbone, HELM_fock_head, orbital_basis)

traj = Trajectory("water.traj", "r")
for i, atoms in enumerate(traj):

    # 2. Make graph
    atom_graph = run_HELM.make_graph(atoms)

    # 3. Get Fock matrix
    fock_matrix = run_HELM.run_HELM_fock(atom_graph, HELM)

    # 4. Compute electron density from fock
    label = f"{i:04d}"
    electron_density = get_density_from_fock(atom_graph, fock_matrix, HELM, label=label)

write("water.xyz", traj)
