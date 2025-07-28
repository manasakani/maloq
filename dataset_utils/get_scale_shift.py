import torch
import numpy as np

from fock_utils import utils_orca_out, fock_targets, basis_sets
from ase import Atoms
from .get_loader import orbital_basis_def2_svp_nabla, orbital_basis_def2_svp_QM7
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset

def get_scale_shift(database, dataset_name, rcut=5.0, dtype=torch.float32, reduce_edge=False, rank=0):
    """
    Compute scaling and shifting factors for the scalar components of the hamiltonian datasets and save them to file
    """

    num_molecules = len(database)

    # 1. Get the indices of the scalar components in every target
    # -----------------------------------------------------------
    if dataset_name == "QM7":
        orbital_basis = orbital_basis_def2_svp_QM7
    elif dataset_name == "nablaDFT":
        orbital_basis = orbital_basis_def2_svp_nabla
    elif dataset_name == "omol":
        # orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
        orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in ['H', 'O']} # test
        orbital_basis = {k: sorted(v) for k, v in orbital_basis.items()} # The basis must be in l-major
    else: 
        print("Unknown dataset name!")
    orbital_basis = {k: torch.tensor(v) for k, v in orbital_basis.items()}
    
    ls_list = []
    for l in range(5): # searching for up to g orbitals
        counts = [torch.sum(orbital_basis[el] == l) for el in orbital_basis]
        ls_list.append(torch.tensor(max(counts) * [l], dtype=torch.int))

    ls_list = torch.cat(ls_list)        # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    print(f"ls_list: {ls_list}")

    # get the indices of the scalar components within each node-block
    scalar_indices = []
    irrep_track = 0
    for i, l1 in enumerate(ls_list):
        for j, l2 in enumerate(ls_list):
            l3s = range(abs(l1 - l2), l1 + l2 + 1) 
            len_l3s = sum([(2*l3+1) for l3 in l3s])

            # print(f"l1: {l1}, l2: {l2}, l3s: {list(l3s)}, len_l3s: {len_l3s}, irrep_track: {irrep_track}")

            # --> Consider all of the scalar components in the l1-l2 block
            for l3 in l3s:
                if l3 == 0: # found a scalar component
                    scalar_indices.append(irrep_track)
                irrep_track += (2 * l3 + 1)

            # --> Consider only the orbital self-interactions. Then, the first l3 is always 0 (and that's what we want)
            # if l1 == l2 and i == j:
            #     scalar_indices.append(irrep_track)
            # irrep_track += len_l3s

    element_scalar_values = {}
    for i in range(num_molecules):
        mol = database[i]

        print("working on molecule", i, "of", num_molecules)
    
        if dataset_name == "QM7":
            energy = mol['energy']
            forces = mol['forces']
            hamiltonian = mol['hamiltonian'].numpy()   
            atomic_numbers = mol['_atomic_numbers'].numpy()
            positions=mol['_positions'].numpy()
            orbital_basis = orbital_basis_def2_svp_QM7
        
        elif dataset_name == "nablaDFT":
            atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = database[i]
            orbital_basis = orbital_basis_def2_svp_nabla
        
        elif dataset_name == "omol":
            atomic_numbers = mol.atomic_numbers
            node_labels = mol.node_y
            edge_labels = mol.y
            energies = mol.energies
            forces = mol.forces
            required_irreps = mol.required_irreps

        else: 
            print("Unknown database!")
        
        # if the dataset is not omol, we need to create the atomic structure and set up the graph targets
        if dataset_name != "omol":
        
            # 1. Make the atomic structure
            mol_atoms = Atoms(symbols=atomic_numbers, positions=positions)

            # 2. Set up the Graph targets 
            if dataset_name == "QM7":                 
                hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)      # QM7 comes in zxy coordinates from ORCA, so need to rotate 
            
            graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian, dtype=dtype, reflection_symmetry=reduce_edge)
            required_irreps = graph_targets.req_output_irreps

            node_labels = graph_targets.node_labels

        # 3. Compute the scale and shift for each atomic number
        for atomic_number, node_block in zip(atomic_numbers, node_labels):
            atomic_number = int(atomic_number.item())
            if atomic_number not in element_scalar_values:
                element_scalar_values[atomic_number] = []

            orbital_onsite_scalars = node_block[scalar_indices]
            element_scalar_values[atomic_number].append(orbital_onsite_scalars)

    # print(f"Element scalar values: {element_scalar_values}")

    # get the mean/std per element
    element_scalar_means = {}
    element_scalar_stds = {}

    for Z, tensor_list in element_scalar_values.items():

        # Stack into a single 2D tensor: shape [num_molecules, num_scalars_per_atom]
        stacked = torch.stack(tensor_list)  # shape: [N, 6] for example with H2O

        means = stacked.mean(dim=0)  # shape: [6]
        stds = stacked.std(dim=0, unbiased=False)  

        # Fix always-zero positions
        threshold = 1e-4
        zero_mask = (means == 0.0) 
        means[zero_mask] = 0.0
        zero_mask = (stds < threshold)
        stds[zero_mask] = 1.0

        element_scalar_means[Z] = means.tolist()
        element_scalar_stds[Z] = stds.tolist()

    print(f"Indices of scalar components: {scalar_indices}")
    print(f"Element scalar means (averaged): {element_scalar_means}")
    print(f"Element scalar stds (averaged): {element_scalar_stds}")

    scale_shift_data = {
        "element_scalar_means": element_scalar_means,  # dict[int -> list[float]]
        "element_scalar_stds": element_scalar_stds,    # dict[int -> list[float]]
        "scalar_irrep_indices": scalar_indices         # list[int]
    }
    torch.save(scale_shift_data, "./fock_datasets/element_scale_shifts_water_" + dataset_name + ".pt")