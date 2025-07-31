import torch
import numpy as np

from fock_utils import utils_orca_out, fock_targets, basis_sets, utils_tensor_decomp
from ase import Atoms
from .get_loader import orbital_basis_def2_svp_nabla, orbital_basis_def2_svp_QM7
import matplotlib.pyplot as plt
from e3nn.o3 import Irreps

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
        orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
        orbital_basis = {k: sorted(v) for k, v in orbital_basis.items()} # The basis must be in l-major
    else: 
        print("Unknown dataset name!")

    # 0. Compute locations of scalars from required_irreps for this dataset's basis:
    _, required_irreps, simplified_out_irreps = utils_tensor_decomp.make_output_irreps(orbital_basis) 
    required_irreps = Irreps(required_irreps)
    scalar_indices = []
    irrep_track = 0
    for mul, irrep in required_irreps:
        if irrep.l == 0:
            for _ in range(mul):
                scalar_indices.append(irrep_track)
                irrep_track += (2 * irrep.l + 1)
        else:
            irrep_track += mul * (2 * irrep.l + 1)
    print(f"Scalar indices: {scalar_indices}")

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
            # required_irreps = mol.required_irreps
            # simplified_out_irreps = Irreps(required_irreps).sort()[0].simplify()

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
            # required_irreps = graph_targets.req_output_irreps
            # simplified_out_irreps = Irreps(required_irreps).sort()[0].simplify()

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
    torch.save(scale_shift_data, "./fock_datasets/element_scale_shifts_water_fullbasis" + dataset_name + ".pt")