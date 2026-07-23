# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
import torch
import os
import numpy as np

from ..fock_utils import utils_orca_out, basis_sets, utils_tensor_decomp, fock_targets_batched
from ..dataset_utils.get_loader import load_periodic_cp2k_structure, read_cp2k_matrix

from ase import Atoms
import matplotlib.pyplot as plt
from e3nn.o3 import Irreps
import time
import copy
import re

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset, Batch
import torch.distributed as dist

def get_scale_shift(database, dataset_name, rcut=5.0, dtype=torch.float32, reduce_edge=False, rank=0, filename='scale_shifts.pt', open_shell=False):
    """
    Compute scaling and shifting factors for the scalar components of the hamiltonian datasets and save them to file
    NOTE: Distributed scale calculation is not verified! Need to test that.
    """

    num_molecules = len(database)

    molecular_database = True if dataset_name in ["QM7", "nablaDFT", "omol", "omol_csh_58k", "omol_electronic_structures"] else False

    # 1. Get the indices of the scalar components in every target
    # -----------------------------------------------------------
    if dataset_name == "QM7":
        orbital_basis = basis_sets.orbital_basis_def2_svp_QM7
    elif dataset_name == "nablaDFT":
        orbital_basis = basis_sets.orbital_basis_def2_svp_nabla
    elif dataset_name in ["omol", "omol_csh_58k", "omol_electronic_structures"]:
        orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
    elif dataset_name == "cp2k_material":
        orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.orbital_basis_def2_svp_cp2k[element] for element in basis_sets.orbital_basis_def2_svp_cp2k.keys()}
    else:
        print("Unknown dataset name!")

    # orbital_basis = {k: sorted(v) for k, v in orbital_basis.items()} # If need to sort by l (but this is not needed)


    # Extract the magnitudes of those irreps to make scaling/shifting factors
    if open_shell:
        element_scalar_values_alpha = {}
        element_scalar_values_beta = {}
    else:
        element_scalar_values = {}


    time_start = time.perf_counter()

    if molecular_database:
        periodic_dataset = False
        for i in range(num_molecules):
            if dataset_name == "QM7":
                energy = mol['energy']
                forces = mol['forces']
                hamiltonian = mol['hamiltonian'].numpy()
                atomic_numbers = mol['_atomic_numbers'].numpy()
                positions=mol['_positions'].numpy()

            elif dataset_name == "nablaDFT":
                atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = database[i]

            elif dataset_name == "omol_electronic_structures":
                atomic_numbers = mol.atomic_numbers
                if open_shell:
                    node_labels = [mol.node_y_alpha, mol.node_y_beta]
                else:
                    node_labels = mol.node_y
                energies = mol.energies

            elif dataset_name == "omol_csh_58k":
                atomic_numbers = [mol['z'].numpy() for mol in database]
                positions = [mol['pos'].numpy() for mol in database]
                hamiltonians = [mol['fock'].numpy() for mol in database]
    
    # materials databases with one folder per structure
    else:
        if dataset_name == "cp2k_material":
            periodic_dataset = True

            start_idx = 0
            end_idx = len(database) 
            
            hamiltonians = []
            overlaps = []
            positions = []
            atomic_numbers = []
            energy = []
            forces = []
            periodic_boxes = []
            charges = [0 for i in range(start_idx, end_idx)]
            spins = [1 for i in range(start_idx, end_idx)]

            for data_folder in database[start_idx : end_idx]:

                # parse xyx file
                xyz_file = [f for f in os.listdir(data_folder) if f.endswith('.xyz')][0]
                structure = load_periodic_cp2k_structure(os.path.join(data_folder, xyz_file))
                # print(f"Structure has {len(structure)} atoms and cell dimensions {structure.get_cell()} with PBC {structure.get_pbc()}", flush=True)
                
                atomic_numbers.append(structure.get_atomic_numbers())
                positions.append(structure.get_positions())
                periodic_boxes.append(structure.get_cell())

                # find the Hamiltonian file of type *..-KS_Spin_1-1_0.csr:
                hamiltonian_file = [f for f in os.listdir(data_folder) if '-KS_SPIN_1-1_0' in f or 'H.csr' in f][0]
                print(f"Loading Hamiltonian from {hamiltonian_file}...", flush=True)
                hamiltonian = read_cp2k_matrix(os.path.join(data_folder, hamiltonian_file), dtype=dtype)
                # print(f"Hamiltonian loaded with shape {hamiltonian.shape} and {hamiltonian.nnz} non-zero elements", flush=True)

                overlap_file = [f for f in os.listdir(data_folder) if '-S_SPIN_1-1_0' in f][0]
                overlap = read_cp2k_matrix(os.path.join(data_folder, overlap_file))
                # print(f"Overlap loaded with shape {overlap.shape} and {overlap.nnz} non-zero elements", flush=True)

                shift_fermi = True
                if shift_fermi:
                    print(f"Shifting Hamiltonian by Fermi energy...", flush=True)
                    out_file = [f for f in os.listdir(data_folder) if f.endswith('.out') and 'slurm' not in f][0]
                    mu = get_fermi_energy(os.path.join(data_folder, out_file))
                    print(f"Fermi energy: {mu}", flush=True)

                    # Apply gauge transformation: H' = H - mu * S
                    hamiltonian = hamiltonian - mu * overlap

                hamiltonians.append(hamiltonian)

        else:
            print("Unknown database!")

    # Set up the graph targets:
    print(f"Setting up graph targets for {len(atomic_numbers)} molecules...", flush=True)
    graph_targets = fock_targets_batched.Fock_Targets(atomic_numbers, positions, rcut, orbital_basis, hamiltonians, 
                                                        dtype=dtype, 
                                                        dataset_name=dataset_name,
                                                        scale_shift_data=None,
                                                        periodic_boxes=periodic_boxes if periodic_dataset else None,
                                                        tiling_dims=None)
        
    # Get rid of the molecule dimension:
    node_labels_tensors = [torch.as_tensor(x)[0] for x in graph_targets.node_labels_list]
    atomic_numbers_tensors = [torch.as_tensor(x).flatten() for x in atomic_numbers]
    node_labels = torch.cat(node_labels_tensors, dim=0)       # Shape: [total_atoms, 676]  (e.g., [2088, 676])
    atomic_numbers = torch.cat(atomic_numbers_tensors, dim=0) # Shape: [total_atoms]       (e.g., [2088])

    # Get the locations of the l=0 irreps:
    required_irreps = graph_targets.req_output_irreps

    scalar_indices = []
    irrep_track = 0
    for _, irrep in required_irreps:
        l = irrep.l
        if l == 0:
            scalar_indices.append(irrep_track)
        irrep_track += 2 * l + 1


    # 3. Compute the scale and shift for each atomic number and irrep degree
    for atom_ind, atomic_number in enumerate(atomic_numbers):
        print(f"Processing atom {atom_ind+1}/{len(atomic_numbers)} with atomic number {atomic_number.item()}...", flush=True)
        atomic_number = int(atomic_number.item())

        if open_shell:
            if atomic_number not in element_scalar_values_alpha:
                element_scalar_values_alpha[atomic_number] = []
                element_scalar_values_beta[atomic_number] = []

            orbital_onsite_scalars_alpha = node_labels[0][atom_ind][scalar_indices]
            orbital_onsite_scalars_beta = node_labels[1][atom_ind][scalar_indices]

            element_scalar_values_alpha[atomic_number].append(orbital_onsite_scalars_alpha)
            element_scalar_values_beta[atomic_number].append(orbital_onsite_scalars_beta)
        else:
            node_block = node_labels[atom_ind] # remove spin dimension
            if atomic_number not in element_scalar_values:
                element_scalar_values[atomic_number] = []

            orbital_onsite_scalars = node_block[scalar_indices]
            element_scalar_values[atomic_number].append(orbital_onsite_scalars)

    time_end = time.perf_counter()
    print(f"Time to extract node irreps for all molecules: {time_end - time_start} seconds", flush=True)

    if open_shell:
        # print(f"Element scalar values [alpha]: {element_scalar_values_alpha}")
        # print(f"Element scalar values [beta]: {element_scalar_values_beta}")
        element_scalar_values_alpha = {k: element_scalar_values_alpha[k] for k in sorted(element_scalar_values_alpha.keys())}
        element_scalar_values_beta = {k: element_scalar_values_beta[k] for k in sorted(element_scalar_values_beta.keys())}
    else:
        # print(f"Element scalar values: {element_scalar_values}")
        element_scalar_values = {k: element_scalar_values[k] for k in sorted(element_scalar_values.keys())}

    # print keys on every rank:
    # dist.barrier()
    # print(f"Rank {rank} - Element scalar values keys: {list(element_scalar_values.keys())}", flush=True)
    # dist.barrier()

    rank = 0
    if open_shell:
        combined_element_scalar_values_alpha = element_scalar_values_alpha
        combined_element_scalar_values_beta = element_scalar_values_beta
    else:
        combined_element_scalar_values = element_scalar_values

    if rank == 0:

        if open_shell:
            # get the mean/std per element
            element_scalar_means_alpha = {}
            element_scalar_means_beta = {}
            element_scalar_stds_alpha = {}
            element_scalar_stds_beta = {}

            for Z, tensor_list in combined_element_scalar_values_alpha.items():

                # Stack into a single 2D tensor: shape [num_molecules, num_scalars_per_atom]
                stacked = torch.stack(tensor_list)  # shape: [N, 6] for example with H2O

                means = stacked.mean(dim=0)  # shape: [6]
                stds = stacked.std(dim=0, unbiased=False)

                # Fix always-zero positions - revisit this
                threshold = 1e-4
                zero_mask = (means == 0.0)
                means[zero_mask] = 0.0
                zero_mask = (stds < threshold)
                stds[zero_mask] = 1.0

                element_scalar_means_alpha[Z] = means.tolist()
                element_scalar_stds_alpha[Z] = stds.tolist()

            for Z, tensor_list in combined_element_scalar_values_beta.items():

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

                element_scalar_means_beta[Z] = means.tolist()
                element_scalar_stds_beta[Z] = stds.tolist()

            print(f"Indices of scalar components: {scalar_indices}")
            print(f"Element scalar means (averaged) [alpha]: {element_scalar_means_alpha}")
            print(f"Element scalar means (averaged) [beta]: {element_scalar_means_beta}")
            print(f"Element scalar stds (averaged) [alpha]: {element_scalar_stds_alpha}")
            print(f"Element scalar stds (averaged) [beta]: {element_scalar_stds_beta}")

            scale_shift_data = {
                "element_scalar_means_alpha": element_scalar_means_alpha,  # dict[int -> list[float]]
                "element_scalar_means_beta": element_scalar_means_beta,  # dict[int -> list[float]]
                "element_scalar_stds_alpha": element_scalar_stds_alpha,    # dict[int -> list[float]]
                "element_scalar_stds_beta": element_scalar_stds_beta,    # dict[int -> list[float]]
                "scalar_irrep_indices": scalar_indices         # list[int]
            }
            torch.save(scale_shift_data, "./fock_utils/"+filename)
            print("Saved scale_shift_data to ./fock_utils/"+filename, flush=True)

        else:
            # get the mean/std per element
            element_scalar_means = {}
            element_scalar_stds = {}

            for Z, tensor_list in combined_element_scalar_values.items():

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
            directory = os.path.join(os.path.dirname(__file__), "../fock_utils")
            torch.save(scale_shift_data, os.path.join(directory, filename))
            print("Saved scale_shift_data to maloq/fock_utils/"+filename, flush=True)

    return scale_shift_data


def get_fermi_energy(out_file_path):
    patterns = [
        r'Fermi Energy \[eV\] :\s*([-+]?\d*\.\d+)',  # Matches "Fermi Energy [eV] :   -4.747024"
        r'Fermi energy:\s*([-+]?\d*\.\d+)',          # Matches "Fermi energy: -4.747024"
        r'Fermi level:\s*([-+]?\d*\.\d+)',           # Matches "Fermi level: -4.747024"
    ]
    with open(out_file_path, 'r') as f:
        for line in f:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    # convert to hartree if it's in eV:
                    if 'eV' in pattern:
                        print(f"Fermi energy found in eV: {match.group(1)}. Converting to Hartree...", flush=True)
                        return float(match.group(1)) / 27.211386245988
                    return float(match.group(1))
    raise ValueError(f"Fermi energy not found in {out_file_path}")
    



# --------------------------------------------
# Energy reference functions
# --------------------------------------------

def compute_energy_references(batch, tensor, elem_refs, operation="subtract"):
    """
    Apply element-wise energy references to molecular energies.

    Args:
        batch: Batch object containing atomic_numbers and batch indices
        tensor: Energy tensor to modify (shape: [num_molecules] or [num_molecules, 1])
        elem_refs: Element reference tensor (shape: [max_atomic_number])
        operation: "subtract" or "add"

    Returns:
        Modified energy tensor
    """

    assert tensor.shape[0] == len(torch.unique(batch.batch))

    with torch.autocast(elem_refs.device.type, enabled=False):
        # Ensure all tensors are on the same device
        device = tensor.device
        elem_refs = elem_refs.to(device)
        batch_indices = batch.batch.to(device)
        atomic_numbers = batch.atomic_numbers.to(device)

        # Get atom references - this is the source tensor
        atom_refs = elem_refs[atomic_numbers]

        # Create refs tensor with same shape as input tensor
        refs = torch.zeros(tensor.shape, dtype=elem_refs.dtype, device=device)

        # Handle dimension mismatch if atom_refs is 2D but batch_indices is 1D
        if atom_refs.dim() > batch_indices.dim():
            # Flatten atom_refs to match batch_indices dimension
            atom_refs = atom_refs.flatten()

        refs = refs.scatter_reduce(
            0,
            batch_indices,  # Maps atoms to molecules
            atom_refs,      # Reference for each atom
            reduce="sum",
        )

        if operation == "subtract":
            return tensor - refs
        elif operation == "add":
            return tensor + refs
        else:
            raise ValueError(f"Unknown operation: {operation}")

def apply_energy_refs(batch, tensor, element_references, operation="subtract"):
    """Apply energy reference subtraction to a batch of energies."""
    if element_references is None:
        return tensor

    # Print statistics before scaling
    # original_energies = tensor.clone()
    # print(f"Before energy reference scaling:")
    # print(f"  Average energy: {original_energies.mean().item():.6f} Hartree")
    # print(f"  Energy std: {original_energies.std().item():.6f} Hartree")
    # print(f"  Energy range: [{original_energies.min().item():.6f}, {original_energies.max().item():.6f}] Hartree")

    # Apply scaling
    scaled_energies = compute_energy_references(batch, tensor, element_references, operation=operation)

    # Print statistics after scaling
    # print(f"After energy reference scaling:")
    # print(f"  Average energy: {scaled_energies.mean().item():.6f} Hartree")
    # print(f"  Energy std: {scaled_energies.std().item():.6f} Hartree")
    # print(f"  Energy range: [{scaled_energies.min().item():.6f}, {scaled_energies.max().item():.6f}] Hartree")
    # print(f"  Energy change: {(scaled_energies.mean() - original_energies.mean()).item():.6f} Hartree")

    return scaled_energies