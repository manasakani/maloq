import torch
import numpy as np

from fock_utils import utils_orca_out, fock_targets, basis_sets, utils_tensor_decomp
from ase import Atoms
from .get_loader import orbital_basis_def2_svp_nabla, orbital_basis_def2_svp_QM7
import matplotlib.pyplot as plt
from e3nn.o3 import Irreps
import time
import copy

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import torch.distributed as dist

def get_scale_shift(database, dataset_name, rcut=5.0, dtype=torch.float32, reduce_edge=False, rank=0, filename='scale_shifts.pt'):
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
    else: 
        print("Unknown dataset name!")
    
    # orbital_basis = {k: sorted(v) for k, v in orbital_basis.items()} # If need to sort by l (but this is not needed)
    orbital_basis = dict(sorted(orbital_basis.items(), key=lambda item: len(item[1]), reverse=True)) # put elements with the largest basis first

    # 0. Compute locations of scalars and higher ranks from required_irreps for this dataset's basis:
    _, required_irreps, simplified_out_irreps, _, _, _, _ = utils_tensor_decomp.make_output_irreps(orbital_basis)  
    required_irreps = Irreps(required_irreps)
    scalar_indices = []
    irrep_track = 0
    for _, irrep in required_irreps:
        if irrep.l == 0:
            scalar_indices.append(irrep_track)
        irrep_track += 2 * irrep.l + 1
    
    # Fock matrix analysis parameters:
    equivariant_blocks = None
    orbital_starts = None
    basis_transformation = None

    # Extract the magnitudes of those irreps to make scaling/shifting factors
    element_scalar_values = {}
    for i in range(num_molecules):
        mol = database[i]

        print("working on molecule", i, "of", num_molecules)
        time_start = time.perf_counter()
    
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

        else: 
            print("Unknown database!")
        
        # if the dataset is not omol, we need to create the atomic structure and set up the graph targets
        if dataset_name != "omol":
        
            # 1. Make the atomic structure
            mol_atoms = Atoms(symbols=atomic_numbers, positions=positions)

            # 2. Set up the Graph targets 
            if dataset_name == "QM7":                 
                hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)      # QM7 comes in zxy coordinates from ORCA, so need to rotate 
            
            graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian, dtype=dtype, half_edges=reduce_edge, 
                                                    equivariant_blocks=equivariant_blocks,
                                                    orbital_starts=orbital_starts,
                                                    basis_transformation=basis_transformation)

            # Save the analysis objects to use for the next structure (these depend only on the basis)
            equivariant_blocks = graph_targets.equivariant_blocks
            orbital_starts = graph_targets.orbital_starts
            basis_transformation = graph_targets.basis_transformation

            node_labels = graph_targets.node_labels
        
        # 3. Compute the scale and shift for each atomic number
        for atomic_number, node_block in zip(atomic_numbers, node_labels):
            atomic_number = int(atomic_number.item())
            if atomic_number not in element_scalar_values:
                element_scalar_values[atomic_number] = []

            orbital_onsite_scalars = node_block[scalar_indices]
            element_scalar_values[atomic_number].append(orbital_onsite_scalars)
        
        time_end = time.perf_counter()
        print(f"Time to scale nodes for molecule {i}: {time_end - time_start} seconds", flush=True)

    # print(f"Element scalar values: {element_scalar_values}")

    # sort the keys in increasing order:
    element_scalar_values = {k: element_scalar_values[k] for k in sorted(element_scalar_values.keys())}

    # print keys on every rank:
    # dist.barrier()
    print(f"Rank {rank} - Element scalar values keys: {list(element_scalar_values.keys())}", flush=True)
    # dist.barrier()

    # if distributed, allgather the element_scalar_values dictionary 
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"Rank {rank} - Allgathering element_scalar_values from all ranks...", flush=True)
        gathered_data = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_data, element_scalar_values)
        combined_element_scalar_values = {}
        if rank == 0:
            # Combine the gathered dictionaries
            for data in gathered_data:
                for key, value in data.items():
                    if key not in combined_element_scalar_values:
                        combined_element_scalar_values[key] = []
                    combined_element_scalar_values[key].extend(value)
            print(f"Rank {rank} - Combined element_scalar_values keys: {list(element_scalar_values.keys())}", flush=True)

        # sort the keys in increasing order
        combined_element_scalar_values = {k: combined_element_scalar_values[k] for k in sorted(combined_element_scalar_values.keys())}
    else:
        rank = 0
        combined_element_scalar_values = element_scalar_values
    
    if rank == 0:
    
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
        torch.save(scale_shift_data, "./fock_datasets/"+filename)
        print("Saved scale_shift_data to ./fock_datasets/"+filename, flush=True)


def scale_shift_database(database, start_mol, end_mol, rcut_orbitals, orbital_basis, reduce_edge, scale_shift_data, scale_nodes=False, train_or_eval='train'):
    """
    Scale and shift the node labels in the database using the scale_shift_data
    """

    # For analysis only (not reconstruction of the matrix), we just create a sample structure and fock target object
    print("Making analysis fock target object for the first molecule in the database", flush=True)
    sample_structure = Atoms(symbols=database[start_mol].atomic_numbers, positions=database[start_mol].pos)
    sample_fock_target_object = fock_targets.Fock_Targets(
        sample_structure, rcut_orbitals, orbital_basis, fock_matrix=None,
        half_edges=reduce_edge, scale_shift_data=scale_shift_data
    )

    data_list = []
    for i in range(start_mol, end_mol):
        data_obj = database[i]
        data_obj.fock_target_object = sample_fock_target_object

        # If running evaluation, we need to create a structure-dependent fock target object
        if train_or_eval == 'eval':
            print("Making fock analysis object for molecule", i, flush=True)
            structure = Atoms(symbols=data_obj.atomic_numbers, positions=data_obj.pos)
            fock_target_object = fock_targets.Fock_Targets(
                structure, rcut_orbitals, orbital_basis, fock_matrix=None,
                half_edges=reduce_edge, scale_shift_data=scale_shift_data
            )
            data_obj.fock_target_object = fock_target_object

        if scale_nodes:
            print(f"Scaling and shifting the node labels in database[{i}]", flush=True)
            scaled_data_obj = copy.deepcopy(data_obj)
            scaled_data_obj.node_y = data_obj.fock_target_object.scale_shift_node_blocks(data_obj.node_y)
            data_obj = scaled_data_obj

        data_list.append(data_obj)
    
    return data_list


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

def apply_energy_refs(batch, tensor, element_references):
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
    scaled_energies = compute_energy_references(batch, tensor, element_references, operation="subtract")
    
    # Print statistics after scaling
    # print(f"After energy reference scaling:")
    # print(f"  Average energy: {scaled_energies.mean().item():.6f} Hartree")
    # print(f"  Energy std: {scaled_energies.std().item():.6f} Hartree")
    # print(f"  Energy range: [{scaled_energies.min().item():.6f}, {scaled_energies.max().item():.6f}] Hartree")
    # print(f"  Energy change: {(scaled_energies.mean() - original_energies.mean()).item():.6f} Hartree")
    
    return scaled_energies
