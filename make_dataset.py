import os
import numpy as np
import matplotlib.pyplot as plt
# from ase.visualize import view
from ase import Atoms
from ase.db import connect
from pathlib import Path
from ase.visualize import view

import torch
import torch.distributed as dist
from fock_utils import utils_orca_out, utils_tensor_decomp, fock_targets, basis_sets
import time
import argparse

parser = argparse.ArgumentParser(description='Create a dataset from ORCA output files')
parser.add_argument('-f', '--structures-dir', type=str, required=True, help='Path to the directory where the orca folders are')
parser.add_argument('-o', '--output-db-name', type=str, required=True, help='Name of the output database file')
parser.add_argument('-m', '--max-structures', type=str, default=100000, help='Max number of structures per gpu')

args = parser.parse_args()

# --> get all non-empty folders inside structures_dir
structures_dir = args.structures_dir 
structure_folders = [f for f in os.listdir(structures_dir) 
                    if len(os.listdir(os.path.join(structures_dir, f))) > 0 and 
                    os.path.isdir(os.path.join(structures_dir, f)) ]
orca_file = 'orca.out'
cutoff = 5.0            
num_local_structures = int(args.max_structures) # use to impose only making a subset
dataset_name = 'omol' 

# --> whether to scale and shift scalar values in the node blocks of the dataset (scale_shift_file needs to be precomputed)
scale_and_shift = True
scale_shift_file = 'element_scale_shifts_' + dataset_name + '.pt'

# Orbital basis for the omol tzvpd dataset:
full_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
full_basis = {k: sorted(v) for k, v in full_basis.items()} # The basis must be in l-major!!!

# ----------------------------
# --> Initialize compute setup
# ----------------------------
gpu_id = 0
world_size = int(os.environ['SLURM_NTASKS'])
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
torch.cuda.set_device(gpu_id)

total_num_folders = len(structure_folders)
local_num_folders = total_num_folders //world_size
counts = np.array([local_num_folders]*world_size, dtype=np.int32)
for i in range(total_num_folders % world_size):
    counts[i] += 1

displacements = np.zeros_like(counts)
for i in range(1, len(counts)):
    displacements[i] = displacements[i-1] + counts[i-1]

folder_start_idx = displacements[rank]
folder_end_idx = displacements[rank] + counts[rank]
local_num_folders = counts[rank]
print(f"Processing {total_num_folders} structures between {world_size} GPUs", flush=True)

# ----------------------------
# --> Make the structures
# ----------------------------
local_folder_name_strings = []
structures = []
orca_output_list = []
big_time_start = time.perf_counter()
for folder_idx, structure_folder in enumerate(structure_folders[folder_start_idx:folder_end_idx]):
    if folder_idx >= num_local_structures:
        print("Reached max set structure per rank, exiting")
        break    

    print(f"Rank {rank} of {world_size} is working on folder {structure_folder}", flush=True)
    time_start = time.perf_counter()
    orca_output_filepath = os.path.join(structures_dir, structure_folder, orca_file)
    local_folder_name_strings.append(structure_folder)

    # General ORCA output file elements:
    parsed_orca_output = utils_orca_out.parse_output(Path(orca_output_filepath), source='manasakani')
    orca_output_list.append(parsed_orca_output)

    # Atomic and electronic structure:
    read_time_start = time.perf_counter()
    fock_matrix, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath) 
    #NOTE: The basis returned by utils_orca_out (taken from the output file) is not in the right order for the diffuse functions! So we don't use it directly.

    # Get basis (for this structure) in the correct l-order for rearranging matrix:
    basis = {element: basis_sets.def2_tzvpd[utils_orca_out.periodic_table_number[element]] for element in elements} # not in l-major order yet
    fock_matrix = utils_orca_out.sort_by_m(fock_matrix, basis, np.array(elements))  # Re-arrange matrix blocks to yzx notation (m=0 is in the middle)
    fock_matrix = utils_orca_out.sort_by_l(fock_matrix, basis, np.array(elements))  # Shift into l-major (in case of diffuse functions)
    basis = {k: sorted(v) for k, v in basis.items()} # now the basis is in l-major order

    ### Display the fock matrix:
    # plt.imshow(fock_matrix, cmap='viridis', interpolation='nearest', vmin=-0.5, vmax=0.5)
    # plt.colorbar()
    # plt.savefig(f"fock_matrix_{structure_folder}.png", bbox_inches='tight', dpi=300)
    # plt.close()

    print("basis: ", basis)
    # print("full basis: ", full_basis)

    structure = Atoms(elements, positions=coordinates)  
    read_time_end = time.perf_counter()
    print("Time to make atoms and get matrix: ", read_time_end - read_time_start, flush=True)
    print("Structure: ", structure)

    # Create fock targets:
    target_time_start = time.perf_counter()

    if scale_and_shift:
        print("Getting scale and shift factors...", flush=True)
        print(f"Scale and shift file: {scale_shift_file}", flush=True)
        if scale_shift_file not in os.listdir('./fock_datasets/'):
            print("[Computing element scale and shift factors for the dataset]", flush=True)
            get_scale_shift.get_scale_shift(database, dataset_name, rcut_orbitals, dtype=dtype, reduce_edge=reduce_edge)
        else:
            print("[Loading element scale and shift factors from file]", flush=True)
            scale_shift_data = torch.load('./fock_datasets/' + scale_shift_file)
            scale_shift_data = {
                "element_scalar_means": scale_shift_data["element_scalar_means"],  # dict[int -> list[float]]
                "element_scalar_stds": scale_shift_data["element_scalar_stds"],    # dict[int -> list[float]]
                "scalar_irrep_indices": scale_shift_data["scalar_irrep_indices"]   # list[int]
            }
    else:
        print("Not scaling or shifting the dataset")
        scale_shift_data = None

    structures.append(fock_targets.Fock_Targets(structure, cutoff, full_basis, fock_matrix, reflection_symmetry=False, scale_shift_data=scale_shift_data))
    # structures.append(fock_targets.Fock_Targets(structure, cutoff, basis, fock_matrix, reflection_symmetry=False, scale_shift_data=scale_shift_data)) # minibasis for water! Fix the full basis later
    target_time_end = time.perf_counter()
    print("Time to make targets: ", target_time_end - target_time_start, flush=True)
    
    time_end = time.perf_counter()
    print("Total time for one structure: ", time_end - time_start, flush=True)

big_time_end = time.perf_counter()
num_targets_made = min(world_size * num_local_structures, local_num_folders)
print(f"Time to make {num_targets_made} targets: {big_time_end - big_time_start}", flush=True)

# ----------------------------
# --> make an ASE DB:
# ----------------------------
for current_rank in range(world_size):
    if rank == current_rank:
        with connect(args.output_db_name) as structure_db:
            print("rank ", rank, "is writing stuff")
            for i, (orca_output_dict, structure) in enumerate(zip(orca_output_list, structures)):
                print(f"Writing structure {i}")
                atoms = structure.atoms
                local_folder_name = local_folder_name_strings[i]

                data = {
                    "pos": structure.atoms.get_positions(),
                    "orbital_basis": structure.orbital_basis,
                    "req_output_irreps": structure.req_output_irreps,
                    "edge_index": structure.neighbour_list,
                    "edge_mask": structure.forward_edge_mask,
                    "reverse_edge_map": structure.reverse_edge_map,
                    "edge_dist": structure.edge_dist.detach().cpu().numpy(),
                    "nedges": len(structure.neighbour_list),
                    "natoms": len(structure.atoms.get_positions()),
                    "atomic_numbers": structure.atomic_numbers,
                    "node_labels": structure.node_labels.detach().cpu().numpy(),
                    "edge_labels": structure.edge_labels.detach().cpu().numpy(),
                    "total_energy [Eh]": orca_output_dict["total_energy [Eh]"],
                    "gradient [Eh/bohr]": orca_output_dict["gradient [Eh/bohr]"],
                    "total_charge": orca_output_dict["total_charge"],
                    "multipoles": orca_output_dict["multipoles"],
                    "cutoff": cutoff,
                    "required_irreps": str(structure.req_output_irreps),
                    "num_atoms_in_molecule": len(structure.atomic_numbers),
                    "folder_name": local_folder_name
                }
                structure_db.write(atoms, data=data)
    dist.barrier()
    
print("done!")

# visualization:
# from ase.visualize import view
# view(water_cluster, viewer='x3d')
