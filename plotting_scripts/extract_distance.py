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

print("Extracting interatomic distances from structures...", flush=True)

parser = argparse.ArgumentParser(description='Create a dataset from ORCA output files')
parser.add_argument('-f', '--structures-dir', type=str, required=True, help='Path to the directory where the orca folders are')
parser.add_argument('-m', '--max-structures', type=str, default=100000, help='Max number of structures per gpu')

args = parser.parse_args()

# --> get all non-empty folders inside structures_dir
structures_dir = args.structures_dir 
structure_folders = [f for f in os.listdir(structures_dir) 
                    if len(os.listdir(os.path.join(structures_dir, f))) > 0 and 
                    os.path.isdir(os.path.join(structures_dir, f)) ]
orca_file = 'orca.out'
cutoff = 10.0            
num_local_structures = int(args.max_structures) # use to impose only making a subset
dataset_name = 'omol' 

# --> whether to scale and shift scalar values in the node blocks of the dataset (scale_shift_file needs to be precomputed)
scale_and_shift = False
scale_shift_file = 'element_scale_shifts_' + dataset_name + '.pt'

# Collect and process orbital basis for the omol tzvpd dataset:
full_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
full_basis = {k: sorted(v) for k, v in full_basis.items()} # The basis must be in l-major!!!
full_basis = dict(sorted(full_basis.items(), key=lambda item: len(item[1]), reverse=True)) # put elements with the largest basis first - this is important!!!

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
interatomic_distances = []
largest_atomic_number = max(full_basis.keys())
element_interaction_matrix = torch.zeros((largest_atomic_number, largest_atomic_number), dtype=torch.int32)
print("Largest atomic number in the basis: ", largest_atomic_number, flush=True)

big_time_start = time.perf_counter()
for folder_idx, structure_folder in enumerate(structure_folders[folder_start_idx:folder_end_idx]):
    if folder_idx >= num_local_structures:
        print("Reached max set structure per rank, exiting")
        break    

    print(f"Rank {rank} of {world_size} is working on folder {structure_folder} with index {folder_idx}", flush=True)
    time_start = time.perf_counter()
    orca_output_filepath = os.path.join(structures_dir, structure_folder, orca_file)

    # General ORCA output file elements:
    # parsed_orca_output = utils_orca_out.parse_output(Path(orca_output_filepath), source='manasakani')
    # orca_output_list.append(parsed_orca_output)

    # Atomic and electronic structure:
    read_time_start = time.perf_counter()
    fock_matrix, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath) 
    #NOTE: The basis returned by utils_orca_out (taken from the output file) is not in the right order for the diffuse functions! So we don't use it directly.

    structure = Atoms(elements, positions=coordinates)  
    read_time_end = time.perf_counter()

    # get all the interatomic distances in the structure:
    coordinates = structure.get_positions()
    distances_between_atoms = np.linalg.norm(
        coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1
    )  
    num_atoms = coordinates.shape[0]
    i_indices, j_indices = np.triu_indices(num_atoms, k=1)
    scalar_distances = distances_between_atoms[i_indices, j_indices]
    interatomic_distances.append(scalar_distances)

    print("Time to make atoms and get matrix: ", read_time_end - read_time_start, flush=True)
    print("Structure: ", structure, flush=True)

    # populate the element interaction matrix with every pair of elements:
    for i, el_i in enumerate(elements):
        for j, el_j in enumerate(elements):
            # print("adding interaction for elements: ", el_i, el_j, flush=True)
            element_interaction_matrix[el_i-1, el_j-1] += 1
    
    time_end = time.perf_counter()
    print("Total time for one structure: ", time_end - time_start, flush=True)

# gather the interatomic distances to rank 0 (they are different lengths):
# interatomic_distances = dist.all_gather_object(interatomic_distances, group=None)
gathered = [None for _ in range(world_size)]
dist.all_gather_object(gathered, interatomic_distances)

# convert interatomic distances to tensors and gather them on rank 0:
if rank == 0:
    print("Gathering interatomic distances to rank 0...", flush=True)

    # Flatten all arrays from all ranks into one long array
    all_arrays = []
    for rank_list in gathered:
        for arr in rank_list:
            all_arrays.append(np.asarray(arr).flatten())
    all_distances = np.concatenate(all_arrays)
    print("Interatomic distances shape: ", all_distances.shape, flush=True)

    # print interatomic distances to a file:
    output_file = 'omol_interatomic_distances.txt'
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, all_distances)))


dist.all_reduce(element_interaction_matrix, op=dist.ReduceOp.SUM)
if rank == 0:
    # make element interaction matrix into one tensor:
    print("Element interaction matrix shape: ", element_interaction_matrix.shape, flush=True)

    # print element interaction matrix to a file:
    output_file = 'omol_element_interaction_matrix.txt'
    with open(output_file, 'w') as f:
        for row in element_interaction_matrix.numpy():
            f.write(' '.join(map(str, row)) + '\n')