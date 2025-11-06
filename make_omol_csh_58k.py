import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.db import connect
from pathlib import Path
from ase.visualize import view

import torch
import torch.distributed as dist
from fock_utils import utils_orca_out
import time
import argparse

parser = argparse.ArgumentParser(description='Create a Hamiltonian dataset from ORCA output files')
parser.add_argument('-f', '--structures-dir', type=str, required=True, help='Path to the directory where the orca folders are')
parser.add_argument('-o', '--output-db-name', type=str, required=True, help='Name of the output database file')
parser.add_argument('-m', '--max-structures', type=str, default=100000, help='Max number of structures per gpu')
args = parser.parse_args()

print("Getting structures")
structures_dir = args.structures_dir
structure_folders = [f for f in os.listdir(structures_dir)
                    if len(os.listdir(os.path.join(structures_dir, f))) > 0 and
                    os.path.isdir(os.path.join(structures_dir, f)) ]
orca_file = 'orca.out'
num_local_structures = int(args.max_structures) # impose only making a subset
print("Got structures")

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
fock_matrices = []
orca_output_list = []
big_time_start = time.perf_counter()
for folder_idx, structure_folder in enumerate(structure_folders[folder_start_idx:folder_end_idx]):
    if folder_idx >= num_local_structures:
        print("Reached max set structure per rank, exiting")
        break

    try:
        print(f"Rank {rank} of {world_size} is working on folder {structure_folder}", flush=True)
        time_start = time.perf_counter()

        orca_output_filepath = os.path.join(structures_dir, structure_folder, orca_file)
        local_folder_name_strings.append(structure_folder)

        # Get data:
        parse_time_start = time.perf_counter()
        parsed_orca_output = utils_orca_out.parse_output(Path(orca_output_filepath), source='manasakani')
        parse_time_end = time.perf_counter()
        print("Time to parse orca output: ", parse_time_end - parse_time_start, flush=True)

        fock_time_start = time.perf_counter()
        fock_matrix, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath)
        fock_time_end = time.perf_counter()
        print("Time to make fock matrix: ", fock_time_end - fock_time_start, flush=True)

        # Make structure:
        structure_time_start = time.perf_counter()
        structure = Atoms(elements, positions=coordinates)
        structure_time_end = time.perf_counter()
        print("Time to make structure: ", structure_time_end - structure_time_start, flush=True)
        print("Structure: ", structure, flush=True)

        flatten_time_start = time.perf_counter()
        assert fock_matrix.ndim > 0
        assert fock_matrix.size > 0
        flattened_fock = fock_matrix[np.triu_indices_from(fock_matrix)]
        flatten_time_end = time.perf_counter()
        print("Time to flatten fock: ", flatten_time_end - flatten_time_start, flush=True)

        orca_output_list.append(parsed_orca_output)
        structures.append(structure)
        fock_matrices.append(flattened_fock)

        time_end = time.perf_counter()
        print("Total time for one structure: ", time_end - time_start, flush=True)

    except Exception as e:
            print(f"ERROR: Job {args.job_id} skipping structure {structure_folder} due to error: {str(e)}", flush=True)
            skipped_structures.append(structure_folder)
            continue

big_time_end = time.perf_counter()
num_targets_made = min(world_size * num_local_structures, local_num_folders)
print(f"Time to make {num_targets_made} targets: {big_time_end - big_time_start}", flush=True)

# ----------------------------
# --> make an ASE DB:
# ----------------------------
for current_rank in range(world_size):
    if rank == current_rank:
        with connect(args.output_db_name) as structure_db:
            for i, (local_folder_name, orca_output_dict, atoms, fock) in enumerate(zip(local_folder_name_strings, orca_output_list, structures, fock_matrices)):

                data = {
                    "total_energy [Eh]": orca_output_dict["total_energy [Eh]"],
                    "gradient [Eh/bohr]": orca_output_dict["gradient [Eh/bohr]"],
                    "Hamiltonian [Eh]": fock,
                    "omol_folder_name": local_folder_name
                }

                print(f"Writing structure {i}")
                structure_db.write(atoms, data=data)
    dist.barrier()

print("done!")
