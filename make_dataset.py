import os
import numpy as np
import matplotlib.pyplot as plt

from ase.visualize import view
from ase import Atoms
from ase.db import connect
from pathlib import Path

import torch
import torch.distributed as dist
import utils_orca_out, utils_tensor_decomp, fock_targets
import time
import argparse

parser = argparse.ArgumentParser(description='Create a dataset from ORCA output files')
parser.add_argument('-f', '--structures-dir', type=str, required=True, help='Path to the directory where the orca folders are')
parser.add_argument('-o', '--output-db-name', type=str, required=True, help='Name of the output database file')
parser.add_argument('-m', '--max-structures', type=str, default=100000, help='Max number of structures per gpu')

args = parser.parse_args()

# --> get all non-empty folders inside structures_dir
structures_dir = args.structures_dir #'/home/manasakani/water_clusters'
structure_folders = [f for f in os.listdir(structures_dir) 
                    if len(os.listdir(os.path.join(structures_dir, f))) > 0 and 
                    os.path.isdir(os.path.join(structures_dir, f)) ]
orca_file = 'orca.out'
cutoff = 6.0            
num_local_structures = int(args.max_structures) # use to impose only making a subset

# --> initialize compute setup
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
print(f"Processing {total_num_folders} structures between {world_size} GPUs")

# --> Make the structures
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

    # General ORCA output file elements:
    parsed_orca_output = utils_orca_out.parse_output(Path(orca_output_filepath), source='manasakani')
    orca_output_list.append(parsed_orca_output)

    # Atomic structure
    read_time_start = time.perf_counter()
    fock_matrix, elements, coordinates, basis = utils_orca_out.read_orca_out(orca_output_filepath)
    fock_matrix = utils_orca_out.sort_by_m(fock_matrix, basis, elements)
    structure = Atoms(elements, positions=coordinates)  
    read_time_end = time.perf_counter()
    print("Time to make atoms and get matrix: ", read_time_end - read_time_start, flush=True)

    # Targets:
    target_time_start = time.perf_counter()
    structures.append(fock_targets.Fock_Targets(structure, cutoff, basis, fock_matrix))
    target_time_end = time.perf_counter()
    print("Time to make targets: ", target_time_end - target_time_start, flush=True)
    
    time_end = time.perf_counter()
    print("Total time for one structure: ", time_end - time_start, flush=True)

big_time_end = time.perf_counter()
print(f"Time to make {local_num_folders} targets: {big_time_end - big_time_start}", flush=True)

# --> make an ASE DB:
with connect(args.output_db_name) as structure_db:
    for current_rank in range(world_size):
        if rank == current_rank:
            print("rank ", rank, "is writing stuff")
            for i, (orca_output_dict, structure) in enumerate(zip(orca_output_list, structures)):
                print(f"Writing structure {i}")
                atoms = structure.atoms

                data = {
                    "pos": structure.atoms.get_positions(),
                    "orbital_basis": structure.orbital_basis,
                    "req_output_irreps": structure.req_output_irreps,
                    "edge_index": structure.neighbour_list,
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
                    "required_irreps": str(structure.req_output_irreps)
                }
                structure_db.write(atoms, data=data)
print("done")

# visualization:
# from ase.visualize import view
# view(water_cluster, viewer='x3d')