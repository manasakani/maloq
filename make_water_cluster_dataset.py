import os
import numpy as np
import matplotlib.pyplot as plt

from ase.visualize import view
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.db import connect

import torch
import torch.distributed as dist
import utils_orca_out, utils_tensor_decomp, fock_targets
import time

# --> get all non-empty folders inside water_clusters_dir
water_clusters_dir = '/home/manasakani/water_clusters'
water_cluster_folders = [f for f in os.listdir(water_clusters_dir) 
                        if len(os.listdir(os.path.join(water_clusters_dir, f))) > 0 and 
                        os.path.isdir(os.path.join(water_clusters_dir, f)) ]
orca_file = 'orca.out'

cutoff = 6.0            
num_local_structures = 2 # use to impose only making a subset

# --> initialize compute setup
gpu_id = 0
world_size = int(os.environ['SLURM_NTASKS'])
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
torch.cuda.set_device(gpu_id)

total_num_folders = len(water_cluster_folders)
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
big_time_start = time.perf_counter()
for folder_idx, water_cluster_folder in enumerate(water_cluster_folders[folder_start_idx:folder_end_idx]):
    if folder_idx >= num_local_structures:
        print("Reached max set structure per rank, exiting")
        break

    print(f"Rank {rank} of {world_size} is working on folder {water_cluster_folder}", flush=True)
    time_start = time.perf_counter()

    # Atomic structure
    read_time_start = time.perf_counter()
    fock_matrix, elements, coordinates, basis = utils_orca_out.read_orca_out(os.path.join(water_clusters_dir, water_cluster_folder, orca_file))
    fock_matrix = utils_orca_out.sort_by_m(fock_matrix, basis, elements)
    water_cluster = Atoms(elements, positions=coordinates)  
    read_time_end = time.perf_counter()
    print("Time to make atoms and get matrix: ", read_time_end - read_time_start, flush=True)

    # Connectivity list:
    num_atoms = len(elements)
    neighbours = NeighborList(np.ones(num_atoms)*cutoff, skin=0, self_interaction=False, bothways=True)
    neighbours.update(water_cluster)
    neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
    neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])

    # Targets:
    target_time_start = time.perf_counter()
    structures.append(fock_targets.Fock_Targets(water_cluster, neighbour_list, basis, fock_matrix))
    target_time_end = time.perf_counter()
    print("Time to make targets: ", target_time_end - target_time_start, flush=True)
    
    time_end = time.perf_counter()
    print("Total time for one structure: ", time_end - time_start, flush=True)

big_time_end = time.perf_counter()
print(f"Time to make {local_num_folders} targets: {big_time_end - big_time_start}", flush=True)

# --> make an ASE DB:
if rank == 0:
    all_structures = [None] * world_size
    dist.gather_object(structures, all_structures, dst=0)
    all_structures = [item for sublist in all_structures for item in sublist]
    # print(f"Rank {rank} to write: {len(all_structures)} structures", flush=True)
else:
    dist.gather_object(structures, dst=0)

if rank == 0:
    with connect("water_clusters_rcut_6.0_"+str(world_size)+"gpus.db") as structure_db:
        for i, structure in enumerate(all_structures):
            print(f"Writing structure {i}")
            atoms = structure.atoms
            data = {
                'neighbour_list': structure.neighbour_list,
                'orbital_basis': structure.orbital_basis,
                'fock_matrix': structure.fock_matrix.detach().cpu().numpy(),
                'node_labels': structure.node_labels.detach().cpu().numpy(),
                'edge_labels': structure.edge_labels.detach().cpu().numpy(),
                'edge_dist': structure.edge_dist.detach().cpu().numpy(),
            }
            structure_db.write(atoms, data=data)


# if rank == 0:
#     structure_db = connect("water_clusters_rcut_6.0_"+str(world_size)+"gpus.db")
#     for i, structure in enumerate(all_structures):
#         print(f"Writing structure {i}")
#         atoms = structure.atoms
#         data = {
#             'neighbour_list': structure.neighbour_list,
#             'orbital_basis': structure.orbital_basis,
#             'fock_matrix': structure.fock_matrix.detach().cpu().numpy(),
#             'node_labels': structure.node_labels.detach().cpu().numpy(),
#             'edge_labels': structure.edge_labels.detach().cpu().numpy(),
#             'edge_dist': structure.edge_dist.detach().cpu().numpy(),
#         }
#         structure_db.write(atoms, data=data)



# visualization:
# view(water_cluster, viewer='x3d')