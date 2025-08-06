import torch
import numpy as np
import multiprocessing as mp

from fock_utils import utils_orca_out, fock_targets
from ase import Atoms
from .get_loader import orbital_basis_def2_svp_nabla, orbital_basis_def2_svp_QM7
from .ASEDataset import ASEAtomsData

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import torch.distributed as dist
import matplotlib.pyplot as plt


def dataset_analysis(database, dataset_name, rcut=5.0, dtype=torch.float64, reduce_edge=False, scale_shift_data=None, rank=0):
    """
    Analysizes the fock target node block values for each element in the dataset.
    """

    num_molecules = len(database)

    element_node_block_values = {}
    for i in range(num_molecules):
        mol = database[i]

        print("working on molecule", i, "of", num_molecules, flush=True)

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
            
            graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian, dtype=dtype, reflection_symmetry=reduce_edge, scale_shift_data=scale_shift_data)
            required_irreps = graph_targets.req_output_irreps

            node_labels = graph_targets.node_labels

        # 3. Extract the node block values for each element
        for atomic_number, node_block in zip(atomic_numbers, node_labels):
            atomic_number = int(atomic_number.item())
            if atomic_number not in element_node_block_values:
                element_node_block_values[atomic_number] = []
            
            # append all the nonzero values:
            tol = 1e-8
            node_block = node_block.cpu().numpy()
            nonzero_values = node_block[np.abs(node_block) >= tol]  
            element_node_block_values[atomic_number].append(nonzero_values)
            
    # sort the keys in increasing order:
    element_node_block_values = {k: element_node_block_values[k] for k in sorted(element_node_block_values.keys())}

    # print keys on every rank:
    dist.barrier()  # Ensure all ranks have completed the above loop before printing
    print(f"Rank {rank} keys of element_node_block_values: {element_node_block_values.keys()}")
    dist.barrier()  # Ensure all ranks have printed before proceeding

    # if distributed, gather the element_node_block_values dictionary to rank 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank() 
        world_size = dist.get_world_size()

        print(f"Rank {rank} gathering element_node_block_values...")
        gathered_data = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_data, element_node_block_values)
        if rank == 0:
            total_element_node_block_values = {}
            for data in gathered_data:
                for key, values in data.items():
                    if key not in total_element_node_block_values:
                        total_element_node_block_values[key] = []
                    total_element_node_block_values[key].extend(values)
            print("keys of Gathered dictionary on rank 0:", total_element_node_block_values.keys())

            # sort the keys in increasing order
            total_element_node_block_values = {k: total_element_node_block_values[k] for k in sorted(total_element_node_block_values.keys())}

    else:
        rank = 0

    # Plotting and output stats
    if rank == 0:

        # save the dictionary to a file:
        np.save(f"{dataset_name}_element_node_block_values_scaled.npy", total_element_node_block_values)

        # # make a scatter plot of the node block values in each element, colored by element type:
        # plt.figure(figsize=(5, 4))
        # plt.xlabel("Element")
        # plt.ylabel("Node Block Values")
        # plt.grid(True)
        # index_track = 0
        # element_labels = []
        # for atomic_number, node_blocks in total_element_node_block_values.items():
        #     node_block_values = np.concatenate(node_blocks)
        #     element_name = utils_orca_out.periodic_table_number[atomic_number]
        #     plt.scatter([index_track] * len(node_block_values), node_block_values, s=5.0, alpha=0.50)
        #     index_track += 1
        #     element_labels.append(element_name)
        
        # # plt.yscale('log')
        # plt.xticks(range(len(element_labels)), element_labels)
        # plt.savefig(f"{dataset_name}_node_block_values_rank_{rank}.png", bbox_inches='tight', dpi=300)

        # # violin plot:
        # plt.figure(figsize=(5, 4))
        # plt.xlabel("Element")
        # plt.ylabel("Node Block Values")
        # plt.grid(True)

        # data = []
        # element_labels = []
        # for atomic_number, node_blocks in total_element_node_block_values.items():
        #     node_block_values = np.concatenate(node_blocks)
        #     element_name = utils_orca_out.periodic_table_number[atomic_number]
        #     data.append(node_block_values)
        #     element_labels.append(element_name)

        # plt.violinplot(data, showmeans=False, showmedians=True, widths=0.3, bw_method=0.1)
        # plt.xticks(range(1, len(element_labels) + 1), element_labels)
        # plt.savefig(f"{dataset_name}_node_block_values_violin_rank_{rank}.png", bbox_inches='tight', dpi=300)
    
    print("Done analysis of dataset:", dataset_name, flush=True)