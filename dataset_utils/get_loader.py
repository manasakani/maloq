import torch
import numpy as np

from fock_utils import utils_orca_out, fock_targets
from dataset_utils.ASEDataset import ASEDataset, ASEAtomsData, sampleDataset
from ase import Atoms

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import torch.distributed as dist

orbital_basis_def2_svp = {35: [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], 
                          17: [0, 0, 0, 0, 1, 1, 1, 2], 
                          16: [0, 0, 0, 0, 1, 1, 1, 2], 
                           9: [0, 0, 0, 1, 1, 2], 
                           8: [0, 0, 0, 1, 1, 2], 
                           7: [0, 0, 0, 1, 1, 2], 
                           6: [0, 0, 0, 1, 1, 2], 
                           1: [0, 0, 1]}

def get_loader(database, start_idx, end_idx, dataset_name, rcut, batch_size, dtype=torch.float32):
    """
    Make dataloader with the given indices of the mocules in the input database
    """
    rank = dist.get_rank()

    datalist = []
    for i in range(start_idx, end_idx):
        mol = database[i]
        # print(f"Rank {rank} making molecule {i}")

        if dataset_name == "QM7":
            energy = mol['energy']
            forces = mol['forces']
            hamiltonian = mol['hamiltonian'].numpy()   
            atomic_numbers = mol['_atomic_numbers'].numpy()
            positions=mol['_positions'].numpy()
            orbital_basis = orbital_basis_def2_svp
        
        elif dataset_name == "nablaDFT":
            atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = database[i]
            orbital_basis = orbital_basis_def2_svp

        else: 
            print("Unknown database!")

        # 1. Make the atomic structure
        mol_atoms = Atoms(symbols=atomic_numbers, positions=positions)

        # 2. Set up the Fock matrix targets:
        if dataset_name == "QM7":                 
            hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)      # QM7 comes in zxy coordinates from ORCA, so need to rotate 
        
        graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian, dtype=dtype)

        # collect only a subset of the edges (use reflection symmetry in the network)
        forward_edge_mask = graph_targets.neighbour_list[0] < graph_targets.neighbour_list[1]
        print("NOTE: Using half the edges + reflection symmetry!")
        # use all edges:
        # forward_edge_mask = [True]*len(graph_targets.neighbour_list[0])

        # 3. Make the data object
        data = gnnData(
                        pos=torch.tensor(graph_targets.atoms.positions, dtype=dtype),
                        edge_index=torch.tensor(graph_targets.neighbour_list), 
                        edge_mask=torch.tensor(forward_edge_mask),
                        edge_attr=graph_targets.edge_dist, 
                        y=graph_targets.edge_labels,
                        node_y=graph_targets.node_labels,
                        atomic_numbers=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long).cpu(),  
                        energies=torch.tensor(energy, dtype=dtype),
                        forces=torch.tensor(forces, dtype=dtype),                                      # Hartree/Angstrom
                        num_atoms_in_molecule=len(graph_targets.atomic_numbers)
                    )
        datalist.append(data)

    required_irreps = graph_targets.req_output_irreps
    print("required irreps: ", required_irreps)

    basis_transform = graph_targets.basis_transformation

    dataset = sampleDataset(datalist)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader, required_irreps, basis_transform