import torch
import numpy as np

from fock_utils import utils_orca_out, fock_targets
from dataset_utils.ASEDataset import ASEDataset, ASEAtomsData, sampleDataset
from ase import Atoms

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import torch.distributed as dist

orbital_basis_def2_svp_nabla = {35: [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], 
                                17: [0, 0, 0, 0, 1, 1, 1, 2], 
                                16: [0, 0, 0, 0, 1, 1, 1, 2], 
                                9: [0, 0, 0, 1, 1, 2], 
                                8: [0, 0, 0, 1, 1, 2], 
                                7: [0, 0, 0, 1, 1, 2], 
                                6: [0, 0, 0, 1, 1, 2], 
                                1: [0, 0, 1]}

orbital_basis_def2_svp_QM7 = {9: [0, 0, 0, 1, 1, 2], 
                              8: [0, 0, 0, 1, 1, 2], 
                              7: [0, 0, 0, 1, 1, 2], 
                              6: [0, 0, 0, 1, 1, 2], 
                              1: [0, 0, 1]}


def get_loader(database, start_idx, end_idx, dataset_name, rcut, batch_size, dtype=torch.float32, reflection_symmetry=True):
    """
    Make dataloader with the given indices of the mocules in the input database
    """
    rank = dist.get_rank()
    assert end_idx > start_idx

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
            orbital_basis = orbital_basis_def2_svp_QM7
        
        elif dataset_name == "nablaDFT":
            atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = database[i]
            orbital_basis = orbital_basis_def2_svp_nabla

        else: 
            print("Unknown database!")

        # 1. Make the atomic structure
        mol_atoms = Atoms(symbols=atomic_numbers, positions=positions)

        # 2. Set up the Fock matrix targets:
        if dataset_name == "QM7":                 
            hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)      # QM7 comes in zxy coordinates from ORCA, so need to rotate 
        
        graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian, dtype=dtype, reflection_symmetry=reflection_symmetry)

        # collect only a subset of the edges (use reflection symmetry in the network)
        forward_edge_mask = graph_targets.forward_edge_mask
        reverse_edge_map = graph_targets.reverse_edge_map
        
        # 3. Make the data object
        data = gnnData(
                        pos=torch.tensor(graph_targets.atoms.positions, dtype=dtype),
                        edge_index=torch.tensor(graph_targets.neighbour_list), 
                        edge_mask=torch.tensor(forward_edge_mask),
                        reverse_edge_map=torch.tensor(reverse_edge_map),
                        edge_attr=graph_targets.edge_dist, 
                        y=graph_targets.edge_labels,
                        node_y=graph_targets.node_labels,
                        atomic_numbers=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long).cpu(),  
                        energies=torch.tensor(energy, dtype=dtype),
                        forces=torch.tensor(forces, dtype=dtype),                                      # Hartree/Angstrom
                        num_atoms_in_molecule=len(graph_targets.atomic_numbers),
                        fock_target_object=graph_targets,
                    )
        datalist.append(data)

    orbital_basis = {k: torch.tensor(v) for k, v in graph_targets.orbital_basis.items()}
    required_irreps = graph_targets.req_output_irreps
    print("required irreps: ", required_irreps)

    basis_transform = graph_targets.basis_transformation

    dataset = sampleDataset(datalist)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader, required_irreps, basis_transform, orbital_basis