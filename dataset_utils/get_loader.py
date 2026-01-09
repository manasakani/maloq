import torch
import numpy as np

from fock_utils import utils_orca_out, fock_targets_batched, matrix2labels_kernels, basis_sets
from dataset_utils.ASEDataset import ASEDataset, ASEAtomsData, sampleDataset
from ase import Atoms
from ase.neighborlist import NeighborList

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import torch.distributed as dist

def get_loader(database, start_idx, end_idx, dataset_name, rcut, batch_size, dtype=torch.float32, half_edges=True, make_fock_targets=True, scale_shift_data=None):
    """
    Make dataloader with the given indices of the mocules in the input database
    NOTE: closedshell only
    """
    rank = dist.get_rank()
    num_molecules_to_process = end_idx - start_idx

    datalist = []

    if dataset_name == "QM7":
        orbital_basis = basis_sets.orbital_basis_def2_svp_QM7
        energy = [database[i]['energy'] for i in range(start_idx, end_idx)]
        forces = [database[i]['forces'] for i in range(start_idx, end_idx)]
        atomic_numbers = [database[i]['_atomic_numbers'].numpy() for i in range(start_idx, end_idx)]
        positions = [database[i]['_positions'].numpy() for i in range(start_idx, end_idx)]

        hamiltonians = [database[i]['hamiltonian'].numpy() for i in range(start_idx, end_idx)]
        hamiltonians = [utils_orca_out.sort_by_m(h, orbital_basis, z) for h, z in zip(hamiltonians, atomic_numbers)] # QM7 comes in zxy coordinates from ORCA, so need to rotate

        overlaps = [database[i]['overlap'].numpy() for i in range(start_idx, end_idx)] # we don't rotate the overlap

    elif dataset_name == "nablaDFT":
        # atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = database[i]
        orbital_basis = basis_sets.orbital_basis_def2_svp_nabla
        
        atomic_numbers = []
        positions = []
        energy = []
        forces = []
        hamiltonians = []
        overlaps = []
        
        for i in range(start_idx, end_idx):
            z, pos, en, f, ham, ov, coeff, m_id, c_id = database[i]
            atomic_numbers.append(z)
            positions.append(pos)
            energy.append(en)
            forces.append(f)
            hamiltonians.append(ham)
            overlaps.append(ov)

    else:
        raise ValueError("Unknown database!")

    # Set up the Graph targets
    if make_fock_targets:
        
        graph_targets = fock_targets_batched.Fock_Targets(atomic_numbers, positions, rcut, orbital_basis, hamiltonians, 
                                                        dtype=dtype, 
                                                        dataset_name=dataset_name,
                                                        scale_shift_data=scale_shift_data)

    # Add the molecules into the dataloader
    for i in range(num_molecules_to_process):

        # closed shell only, needed for custom collate
        if graph_targets.node_labels_list[i].ndim == 3:
            edge_labels = graph_targets.edge_labels_list[i][0]
            node_labels = graph_targets.node_labels_list[i][0]
        else:
            edge_labels = graph_targets.edge_labels_list[i]
            node_labels = graph_targets.node_labels_list[i]

        # 3. Make the data object
        data = gnnData(
                        pos=torch.tensor(graph_targets.atomic_positions_list[i], dtype=dtype),
                        edge_index=torch.tensor(graph_targets.neighbour_list_list[i]),
                        edge_attr=graph_targets.edge_dist_list[i],
                        y=edge_labels,
                        node_y=node_labels,
                        atomic_numbers=torch.tensor(graph_targets.atomic_numbers_list[i], dtype=torch.long).cpu(),
                        energies=torch.tensor(energy[i], dtype=dtype),
                        forces=torch.tensor(forces[i], dtype=dtype),                                      # Hartree/Angstrom
                        num_atoms_in_molecule=len(graph_targets.atomic_numbers_list[i]),
                        fock_target_object=graph_targets,
                        overlap_matrix=torch.tensor(overlaps[i], dtype=dtype) if make_fock_targets else None,
                    )
        datalist.append(data)

    orbital_basis = {k: torch.tensor(v) for k, v in graph_targets.orbital_basis.items()}
    ls_list = graph_targets.ls_list
    required_irreps = graph_targets.req_output_irreps
    print("required irreps: ", required_irreps)

    basis_transform = graph_targets.basis_transformation
    orbital_starts = graph_targets.orbital_starts

    dataset = sampleDataset(datalist)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader, required_irreps, basis_transform, orbital_basis, ls_list



def get_datalist(dataset, start_idx, end_idx, dataset_name, rcut, element_references, dtype=torch.float32, half_edges=True, make_fock_targets=True, scale_shift_data=None):
    """
    This is for the omol 4 mil energy dataset!
    """

    nonzero_mask = torch.abs(element_references) > 1e-10
    existing_elements = torch.where(nonzero_mask)[0]

    print("Using elements: ", existing_elements)

    datalist = []
    for i in range(start_idx, end_idx):
        if i % 1000 == 0:
            print("Working on atom ", i, flush=True)

        # 1. Get the atomic structure
        atoms = dataset.get_atoms(i)
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        energy = atoms.get_potential_energy() # in eV
        # forces = atoms.get_forces()           # in eV/A

        # convert energy to hartree:
        energy = energy / 27.211386245988 # in Hartree

        # if any of the atomic_numbers are not in element_references, skip this molecule
        if any(z not in existing_elements for z in atomic_numbers):
            # missing_atomic_numbers = [z for z in atomic_numbers if z not in existing_elements]
            # print(f"Skipping molecule with missing atomic numbers: {missing_atomic_numbers}")
            print(f"Skipping molecule with missing atomic numbers")
            continue

        # 2. Set up the Graph
        # --> Neighbour list
        num_atoms = len(atoms)
        neighbours = NeighborList(np.ones(num_atoms)*rcut, skin=0, self_interaction=False, bothways=True)
        neighbours.update(atoms)
        neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
        neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])

        # --> Edge distances
        indices0 = neighbour_list[0]  # First atom indices
        indices1 = neighbour_list[1]  # Second atom indices
        edge_dist = torch.zeros((len(indices0), 4), dtype=dtype)
        edge_dist[:, 1:4] = torch.from_numpy(atoms.get_distances(indices1, indices0, vector=True))    # Vector components
        edge_dist[:, 0] = torch.linalg.norm(edge_dist[:, 1:4], dim=-1, keepdim=False)                 # Scalar distances

        # 3. Make the data object
        data = gnnData(
                        pos=torch.tensor(positions, dtype=dtype),
                        edge_index=torch.tensor(neighbour_list),
                        edge_attr=edge_dist,
                        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long).cpu(),
                        energies=torch.tensor(energy, dtype=dtype),
                        # forces=torch.tensor(forces, dtype=dtype),                                      # Hartree/Angstrom
                        num_atoms_in_molecule=len(atomic_numbers),
                        nedges=len(neighbour_list[0]),
                    )
        datalist.append(data)

    return datalist
