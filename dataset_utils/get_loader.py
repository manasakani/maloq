import torch
import numpy as np

from fock_utils import utils_orca_out, fock_targets, matrix2labels_kernels, basis_sets
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
    assert end_idx > start_idx

    # Fock matrix analysis parameters:
    orbital_starts = None
    out_js_list = None
    orbital_template = None
    req_output_irreps = None
    ls_list = None

    datalist = []
    for i in range(start_idx, end_idx):
        # print(f"Rank {rank} making molecule {i}")
        mol = database[i]

        if dataset_name == "QM7":
            energy = mol['energy']
            forces = mol['forces']
            hamiltonian = mol['hamiltonian'].numpy()
            overlap = mol['overlap'].numpy()
            atomic_numbers = mol['_atomic_numbers'].numpy()
            positions=mol['_positions'].numpy()
            orbital_basis = basis_sets.orbital_basis_def2_svp_QM7

        elif dataset_name == "nablaDFT":
            atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = database[i]
            orbital_basis = basis_sets.orbital_basis_def2_svp_nabla

        else:
            raise ValueError("Unknown database!")

        # 1. Make the atomic structure
        mol_atoms = Atoms(symbols=atomic_numbers, positions=positions)

        # 2. Set up the Graph targets
        if make_fock_targets:
            if dataset_name == "QM7":
                hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)      # QM7 comes in zxy coordinates from ORCA, so need to rotate

            graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian, dtype=dtype, half_edges=half_edges,
                                                      scale_shift_data=scale_shift_data,
                                                      orbital_starts=orbital_starts,
                                                      out_js_list=out_js_list,
                                                      orbital_template=orbital_template,
                                                      req_output_irreps=req_output_irreps,
                                                      ls_list=ls_list)

            orbital_starts = graph_targets.orbital_starts
            out_js_list = graph_targets.out_js_list
            orbital_template = graph_targets.orbital_template
            req_output_irreps = graph_targets.req_output_irreps
            ls_list = graph_targets.ls_list

        else:
            graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, None, dtype=dtype, half_edges=half_edges,
                                                      scale_shift_data=scale_shift_data)

        # collect only a subset of the edges (use reflection symmetry in the network)
        forward_edge_mask = graph_targets.forward_edge_mask
        reverse_edge_map = graph_targets.reverse_edge_map

        # closed shell only, needed for custom collate
        if graph_targets.node_labels.ndim == 3:
            edge_labels = graph_targets.edge_labels[0]
            node_labels = graph_targets.node_labels[0]
        else:
            edge_labels = graph_targets.edge_labels
            node_labels = graph_targets.node_labels

        # 3. Make the data object
        data = gnnData(
                        pos=torch.tensor(graph_targets.atoms.positions, dtype=dtype),
                        edge_index=torch.tensor(graph_targets.neighbour_list),
                        edge_mask=torch.tensor(forward_edge_mask),
                        reverse_edge_map=torch.tensor(reverse_edge_map),
                        edge_attr=graph_targets.edge_dist,
                        y=edge_labels,
                        node_y=node_labels,
                        atomic_numbers=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long).cpu(),
                        energies=torch.tensor(energy, dtype=dtype),
                        forces=torch.tensor(forces, dtype=dtype),                                      # Hartree/Angstrom
                        num_atoms_in_molecule=len(graph_targets.atomic_numbers),
                        fock_target_object=graph_targets,
                        overlap_matrix=torch.tensor(overlap, dtype=dtype) if make_fock_targets else None,
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
