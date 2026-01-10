import torch
import numpy as np

from fock_utils import utils_orca_out, fock_targets_batched, matrix2labels_kernels, basis_sets
from dataset_utils.ASEDataset import ASEDataset, ASEAtomsData, sampleDataset
from ase import Atoms
from ase.neighborlist import NeighborList

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import torch.distributed as dist

def get_loader(database, start_idx, end_idx, dataset_name, rcut, batch_size, dtype=torch.float32, half_edges=True, make_fock_targets=True, scale_shift_data=None, is_open_shell=False, loss_target_string='fock_matrix'):
    """
    Make dataloader with the given indices of the mocules in the input database
    Currently set up for three datasets: QM7, nablaDFT, omol. Need to modify for others.
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
        charge = [0 for i in range(start_idx, end_idx)]
        spins = [1 for i in range(start_idx, end_idx)]

        hamiltonians = [database[i]['hamiltonian'].numpy() for i in range(start_idx, end_idx)]
        hamiltonians = [utils_orca_out.sort_by_m(h, orbital_basis, z) for h, z in zip(hamiltonians, atomic_numbers)] # QM7 comes in zxy coordinates from ORCA, so need to rotate

        overlaps = [database[i]['overlap'].numpy() for i in range(start_idx, end_idx)] # we don't rotate the overlap

    elif dataset_name == "nablaDFT":
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
            charge.append(0)
            spins.append(1)

    elif dataset_name == "omol":
        orbital_basis = basis_sets.def2_tzvpd
        orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
        orbital_basis = {int(k): v for k, v in orbital_basis.items()}
        
        positions = [database[i]['pos'] for i in range(start_idx, end_idx)]
        atomic_numbers = [database[i]['atomic_numbers'] for i in range(start_idx, end_idx)]
        energy = [database[i]['energies'] for i in range(start_idx, end_idx)]
        forces = [0 for i in range(start_idx, end_idx)]  # dummy forces for now!!!
        charges = [database[i]['charge'] for i in range(start_idx, end_idx)]
        spins = [database[i]['spin_multiplicity'] for i in range(start_idx, end_idx)]

        hamiltonians = [database[i][loss_target_string] for i in range(start_idx, end_idx)]
        overlaps = [0 for i in range(start_idx, end_idx)]

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

        if is_open_shell:
            assert graph_targets.node_labels_list[i].shape[0] == 2, "Open shell requested, but did not find two spins!"

        # 3. Make the data object
        if not is_open_shell:
            data = gnnData(
                        pos=torch.tensor(graph_targets.atomic_positions_list[i], dtype=dtype),
                        edge_index=torch.tensor(graph_targets.neighbour_list_list[i]),
                        edge_attr=graph_targets.edge_dist_list[i],
                        y=graph_targets.edge_labels_list[i][0],
                        node_y=graph_targets.node_labels_list[i][0],
                        atomic_numbers=torch.tensor(graph_targets.atomic_numbers_list[i], dtype=torch.long).cpu(),
                        energies=torch.tensor(energy[i], dtype=dtype),
                        forces=torch.tensor(forces[i], dtype=dtype),                                      # Hartree/Angstrom
                        num_atoms_in_molecule=len(graph_targets.atomic_numbers_list[i]),
                        fock_target_object=graph_targets,
                        overlap_matrix=torch.tensor(overlaps[i], dtype=dtype) if make_fock_targets else None,
                        charge=charges[i],
                        spin_multiplicity=spins[i],
                    )
        else:
            data = gnnData(
                        pos=torch.tensor(graph_targets.atomic_positions_list[i], dtype=dtype),
                        edge_index=torch.tensor(graph_targets.neighbour_list_list[i]),
                        edge_attr=graph_targets.edge_dist_list[i],
                        y_alpha=graph_targets.edge_labels_list[i][0],
                        y_beta=graph_targets.edge_labels_list[i][1],
                        node_y_alpha=graph_targets.node_labels_list[i][0],
                        node_y_beta=graph_targets.node_labels_list[i][1],
                        atomic_numbers=torch.tensor(graph_targets.atomic_numbers_list[i], dtype=torch.long).cpu(),
                        energies=torch.tensor(energy[i], dtype=dtype),
                        forces=torch.tensor(forces[i], dtype=dtype),                                      # Hartree/Angstrom
                        num_atoms_in_molecule=len(graph_targets.atomic_numbers_list[i]),
                        fock_target_object=graph_targets,
                        overlap_matrix=torch.tensor(overlaps[i], dtype=dtype) if make_fock_targets else None,
                        charge=charges[i],
                        spin_multiplicity=spins[i],
                    )
        datalist.append(data)

    orbital_basis = {k: torch.tensor(v) for k, v in graph_targets.orbital_basis.items()}

    ls_list = graph_targets.ls_list
    required_irreps = graph_targets.req_output_irreps
    basis_transform = graph_targets.basis_transformation
    orbital_starts = graph_targets.orbital_starts

    dataset = sampleDataset(datalist)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    print("required irreps: ", required_irreps)

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

# --------------------------------------------
# Loading balancing batches 
# --------------------------------------------

def create_atom_balanced_batches(data_list, target_atoms_per_batch, tolerance=0.1):
    """
    Create batches where each batch has approximately the same total number of atoms.

    Args:
        data_list: List of molecular data objects
        target_atoms_per_batch: Target total atoms per batch
        tolerance: Tolerance for batch size (0.1 = 10% tolerance)

    Returns:
        List of batches, where each batch has similar total atom counts
    """

    # Get molecule sizes and sort by size (largest first for better packing)
    molecule_data = [(i, len(data.atomic_numbers), data) for i, data in enumerate(data_list)]
    molecule_data.sort(key=lambda x: x[1], reverse=True)  # Sort by atoms (largest first)

    print(f"Molecule size distribution:")
    sizes = [size for _, size, _ in molecule_data]
    print(f"  Min atoms: {min(sizes)}, Max atoms: {max(sizes)}")
    print(f"  Mean atoms: {sum(sizes)/len(sizes):.1f}, Total atoms: {sum(sizes)}")
    print(f"  Target atoms per batch: {target_atoms_per_batch}")

    batches = []
    remaining_molecules = molecule_data.copy()

    while remaining_molecules:
        current_batch = []
        current_batch_atoms = 0
        max_atoms = int(target_atoms_per_batch * (1 + tolerance))
        min_atoms = int(target_atoms_per_batch * (1 - tolerance))

        # Try to fill current batch to target size
        i = 0
        while i < len(remaining_molecules):
            mol_idx, mol_size, mol_data = remaining_molecules[i]

            # Check if adding this molecule would exceed the maximum
            if current_batch_atoms + mol_size <= max_atoms:
                # Add molecule to current batch
                current_batch.append(mol_data)
                current_batch_atoms += mol_size
                remaining_molecules.pop(i)  # Remove from remaining

                # If we've reached a good batch size, stop adding
                if current_batch_atoms >= min_atoms:
                    break
            else:
                i += 1  # Try next molecule

        # If we couldn't add any molecule, just add the first remaining one
        # (this handles cases where a single molecule is larger than target)
        if not current_batch and remaining_molecules:
            mol_idx, mol_size, mol_data = remaining_molecules.pop(0)
            current_batch.append(mol_data)
            current_batch_atoms = mol_size

        if current_batch:
            batches.append(current_batch)

    # Print batch statistics
    print(f"\nCreated {len(batches)} atom-balanced batches:")
    total_atoms_all = 0
    atom_counts = []

    for i, batch in enumerate(batches):
        batch_sizes = [len(data.atomic_numbers) for data in batch]
        total_atoms = sum(batch_sizes)
        atom_counts.append(total_atoms)
        total_atoms_all += total_atoms

        if i < 10:  # Show first 10 batches
            print(f"  Batch {i}: {len(batch)} molecules, {total_atoms} total atoms, "
                  f"size range: {min(batch_sizes)}-{max(batch_sizes)}")

    if len(batches) > 10:
        print(f"  ... and {len(batches) - 10} more batches")

    print(f"\nBatch atom count statistics:")
    print(f"  Target: {target_atoms_per_batch}, Mean: {np.mean(atom_counts):.1f}")
    print(f"  Min: {min(atom_counts)}, Max: {max(atom_counts)}")
    print(f"  Std dev: {np.std(atom_counts):.1f}")

    return batches

def create_atom_balanced_dataloader(data_list, target_atoms_per_batch, tolerance=0.1,
                                  shuffle=True, num_workers=0):
    """
    Create a single DataLoader with atom-balanced batches.
    """

    # Create balanced batches
    batches = create_atom_balanced_batches(data_list, target_atoms_per_batch, tolerance)

    # Convert each batch to PyTorch Geometric Batch objects immediately
    batched_data = []
    for i, molecule_list in enumerate(batches):
        try:
            batch_obj = Batch.from_data_list(molecule_list)
            batched_data.append(batch_obj)
            print(f"Created batch {i}: {type(batch_obj)}, {len(molecule_list)} molecules")
        except Exception as e:
            print(f"Error creating batch {i}: {e}")
            raise

    # Now shuffle the pre-created batches if requested
    if shuffle:
        import random
        random.shuffle(batched_data)

    print(f"Created {len(batched_data)} pre-batched PyG Batch objects")

    return SimpleBatchIterator(batched_data)

def create_edge_balanced_batches(data_list, target_edges_per_batch, tolerance=0.1):
    """
    Create batches where each batch has approximately the same total number of edges.

    Args:
        data_list: List of molecular data objects with .nedges attribute
        target_edges_per_batch: Target total edges per batch
        tolerance: Tolerance for batch size (0.1 = 10% tolerance)

    Returns:
        List of batches, where each batch has similar total edge counts
    """

    # Get molecule edge counts and sort by size (largest first for better packing)
    molecule_data = [(i, data.nedges, data) for i, data in enumerate(data_list)]
    molecule_data.sort(key=lambda x: x[1], reverse=True)  # Sort by edges (largest first)

    print(f"Molecule edge distribution:")
    edge_counts = [nedges for _, nedges, _ in molecule_data]
    print(f"  Min edges: {min(edge_counts)}, Max edges: {max(edge_counts)}")
    print(f"  Mean edges: {sum(edge_counts)/len(edge_counts):.1f}, Total edges: {sum(edge_counts)}")
    print(f"  Target edges per batch: {target_edges_per_batch}")

    batches = []
    remaining_molecules = molecule_data.copy()

    # Remove molecules that are too large upfront
    max_edges = int(target_edges_per_batch)

    # Filter out molecules that are too large
    valid_molecules = []
    removed_count = 0

    for i, data in enumerate(data_list):
        if data.nedges <= max_edges:
            valid_molecules.append((i, data.nedges, data))
        else:
            removed_count += 1

    print(f"Removed {removed_count} molecules with >{max_edges} edges")
    print(f"Processing {len(valid_molecules)} molecules")

    # Sort by size (largest first for better packing)
    molecule_data = sorted(valid_molecules, key=lambda x: x[1], reverse=True)

    while remaining_molecules:
        current_batch = []
        current_batch_edges = 0
        max_edges = int(target_edges_per_batch * (1 + tolerance))
        min_edges = int(target_edges_per_batch * (1 - tolerance))

        # Try to fill current batch to target size
        i = 0
        while i < len(remaining_molecules):
            mol_idx, mol_edges, mol_data = remaining_molecules[i]

            # Check if adding this molecule would exceed the maximum
            if current_batch_edges + mol_edges <= max_edges:
                # Add molecule to current batch
                current_batch.append(mol_data)
                current_batch_edges += mol_edges
                remaining_molecules.pop(i)  # Remove from remaining

                # If we've reached a good batch size, stop adding
                if current_batch_edges >= min_edges:
                    break
            else:
                i += 1  # Try next molecule

        # If we couldn't add any molecule, just add the first remaining one
        # (this handles cases where a single molecule has more edges than target)
        print("Current batch edges:", current_batch_edges, "Remaining molecules:", len(remaining_molecules), " - not adding the rest!!!")
        if not current_batch and remaining_molecules:
            mol_idx, mol_edges, mol_data = remaining_molecules.pop(0)
            current_batch.append(mol_data)
            current_batch_edges = mol_edges

        if current_batch:
            batches.append(current_batch)

    # Print batch statistics
    print(f"\nCreated {len(batches)} edge-balanced batches:")
    total_edges_all = 0
    batch_edge_counts = []

    for i, batch in enumerate(batches):
        batch_edges = [data.nedges for data in batch]
        total_edges = sum(batch_edges)
        batch_edge_counts.append(total_edges)
        total_edges_all += total_edges

        if i < 10:  # Show first 10 batches
            print(f"  Batch {i}: {len(batch)} molecules, {total_edges} total edges, "
                  f"edge range: {min(batch_edges)}-{max(batch_edges)}")

    if len(batches) > 10:
        print(f"  ... and {len(batches) - 10} more batches")

    print(f"\nBatch edge count statistics:")
    print(f"  Target: {target_edges_per_batch}, Mean: {np.mean(batch_edge_counts):.1f}")
    print(f"  Min: {min(batch_edge_counts)}, Max: {max(batch_edge_counts)}")
    print(f"  Std dev: {np.std(batch_edge_counts):.1f}")

    return batches

def create_edge_balanced_dataloader(data_list, target_edges_per_batch, tolerance=0.1,
                                   shuffle=True, num_workers=0):
    """
    Create a dataloader with edge-balanced batches using the existing SimpleBatchIterator.
    """

    # Create balanced batches
    batches = create_edge_balanced_batches(data_list, target_edges_per_batch, tolerance)

    # Convert each batch to PyTorch Geometric Batch objects immediately
    batched_data = []
    for i, molecule_list in enumerate(batches):
        try:
            batch_obj = Batch.from_data_list(molecule_list)
            batched_data.append(batch_obj)
            print(f"Created batch {i}: {type(batch_obj)}, {len(molecule_list)} molecules, "
                  f"{sum(data.nedges for data in molecule_list)} total edges")
        except Exception as e:
            print(f"Error creating batch {i}: {e}")
            raise

    # Now shuffle the pre-created batches if requested
    if shuffle:
        import random
        random.shuffle(batched_data)

    print(f"Created {len(batched_data)} edge-balanced PyG Batch objects")

    # Use the existing SimpleBatchIterator
    return SimpleBatchIterator(batched_data)

# Return a simple list-based iterator instead of DataLoader
class SimpleBatchIterator:
    def __init__(self, batches):
        self.batches = batches
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.index]
        self.index += 1
        return batch

    def __len__(self):
        return len(self.batches)
