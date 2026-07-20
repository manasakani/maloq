# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
import torch
import numpy as np
import os
from scipy.sparse import coo_matrix, diags
import matplotlib.pyplot as plt

from ..fock_utils import utils_orca_out, fock_targets_batched, matrix2labels_kernels, basis_sets
from .ASEDataset import ASEDataset, ASEAtomsData, sampleDataset

from ase import Atoms
from ase.neighborlist import NeighborList
from ase.io import read

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI
import re

def get_loader(database, 
                start_idx, 
                end_idx, 
                dataset_name, 
                rcut, 
                batch_size, 
                dtype=torch.float32,
                half_edges=True, 
                make_fock_targets=True, 
                scale_shift_data=None, 
                is_open_shell=False, 
                loss_target_string='fock_matrix', 
                distribute_graphs=False,
                tiling_dims=None,
                partition_type='linear',
                train_or_eval='train',
                basis_transform_backend='torch'):
    """
    Make dataloader with the given indices of the mocules in the input database
    Currently set up for three datasets: QM7, nablaDFT, omol. Need to modify for others.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_molecules_to_process = end_idx - start_idx
    dist.barrier()

    if dataset_name == "QM7":

        orbital_basis = basis_sets.orbital_basis_def2_svp_QM7
        periodic_dataset = False

        # Extract all the data on every rank (avoids strange indexing errors for the QM7 databases)
        all_counts = [None] * world_size
        dist.all_gather_object(all_counts, num_molecules_to_process)
        total_num_molecules = sum(all_counts)
        device = torch.device('cuda:'+ str(torch.cuda.current_device()))

        # get rank 0's start idx and end idx on all ranks:
        global_start_idx = torch.tensor(start_idx).to(device) if rank == 0 else torch.tensor(0).to(device)
        dist.broadcast(global_start_idx, src=0)
        global_start_idx = global_start_idx.item()
        global_end_idx = global_start_idx + total_num_molecules
        all_data = [database[i] for i in range(global_start_idx, global_end_idx)]
        
        local_start = start_idx - global_start_idx
        local_end = end_idx - global_start_idx
        local_data = all_data[local_start : local_end]

        energy = [row['energy'] for row in local_data]
        forces = [row['forces'] for row in local_data]
        atomic_numbers = [row['_atomic_numbers'].numpy() for row in local_data]
        positions = [row['_positions'].numpy() for row in local_data]
        charges = [0 for i in range(start_idx, end_idx)]
        spins = [1 for i in range(start_idx, end_idx)]

        hamiltonians = [row['hamiltonian'].numpy() for row in local_data]
        hamiltonians = [utils_orca_out.sort_by_m(h, orbital_basis, z) for h, z in zip(hamiltonians, atomic_numbers)] # QM7 comes in zxy coordinates from ORCA, so need to rotate
        overlaps = [row['overlap'].numpy() for row in local_data] # we don't rotate the overlap

    elif dataset_name == "nablaDFT":
        orbital_basis = basis_sets.orbital_basis_def2_svp_nabla
        periodic_dataset = False
        
        atomic_numbers = []
        positions = []
        energy = []
        forces = []
        hamiltonians = []
        overlaps = []
        charges = []
        spins = []
        
        for i in range(start_idx, end_idx):
            z, pos, en, f, ham, ov, coeff, m_id, c_id = database[i]
            atomic_numbers.append(z)
            positions.append(pos)
            energy.append(en)
            forces.append(f)
            hamiltonians.append(ham)
            overlaps.append(ov)
            charges.append(0)
            spins.append(1)

    elif dataset_name == "omol":
        orbital_basis = basis_sets.def2_tzvpd
        orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
        orbital_basis = {int(k): v for k, v in orbital_basis.items()}
        periodic_dataset = False

        positions = [database[i]['pos'] for i in range(start_idx, end_idx)]
        atomic_numbers = [database[i]['atomic_numbers'] for i in range(start_idx, end_idx)]
        energy = [database[i]['energies'] for i in range(start_idx, end_idx)]
        forces = [0 for i in range(start_idx, end_idx)]  # dummy forces for now!!!
        charges = [database[i]['charge'] for i in range(start_idx, end_idx)]
        spins = [database[i]['spin_multiplicity'] for i in range(start_idx, end_idx)]

        hamiltonians = [database[i][loss_target_string] for i in range(start_idx, end_idx)]
        overlaps = [0 for i in range(start_idx, end_idx)] # dummy overlaps for now!!!

    # 'database' is a folder in this case
    elif dataset_name == "cp2k_material":
        orbital_basis = basis_sets.orbital_basis_def2_svp_cp2k
        orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.orbital_basis_def2_svp_cp2k[element] for element in basis_sets.orbital_basis_def2_svp_cp2k.keys()}
        orbital_basis = {int(k): v for k, v in orbital_basis.items()}
        periodic_dataset = True

        hamiltonians = []
        overlaps = []
        positions = []
        atomic_numbers = []
        energy = []
        forces = []
        periodic_boxes = []
        charges = [0 for i in range(start_idx, end_idx)]
        spins = [1 for i in range(start_idx, end_idx)]

        for data_folder in database[start_idx : end_idx]:

            print(f"Rank {rank}: Processing folder {data_folder}...", flush=True)

            # parse xyx file
            xyz_file = [f for f in os.listdir(data_folder) if f.endswith('.xyz')][0]
            structure = load_periodic_cp2k_structure(os.path.join(data_folder, xyz_file))
            print(f"Structure has {len(structure)} atoms and cell dimensions {structure.get_cell()} with PBC {structure.get_pbc()}", flush=True)
            
            # Randomly translate training structures so something will look obviously wrong if the periodicity is incorrect
            if train_or_eval == 'train':
                apply_random_translation_ase(structure)

            # coordinates MUST be wrapped
            structure.wrap()

            atomic_numbers.append(structure.get_atomic_numbers())
            positions.append(structure.get_positions())
            energy.append(0) 
            forces.append(0) 
            periodic_boxes.append(structure.get_cell())
            
            if train_or_eval != 'infer':

                overlap_file = [f for f in os.listdir(data_folder) if '-S_SPIN_1-1_0' in f][0]
                overlap = read_cp2k_matrix(os.path.join(data_folder, overlap_file))
                overlaps.append(overlap)

                # find the Hamiltonian file of type *..-KS_Spin_1-1_0.csr:
                hamiltonian_file = [f for f in os.listdir(data_folder) if '-KS_SPIN_1-1_0' in f or 'H.csr' in f][0]
                print(f"Loading Hamiltonian from {hamiltonian_file}...", flush=True)
                hamiltonian = read_cp2k_matrix(os.path.join(data_folder, hamiltonian_file), dtype=dtype)
                print(f"Hamiltonian loaded with shape {hamiltonian.shape} and {hamiltonian.nnz} non-zero elements", flush=True)

                shift_fermi = True
                if shift_fermi:
                    # find files that end with .out and dont have 'slurm' in the name - likely the cp2k output file
                    out_file = [f for f in os.listdir(data_folder) if f.endswith('.out') and 'slurm' not in f][0]
                    mu = get_fermi_energy(os.path.join(data_folder, out_file))
                    print(f"Fermi energy: {mu}", flush=True)

                    # Apply gauge transformation: H' = H - mu * S
                    hamiltonian = hamiltonian - mu * overlap

                hamiltonians.append(hamiltonian)

            else:
                print(f"Rank {rank}: Inference mode - skipping Hamiltonian and overlap loading for {data_folder}", flush=True)

    else:
        raise ValueError("Unknown database!")

    print(f"Rank {rank}: Loaded data for {num_molecules_to_process} molecules.", flush=True)
    dist.barrier()

    datalist = []

    # If we are distributing, we collapse all loaded data into 1 single Graph object
    if distribute_graphs:

        # Use 'one_rank_contributes' if the structures are periodic:
        if periodic_dataset or train_or_eval == 'eval':
            dist_type = 'one_rank_contributes'
            assert batch_size == 1, "When using 'one_rank_contributes' distribution, the batch size must be 1 since only one rank contributes to each graph!"
        else:
            dist_type = 'all_ranks_contribute'

        # No periodicity - Molecular dataset
        if dist_type == 'all_ranks_contribute':
            comm = MPI.COMM_WORLD

            print(f"Rank {rank}: Distributing graphs with batch size {batch_size}. Total molecules: {num_molecules_to_process}. Number of batches: {(num_molecules_to_process + batch_size - 1) // batch_size}", flush=True)

            local_count = len(energy)
            max_molecules_across_ranks = torch.tensor(local_count).cuda()
            dist.all_reduce(max_molecules_across_ranks, op=dist.ReduceOp.MAX)

            #  The 'batch_size' set determines the maximum number of local molecules that each rank 
            #  contributes to a single 'supergraph', which is then partitioned & distributed across the ranks.
            for i in range(0, max_molecules_across_ranks.item(), batch_size):

                # Each rank prepares a batch of its local molecules (up to 'batch_size'), 
                # and contributes empty data if it has fewer than 'batch_size' molecules left.
                if i < local_count:
                    stop_idx = min(i + batch_size, local_count)
                    batch_idxs = slice(i, stop_idx)
                    b_energy = energy[batch_idxs]
                    b_forces = forces[batch_idxs]
                    b_charges = charges[batch_idxs]
                    b_spins = spins[batch_idxs]
                else:
                    batch_idxs = slice(0, 0)
                    b_energy, b_forces, b_charges, b_spins = [], [], [], []
                
                all_energies_list = comm.allgather(b_energy)
                all_forces_list = comm.allgather(b_forces)
                all_charges_list = comm.allgather(b_charges)
                all_spins_list = comm.allgather(b_spins)

                global_energy = np.array([item for sublist in all_energies_list for item in sublist])
                global_forces = np.array([item for sublist in all_forces_list for item in sublist])
                global_charges = np.array([item for sublist in all_charges_list for item in sublist])
                global_spins = np.array([item for sublist in all_spins_list for item in sublist])

                # Set up the Graph targets for this batch           
                graph_targets = fock_targets_batched.Fock_Targets(atomic_numbers[batch_idxs], positions[batch_idxs], rcut, orbital_basis, hamiltonians[batch_idxs], 
                                                                    partition_type=partition_type,
                                                                    dtype=dtype, 
                                                                    dataset_name=dataset_name,
                                                                    scale_shift_data=scale_shift_data,
                                                                    periodic_boxes=periodic_boxes[batch_idxs] if periodic_dataset else None,
                                                                    tiling_dims=tiling_dims,
                                                                    distribute_graphs=distribute_graphs,
                                                                    basis_transform_backend=basis_transform_backend)

                atom_mol_id = graph_targets.atom_mol_id
                batch_atomic_numbers = graph_targets.atomic_numbers_list
                
                data = gnnData(
                    pos=torch.tensor(graph_targets.atomic_positions_list, dtype=dtype),
                    edge_index=torch.tensor(graph_targets.neighbour_list_list),
                    edge_attr=graph_targets.edge_dist_list,
                    y=graph_targets.edge_labels_list, 
                    node_y=graph_targets.node_labels_list,
                    edge_padding_mask=graph_targets.edge_unpadding_mask_list,
                    node_padding_mask=graph_targets.node_unpadding_mask_list,
                    atomic_numbers=torch.tensor(batch_atomic_numbers, dtype=torch.long).cpu(),
                    energies=torch.tensor(global_energy[atom_mol_id], dtype=dtype), 
                    forces=torch.tensor(global_forces[atom_mol_id], dtype=dtype),  
                    num_atoms_in_molecule=len(graph_targets.atomic_numbers_list),                                   
                    atom_mol_id=atom_mol_id,
                    fock_target_object=graph_targets,
                    overlap_matrix=None,
                    charge=torch.tensor(global_charges[atom_mol_id], dtype=torch.long),
                    spin_multiplicity=torch.tensor(global_spins[atom_mol_id], dtype=torch.long), 
                    distributed_graph_training=distribute_graphs,
                )
                datalist.append(data)
        
        # Periodicity - Round-robin rank contribution, where each rank takes turn being the "Source" of the full graph data
        else:

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            # 1. Share how many molecules EVERY rank has so we can loop safely
            local_count = len(energy)
            all_counts = [None] * world_size
            dist.all_gather_object(all_counts, local_count)

            print(f"Rank {my_rank}: Starting sequential distribution. Counts per rank: {all_counts}", flush=True)

            # 2. Loop through each rank, letting them be the "Source" one by one
            for source_rank in range(world_size):
                source_total_mols = all_counts[source_rank]

                # 3. Loop through the Source Rank's data in chunks of 'batch_size'
                for i in range(0, source_total_mols, batch_size):
                    
                    # If I am the source, I pack my real data
                    if my_rank == source_rank:
                        stop_idx = min(i + batch_size, source_total_mols)
                        batch_idxs = slice(i, stop_idx)
                        
                        b_energy = energy[batch_idxs]
                        b_forces = forces[batch_idxs]
                        b_charges = charges[batch_idxs]
                        b_spins = spins[batch_idxs]
                        b_atomic_numbers = atomic_numbers[batch_idxs]
                        b_positions = positions[batch_idxs]
                        b_hamiltonians = hamiltonians[batch_idxs]
                        b_overlaps = overlaps[batch_idxs]
                        b_periodic_boxes = periodic_boxes[batch_idxs] if periodic_dataset else None
                    
                    # If I am a receiver, I prepare empty variables
                    else:
                        b_energy, b_forces, b_charges, b_spins, b_overlaps = None, None, None, None, None
                        b_atomic_numbers, b_positions, b_hamiltonians, b_periodic_boxes = [], [], [], []

                    # 4. Pack everything into a single list for easy broadcasting
                    data_pack = [[
                        b_energy, b_forces, b_charges, b_spins, b_overlaps
                    ]] if my_rank == source_rank else [None]

                    # 5. The Source Rank broadcasts the pack to everyone else
                    dist.broadcast_object_list(data_pack, src=source_rank)

                    # 6. Unpack the data. Now EVERY rank has the exact same batch of molecules
                    (b_energy, b_forces, b_charges, b_spins, b_overlaps) = data_pack[0]

                    global_energy = np.array(b_energy)
                    global_forces = np.array(b_forces)
                    global_charges = np.array(b_charges)
                    global_spins = np.array(b_spins)
                    global_overlaps = np.array(b_overlaps)

                    # 7. Set up the Graph targets for this batch 
                    # (Fock_Targets will now partition this specific batch across all ranks)
                    graph_targets = fock_targets_batched.Fock_Targets(
                        b_atomic_numbers, b_positions, rcut, orbital_basis, b_hamiltonians, 
                        partition_type=partition_type,
                        dtype=dtype, 
                        dataset_name=dataset_name,
                        scale_shift_data=scale_shift_data,
                        periodic_boxes=b_periodic_boxes,
                        tiling_dims=tiling_dims,
                        distribute_graphs=distribute_graphs,
                        basis_transform_backend=basis_transform_backend
                    )

                    atom_mol_id = graph_targets.atom_mol_id
                    batch_atomic_numbers = graph_targets.atomic_numbers_list
                    
                    data = gnnData(
                        pos=torch.tensor(graph_targets.atomic_positions_list, dtype=dtype),
                        edge_index=torch.tensor(graph_targets.neighbour_list_list),
                        edge_attr=graph_targets.edge_dist_list,
                        y=graph_targets.edge_labels_list if train_or_eval!='infer' else None,
                        node_y=graph_targets.node_labels_list  if train_or_eval!='infer' else None,
                        edge_padding_mask=graph_targets.edge_unpadding_mask_list if train_or_eval!='infer' else None,
                        node_padding_mask=graph_targets.node_unpadding_mask_list  if train_or_eval!='infer' else None,
                        atomic_numbers=torch.tensor(batch_atomic_numbers, dtype=torch.long).cpu(),
                        energies=torch.tensor(global_energy[atom_mol_id], dtype=dtype), 
                        forces=torch.tensor(global_forces[atom_mol_id], dtype=dtype),  
                        num_atoms_in_molecule=len(graph_targets.atomic_numbers_list),                                   
                        atom_mol_id=atom_mol_id,
                        fock_target_object=graph_targets,
                        overlap_matrix=global_overlaps[0] if train_or_eval=='eval' else None,
                        charge=torch.tensor(global_charges[atom_mol_id], dtype=torch.long),
                        spin_multiplicity=torch.tensor(global_spins[atom_mol_id], dtype=torch.long), 
                        distributed_graph_training=distribute_graphs,
                    )
                    datalist.append(data)
                    print(f"Rank {my_rank}: Finished processing batch {i // batch_size + 1} from source rank {source_rank}.", flush=True)
                    dist.barrier()
    else:

        # Set up the Graph targets            
        graph_targets = fock_targets_batched.Fock_Targets(atomic_numbers, positions, rcut, orbital_basis, hamiltonians, 
                                                            dtype=dtype, 
                                                            dataset_name=dataset_name,
                                                            scale_shift_data=scale_shift_data,
                                                            periodic_boxes=periodic_boxes if periodic_dataset else None,
                                                            tiling_dims=tiling_dims,
                                                            partition_type=partition_type,
                                                            basis_transform_backend=basis_transform_backend)

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
                            edge_padding_mask=graph_targets.edge_unpadding_mask_list[i][0],
                            node_padding_mask=graph_targets.node_unpadding_mask_list[i][0],
                            atomic_numbers=torch.tensor(graph_targets.atomic_numbers_list[i], dtype=torch.long).cpu(),
                            energies=torch.tensor(energy[i], dtype=dtype),
                            forces=torch.tensor(forces[i], dtype=dtype),                                      # Hartree/Angstrom
                            num_atoms_in_molecule=len(graph_targets.atomic_numbers_list[i]),
                            fock_target_object=graph_targets,
                            overlap_matrix=overlaps[i] if make_fock_targets else None,
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
                            overlap_matrix=overlaps[i] if make_fock_targets else None,
                            charge=charges[i],
                            spin_multiplicity=spins[i],
                        )
            datalist.append(data)

    orbital_basis = {k: torch.tensor(v) for k, v in graph_targets.orbital_basis.items()}
    ls_list = graph_targets.ls_list
    required_irreps = graph_targets.req_output_irreps
    basis_transform = graph_targets.basis_transformation
    orbital_starts = graph_targets.orbital_starts

    # when distributing graphs, the batch size seen by the dataloader needs to be 1 since each graph is a 'supergraph' with multiple molecules. 
    if distribute_graphs:
        batch_size = 1

    dataset = sampleDataset(datalist)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    # print number of batches owned by each rank:
    print(f"Rank {rank}: Number of batches = {len(data_loader)} with batch size {batch_size}", flush=True)

    if rank == 0:
        print("Required irreps to cover all orbital interactions: ", required_irreps)

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

def read_cp2k_matrix(file_name, dtype=torch.float32):
    
    # Attempt Binary Reading first
    try:
        file_size = os.path.getsize(file_name)
        record_size = 24
        
        # Quick check: If it's binary CSR from CP2K, it MUST be a multiple of 24
        # if file_size % record_size != 0 or file_size == 0:
        #     raise ValueError("Not a binary CSR file!")

        # Define the Fortran unformatted record structure
        # (4-byte pad, 4-byte int, 4-byte int, 8-byte float, 4-byte pad)
        dt = np.dtype([
            ('pad1', '<i4'), ('x', '<u4'), ('y', '<u4'), 
            ('data', '<f8'), ('pad2', '<i4')
        ])
        
        raw_data = np.fromfile(file_name, dtype=dt)
        # Fortran unformatted records for this structure must have pads = 16 bytes
        # We check the first few records to ensure they look like CP2K binary
        if len(raw_data) > 0:
            if not np.all(raw_data['pad1'][:10] == 16) or not np.all(raw_data['pad2'][:10] == 16):
                raise ValueError("Invalid binary markers: Likely an ASCII file.", flush=True)
            
            # Additional safety: indices shouldn't be astronomically high (adjust if expecting billions of atoms...)
            if np.any(raw_data['x'][:10] > 100_000_000):
                raise ValueError("Indices too large: Likely garbage data from ASCII.", flush=True)

        print(f"Detected Binary Format: {file_name}", flush=True)

        x_indices = raw_data['x']
        y_indices = raw_data['y']
        data = raw_data['data']

    except (ValueError, OSError, RuntimeError):
        # Fallback to ASCII
        print(f"Reading as ASCII Format: {file_name}")
        try:
            # read 3 columns: Row, Col, Value
            raw_data = np.loadtxt(file_name)
            x_indices = raw_data[:, 0].astype(np.uint32)
            y_indices = raw_data[:, 1].astype(np.uint32)
            data = raw_data[:, 2]

        except Exception as e:
            raise IOError(f"Could not read {file_name} as Binary or ASCII. Error: {e}")
    
    # convert to desired dtype
    type_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64
    }
    # Fallback to float32 if type is not in map
    np_dtype = type_map.get(dtype, np.float32)
    data = data.astype(np_dtype)

    # --- Construct Sparse Matrix (Unified Logic) ---
    # Handle 1-based indexing from CP2K
    max_idx = int(max(np.max(x_indices), np.max(y_indices)))
    matsize = (max_idx, max_idx)

    # Create COO and convert to CSR
    H = coo_matrix((data, (x_indices - 1, y_indices - 1)), shape=matsize)
    H = H.tocsr()

    # Enforce Symmetry (H_full = H + H.T - diag(H))
    # This is necessary because CP2K often only prints the upper triangle
    D = diags(H.diagonal(), offsets=0, shape=H.shape, format='csr')
    H_full = H + H.T - D

    return H_full

def apply_random_translation_ase(atoms):
    """
    Applies a random translation to an ASE Atoms object, keeping 
    atoms within the unit cell.
    """

    cell = atoms.get_cell()
    translation_vector = np.dot(np.random.rand(3), cell)
    atoms.translate(translation_vector)
    atoms.wrap()
    return atoms

def load_periodic_cp2k_structure(file_path):
    """
    Reads an XYZ file and manually extracts cell dimensions from the 
    second line (comment line) if they are not automatically parsed.
    """
    print(f"Loading structure from {file_path}...", flush=True)

    # atoms = read(file_path, format='xyz') # somehow this hangs when distributed
    with open(file_path, 'r') as f:
        lines = f.readlines()
            
        # Line 0: Number of atoms
        num_atoms = int(lines[0].strip())
            
        # Lines 2 to 2+num_atoms: Atomic data
        symbols = []
        positions = []
        for i in range(2, 2 + num_atoms):
            parts = lines[i].split()
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])
            
        # Create ASE object 
        atoms = Atoms(symbols=symbols, positions=positions)

        # Line 1: Comment/Cell line
        header_line = lines[1].strip()

        # --> Try parsing Extended XYZ Lattice format: Lattice="..."
        lattice_match = re.search(r'Lattice="([^"]+)"', header_line)
        
        if lattice_match:
            lattice_vals = [float(x) for x in lattice_match.group(1).split()]
            if len(lattice_vals) == 9:
                # Extract diagonal elements [X_x, Y_y, Z_z] from the 3x3 matrix
                cell_dims = [lattice_vals[0], lattice_vals[4], lattice_vals[8]]
            elif len(lattice_vals) == 3:
                cell_dims = lattice_vals
            else:
                cell_dims = None
                print("Warning: Unexpected number of values in Lattice string.")
                
            if cell_dims:
                atoms.set_cell(cell_dims)
                atoms.set_pbc([True, True, True])
                
        # --> Fallback to the original CP2K "Cell:" format
        else:
            header_parts = header_line.split()
            if header_parts and (header_parts[0].lower() == 'cell' or header_parts[0].lower() == 'cell:'):
                cell_dims = [float(x) for x in header_parts[1:4]]
                atoms.set_cell(cell_dims)
                atoms.set_pbc([True, True, True])
            else:
                print("Warning: Neither 'Lattice' nor 'Cell' keyword found in header line.")

        # if header_parts[0].lower() == 'cell' or header_parts[0].lower() == 'cell:':
        #     # Extract the 3 dimensions: [X Y Z]
        #     cell_dims = [float(x) for x in header_parts[1:4]]
        #     atoms.set_cell(cell_dims)
        #     atoms.set_pbc([True, True, True])
        # else:
        #     print("Warning: 'Cell' keyword not found in header line.")
    
    return atoms

def get_fermi_energy(out_file_path):
    patterns = [
        r'Fermi Energy \[eV\] :\s*([-+]?\d*\.\d+)',  # Matches "Fermi Energy [eV] :   -4.747024"
        r'Fermi energy:\s*([-+]?\d*\.\d+)',          # Matches "Fermi energy: -4.747024"
        r'Fermi level:\s*([-+]?\d*\.\d+)',           # Matches "Fermi level: -4.747024"
    ]
    with open(out_file_path, 'r') as f:
        for line in f:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    # convert to hartree if it's in eV:
                    if 'eV' in pattern:
                        print(f"Fermi energy found in eV: {match.group(1)}. Converting to Hartree...", flush=True)
                        return float(match.group(1)) / 27.211386245988
                    return float(match.group(1))
    raise ValueError(f"Fermi energy not found in {out_file_path}")
    
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
