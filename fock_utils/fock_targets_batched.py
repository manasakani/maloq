import torch
import time
import random
import pickle, os

import numpy as np
import cupy as cp
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from ase import Atoms
from ase.neighborlist import NeighborList
from ase.io import write

from . import utils_tensor_decomp, matrix2labels_kernels, reorder
from fock_utils.domain_decomp import MergedStructure, Domain_Decomp

from torch.utils.dlpack import from_dlpack
import torch.distributed as dist
from mpi4py import MPI
from collections import defaultdict

class Fock_Targets:
    """
    Fock matrix analysis object, consists of two main components:
    1. Atomic graph and connectivity list for the input structure
    2. Fock target analysis components for a given atomic basis (this is the same for any structure sharing the same atomic basis)
    3. If fock_matrix input is not None, computes the fock matrix decomposition into orbital blocks (for each pair of atoms)
    Sets up inputs/targets for supervised (atomic_structure -> Fock matrix) training for an set of atoms.
    """

    def __init__(self, atomic_numbers, atomic_positions, cutoff, orbital_basis,
                fock_matrices=None,
                periodic_boxes=None,
                dataset_name='temp',
                dtype=torch.float32,
                compute_fock_eigenvalues=False,
                scale_shift_data=None,
                distribute_graphs=False,
                orbital_starts=None,
                orbital_template=None,
                req_output_irreps=None,
                out_js_list=None,
                ls_list=None):
        """
        neighbor_list - H2O: [[0, 0, 1, 1, 2, 2], [1, 2, 2, 0, 0, 1]]
        orbital_basis - H2O: {8: [0, 0, 0, 1, 1, 2], 1: [0, 0, 1]} (ex. dzvp)
        fock_matrix - Norb x Norb fock matrix (dense)
        """

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.comm = MPI.COMM_WORLD

        # Graph-wise distribution of the underlying data 
        self.distribute_graphs = distribute_graphs

        self.orbital_basis = orbital_basis
        self.dtype = dtype

        # Create structures and neighbor lists (outer index is the molecule index) - follows the fock matrices owned by each rank
        num_structures = len(atomic_numbers)
        self.atomic_numbers_list = []
        self.atomic_positions_list = []
        self.neighbour_list_list = []
        self.edge_dist_list = []
        self.orbitals_per_atom_list = []
        self.block_starts_list = []
        self.make_atomic_graphs(atomic_numbers, atomic_positions, cutoff, periodic_boxes)

        # Create a merged distributed graph
        if self.distribute_graphs:
            self.make_distributed_atomic_graph(atomic_numbers, atomic_positions, cutoff, periodic_boxes)

        # --> Analyze structure of orbital interactions
        if orbital_template is None or out_js_list is None or req_output_irreps is None or ls_list is None:
            cache_path = "orbital_cache_"+str(dataset_name)+".pkl"
            if os.path.exists(cache_path):
                print("Reading orbital info from cache")
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                self.req_output_irreps = cache["req_output_irreps"]
                self.out_js_list = cache["out_js_list"]
                self.orbital_starts = cache["orbital_starts"]
                self.orbital_template = cache["orbital_template"]
                self.ls_list = cache["ls_list"]
            else:
                print("Recomputing orbital interactions...")
                targets, self.req_output_irreps, simplified_out_irreps, ls_list, self.out_js_list, self.orbital_starts, full_orb_interaction_list = utils_tensor_decomp.make_output_irreps(self.orbital_basis)
                equivariant_blocks = utils_tensor_decomp.process_targets(self.orbital_basis, targets, ls_list, self.out_js_list, full_orb_interaction_list)
                self.orbital_template = matrix2labels_kernels.get_orbital_template(equivariant_blocks, self.orbital_starts)

                # ls list will define the max basis needed (eg, for OMOL: tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 1, 2])
                ls_list = []
                for l in range(20): # large to account for possible diffuse functions which are incremented by 10
                    counts = [torch.sum(torch.tensor(self.orbital_basis[el]) == l) for el in self.orbital_basis]
                    max_count = max(counts).item()
                    ls_list.append(torch.tensor(max_count * [l], dtype=torch.int))

                # Shift back all the diffuse orbitals (which were incremented by 10 in utils_tensor_decomp.py)
                for atom, orbitals in self.orbital_basis.items():
                    self.orbital_basis[atom] = [orb % 10 for orb in orbitals]

                for atom, orbitals in orbital_basis.items():
                    orbital_basis[atom] = [orb % 10 for orb in orbitals]

                self.ls_list = torch.cat(ls_list)        # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2].
                self.ls_list = self.ls_list % 10         # for OMOL: tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 1, 2]

                cache = {
                    "req_output_irreps": self.req_output_irreps,
                    "out_js_list": self.out_js_list,
                    "orbital_starts": self.orbital_starts,
                    "orbital_template": self.orbital_template,
                    "ls_list": self.ls_list
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        # To avoid file write/read during database creation, can directly pass in the orbital information
        else:
            self.orbital_template = orbital_template
            self.orbital_starts = orbital_starts
            self.out_js_list = out_js_list
            self.req_output_irreps = req_output_irreps
            self.ls_list = ls_list

        # --> Create coupled/uncoupled basis transformation
        self.basis_transformation = utils_tensor_decomp.e3TensorDecomp(self.req_output_irreps,
                                                                       self.out_js_list,
                                                                       default_dtype_torch=dtype,
                                                                       if_sort=False,
                                                                       device_torch=self.device)

        # print(f'Required irreps to represent orbital interactions: {self.req_output_irreps}')
        self.scale_shift_data = scale_shift_data

        # Decompose the Fock matrix into orbital blocks and insert them into the targets
        self.target_len = None
        if fock_matrices is not None:
            self.make_targets(fock_matrices)

        # Whether to redistribute the partitioned data based on the domain decomposition of the merged graph (only relevant if distribute_graphs is True)
        self.redistribute_partition_data = True
        partition_type = "metis"
        if self.distribute_graphs and self.redistribute_partition_data:

            periodicity = False if periodic_boxes is None else True # single-structure periodicity only

            # check that if periodicity is true, there is only one structure (ie we are not trying to apply periodic partitioning across multiple 
            # different structures, which would be more complex and is not currently implemented)
            if periodicity and num_structures > 1:
                raise ValueError("Periodic partitioning only implemented for single structure for now!")

            # determine the new partioning based on the partition_type (eg "metis" or "random")
            # reordered_partitions returns the set of new partitions  (eg, [[3 5 0], [1 2 4]] for two partitions with 3 nodes each), 
            # atom_reorder_perm returns a permutation for the nodes in the global graph (eg, [3 5 0 1 2 4]), 
            # atoms_per_partition returns how many belong to each partition (eg, [2, 2, 2] for 3 partitions with 6 total nodes)
            reordered_partitions, atom_reorder_perm, atoms_per_partition = self.get_partition_map(cutoff, partition_type=partition_type, periodicity=periodicity)

            # redistribute the data based on the new partitioning (this will involve communication across ranks to send the relevant node/edge l
            # abels to the ranks that now own those nodes/edges after repartitioning)
            self.redistribute_graph(reordered_partitions, atom_reorder_perm, atoms_per_partition)


    def make_atomic_graphs(self, atomic_numbers, atomic_positions, cutoff, periodic_boxes):

        # The outer most index is the molecule index
        num_molecules = len(atomic_numbers)

        # --> Atoms and connectivity list:
        for i, (numbers, positions) in enumerate(zip(atomic_numbers, atomic_positions)):

            # Atoms for every molecule
            atoms = Atoms(symbols=numbers, positions=positions)
            num_atoms = len(numbers)
            self.atomic_numbers_list.append(atoms.get_atomic_numbers())
            self.atomic_positions_list.append(atoms.get_positions())

            if periodic_boxes is not None:
                atoms.set_cell(periodic_boxes[i])
                atoms.set_pbc([True, True, True])

            # Neighbor lists for every molecule
            neighbours = NeighborList(np.ones(num_atoms)*cutoff, skin=0, self_interaction=False, bothways=True)
            neighbours.update(atoms)
            neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
            mol_neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])
            self.neighbour_list_list.append(mol_neighbour_list)

            # Edge distances for every molecule
            indices0 = mol_neighbour_list[0]  # First atom indices 
            indices1 = mol_neighbour_list[1]  # Second atom indices

            # NOTE: in the distribution, we have swapped the labels for src and dst across every edge, 
            # so we recover that by simply switching the order of indices when we compute the edge distances (so the vector points from src to dst in both cases)
            mol_edge_dist = torch.zeros((len(indices0), 4), dtype=self.dtype)
            if self.distribute_graphs:
                mol_edge_dist[:, 1:4] = torch.from_numpy(atoms.get_distances(indices0, indices1, vector=True))    # Vector components
                print("Computed edge distances with swapped indices for distributed graph!", flush=True)
            else:
                mol_edge_dist[:, 1:4] = torch.from_numpy(atoms.get_distances(indices1, indices0, vector=True))    # Vector components
            mol_edge_dist[:, 0] = torch.linalg.norm(mol_edge_dist[:, 1:4], dim=-1, keepdim=False)             # Scalar distances
            self.edge_dist_list.append(mol_edge_dist)

            # Orbital block locations
            orbitals_per_atom = ([ sum([(2*l+1)
                                for l in self.orbital_basis[atom_number]])
                                for atom_number in numbers ])
            block_starts = np.hstack([0, np.cumsum(orbitals_per_atom)]) # start index of atom i in the matrix (and block_starts[-1] is the matrix size)
            self.orbitals_per_atom_list.append(orbitals_per_atom)
            self.block_starts_list.append(block_starts)

    def make_distributed_atomic_graph(self, atomic_numbers, atomic_positions, cutoff, periodic_boxes):
        """
        1. Process local molecules.
        2. Synchronize all molecules across all ranks.
        3. Build one global graph.
        4. Apply Domain Decomposition.
        """
        # --- 1: Local Pre-processing ---
        local_mol_data = []
        for i, (numbers, positions) in enumerate(zip(atomic_numbers, atomic_positions)):
            atoms = Atoms(symbols=numbers, positions=positions)
            num_atoms = len(numbers)

            if periodic_boxes is not None:
                atoms.set_cell(periodic_boxes[i])
                atoms.set_pbc([True, True, True])
            
            # Get neighbor list for this single molecule
            nl = NeighborList(np.ones(num_atoms)*cutoff, skin=0, self_interaction=False, bothways=True)
            nl.update(atoms)
            nb_matrix = nl.get_connectivity_matrix(sparse=True).tocoo()
            edge_index = np.vstack([nb_matrix.row, nb_matrix.col])
            
            local_mol_data.append({
                'z': atoms.get_atomic_numbers(),
                'pos': atoms.get_positions(),
                'edge_index': edge_index,
            })

        # --- 2: Global Synchronization ---
        # Gather the molecule lists from every rank to every rank and flatten it into a list
        global_mol_data_list = [None for _ in range(self.world_size)]
        dist.all_gather_object(global_mol_data_list, local_mol_data)
        all_molecules = [mol for rank_list in global_mol_data_list for mol in rank_list]

        # Track offsets of where this molecule's fock blocks start in the global graph
        self.global_fock_ii_start_offsets = []
        self.global_fock_ij_start_offsets = []
        self.global_fock_ii_end_offsets = []
        self.global_fock_ij_end_offsets = []
        node_offset = 0
        edge_offset = 0
        for mol in global_mol_data_list:
            self.global_fock_ii_start_offsets.append(node_offset)
            self.global_fock_ij_start_offsets.append(edge_offset)
            for m in mol:
                node_offset += len(m['z'])
                edge_offset += m['edge_index'].shape[1]
            self.global_fock_ii_end_offsets.append(node_offset)
            self.global_fock_ij_end_offsets.append(edge_offset)
            
        # print(f"Offsets for global node indices: {self.global_fock_ii_start_offsets}")
        # print(f"Offsets for global edge indices: {self.global_fock_ij_start_offsets}")
        # print(f"End Offsets for global node indices: {self.global_fock_ii_end_offsets}")
        # print(f"End Offsets for global edge indices: {self.global_fock_ij_end_offsets}")

        # --- 3: Merging into a Global Super-Graph ---
        all_z = []
        all_pos = []
        global_edges = []
        all_mol_ids = []
        
        current_node_offset = 0
        for mol_idx, mol in enumerate(all_molecules):
            num_nodes = len(mol['z'])
            all_z.append(mol['z'])
            all_pos.append(mol['pos'])

            # Create a vector of IDs for every atom in this specific molecule
            all_mol_ids.append(np.full(num_nodes, mol_idx, dtype=np.int64))
            
            # Offset the local molecule edge indices to the global index space
            global_edges.append(mol['edge_index'] + current_node_offset)
            current_node_offset += num_nodes

        # Convert to giant arrays/tensors
        global_structure_z = np.concatenate(all_z)
        global_structure_pos = np.concatenate(all_pos)
        global_structure_edges = np.hstack(global_edges)
        global_mol_ids = np.concatenate(all_mol_ids)

        # --- 4: Domain Decomposition ---
        dist.barrier()
        self.merged_atomic_graph = MergedStructure(global_structure_z, global_structure_pos, global_structure_edges)
        self.domain = Domain_Decomp(self.merged_atomic_graph, device=self.device)
        self.domain.print_info()
        dist.barrier()

        # Store global molecule ID of every atom this rank owns
        self.atom_mol_id = global_mol_ids[self.domain.local_node_index]        

        if self.rank == 0:
            print(f"Global graph created with {len(global_structure_z)} nodes and {global_structure_edges.shape[1]} edges.")

        dist.barrier()      # Clear PyTorch/Gloo buffer
        self.comm.Barrier() # Clear MPI buffer

    def make_targets(self, fock_matrices):
        """
        Creates padded node/edge labels from the fock matrix
        """

        method = 'cupy_kernel' # 'numpy_kernel' or 'cupy_kernel' 
        single_matrix = True  # whether to process one matrix at a time (if false, all at once) 
        spin_strings = ['_alpha', '_beta']

        # each target should fit in a NxN matrix (to be flattened)
        self.target_len = self.get_target_len()

        if self.distribute_graphs:
            node_labels_list = []
            edge_labels_list = []

        self.node_labels_list = []
        self.edge_labels_list = []

        if method == 'cupy_kernel':
            print("Creating orbital template pointers for matrix->label kernel... ")
            orbital_template_ptrs = []
            orbital_template_tmp = []
            for o in self.orbital_template:
                inner_size = 5 * len(o)
                tmp = np.zeros((inner_size,), dtype=cp.int32)
                for j, (row_slice, col_slice, output_slice) in enumerate(o):
                    tmp[j * 5 + 0] = row_slice.start
                    tmp[j * 5 + 1] = row_slice.stop
                    tmp[j * 5 + 2] = col_slice.start
                    tmp[j * 5 + 3] = col_slice.stop
                    tmp[j * 5 + 4] = output_slice.start
                tmp = cp.array(tmp, dtype=cp.int32)
                orbital_template_tmp.append(tmp)
                orbital_template_ptrs.append(matrix2labels_kernels.get_ptr(tmp))

            # template: for interation Z1-Z2, [row slice, col slice] of matrix goes to [output slice] of label
            orbital_template_ptrs = cp.array(orbital_template_ptrs, dtype=cp.uintp)
            cp.cuda.Stream.null.synchronize()

        if single_matrix:
            for i, fock_matrix in enumerate(fock_matrices):

                open_shell = fock_matrix.ndim == 3

                # Handle sparse fock
                if sp.issparse(fock_matrix):
                    fock_matrix = fock_matrix.toarray()

                # Move fock matrix to device
                if not isinstance(fock_matrix, torch.Tensor):
                    fock_matrix = torch.from_numpy(fock_matrix)
                fock_matrix = fock_matrix.to(device=self.device)

                neighbour_list = self.neighbour_list_list[i]
                num_atoms = len(self.atomic_numbers_list[i])
                num_edges = len(neighbour_list[0])

                # Augment neighbor list with node self-neighbors, because we will stack the nodes together with the edges
                src_idx, target_idx = neighbour_list[0], neighbour_list[1]
                src_idxes = np.concatenate([src_idx, np.arange(num_atoms)])
                target_idxes = np.concatenate([target_idx, np.arange(num_atoms)])
                fock_block_offsets = np.concatenate([np.array([0]), np.cumsum(self.orbitals_per_atom_list[i])])

                # initialize tensors for node and edge labels for training (all other values are 0!)
                num_spins = 2 if open_shell else 1
                node_labels = torch.zeros((num_spins, num_atoms, self.target_len), device=self.device)
                edge_labels = torch.zeros((num_spins, num_edges, self.target_len), device=self.device)

                mol_atomic_numbers = self.atomic_numbers_list[i]

                # Populate the matrix elements into the correct positions in the labels
                for spin in range(num_spins):

                    if method == 'numpy_kernel':
                        labels = torch.zeros((num_edges + num_atoms, self.target_len), device=self.device)
                        matrix = fock_matrix[spin] if open_shell else fock_matrix
                        matrix2labels_kernels.numpy_single_matrix2label(
                                                                            self.orbital_template,
                                                                            fock_block_offsets,
                                                                            mol_atomic_numbers,
                                                                            src_idxes,
                                                                            target_idxes,
                                                                            matrix,
                                                                            labels,
                                                                            forward=True
                                                                        )
                    # call cupy kernel
                    else: 
                        cupy_dtype = self.torch_dtype_to_cupy_dtype(self.dtype)
                        matrix = cp.array(fock_matrix[spin], dtype=cupy_dtype) if open_shell else cp.array(fock_matrix, dtype=cupy_dtype)
                        labels = cp.zeros((num_edges + num_atoms, self.target_len), dtype=cupy_dtype)
                        matrix2labels_kernels.cupy_single_matrix2label(
                                                                        self.orbital_template,
                                                                        fock_block_offsets,
                                                                        mol_atomic_numbers,
                                                                        src_idxes,
                                                                        target_idxes,
                                                                        matrix,
                                                                        labels,
                                                                        orbital_template_ptrs,
                                                                        forward=True
                                                                    )
                        # cupy -> torch
                        labels = from_dlpack(labels.toDlpack())

                    # Basis transformation:
                    labels = self.basis_transformation.get_net_out(labels)

                    # ---------------------------------------------
                    node_labels[spin] = labels[num_edges:, :]
                    edge_labels[spin] = labels[:num_edges, :]
                    # ----------------------------------------------

                    # scale and shift the node labels (l=0 irreps) in the targets
                    if self.scale_shift_data is not None:
                        print("max and min before scaling: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item(), flush=True)
                        if open_shell:
                            node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers, spin_string=spin_strings[spin])
                        else:
                            node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers)
                    print("node label magnitude range: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item(), flush=True)

                # No distribution, store directly
                if not self.distribute_graphs:
                    self.node_labels_list.append(node_labels)
                    self.edge_labels_list.append(edge_labels)
                else:
                    node_labels_list.append(node_labels)
                    edge_labels_list.append(edge_labels)
            
        # process all incoming fock matrices at once
        else:

            open_shell = fock_matrices[0].ndim == 3
            num_spins = 2 if open_shell else 1

            # Populate the matrix elements into the correct positions in the labels
            all_labels = []
            all_src_idx = []
            all_target_idx = []
            all_mol_atomic_numbers = []
            all_fock_block_offsets = []
            all_node_labels = []
            all_edge_labels = []
            for i, neighbour_list in enumerate(self.neighbour_list_list):
                num_atoms = len(self.atomic_numbers_list[i])
                num_edges = len(neighbour_list[0])
                cupy_dtype = self.torch_dtype_to_cupy_dtype(self.dtype)
                all_labels.append(
                    cp.zeros((num_edges + num_atoms, self.target_len), dtype=cupy_dtype)
                )


                # Augment neighbor list with node self-neighbors, because we will stack the nodes together with the edges
                src_idx, target_idx = neighbour_list[0], neighbour_list[1]
                src_idxes = np.concatenate([src_idx, np.arange(num_atoms)])
                target_idxes = np.concatenate([target_idx, np.arange(num_atoms)])
                fock_block_offsets = np.concatenate([np.array([0]), np.cumsum(self.orbitals_per_atom_list[i])])

                all_src_idx.append(src_idxes)
                all_target_idx.append(target_idxes)

                mol_atomic_numbers = self.atomic_numbers_list[i]
                all_mol_atomic_numbers.append(mol_atomic_numbers)
                all_fock_block_offsets.append(fock_block_offsets)

                node_labels = torch.zeros((num_spins, num_atoms, self.target_len), device=self.device)
                edge_labels = torch.zeros((num_spins, num_edges, self.target_len), device=self.device)
                all_node_labels.append(node_labels)
                all_edge_labels.append(edge_labels)


            for spin in range(num_spins):

                cupy_dtype = self.torch_dtype_to_cupy_dtype(self.dtype)
                matrices = [cp.array(fock_matrix[spin], dtype=cupy_dtype) if open_shell else cp.array(fock_matrix, dtype=cupy_dtype) for fock_matrix in fock_matrices]

                if method == 'numpy_kernel':
                    raise NotImplementedError("Numpy kernel not implemented for multiple matrices yet!")
                
                # call cupy kernel
                else: 
                    matrix2labels_kernels.cupy_multiple_matrix2label(
                                                                    len(matrices),
                                                                    self.orbital_template,
                                                                    all_fock_block_offsets,
                                                                    all_mol_atomic_numbers,
                                                                    all_src_idx,
                                                                    all_target_idx,
                                                                    matrices,
                                                                    all_labels,
                                                                    orbital_template_ptrs,
                                                                    forward=True
                                                                )
                    # cupy -> torch
                    all_labels = [from_dlpack(label.toDlpack()) for label in all_labels]


                offsets = [0] + list(np.cumsum([len(self.neighbour_list_list[i][0]) + len(self.atomic_numbers_list[i]) for i in range(len(self.neighbour_list_list))]))
                all_labels_ = torch.cat(all_labels, dim=0)
                all_labels = self.basis_transformation.get_net_out(all_labels_)
                all_labels = [all_labels[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]

                for i, (node_labels, edge_labels, labels) in enumerate(zip(all_node_labels, all_edge_labels, all_labels)):

                    num_edges = len(self.neighbour_list_list[i][0])
                    mol_atomic_numbers = self.atomic_numbers_list[i]

                    # ---------------------------------------------
                    node_labels[spin] = labels[num_edges:, :]
                    edge_labels[spin] = labels[:num_edges, :]
                    # ----------------------------------------------

                    # scale and shift the node labels (l=0 irreps) in the targets
                    if self.scale_shift_data is not None:
                        print("max and min before scaling: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item(), flush=True)
                        if open_shell:
                            node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers, spin_string=spin_strings[spin])
                        else:
                            node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers)
                    print("node label magnitude range: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item(), flush=True)


            # No distribution, store directly
            if not self.distribute_graphs:
                self.node_labels_list = all_node_labels
                self.edge_labels_list = all_edge_labels
            else:
                node_labels_list = all_node_labels
                edge_labels_list = all_edge_labels

        # --------------------------- Redistribute targets based on domain decomposition ---------------------------

        if self.distribute_graphs:
            print("Redistributing fock labels based on domain decomposition... ", flush=True)

            comm = self.domain.comm
            num_local_fock_matrices = len(fock_matrices)
            total_num_fock_matrices = comm.allreduce(len(fock_matrices), op=MPI.SUM)

            # these are the indices of this rank's local fock matrix blocks in the global index space
            local_fock_ii_idxes = range(sum([len(self.atomic_numbers_list[i]) for i in range(num_local_fock_matrices)]))
            local_fock_ii_idxes = [i + self.global_fock_ii_start_offsets[self.rank] for i in local_fock_ii_idxes]
            local_fock_ij_idxes = range(sum([len(self.neighbour_list_list[i][0]) for i in range(num_local_fock_matrices)]))
            local_fock_ij_idxes = [i + self.global_fock_ij_start_offsets[self.rank] for i in local_fock_ij_idxes]

            # self.comm.Barrier()
            # print(f"Rank {self.rank} has local fock_ii indices: {local_fock_ii_idxes}", flush=True)
            # print(f"Rank {self.rank} has local fock_ij indices: {local_fock_ij_idxes}", flush=True)
            # print(f"rank {self.rank} needs fock_ii indices: ", range(self.domain.start_node, self.domain.end_node), flush=True)
            # print(f"rank {self.rank} needs fock_ij indices: ", range(self.domain.start_edge, self.domain.end_edge), flush=True)
            # self.comm.Barrier()

            # flatten node labels list to remove the molecule dimension:    
            # NOTE: Figure out how to add spin dimension back in later!
            if len(node_labels_list) > 0:
                node_labels_list = torch.cat(node_labels_list, dim=1).squeeze(0) 
                edge_labels_list = torch.cat(edge_labels_list, dim=1).squeeze(0)

                # print(f"Rank {self.rank} initial node_labels_list: ", node_labels_list[:, :5], flush=True)
                # print(f"Rank {self.rank} initial edge_labels_list: ", edge_labels_list[:, :5], flush=True)

            # ------------ Figure out the node re-distribution --------------

            # Need to recieve from rank "key" node_idx value(0) to position value(1)
            nodes_to_recv = defaultdict(list)
            self.node_labels_list = [None for _ in range(self.domain.local_num_nodes)]
            local_start = self.global_fock_ii_start_offsets[self.rank]
            local_end = self.global_fock_ii_end_offsets[self.rank]

            for pos_idx, node_idx in enumerate(range(self.domain.start_node, self.domain.end_node)):
                # This rank already owns the fock matrix that contains this node
                if local_start <= node_idx < local_end:
                    local_idx = node_idx - local_start
                    self.node_labels_list[pos_idx] = node_labels_list[local_idx]
                    
                # Fock label is not owned by this rank, need to recieve from another rank
                else:
                    for rank_idx in range(self.world_size):
                        if self.global_fock_ii_start_offsets[rank_idx] <= node_idx < self.global_fock_ii_end_offsets[rank_idx]:
                            nodes_to_recv[rank_idx].append((node_idx, pos_idx))
                            break

            # print(f"Rank {self.rank} needs to recv nodes: ", nodes_to_recv, flush=True)

            # Need to send global node idx "key" to rank "value"
            nodes_to_send = defaultdict(list) 
            for node_idx in local_fock_ii_idxes:
                for rank_idx in range(self.world_size):
                    if rank_idx == self.rank:
                        continue

                    rank_start = self.domain.displacements[rank_idx]
                    rank_end = rank_start + self.domain.counts[rank_idx]
                    if rank_start <= node_idx < rank_end:
                        nodes_to_send[rank_idx].append(node_idx)
                        break
                            
            # print(f"Rank {self.rank} needs to send nodes: {dict(nodes_to_send)}", flush=True)

            request_objects = []

            # ------------ Figure out the edge re-distribution --------------

            # Need to receive from rank "key" edge_idx value(0) to position value(1)
            edges_to_recv = defaultdict(list)
            self.edge_labels_list = [None for _ in range(self.domain.local_num_edges)]
            edge_local_start = self.global_fock_ij_start_offsets[self.rank]
            edge_local_end = self.global_fock_ij_end_offsets[self.rank]

            # 1. Map where we get our required edges from
            for pos_idx, edge_idx in enumerate(range(self.domain.start_edge, self.domain.end_edge)):
                if edge_local_start <= edge_idx < edge_local_end:
                    # We already own this edge
                    local_idx = edge_idx - edge_local_start
                    self.edge_labels_list[pos_idx] = edge_labels_list[local_idx]
                else:
                    # Search for which rank owns this edge in the fock matrix distribution
                    for rank_idx in range(self.world_size):
                        if self.global_fock_ij_start_offsets[rank_idx] <= edge_idx < self.global_fock_ij_end_offsets[rank_idx]:
                            edges_to_recv[rank_idx].append((edge_idx, pos_idx))
                            break

            # print(f"Rank {self.rank} needs to recv edges: ", dict(edges_to_recv), flush=True)

            # 2. Map where we need to send our current edges to
            edges_to_send = defaultdict(list)
            for edge_idx in local_fock_ij_idxes:
                for rank_idx in range(self.world_size):
                    if rank_idx == self.rank:
                        continue
                    
                    # Boundary in the domain decomposition (what the other rank needs)
                    r_edge_start = self.domain.edge_displacements[rank_idx]
                    r_edge_end = r_edge_start + self.domain.edge_counts[rank_idx]

                    if r_edge_start <= edge_idx < r_edge_end:
                        edges_to_send[rank_idx].append(edge_idx)
                        break

            # print(f"Rank {self.rank} needs to send edges: ", dict(edges_to_send), flush=True)

            # ------------- Perform Communication --------------

            NODE_TAG = 101
            EDGE_TAG = 102
            self.comm.Barrier() 

            # ----- Nodes -----

            # Exchange counts: How many NODES is each rank sending?
            node_send_nums = [len(nodes_to_send[r]) for r in range(self.world_size)]
            node_recv_nums = comm.alltoall(node_send_nums)

            node_recv_reqs = []
            node_recv_buffers = {}
            for src, count in enumerate(node_recv_nums):
                if count > 0 and src != self.rank:
                    # Pre-allocate buffer: [Count, Target_Len]
                    buf = np.empty((count, self.target_len), dtype=np.float32)  # REMOVE HARDCODED DTYPE
                    node_recv_buffers[src] = buf
                    node_recv_reqs.append(comm.Irecv(buf, source=src, tag=NODE_TAG))

            node_send_reqs = []
            node_send_buffers = {} 
            for target_rank, indices in nodes_to_send.items():
                if target_rank != self.rank and len(indices) > 0:
                    # Stack indices into a contiguous numpy array
                    try:                        
                        data = torch.stack([node_labels_list[idx - local_start] for idx in indices])                    
                    except Exception as e:                        
                        print(f"Rank {self.rank} failed stacking nodes for Rank {target_rank}. Indices: {indices}, local_start: {local_start}, list_len: {len(node_labels_list)}")                        
                        raise e
                    buf = data.detach().cpu().numpy().astype(np.float32)
                    node_send_buffers[target_rank] = buf
                    node_send_reqs.append(comm.Isend(buf, dest=target_rank, tag=NODE_TAG))

            # Wait and Slot
            if node_recv_reqs:
                MPI.Request.Waitall(node_recv_reqs)
                for src, buf in node_recv_buffers.items():
                    for i, (g_idx, pos_idx) in enumerate(nodes_to_recv[src]):
                        self.node_labels_list[pos_idx] = torch.from_numpy(buf[i]).to(self.device)

            if node_send_reqs:
                MPI.Request.Waitall(node_send_reqs)

            comm.Barrier()

            # ----- Edges -----

            # Exchange counts: How many EDGES is each rank sending?
            edge_send_nums = [len(edges_to_send[r]) for r in range(self.world_size)]
            edge_recv_nums = comm.alltoall(edge_send_nums)

            edge_recv_reqs = []
            edge_recv_buffers = {}
            for src, count in enumerate(edge_recv_nums):
                if count > 0 and src != self.rank:
                    buf = np.empty((count, self.target_len), dtype=np.float32)  # REMOVE HARDCODED DTYPE
                    edge_recv_buffers[src] = buf
                    edge_recv_reqs.append(comm.Irecv(buf, source=src, tag=EDGE_TAG))

            edge_send_reqs = []
            edge_send_buffers = {} 
            for target_rank, indices in edges_to_send.items():
                if target_rank != self.rank and len(indices) > 0:
                    try:
                        data = torch.stack([edge_labels_list[idx - edge_local_start] for idx in indices])
                    except Exception as e:
                        print(f"Rank {self.rank} failed stacking edges for Rank {target_rank}. Indices sample: {indices[:5]}, edge_local_start: {edge_local_start}")
                        raise e
                    buf = data.detach().cpu().numpy().astype(np.float32)
                    edge_send_buffers[target_rank] = buf
                    edge_send_reqs.append(comm.Isend(buf, dest=target_rank, tag=EDGE_TAG))

            # Wait and Slot
            if edge_recv_reqs:
                MPI.Request.Waitall(edge_recv_reqs)
                for src, buf in edge_recv_buffers.items():
                    for i, (g_idx, pos_idx) in enumerate(edges_to_recv[src]):
                        self.edge_labels_list[pos_idx] = torch.from_numpy(buf[i]).to(self.device)

            if edge_send_reqs:
                MPI.Request.Waitall(edge_send_reqs)

            self.comm.Barrier()

            # Finalize by stacking into tensors
            self.node_labels_list = torch.stack(self.node_labels_list)
            self.edge_labels_list = torch.stack(self.edge_labels_list)

            # self.comm.Barrier()
            # print(f"Rank {self.rank} final node_labels_list: ", self.node_labels_list[:, :5], flush=True)
            # print(f"Rank {self.rank} final edge_labels_list: ", self.edge_labels_list[:, :5], flush=True)
            # self.comm.Barrier()

            # --- 'Truly Local' edge reorder the fock edge and edge dist info to [truly local (rank owns src and dst), and the rest] ---
            src_fock_edges = torch.cat([self.edge_labels_list[self.domain.is_truly_local_edge, :], 
                                        self.edge_labels_list[~self.domain.is_truly_local_edge, :]], dim=0)
            self.edge_labels_list = src_fock_edges

            # src_edge_nodes = np.concatenate([self.local_edge_index[0, :][is_local], self.local_edge_index[0, :][~is_local]])
            # dst_edge_nodes = np.concatenate([self.local_edge_index[1, :][is_local], self.local_edge_index[1, :][~is_local]])
            # self.local_edge_index = np.stack([src_edge_nodes, dst_edge_nodes], axis=0)
            # self.truly_local_num_edges = np.sum(is_local)

            # Add the molecule index back in, but we pretend this is just one big molecule (so only molecule #0 exists)
            self.comm.Barrier()
            self.node_labels_list = self.node_labels_list.unsqueeze(0)
            self.edge_labels_list = self.edge_labels_list.unsqueeze(0)

            # ------ Flatten all the structure data into one molecule ------

            if len(self.atomic_numbers_list) > 0:
                self.atomic_positions_list = np.vstack(self.atomic_positions_list)
                self.edge_dist_list = torch.vstack(self.edge_dist_list)
                self.atomic_numbers_list = np.hstack(self.atomic_numbers_list)

            # allgather the data and then index only the current rank's portion
            all_atomic_numbers = comm.allgather(self.atomic_numbers_list)
            all_atomic_positions = comm.allgather(self.atomic_positions_list)
            all_edge_dists = comm.allgather(self.edge_dist_list)

            self.local_node_indices = self.domain.local_node_index
            local_edge_range = range(self.domain.start_edge, self.domain.end_edge)

            # print(f"Rank {self.rank} all_atomic_numbers: ", all_atomic_numbers, flush=True)
            # print(f"Rank {self.rank} all_atomic_positions: ", all_atomic_positions, flush=True)
            # print(f"Rank {self.rank} all_edge_dists: ", all_edge_dists, flush=True)
            # dist.barrier()
            
            # filtering out any empty lists (in case some ranks had no data)
            global_atomic_numbers = np.concatenate([x for x in all_atomic_numbers if len(x) > 0])
            global_pos   = np.concatenate([x for x in all_atomic_positions if len(x) > 0], axis=0)
            global_dist  = torch.cat([x for x in all_edge_dists if len(x) > 0], dim=0)

            # dist.barrier()
            # print(f"Rank {self.rank} global_atomic_numbers: ", global_atomic_numbers, flush=True)
            # print(f"Rank {self.rank} global_pos: ", global_pos, flush=True)
            # print(f"Rank {self.rank} global_dist: ", global_dist, flush=True)
            # dist.barrier()

            self.atomic_numbers_list = global_atomic_numbers[self.local_node_indices]
            self.atomic_positions_list = global_pos[self.local_node_indices]
            self.neighbour_list_list = self.domain.local_edge_index
            self.edge_dist_list = global_dist[local_edge_range, :]

            # Perform Truly Local reorder on edge dists:
            edge_dists = torch.cat([self.edge_dist_list[self.domain.is_truly_local_edge, :],
                                    self.edge_dist_list[~self.domain.is_truly_local_edge, :]], dim=0)
            self.edge_dist_list = edge_dists

            print("Final distributed atomic graph has ", len(self.atomic_numbers_list), " atoms and ", self.neighbour_list_list.shape[1], " edges on Rank ", self.rank, flush=True)
            # print(f"Rank {self.rank} self.atomic_numbers_list after allgather: ", self.atomic_numbers_list, flush=True)
            # print(f"Rank {self.rank} self.atomic_positions_list after allgather: ", self.atomic_positions_list, flush=True)
            # print(f"Rank {self.rank} self.neighbour_list_list after allgather: ", self.neighbour_list_list, flush=True)
            # print(f"Rank {self.rank} self.edge_dist_list after allgather: ", self.edge_dist_list, flush=True)
            self.comm.Barrier()
    
    def get_partition_map(self, cutoff, partition_type='linear', periodicity=False):
        """
        Returns a mapping from global node indices to partition IDs based on the specified partitioning strategy.
        Partition type: 
        'linear' (default) - simple contiguous blocks of nodes
        'metis' - use METIS graph partitioning
        'random' - assign nodes randomly to partitions
        """

        levels = self.world_size if partition_type in ['metis', 'random', 'linear', 'worstcase'] else int(np.log2(self.world_size))
        atomic_positions = self.merged_atomic_graph.atomic_positions
        edges = self.merged_atomic_graph.edge_matrix

        n_nodes = np.max(edges) + 1
        n_edges = len(edges[0,:])
        data = np.ones(n_edges)
        rows = np.array(edges[0,:])
        cols = np.array(edges[1,:])
        adj_matrix = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        
        if periodicity:
            lx = np.max(atomic_positions[:,0]) - np.min(atomic_positions[:,0])
            ly = np.max(atomic_positions[:,1]) - np.min(atomic_positions[:,1]) 
            lz = np.max(atomic_positions[:,2]) - np.min(atomic_positions[:,2]) 
            cell_size = np.array([lx, ly, lz])
        else:
            cell_size = None

        reordered_partitions = reorder.parition_wrapper(levels, atomic_positions, cell_size,
                            adj_matrix, cutoff, partition_type, 'num_neighbors')

        atom_reorder_perm = np.concatenate([o.reshape(-1) for o in reordered_partitions], axis=-1)
        atoms_per_partition = np.array([len(o) for o in reordered_partitions])

        ### PLOTTING THE STRUCTURE
        print("Writing atomic structure partition image...")
        parts_per_rank = [count for count in atoms_per_partition]
        cmap = plt.cm.rainbow(np.linspace(0, 1, len(parts_per_rank)))
        cmap = [(color[0], color[1], color[2], 0.5) for color in cmap]
        points = np.arange(0, len(parts_per_rank))
        np.random.shuffle(points)
        discrete_colormap = [cmap[int(point)] for point in points]
        color_parts = []
        for i, p in enumerate(parts_per_rank):
            tmp = np.ones((p, 4))
            tmp[:,:] *= discrete_colormap[i]    
            color_parts.extend(tmp)
        
        reordered_numbers = self.merged_atomic_graph.atomic_numbers[atom_reorder_perm]
        reordered_positions = self.merged_atomic_graph.atomic_positions[atom_reorder_perm]
        rotated_structure = Atoms(symbols=reordered_numbers, positions=reordered_positions)

        rotated_structure.rotate(5, 'x', center='COM')
        rotated_structure.rotate(20, 'y', center='COM')
        write('atomic_structure_' + partition_type + '_size={}.png'.format(self.world_size), rotated_structure, show_unit_cell=2, colors=color_parts)

        return reordered_partitions, atom_reorder_perm, atoms_per_partition

    def redistribute_graph(self, reordered_partitions, atom_reorder_perm, atoms_per_partition):
        """
        Redistributes the graph according to the provided partition map. This involves:
        1. Determining which nodes and edges belong to which partitions based on the partition map.
        2. Communicating the necessary node and edge data to the appropriate ranks so that each rank 
           ends up with the nodes and edges corresponding to its assigned partition.
        """

        # these atoms are owned by this rank according to the partition map
        original_own_partition_atoms = range(self.domain.start_node, self.domain.end_node)
        reordered_own_partition_atoms = reordered_partitions[self.rank]

        # get the incoming edges for these atoms from the global edge list
        global_edges = self.merged_atomic_graph.edge_matrix
        original_is_own_partition_edge = np.isin(global_edges[0,:], original_own_partition_atoms)
        reodered_is_own_partition_edge = np.isin(global_edges[0,:], reordered_own_partition_atoms)

        dist.barrier()
        for i in range(self.world_size):
            if i == self.rank:
                print(f"Rank {self.rank} owns {original_own_partition_atoms} atoms and {original_is_own_partition_edge} edges before re-ordered partitioning.", flush=True)
                print(f"Rank {self.rank} owns {reordered_own_partition_atoms} atoms and {reodered_is_own_partition_edge} edges after re-ordered partitioning.", flush=True)
            dist.barrier()

        raise NotImplementedError("Graph redistribution not implemented yet!")  
        


    def torch_dtype_to_cupy_dtype(self, torch_dtype):
        if torch_dtype == torch.float32:
            return cp.float32
        elif torch_dtype == torch.float64:
            return cp.float64
        else:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}, add to conversion")


    def scale_shift_node_blocks(self, node_blocks, node_atomic_numbers, spin_string=''):
        """
        Scale the l=0 values in the targets
        scales - a list of scaling factors for each l=0 irrep component
        shifts - a list of shifts for each l=0 irrep component
        scalar_indices - a list of indices in the node_labels that correspond to the l=0 irreps
        self.node_labels - the node labels that will be scaled
        NOTE: if an element does not have that scalar value, the corresponding mean is 0.0 and std is 1.0
        """
 
        # if node_atomic_numbers is None:
            # node_atomic_numbers = self.atomic_numbers

        means = self.scale_shift_data['element_scalar_means'+spin_string]
        stds = self.scale_shift_data['element_scalar_stds'+spin_string]
        scalar_indices = self.scale_shift_data['scalar_irrep_indices']

        # check for leading spin dimension (only one spin is passed in)
        unsqueeze = False
        if node_blocks.ndim == 3:
            unsqueeze = True
            node_blocks = node_blocks[0]

        # Process each node block
        for i, (node_block, z) in enumerate(zip(node_blocks, node_atomic_numbers)):
            z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)
            mean_vals = means[z]
            std_vals = stds[z]

            # Scale and shift the l=0 values in the node block
            for idx_offset, idx in enumerate(scalar_indices):
                node_block[idx] = (node_block[idx] - mean_vals[idx_offset]) / std_vals[idx_offset]

        if unsqueeze:
            node_blocks = node_blocks.unsqueeze(0)

        return node_blocks

    def unscale_shift_node_blocks(self, node_blocks, atomic_numbers):
        """
        Undo the scaling and shifting applied to the targets (l=0 values and optionally all irrep degrees).
        """


        new_node_blocks = node_blocks.clone()  # Create a copy to avoid modifying the original list

        means = self.scale_shift_data['element_scalar_means']
        stds = self.scale_shift_data['element_scalar_stds']
        scalar_indices = self.scale_shift_data['scalar_irrep_indices']

        for i, (node_block, z) in enumerate(zip(node_blocks, atomic_numbers)):
            z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)

            mean_vals = means[z]
            std_vals = stds[z]

            for idx_offset, idx in enumerate(scalar_indices):
                new_node_blocks[i][idx] = node_block[idx] * std_vals[idx_offset] + mean_vals[idx_offset]

        return new_node_blocks


    def get_target_len(self):
        """
        Returns the expected size of the targets which contain the maximum orbital interactions.
        This corresponds to max(Ns)x1 + max(Np)x3 + max(Nd)x5 + max(Nf)x7 + max(Ng)x9
        Searches for up to h-orbitals
        """

        N = 0
        for l in range(6):
            max_l_multiplicity = np.max([self.orbital_basis[el].count(l) for el in self.orbital_basis])
            N += (2*l + 1) * max_l_multiplicity

        return N**2

    def undo_scale_shift(self, node_blocks, atomic_numbers):

        # Unscale and shift the node blocks!
        if self.scale_shift_data is None:
            print("Possible Error! No scale/shift data provided! Not unscaling")
            return node_blocks
        else:
            print("Unscaling node blocks with scale/shift data")
            return self.unscale_shift_node_blocks(node_blocks, atomic_numbers)
            
    
    def to(self, device):
        """
        Moves all internal torch tensors to the specified device.
        """
        self.device = torch.device(device)
        
        # Move primary data tensors
        if self.node_labels is not None:
            self.node_labels = self.node_labels.to(device)
        if self.edge_labels is not None:
            self.edge_labels = self.edge_labels.to(device)
        if self.edge_dist is not None:
            self.edge_dist = self.edge_dist.to(device)
            
        # Move basis/metadata tensors
        if isinstance(self.ls_list, torch.Tensor):
            self.ls_list = self.ls_list.to(device)
            
        # Clear the large input matrix if it's a tensor
        if isinstance(self.fock_matrix, torch.Tensor):
            self.fock_matrix = self.fock_matrix.to(device)
        elif isinstance(self.fock_matrix, list):
            self.fock_matrix = [
                m.to(device) if isinstance(m, torch.Tensor) else m 
                for m in self.fock_matrix
            ]
        
        return self