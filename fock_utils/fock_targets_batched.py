import torch
import time
import random
import pickle, os

import numpy as np
import cupy as cp
import scipy.sparse as sp
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
                partition_type="linear", # 'metis', 'random', 'worstcase', 'low_nn'
                dataset_name='temp',
                dtype=torch.float32,
                compute_fock_eigenvalues=False,
                scale_shift_data=None,
                distribute_graphs=False,
                tiling_dims=None,
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

        # Option to tile the structure, used for weak scaling
        if tiling_dims is not None and periodic_boxes is not None:
            if self.rank == 0:
                print(f"Tiling the structure with tiling dimensions: {tiling_dims}, periodic boxes: {periodic_boxes}", flush=True)
                atomic_numbers, atomic_positions, periodic_boxes = self.tile_structure(
                    atomic_numbers, atomic_positions, periodic_boxes, tiling_dims
                )

        # Graph-wise distribution of the underlying data 
        self.distribute_graphs = distribute_graphs
        self.partition_type = partition_type if distribute_graphs else "linear"

        self.orbital_basis = orbital_basis
        self.dtype = dtype

        # Create structures and neighbor lists (outer index is the molecule index) - follows the fock matrices owned by each rank
        self.num_structures = len(atomic_numbers)
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
        else:
            self.make_no_targets()
            

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
                # print("Computed edge distances with swapped indices for distributed graph!", flush=True)
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

        # Determine periodicity
        periodicity = False if periodic_boxes is None else True # single-structure periodicity only
        if periodicity and self.num_structures > 1:
            raise ValueError("Periodic partitioning only implemented for single structure for now!")

        # --- 4: Domain Decomposition ---
        dist.barrier()
        self.merged_atomic_graph = MergedStructure(global_structure_z, global_structure_pos, global_structure_edges, cutoff, periodicity)
        self.domain = Domain_Decomp(self.merged_atomic_graph, device=self.device, partition_type=self.partition_type)
        self.domain.print_info()
        dist.barrier()

        # Store global molecule ID of every atom this rank owns
        self.atom_mol_id = global_mol_ids[self.domain.local_node_indices]        

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
        self.target_len = self.basis_transformation.required_irreps_out.dim

        if self.distribute_graphs:
            node_labels_list = []
            edge_labels_list = []

        self.node_labels_list = []
        self.edge_labels_list = []

        if method == 'cupy_kernel':
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
                        if open_shell:
                            node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers, spin_string=spin_strings[spin])
                        else:
                            node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers)
                        
                # No distribution, store directly
                if not self.distribute_graphs:
                    self.node_labels_list.append(node_labels)
                    self.edge_labels_list.append(edge_labels)
                else:
                    node_labels_list.append(node_labels)
                    edge_labels_list.append(edge_labels)
            
            if len(fock_matrices) > 0:
                print("Rank ", self.rank, ": Node label magnitude range: ", torch.max(node_labels).item(), torch.min(node_labels).item(), flush=True)
            
        # process all incoming fock matrices at once
        else:
            print("Processing all fock matrices at once with cupy kernel... ")

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
                        if open_shell:
                            node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers, spin_string=spin_strings[spin])
                        else:
                            node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers)
                        print("node label magnitude range: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item(), flush=True)

            if len(fock_matrices) > 0:
                print("Rank ", self.rank, ": Node label magnitude range: ", torch.max(all_node_labels).item(), torch.min(all_node_labels).item(), flush=True)

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
            # self.comm.Barrier()

            # flatten node labels list to remove the molecule dimension:    
            # NOTE: Figure out how to add spin dimension back in later!
            if len(node_labels_list) > 0:
                node_labels_list = torch.cat(node_labels_list, dim=1).squeeze(0) 
                edge_labels_list = torch.cat(edge_labels_list, dim=1).squeeze(0)

            # ------------ Figure out the node re-distribution --------------

            # Need to recieve from rank "key" node_idx value(0) to position value(1)
            nodes_to_recv = defaultdict(list)
            self.node_labels_list = [None for _ in range(self.domain.local_num_nodes)]
            local_start = self.global_fock_ii_start_offsets[self.rank]
            local_end = self.global_fock_ii_end_offsets[self.rank]

            for pos_idx, node_idx in enumerate(self.domain.local_node_indices):
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

                    if node_idx in self.domain.all_local_node_indices[rank_idx]: 
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
            # for pos_idx, edge_idx in enumerate(range(self.domain.start_edge, self.domain.end_edge)):
            for pos_idx, edge_idx in enumerate(self.domain.local_edge_indices):
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
                    
                    # what the other rank needs
                    if edge_idx in self.domain.all_local_edge_indices[rank_idx]:
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
            # src_fock_edges = torch.cat([self.edge_labels_list[self.domain.is_truly_local_edge, :], 
            #                             self.edge_labels_list[~self.domain.is_truly_local_edge, :]], dim=0)
            # self.edge_labels_list = src_fock_edges

            # src_edge_nodes = np.concatenate([self.local_edges[0, :][is_local], self.local_edges[0, :][~is_local]])
            # dst_edge_nodes = np.concatenate([self.local_edges[1, :][is_local], self.local_edges[1, :][~is_local]])
            # self.local_edges = np.stack([src_edge_nodes, dst_edge_nodes], axis=0)
            # self.truly_local_num_edges = np.sum(is_local)

            # Add the molecule index back in, but we pretend this is just one big molecule (so only molecule #0 exists)
            self.comm.Barrier()
            self.node_labels_list = self.node_labels_list.unsqueeze(0)
            self.edge_labels_list = self.edge_labels_list.unsqueeze(0)

            # ------ Flatten all the structure data into one molecule ------

            if len(self.atomic_numbers_list) > 0:
                self.edge_dist_list = torch.vstack(self.edge_dist_list)

            # allgather the data and then index only the current rank's portion
            all_edge_dists = comm.allgather(self.edge_dist_list)

            # filtering out any empty lists (in case some ranks had no data)
            global_dist  = torch.cat([x for x in all_edge_dists if len(x) > 0], dim=0)

            # dist.barrier()
            # print(f"Rank {self.rank} global_dist: ", global_dist, flush=True)
            # dist.barrier()

            # Global
            self.atomic_numbers_list = self.merged_atomic_graph.atomic_numbers

            # Local
            self.atomic_positions_list = self.merged_atomic_graph.atomic_positions[self.domain.local_node_indices]
            self.neighbour_list_list = self.domain.local_edges
            self.edge_dist_list = global_dist[self.domain.local_edge_indices, :]

            # Perform Truly Local reorder on edge dists:
            # edge_dists = torch.cat([self.edge_dist_list[self.domain.is_truly_local_edge, :],
            #                         self.edge_dist_list[~self.domain.is_truly_local_edge, :]], dim=0)
            # self.edge_dist_list = edge_dists

            # print("Final distributed atomic graph has ", len(self.atomic_numbers_list), " atoms and ", self.neighbour_list_list.shape[1], " edges on Rank ", self.rank, flush=True)
            # print(f"Rank {self.rank} self.atomic_numbers_list after allgather: ", self.atomic_numbers_list, flush=True)
            # print(f"Rank {self.rank} self.atomic_positions_list after allgather: ", self.atomic_positions_list, flush=True)
            # print(f"Rank {self.rank} self.neighbour_list_list after allgather: ", self.neighbour_list_list, flush=True)
            # print(f"Rank {self.rank} self.edge_dist_list after allgather: ", self.edge_dist_list, flush=True)
            self.comm.Barrier()
    
    def make_no_targets(self):

        if len(self.atomic_numbers_list) > 0:
            self.edge_dist_list = torch.vstack(self.edge_dist_list)

        # allgather the data and then index only the current rank's portion
        all_edge_dists = self.comm.allgather(self.edge_dist_list)

        # filtering out any empty lists (in case some ranks had no data)
        global_dist  = torch.cat([x for x in all_edge_dists if len(x) > 0], dim=0)

        # Global
        self.atomic_numbers_list = self.merged_atomic_graph.atomic_numbers

        # Local
        self.atomic_positions_list = self.merged_atomic_graph.atomic_positions[self.domain.local_node_indices]
        self.neighbour_list_list = self.domain.local_edges
        self.edge_dist_list = global_dist[self.domain.local_edge_indices, :]


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


    def undo_scale_shift(self, node_blocks, atomic_numbers):

        # Unscale and shift the node blocks!
        if self.scale_shift_data is None:
            print("Possible Error! No scale/shift data provided! Not unscaling")
            return node_blocks
        else:
            print("Unscaling node blocks with scale/shift data")
            return self.unscale_shift_node_blocks(node_blocks, atomic_numbers)
    
    def tile_structure(self, atomic_numbers, atomic_positions, periodic_boxes, tiling_dims):
        """
        Tiles the first structure in the input list.
        tiling_dims: list or tuple of 3 integers, e.g., [2, 2, 2]
        """
        # 1. Create the ASE Atoms object for the first structure
        # periodic_boxes[0] should be a 3x3 matrix or [a, b, c, alpha, beta, gamma]
        unit_cell = periodic_boxes[0]
        
        mol = Atoms(
            numbers=atomic_numbers[0],
            positions=atomic_positions[0],
            cell=unit_cell,
            pbc=True # Tiling usually implies periodic boundary conditions
        )

        # 2. Tile the structure
        # If tiling_dims is [2, 2, 2], this creates an 8x larger supercell
        tiled_mol = mol * tiling_dims 

        # 3. Extract the new data
        new_atomic_numbers = [tiled_mol.get_atomic_numbers()]
        new_atomic_positions = [tiled_mol.get_positions()]
        
        # The new box is the original box scaled by the tiling dimensions
        # ASE updates the cell automatically during the multiplication
        new_periodic_boxes = [tiled_mol.get_cell().array]

        return new_atomic_numbers, new_atomic_positions, new_periodic_boxes
    
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