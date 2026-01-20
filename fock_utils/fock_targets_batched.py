import numpy as np
import torch
from . import utils_tensor_decomp, matrix2labels_kernels
import torch
import time
from ase.neighborlist import NeighborList
import random
import cupy as cp
import pickle, os
from ase import Atoms
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
                dataset_name='temp',
                dtype=torch.float32,
                compute_fock_eigenvalues=False,
                scale_shift_data=None,
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

        # Graph-wise distribution of the underlying data - IMPLEMENTING
        self.distribute_graphs = True

        # self.atoms = atoms
        self.orbital_basis = orbital_basis
        self.dtype = dtype

        # Create structures and neighbor lists (outer index is the molecule index) - follows the fock matrices owned by each rank
        self.atomic_numbers_list = []
        self.atomic_positions_list = []
        self.neighbour_list_list = []
        self.edge_dist_list = []
        self.orbitals_per_atom_list = []
        self.block_starts_list = []
        self.make_atomic_graphs(atomic_numbers, atomic_positions, cutoff)

        # Create a merged distributed graph
        if self.distribute_graphs:
            self.make_distributed_atomic_graph(atomic_numbers, atomic_positions, cutoff)

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

        # If the fock targets should be computed on-the-fly rather than loaded from the db:
        self.target_len = None

        # Decompose the Fock matrix into orbital blocks and insert them into the targets
        if fock_matrices is not None:
            self.make_targets(fock_matrices)


    def make_atomic_graphs(self, atomic_numbers, atomic_positions, cutoff):

        # The outer most index is the molecule index
        num_molecules = len(atomic_numbers)

        # --> Atoms and connectivity list:
        for numbers, positions in zip(atomic_numbers, atomic_positions):

            # Atoms for every molecule
            atoms = Atoms(symbols=numbers, positions=positions)
            num_atoms = len(numbers)
            self.atomic_numbers_list.append(atoms.get_atomic_numbers())
            self.atomic_positions_list.append(atoms.get_positions())

            # Neighbor lists for every molecule
            neighbours = NeighborList(np.ones(num_atoms)*cutoff, skin=0, self_interaction=False, bothways=True)
            neighbours.update(atoms)
            neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
            mol_neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])
            self.neighbour_list_list.append(mol_neighbour_list)

            # Edge distances for every molecule
            indices0 = mol_neighbour_list[0]  # First atom indices
            indices1 = mol_neighbour_list[1]  # Second atom indices

            mol_edge_dist = torch.zeros((len(indices0), 4), dtype=self.dtype)
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

    def make_distributed_atomic_graph(self, atomic_numbers, atomic_positions, cutoff):
        """
        1. Process local molecules.
        2. Synchronize all molecules across all ranks.
        3. Build one global graph.
        4. Apply Domain Decomposition.
        """
        # --- 1: Local Pre-processing ---
        local_mol_data = []
        for numbers, positions in zip(atomic_numbers, atomic_positions):
            atoms = Atoms(symbols=numbers, positions=positions)
            num_atoms = len(numbers)
            
            # Get neighbor list for this single molecule
            nl = NeighborList(np.ones(num_atoms)*cutoff, skin=0, self_interaction=False, bothways=True)
            nl.update(atoms)
            nb_matrix = nl.get_connectivity_matrix(sparse=True).tocoo()
            edge_index = np.vstack([nb_matrix.row, nb_matrix.col])
            
            # Calculate block starts for Fock targets later
            # orbitals_per_atom = [sum([(2*l+1) for l in self.orbital_basis[z]]) for z in atoms.get_atomic_numbers()]
            # block_starts = np.hstack([0, np.cumsum(orbitals_per_atom)])

            local_mol_data.append({
                'z': atoms.get_atomic_numbers(),
                'pos': atoms.get_positions(),
                'edge_index': edge_index,
                # 'orbitals_per_atom': orbitals_per_atom,
                # 'block_starts': block_starts
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
            
        print(f"Offsets for global node indices: {self.global_fock_ii_start_offsets}")
        print(f"Offsets for global edge indices: {self.global_fock_ij_start_offsets}")
        print(f"End Offsets for global node indices: {self.global_fock_ii_end_offsets}")
        print(f"End Offsets for global edge indices: {self.global_fock_ij_end_offsets}")

        # --- 3: Merging into a Global Super-Graph ---
        all_z = []
        all_pos = []
        global_edges = []
        
        current_node_offset = 0
        for mol in all_molecules:
            num_nodes = len(mol['z'])
            all_z.append(mol['z'])
            all_pos.append(mol['pos'])
            
            # Offset the local molecule edge indices to the global index space
            global_edges.append(mol['edge_index'] + current_node_offset)
            current_node_offset += num_nodes

        # Convert to giant arrays/tensors
        global_structure_z = np.concatenate(all_z)
        global_structure_pos = np.concatenate(all_pos)
        global_structure_edges = np.hstack(global_edges)

        # --- 4: Domain Decomposition ---
        self.merged_atomic_graph = MergedStructure(global_structure_z, global_structure_edges)
        self.domain = Domain_Decomp(self.merged_atomic_graph, device=self.device)

        # --- 5: Assign Local Properties ---
        # Now that Domain_Decomp has decided which nodes this rank owns:
        self.local_node_indices = self.domain.local_node_index
        self.local_atomic_numbers = torch.tensor(global_structure_z[self.local_node_indices], device=self.device)
        self.local_pos = torch.tensor(global_structure_pos[self.local_node_indices], device=self.device, dtype=self.dtype)
        
        # Extract edges owned by this rank (incoming edge split)
        # self.domain.local_edge_index already contains the subset of global_structure_edges for this rank
        # self.local_edge_index = torch.tensor(self.domain.local_edge_index, device=self.device)

        if self.rank == 0:
            print(f"Global graph created with {len(global_structure_z)} nodes and {global_structure_edges.shape[1]} edges.")
        self.domain.print_info()

    def make_targets(self, fock_matrices):
        """
        Creates padded node/edge labels from the fock matrix
        """

        method = 'cupy_kernel' # 'numpy_kernel' or 'cupy_kernel' 

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
                tmp = cp.zeros((inner_size,), dtype=cp.int32)
                for j, (row_slice, col_slice, output_slice) in enumerate(o):
                    tmp[j * 5 + 0] = row_slice.start
                    tmp[j * 5 + 1] = row_slice.stop
                    tmp[j * 5 + 2] = col_slice.start
                    tmp[j * 5 + 3] = col_slice.stop
                    tmp[j * 5 + 4] = output_slice.start
                orbital_template_tmp.append(tmp)
                orbital_template_ptrs.append(matrix2labels_kernels.get_ptr(tmp))

            orbital_template_ptrs = cp.array(orbital_template_ptrs, dtype=cp.uintp)
            cp.cuda.Stream.null.synchronize()

            cupy_dtype = self.torch_dtype_to_cupy_dtype(self.dtype)

        for i, fock_matrix in enumerate(fock_matrices):

            open_shell = fock_matrix.ndim == 3

            # Move fock matrix to device
            if not isinstance(fock_matrix, torch.Tensor):
                fock_matrix = torch.from_numpy(fock_matrix)
            fock_matrix = fock_matrix.to(device=self.device)

            neighbour_list = self.neighbour_list_list[i]
            num_atoms = len(self.atomic_numbers_list[i])
            num_edges = len(neighbour_list[0])
            spin_strings = ['_alpha', '_beta']

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
                print("max and min before scaling: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item(), flush=True)
                if self.scale_shift_data is not None:
                    if open_shell:
                        node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers, spin_string=spin_strings[spin])
                    else:
                        node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers)
                print("max and min after scaling: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item(), flush=True)

            # No distribution, store directly
            if not self.distribute_graphs:
                self.node_labels_list.append(node_labels)
                self.edge_labels_list.append(edge_labels)
            else:
                node_labels_list.append(node_labels)
                edge_labels_list.append(edge_labels)
        
        # --------------------------- Redistribute targets based on domain decomposition ---------------------------

        # In this case, need to communicate the targets to the correct ranks based on the domain decomposition
        if self.distribute_graphs:

            comm = self.domain.comm
            num_local_fock_matrices = len(fock_matrices)
            total_num_fock_matrices = comm.allreduce(len(fock_matrices), op=MPI.SUM)

            # these are the indices of this rank's local fock matrix blocks in the global index space
            local_fock_ii_idxes = range(sum([len(self.atomic_numbers_list[i]) for i in range(num_local_fock_matrices)]))
            local_fock_ii_idxes = [i + self.global_fock_ii_start_offsets[self.rank] for i in local_fock_ii_idxes]
            local_fock_ij_idxes = range(sum([len(self.neighbour_list_list[i][0]) for i in range(num_local_fock_matrices)]))
            local_fock_ij_idxes = [i + self.global_fock_ij_start_offsets[self.rank] for i in local_fock_ij_idxes]

            # print("self.global_fock_ii_start_offsets: ", self.global_fock_ii_start_offsets, flush=True)
            # print(f"Rank {self.rank} has local fock_ii indices: {local_fock_ii_idxes}", flush=True)
            # print(f"Rank {self.rank} has local fock_ij indices: {local_fock_ij_idxes}", flush=True)
            # print(f"rank {self.rank} needs fock_ii indices: ", range(self.domain.start_node, self.domain.end_node), flush=True)
            # print(f"rank {self.rank} needs fock_ij indices: ", 


            # flatten node labels list to remove the molecule dimension:    
            # NOTE: Figure out how to add spin dimension back in later!
            node_labels_list = torch.cat(node_labels_list, dim=1).squeeze(0) 
            edge_labels_list = torch.cat(edge_labels_list, dim=1).squeeze(0)

            print(f"Rank {self.rank} initial node_labels_list: ", node_labels_list[:, :5], flush=True)

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

            print(f"Rank {self.rank} needs to recv nodes: ", nodes_to_recv, flush=True)

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
                            
            print(f"Rank {self.rank} needs to send nodes: {dict(nodes_to_send)}", flush=True)

            request_objects = []

            # ------------- Perform Communication --------------

            # 1. Prepare Receives 
            recv_requests = []
            recv_buffers = {}

            for remote_rank, items in nodes_to_recv.items():
                # We expect a list of labels equal to the number of nodes we requested
                req = self.domain.comm.irecv(source=remote_rank, tag=11)
                recv_requests.append(req)
                recv_buffers[remote_rank] = req

            # 2. Prepare and Post Sends
            for remote_rank, node_indices in nodes_to_send.items():
                data_to_send = [node_labels_list[idx - local_start] for idx in node_indices]
                self.domain.comm.isend(data_to_send, dest=remote_rank, tag=11)

            # 3. Wait and Slot Data
            for remote_rank, req in recv_buffers.items():
                received_data = req.wait() # This is the list of labels from remote_rank
                
                # Use the stored pos_idx to put the data in the right spot
                # The order in received_data matches the order in nodes_to_recv[remote_rank]
                for i, (g_idx, pos_idx) in enumerate(nodes_to_recv[remote_rank]):
                    self.node_labels_list[pos_idx] = received_data[i]
                    
            print("Done communication")

            self.node_labels_list = torch.stack(self.node_labels_list)
            print(f"Rank {self.rank} final node_labels_list: ", self.node_labels_list[:, :5], flush=True)
            dist.barrier()
            exit()

            # ------------ Figure out the edge re-distribution --------------


        # now need to restructure node_labels_list in terms of the batch size.. use full batch for now (one giant molecule)



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

class MergedStructure:
    def __init__(self, z, edges):
        self.atomic_numbers = z
        self.edge_matrix = edges
        self.counts = None # Domain_Decomp will calculate uniform split if None

        print("Created MergedStructure with ", len(z), " atoms and edge list ", edges)

class Domain_Decomp():
    def __init__(self, structure, device, use_nccl=True):
        
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.comm = MPI.COMM_WORLD
        self.device = device
        
        # --> Split nodes between ranks
        total_num_nodes = len(structure.atomic_numbers) 
        local_num_nodes = total_num_nodes // self.size
        counts = np.array([local_num_nodes] * self.size, dtype=np.int32)
        for i in range(total_num_nodes % self.size):
            counts[i] += 1

        displacements = np.zeros_like(counts)
        for i in range(1, len(counts)):
            displacements[i] = displacements[i-1] + counts[i-1]

        self.counts = counts
        self.displacements = displacements

        # --> Start with the naive partition assignment (rank 0 gets the first partition, etc)
        start_node = displacements[self.rank]
        end_node = displacements[self.rank] + counts[self.rank]

        self.start_node = start_node
        self.end_node = end_node
        self.local_num_nodes = counts[self.rank]

        self.edge_split_type = "incoming"
        self.use_nccl = use_nccl

        # --> Split edges between ranks (naive split)
        if self.edge_split_type == "uniform":

            total_num_edges = structure.edge_matrix.shape[1]
            local_num_edges = total_num_edges // self.size

            start_edge = self.rank * local_num_edges
            end_edge = start_edge + local_num_edges

            if self.rank == self.size - 1:
                local_num_edges += total_num_edges % self.size
                end_edge += total_num_edges % self.size
        
            self.start_edge = start_edge
            self.end_edge = end_edge

        # --> Split edges between ranks (split based on nodes, each rank gets all edges of its nodes, no communication needed for aggregation)

        elif self.edge_split_type == "incoming":   

            # start edge is the first edge of the first node in the local node list
            start_edge_idx = 0
            for i, dst_edge in enumerate(structure.edge_matrix[0]):
                if dst_edge == self.start_node:
                    start_edge_idx = i
                    break
            
            # end edge is the last edge of the last node in the local node list
            end_edge_idx = len(structure.edge_matrix[0]) - 1
            for i, dst_edge in enumerate(structure.edge_matrix[0][::-1]):
                if dst_edge == self.end_node - 1:
                    end_edge_idx = len(structure.edge_matrix[0]) - i
                    break
            
            self.start_edge = start_edge_idx
            self.end_edge = end_edge_idx
        
        else: 
            print("Edge split type not recognized.")
                        

        # the numbers correspond to the full set of nodes and edges in the structure
        self.local_node_index = np.arange(start_node, end_node)
        self.local_edge_index = structure.edge_matrix[:, self.start_edge:self.end_edge]
        global_edge_index = structure.edge_matrix
        self.global_edge_index = torch.tensor(global_edge_index, device=self.device)
        self.global_atomic_numbers = torch.tensor(structure.atomic_numbers, device=self.device)

        # created and assigned during data creation:
        # self.global_edge_distance_vec = None
        # self.local_edge_idx = None

        # _________________________________________________________________________________________
        # initialize communication patterns for message passing

        # reorder the edge list so that the local edges are at the start of the list:
        local_node_nums = np.arange(self.start_node, self.end_node)
        is_local = np.isin(self.local_edge_index[1, :], local_node_nums)
        src_edge_nodes = np.concatenate([self.local_edge_index[0, :][is_local], self.local_edge_index[0, :][~is_local]])
        dst_edge_nodes = np.concatenate([self.local_edge_index[1, :][is_local], self.local_edge_index[1, :][~is_local]])
        self.local_edge_index = np.stack([src_edge_nodes, dst_edge_nodes], axis=0)
        self.truly_local_num_edges = np.sum(is_local)

        # print("Number of truly local edges: ", self.truly_local_num_edges, flush=True)
        # print("Number of remote edges: ", np.sum(~is_local), flush=True)

        # message creation
        self.expand_edge_0 = self.init_comm_pattern_expand(self.local_edge_index[0, :])     # dst node   
        self.expand_edge_1 = self.init_comm_pattern_expand(self.local_edge_index[1, :])     # src node
        self.expand_edge_0['use_nccl'] = self.use_nccl
        self.expand_edge_1['use_nccl'] = self.use_nccl

        # aggregation
        self.reduce_edge = self.init_comm_pattern_reduce(self.local_edge_index[0, :])

        # --> Shuffle gpus for topology-optimized partition assignment
        # rank_topology_assignment = redistribute_partitions(self)
        # structure.shuffle_partitions(rank_topology_assignment) ?
        # call init on self?


    def print_info(self):
        dist.barrier()
        for i in range(self.size):
            if self.rank == i:
                print("________________________________________________________")
                print(f"Rank {self.rank} has {self.end_node - self.start_node} nodes and {self.end_edge - self.start_edge} edges:")
                print(f"Rank {self.rank} has nodes from {self.start_node} to {self.end_node}: {self.local_node_index}")
                print(f"Rank {self.rank} has edges from {self.start_edge} to {self.end_edge}: {self.local_edge_index}")
            self.comm.Barrier()


    def init_comm_pattern_expand(self, edge_index):

        with torch.no_grad():

            # expand edge:
            local_num_nodes = len(self.local_node_index)
            total_num_nodes = self.comm.allreduce(local_num_nodes, op=MPI.SUM)
            num_nodes_local = total_num_nodes // self.size
            local_node_nums = torch.arange(self.start_node, self.end_node)
            
            # start and end nodes on every rank:
            start_nodes = self.comm.allgather(self.start_node)
            end_nodes = self.comm.allgather(self.end_node)

            #  get 'remote' nodes in this rank to be recieved from remote ranks
            remote_node_ranks = []
            remote_nodes = []
            for node in edge_index:
                # if the node is not local, it is remote
                if node < self.start_node or node >= self.end_node:
                    for i, (start, end) in enumerate(zip(start_nodes, end_nodes)):
                        if node >= start and node < end:
                            if node not in remote_nodes:
                                remote_node_ranks.append(i)
                                remote_nodes.append(node)
                            break 

            # Nodes to recieve on this rank 
            nodes_to_recv = {}
            for i, remote_rank in enumerate(remote_node_ranks):
                if remote_rank not in nodes_to_recv:
                    nodes_to_recv[remote_rank] = []
                if remote_nodes[i].item() not in nodes_to_recv[remote_rank]:
                    nodes_to_recv[remote_rank].append(remote_nodes[i].item())

            # allgatherv the edge_indices on each rank:
            length_local_edge_idx = len(edge_index)
            counts = self.comm.allgather(length_local_edge_idx)
            displacements = [0] + [sum(counts[:i]) for i in range(1, self.size)]

            total_length_edge_idx = sum(counts)        
            # all_edge_idx = torch.zeros(total_length_edge_idx, dtype=torch.int64)
            all_edge_idx = np.empty(total_length_edge_idx, dtype=edge_index.dtype)
            self.comm.Allgatherv(edge_index, (all_edge_idx, counts, displacements, MPI.INT))

            # Nodes to send from this rank
            nodes_to_send = {}
            # iterate over all_edge_idx, if this rank has a node which that rank does not, add it to the nodes to send
            for i, (c, d) in enumerate(zip(counts, displacements)):
                # look at all the nodes in the edge index for rank i
                for node in all_edge_idx[d:d+c]:
                    # if the node is not in the local nodes for rank 1, but is in the current local nodes:
                    if node in local_node_nums:
                        if node < start_nodes[i] or node >= end_nodes[i]:
                            # add the note to the send list, i is the rank to send to
                            if i not in nodes_to_send:
                                nodes_to_send[i] = []

                            if node not in nodes_to_send[i]:
                                nodes_to_send[i].append(int(node))

            indices_to_send = {}
            for target_rank, nodes in nodes_to_send.items():
                if nodes:
                    nodes_tensor = torch.tensor(nodes, dtype=torch.int64, requires_grad=False)
                    indices = torch.empty_like(nodes_tensor)
                    for j, node in enumerate(nodes_tensor):
                        idx = torch.where(local_node_nums == node)[0]
                        indices[j] = idx
                    indices_to_send[target_rank] = indices.to(self.device)

            dist.barrier()
            print("rank ", self.rank, " Nodes to send (during message creation): ", nodes_to_send)
            print("rank ", self.rank, " Nodes to recv (during message creation): ", nodes_to_recv)
            print("rank ", self.rank, " Indices to send (during message creation): ", indices_to_send)
            dist.barrier()

            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)

            local_node_nums = torch.tensor(local_node_nums, dtype=torch.long, device=self.device)
            is_local = torch.isin(edge_index, local_node_nums)
            local_edge_nodes = edge_index[is_local]
            local_indices = edge_index[is_local] - self.start_node

            remote_nodes = torch.tensor(remote_nodes, dtype=torch.long, device=self.device)
            is_remote = torch.isin(edge_index, remote_nodes)
            remote_edge_nodes = edge_index[is_remote]
            remote_indices = torch.ones(len(remote_edge_nodes), dtype=torch.long, device=self.device) # indices of received_embeddings to slot into the new embedding

            node_track = 0
            for i, (source_rank, nodes) in enumerate(nodes_to_recv.items()):
                if nodes: 
                    for node in nodes:  # node is the identity of the recieved node, not the index
                        remote_indices[torch.where(remote_edge_nodes == node)[0]] = node_track # locations in the new embedding where this recieved embedding should go
                        node_track += 1 # track the number of nodes received

            expand_edge_dict = {}
            expand_edge_dict['local_indices'] = local_indices
            expand_edge_dict['remote_indices'] = remote_indices
            expand_edge_dict['is_local'] = is_local
            expand_edge_dict['is_remote'] = is_remote
            expand_edge_dict['nodes_to_send'] = nodes_to_send
            expand_edge_dict['indices_to_send'] = indices_to_send
            expand_edge_dict['nodes_to_recv'] = nodes_to_recv
            expand_edge_dict['remote_nodes'] = remote_nodes
            expand_edge_dict['remote_node_ranks'] = remote_node_ranks
            expand_edge_dict['local_node_nums'] = local_node_nums
            expand_edge_dict['start_node'] = self.start_node
            expand_edge_dict['end_node'] = self.end_node
            expand_edge_dict['displacements'] = displacements
            expand_edge_dict['global_edge_idx'] = all_edge_idx

            # Convert PyTorch tensor directly to CuPy array to use when indexing flattened embeddings in SO3.py
            indices_to_send_cp = {
                target_rank: cp.asarray(nodes.to(self.device))  
                for target_rank, nodes in indices_to_send.items()
            }
            expand_edge_dict['indices_to_send_cp'] = indices_to_send_cp

            # torch versions of some index arrays, to allow for skipping of memory copies
            local_indices_torch = torch.tensor(local_indices, dtype=torch.long, device=self.device)
            remote_indices_torch = torch.tensor(remote_indices, dtype=torch.long, device=self.device)
            expand_edge_dict['local_indices_torch'] = local_indices_torch
            expand_edge_dict['remote_indices_torch'] = remote_indices_torch

        return expand_edge_dict


    def init_comm_pattern_reduce(self, edge_index):

        rank = dist.get_rank()
        size = dist.get_world_size()
        comm = MPI.COMM_WORLD

        local_num_nodes = len(self.local_node_index)
        total_num_nodes = self.comm.allreduce(local_num_nodes, op=MPI.SUM)
        num_nodes_local = total_num_nodes // self.size

        # nodes owned by this rank
        local_node_nums = torch.arange(self.start_node, self.end_node)

        # start and end nodes on every rank:
        start_nodes = self.comm.allgather(self.start_node)
        end_nodes = self.comm.allgather(self.end_node)
        # print("Total number of nodes: ", total_num_nodes, "rank: ", rank, "start nodes: ", start_nodes, " end nodes: ", end_nodes)

        # allgather the edge_indices on each rank to make a global_edge_index and counts and displacements:
        length_local_edge_idx = len(edge_index)
        edge_index_np = edge_index
        counts = comm.allgather(length_local_edge_idx)
        displacements = [0] + [sum(counts[:i]) for i in range(1, size)]

        total_length_edge_idx = sum(counts)
        global_edge_index = torch.zeros(total_length_edge_idx, dtype=torch.int64)
        comm.Allgatherv(edge_index_np, [global_edge_index, counts, displacements, MPI.LONG])
        
        local_edge_idx = torch.arange(self.start_edge, self.end_edge)
        local_edge_idx = local_edge_idx.to(self.device)
        self.local_edge_idx = local_edge_idx

        # messages to send are in the form of {rank: [indices of own self.embedding to send to rank]}
        messages_to_send = {}
        for i, target_node in enumerate(edge_index):
            if target_node in local_node_nums:
                pass # this is where the self-edges are handled
            else:
                for j, (start, end) in enumerate(zip(start_nodes, end_nodes)):
                    if target_node >= start and target_node < end:
                        if j not in messages_to_send:
                            messages_to_send[j] = []
                        messages_to_send[j].append(i)
                        break

        # messages to send are in the form of {rank: [indices of rank's embedding to be recieved]}
        messages_to_recv = {}
        for i, target_node in enumerate(global_edge_index):
            if target_node in local_node_nums and i not in local_edge_idx: # check here
                for j, (c, d) in enumerate(zip(counts, displacements)):
                    if i >= d and i < d + c:
                        if j not in messages_to_recv:
                            messages_to_recv[j] = []
                        messages_to_recv[j].append(i-d)
                        break

        # the messages are the indices of the local embeddings on the source rank
        dist.barrier()
        print(f"Rank {rank}: messages_to_send (during message aggregation) = {messages_to_send}")
        print(f"Rank {rank}: messages_to_recv (during message aggregation) = {messages_to_recv}")
        dist.barrier()
        

        for dest_rank, embedding_idxs in messages_to_send.items():
            if embedding_idxs:
                messages_to_send[dest_rank] = torch.tensor(embedding_idxs, dtype=torch.int64, device=self.device)


        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        local_node_nums = torch.tensor(local_node_nums, dtype=torch.long, device=self.device)

        is_local = (edge_index >= self.start_node) & (edge_index < self.end_node)
        local_indices = edge_index[is_local] - self.start_node

        # get remote indices to write into
        recv_pointer = 0
        slot_pointer = 0
        num_msgs_to_recv = sum([len(msgs) for msgs in messages_to_recv.values()])
        remote_indices = torch.zeros(num_msgs_to_recv, dtype=torch.long, device=self.device)       # already start collecting where the embeddings should go
        for source_rank, embedding_idxs in messages_to_recv.items():

            if embedding_idxs:
                node_start = displacements[source_rank]
                for j, idx in enumerate(embedding_idxs):
                    node_to_sum_into = global_edge_index[node_start + idx]
                    remote_indices[slot_pointer] = torch.where(local_node_nums == node_to_sum_into)[0].item()
                    slot_pointer += 1


        reduce_edge_dict = {}
        reduce_edge_dict['is_local'] = is_local
        reduce_edge_dict['local_indices'] = local_indices
        reduce_edge_dict['remote_indices'] = remote_indices
        reduce_edge_dict['messages_to_send'] = messages_to_send
        reduce_edge_dict['messages_to_recv'] = messages_to_recv
        reduce_edge_dict['local_node_nums'] = local_node_nums
        reduce_edge_dict['global_edge_index'] = global_edge_index
        reduce_edge_dict['start_node'] = self.start_node    
        reduce_edge_dict['end_node'] = self.end_node
        reduce_edge_dict['start_nodes'] = start_nodes
        reduce_edge_dict['counts'] = counts
        reduce_edge_dict['displacements'] = displacements

        return reduce_edge_dict