import numpy as np
import torch
from . import utils_tensor_decomp
import torch
import time
from ase.neighborlist import NeighborList
import random

class Fock_Targets:
    """
    Consists of two main components:
    1. Atomic graph and connectivity list for the input structure
    2. Fock target analysis components for a given atomic basis (this is the same for any structure sharing the same atomic basis)
    3. If fock_matrix input is not None, computes the fock matrix decomposition into orbital blocks (for each pair of atoms)
    Sets up inputs/targets for supervised (atomic_structure -> Fock matrix) training for an set of atoms.
    Input target shape to standardize across molecules with different elements
    """

    def __init__(self, atoms, cutoff, orbital_basis, fock_matrix=None, target_len=0, dtype=torch.float32, reflection_symmetry=False, compute_fock_eigenvalues=False, scale_shift_data=None):
        """
        atoms - ASE atoms object of the atomic structure
        neighbor_list - H2O: [[0, 0, 1, 1, 2, 2], [1, 2, 2, 0, 0, 1]] 
        orbital_basis - H2O: {8: [0, 0, 0, 1, 1, 2], 1: [0, 0, 1]} (ex. dzvp)
        fock_matrix - Norb x Norb fock matrix (dense)
        reflection_symmetry - when True, only considers forward edge blocks (backward edges are constructed as the reflected forward ones)
        """

        if torch.cuda.is_available():
            self.device = torch.device('cuda')         
        else:
            self.device = torch.device('cpu')
                
        self.atoms = atoms                          
        self.orbital_basis = orbital_basis          
        self.dtype = dtype
        self.reflection_symmetry = reflection_symmetry

        # --> Atoms and connectivity list:
        num_atoms = len(atoms)
        neighbours = NeighborList(np.ones(num_atoms)*cutoff, skin=0, self_interaction=False, bothways=True)
        neighbours.update(self.atoms)
        neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
        self.neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])

        self.edge_dist = None
        self.make_edge_vectors()  # make edge vectors (distances and vector components)

        self.NA = len(atoms)
        self.atomic_numbers = self.atoms.get_atomic_numbers()
        self.orbitals_per_atom = ([ sum([(2*l+1)    
                                         for l in orbital_basis[atom_number]]) 
                                         for atom_number in self.atomic_numbers ])
        self.block_starts = np.hstack([0, np.cumsum(self.orbitals_per_atom)]) # start index of atom i in the matrix (and block_starts[-1] is the matrix size)         

        self.edge_type = "i < j"              # keep edges i, j where i < j or i > j
        if self.reflection_symmetry:
            if self.edge_type == "i < j":
                self.forward_edge_mask = self.neighbour_list[0] < self.neighbour_list[1]    
            elif self.edge_type == "i > j":
                self.forward_edge_mask = self.neighbour_list[0] > self.neighbour_list[1]
        else:
            self.forward_edge_mask = [True]*len(self.neighbour_list[0])                 # keep all edges
        
        # index of self.neighbour_list which contains the forward edge 
        self.reverse_edge_map = [-1] * len(self.neighbour_list[0])  # Initialize with -1 for safety
        edge_dict = {(i.item(), j.item()): idx for idx, (i, j) in enumerate(zip(self.neighbour_list[0], self.neighbour_list[1]))}
        for ind, (i, j) in enumerate(zip(self.neighbour_list[0], self.neighbour_list[1])):

            if self.edge_type == "i < j":
                edge_condition = i < j
            elif self.edge_type == "i > j":
                edge_condition = i > j

            if edge_condition:
                self.reverse_edge_map[ind] = ind
            else:
                self.reverse_edge_map[ind] = edge_dict.get((j.item(), i.item()), None)

        # --> Analyze structure of orbital interactions
        targets, self.req_output_irreps, self.simplified_out_irreps = utils_tensor_decomp.make_output_irreps(self.orbital_basis)     # list of all possible irreps required to capture the orbital interactions
        self.equivariant_blocks, out_js_list, self.orbital_starts = utils_tensor_decomp.process_targets(self.orbital_basis, targets)
        self.basis_transformation = utils_tensor_decomp.e3TensorDecomp(self.req_output_irreps,
                                                            out_js_list,
                                                            default_dtype_torch=dtype,
                                                            if_sort=False,
                                                            device_torch=self.device)
        
        # Shift back all the diffuse orbitals (which were incremented by 10 in utils_tensor_decomp.py)
        for atom, orbitals in self.orbital_basis.items():
            self.orbital_basis[atom] = [orb % 10 for orb in orbitals]

        for atom, orbitals in orbital_basis.items():
            orbital_basis[atom] = [orb % 10 for orb in orbitals]
        
        # print(f'Required irreps to represent orbital interactions: {self.req_output_irreps}')
        # print(f'Simplified irreps: {self.simplified_out_irreps}')
        self.scale_shift_data = scale_shift_data

        # If the fock targets should be computed on-the-fly rather than loaded from the db:
        self.node_labels = None
        self.edge_labels = None    
        if fock_matrix is not None:

            self.fock_matrix = torch.from_numpy(fock_matrix).to(self.device)

            if compute_fock_eigenvalues:
                eigenvalues, eigenvectors = torch.linalg.eigh(self.fock_matrix)   # compute eigenvalues and eigenvectors of the fock matrix
                self.eigenvalues = eigenvalues.to(self.device)                    # store eigenvalues
                self.eigenvectors = eigenvectors.to(self.device)                  # store eigenvectors

            self.target_len = target_len if target_len != 0 else None                  

            # Decompose the Fock matrix into orbital blocks and insert them into the targets
            self.make_targets()

            if self.scale_shift_data is not None:
                self.node_labels = self.scale_shift_node_blocks(self.node_labels)  # scale and shift the node labels (l=0 irreps) in the targets

    def make_targets(self):

        self.target_len = self.get_target_len()                                 # each target should fit in a NxN matrix (to be flattened)

        # initialize torch tensors of size N for nodes and (forward) edges
        node_labels = torch.zeros(( len(self.atoms), self.target_len ), dtype=self.dtype, device=self.device)
        edge_labels = torch.zeros(( len(self.neighbour_list[0]), self.target_len ), dtype=self.dtype, device=self.device)

        # Extract blocks from fock matrix:
        self_edges = [list(range(self.NA)), list(range(self.NA))]
        node_orbital_blocks = self.get_orbital_blocks(self_edges)
        edge_orbital_blocks = self.get_orbital_blocks(self.neighbour_list)

        # // !!! do garbage cleanup for the fock matrix here !!! //
        
        flat_blocks = []
        # Iterate over target blocks (each corresponding to an l1-l2 interaction)
        for index_target, equivariant_block in enumerate(self.equivariant_blocks):
            # Collect all the atom-atom interactions which will use this target block
            for N_M_str, block_slice in equivariant_block.items():
                condition_numbers = tuple(map(int, N_M_str.split()))    # atomic numbers
                slice_row = slice(block_slice[0], block_slice[1])
                slice_col = slice(block_slice[2], block_slice[3])
                slice_out = slice(self.orbital_starts[index_target], self.orbital_starts[index_target + 1])
                flat_blocks.append((condition_numbers, slice_row, slice_col, slice_out)) 
                # ^ from the interaction between atomic numbers 'cond1 and cond2', we extract the block defined by slide_row, slice_col and insert 
                # it into slice_out of the corresponding labels
        

        time_label_start = time.perf_counter()
        # Off-diagonal orbital blocks --> Edge labels
        atomic_numbers_i = self.atomic_numbers[self.neighbour_list[0]]
        atomic_numbers_j = self.atomic_numbers[self.neighbour_list[1]]
        for condition_numbers, slice_row, slice_col, slice_out in flat_blocks:
            condition_i, condition_j = condition_numbers
            mask = (atomic_numbers_i == condition_i) & (atomic_numbers_j == condition_j)

            # select relevant edges and accumulate the corresponding slices
            matching_indices = np.where(mask)[0]  
            for edge_idx in matching_indices:

                # only collect from forward edges if we are using reflected edges, the other edge_orbital_blocks are None
                if self.forward_edge_mask[edge_idx]:

                    edge_labels[edge_idx, slice_out] += torch.squeeze(
                        edge_orbital_blocks[edge_idx][slice_row, slice_col].reshape(1, -1)
                    )

        # Keep only the filled forward edges
        edge_labels = edge_labels[self.forward_edge_mask]

        # Diagonal orbital blocks --> Node labels
        atomic_numbers_i = self.atomic_numbers[self_edges[0]]
        atomic_numbers_j = self.atomic_numbers[self_edges[1]]
        for condition_numbers, slice_row, slice_col, slice_out in flat_blocks:
            condition_i, condition_j = condition_numbers
            mask = (atomic_numbers_i == condition_i) & (atomic_numbers_j == condition_j)

            # select relevant nodes and accumulate the corresponding slices
            matching_indices = np.where(mask)[0]  
            for node_idx in matching_indices:
                node_labels[node_idx, slice_out] += torch.squeeze(
                    node_orbital_blocks[node_idx][slice_row, slice_col].reshape(1, -1)
                )
        time_label_end = time.perf_counter()
        # print("time to make labels: ", time_label_end - time_label_start, flush=True)

        # Basis transformation:
        # ---------------------------------------------------------------------------------------------
        self.node_labels = self.basis_transformation.get_net_out(node_labels)
        self.edge_labels = self.basis_transformation.get_net_out(edge_labels)
        # ---------------------------------------------------------------------------------------------

    def make_edge_vectors(self):

        # ---------------------------------------------------------------------------------------------
        indices0 = self.neighbour_list[0]  # First atom indices
        indices1 = self.neighbour_list[1]  # Second atom indices

        self.edge_dist = torch.zeros((len(indices0), 4), dtype=self.dtype)
        self.edge_dist[:, 1:4] = torch.from_numpy(self.atoms.get_distances(indices1, indices0, vector=True))    # Vector components
        self.edge_dist[:, 0] = torch.linalg.norm(self.edge_dist[:, 1:4], dim=-1, keepdim=False)                 # Scalar distances
    
    
    def scale_shift_node_blocks(self, node_blocks):
        """
        Scale the l=0 values in the targets
        scales - a list of scaling factors for each l=0 irrep component
        shifts - a list of shifts for each l=0 irrep component
        scalar_indices - a list of indices in the node_labels that correspond to the l=0 irreps
        self.node_labels - the node labels that will be scaled
        NOTE: if an element does not have that scalar value, the corresponding mean is 0.0 and std is 1.0
        """

        means = self.scale_shift_data['element_scalar_means']
        stds = self.scale_shift_data['element_scalar_stds']
        scalar_indices = self.scale_shift_data['scalar_irrep_indices']

        # Process each node block
        for i, (node_block, z) in enumerate(zip(node_blocks, self.atomic_numbers)):
            z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)
            mean_vals = means[z]
            std_vals = stds[z]

            # Scale and shift the l=0 values in the node block
            for idx_offset, idx in enumerate(scalar_indices):
                node_block[idx] = (node_block[idx] - mean_vals[idx_offset]) / std_vals[idx_offset]
            # print("maximum value in node block for element", z, ":", torch.max(node_block).item(), flush=True)

        return node_blocks 

    def unscale_shift_node_blocks(self, node_blocks):
        """
        Undo the scaling and shifting applied to the l=0 values in the targets.
        """

        if self.scale_shift_data is None:
            print("Possible Error! No scale/shift data provided! Not unscaling")
            return node_blocks
            
        means = self.scale_shift_data['element_scalar_means']
        stds = self.scale_shift_data['element_scalar_stds']
        scalar_indices = self.scale_shift_data['scalar_irrep_indices']

        new_node_blocks = node_blocks.clone()  # Create a copy to avoid modifying the original list

        for i, (node_block, z) in enumerate(zip(node_blocks, self.atomic_numbers)):
            z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)

            mean_vals = means[z]
            std_vals = stds[z]

            for idx_offset, idx in enumerate(scalar_indices):

                # node_block[idx] = node_block[idx] * std_vals[idx_offset] + mean_vals[idx_offset]
                new_node_blocks[i][idx] = node_block[idx] * std_vals[idx_offset] + mean_vals[idx_offset]

            # new_node_blocks[i] = node_block
        return new_node_blocks


    def get_cartesian_and_spherical_rotations_to_yzx(self):
        """
        Specifically gets the cartesian and spherical rotations for xyz -> yzx
        Note: not needed.
        """
        R_cart = torch.tensor([[0.0,  0.0, 1.0],
                               [ 1.0, 0.0, 0.0],
                               [ 0.0, 1.0, 0.0]])
        R_sphere = self.req_output_irreps.D_from_matrix(R_cart).to(self.device)
        return R_cart, R_sphere

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

    def locate_atom_in_matrix(self, idx):
        return self.block_starts[idx], self.orbitals_per_atom[idx]
        
    def get_orbital_blocks(self, edges):
        """
        The order of the orbital blocks returned corresponds to the order of the input edges
        """

        orbital_blocks = {}

        for i in range(len(edges[0])): 

            atom_i_index = int(edges[0][i])
            atom_j_index = int(edges[1][i])

            # if it is a self edge or a forward edge (omit the backward edges)
            if atom_i_index == atom_j_index or self.forward_edge_mask[i]:

                starting_i, num_orbitals_i = self.locate_atom_in_matrix(atom_i_index)
                starting_j, num_orbitals_j = self.locate_atom_in_matrix(atom_j_index)
                mat = self.fock_matrix[starting_i:starting_i+num_orbitals_i, starting_j:starting_j+num_orbitals_j]

                orbital_blocks[i] = mat
            
            # if it is a backward edge  
            else: 
                orbital_blocks[i] = None
                

        return orbital_blocks
    
    def unpad_node_blocks(self, H_pred, atomic_numbers=None):
        
        atom_orbitals = self.orbital_basis
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers

        # Precompute number of orbitals for each atom
        atom_orbitals_count = {key: np.sum(2 * np.array(atom_orbitals[key]) + 1) for key in atom_orbitals}
        
        H_prev = {}
        
        for atom_ind in range(len(atomic_numbers)):
            
            key_term = (atom_ind, atom_ind)  # node key

            # Precompute number of orbitals for atoms i and j
            num_orbitals_i = atom_orbitals_count[atomic_numbers[atom_ind].item()]

            # Initialize H_prev for this edge
            H_prev[key_term] = torch.zeros((num_orbitals_i, num_orbitals_i), dtype=float)

            H_prev_edge = H_prev[key_term]  # just to avoid repeated dictionary lookup 

            for index_target, equivariant_block in enumerate(self.equivariant_blocks):
                slice_out = slice(self.orbital_starts[index_target], self.orbital_starts[index_target + 1])
                
                # Precompute block slices for this equivariant block
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])
                    len_row = block_slice[1] - block_slice[0]
                    len_col = block_slice[3] - block_slice[2]
                    
                    condition_atomic_number_i, condition_atomic_number_j = N_M_str.split()

                    if atomic_numbers[atom_ind].item() == int(condition_atomic_number_i) and atomic_numbers[atom_ind].item() == int(condition_atomic_number_j):
                        H_prev_edge[slice_row, slice_col] = H_pred[atom_ind][slice_out].reshape(len_row, len_col)

        return H_prev
    
    def unpad_edge_blocks(self, H_pred, atomic_numbers=None):
        
        edge_index = self.neighbour_list
        atom_orbitals = self.orbital_basis

        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers

        # Precompute number of orbitals for each atom
        atom_orbitals_count = {key: np.sum(2 * np.array(atom_orbitals[key]) + 1) for key in atom_orbitals}
        
        H_prev = {}
        edge_counter = 0
        
        for index_edge in range(edge_index.shape[1]):
            i = edge_index[0][index_edge].item()  # atom index 
            j = edge_index[1][index_edge].item()
            
            if self.forward_edge_mask[index_edge]:

                key_term = (i, j)  # edge key term 

                # Precompute number of orbitals for atoms i and j
                num_orbitals_i = atom_orbitals_count[atomic_numbers[i].item()]
                num_orbitals_j = atom_orbitals_count[atomic_numbers[j].item()]

                # Initialize H_prev for this edge
                H_prev[key_term] = torch.zeros((num_orbitals_i, num_orbitals_j), dtype=float)
                H_prev_edge = H_prev[key_term]  

                for index_target, equivariant_block in enumerate(self.equivariant_blocks):
                    slice_out = slice(self.orbital_starts[index_target], self.orbital_starts[index_target + 1])
                    
                    # Precompute block slices for this equivariant block
                    for N_M_str, block_slice in equivariant_block.items():
                        slice_row = slice(block_slice[0], block_slice[1])
                        slice_col = slice(block_slice[2], block_slice[3])
                        len_row = block_slice[1] - block_slice[0]
                        len_col = block_slice[3] - block_slice[2]
                        
                        condition_atomic_number_i, condition_atomic_number_j = N_M_str.split()

                        if atomic_numbers[i].item() == int(condition_atomic_number_i) and atomic_numbers[j].item() == int(condition_atomic_number_j):
                            H_prev_edge[slice_row, slice_col] = H_pred[edge_counter][slice_out].reshape(len_row, len_col)
                
                edge_counter += 1

        return H_prev

    def undo_scale_shift(self, node_blocks):

        # Unscale and shift the node blocks!
        if self.scale_shift_data is None:
            print("Possible Error! No scale/shift data provided! Not unscaling")
            return node_blocks
        else:
            print("Unscaling node blocks with scale/shift data")
            return self.unscale_shift_node_blocks(node_blocks)
    

    def reconstruct_matrix(self, node_blocks, edge_blocks, symmetrize_matrix_if_needed=False):
        """
        Note: always returns a symmetric matrix (symmetrizes it if not already symmetric)
        """

        N = self.block_starts[-1]
        reconstructed_matrix = torch.zeros((N, N), dtype=self.dtype, device=self.device)

        # Insert node orbital blocks (diagonal blocks)
        for i, node in enumerate(node_blocks):
            starting_i, num_orbitals_i = self.locate_atom_in_matrix(i)
            reconstructed_matrix[starting_i:starting_i+num_orbitals_i, starting_i:starting_i+num_orbitals_i] = node_blocks[node]

        # Insert edge orbital blocks (off-diagonal blocks)
        edge_counter = 0
        for index_edge in range(len(self.neighbour_list[0])):

            # get just the forward edges if we are using a reduced set of edges
            if self.forward_edge_mask[index_edge]:

                i, j = self.neighbour_list[0][index_edge], self.neighbour_list[1][index_edge]
                starting_i, num_orbitals_i = self.locate_atom_in_matrix(i)
                starting_j, num_orbitals_j = self.locate_atom_in_matrix(j)

                # Insert forward edge
                edge = edge_blocks[(i, j)]
                reconstructed_matrix[starting_i:starting_i+num_orbitals_i, starting_j:starting_j+num_orbitals_j] = edge

                # If reflection symmetry is used, insert the backward edge as the transpose
                if self.reflection_symmetry:
                    reconstructed_matrix[starting_j:starting_j+num_orbitals_j, starting_i:starting_i+num_orbitals_i] = edge.T

                edge_counter += 1
        
        # Check if the matrix is symmetric and symmetrize if not
        if not torch.allclose(reconstructed_matrix, reconstructed_matrix.T, atol=1e-10) and symmetrize_matrix_if_needed:
            print("Matrix is not already symmetrix! Symmetrizing the matrix")
            reconstructed_matrix = (reconstructed_matrix + reconstructed_matrix.T) / 2

        return reconstructed_matrix