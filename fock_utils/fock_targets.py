import numpy as np
import torch
from . import utils_tensor_decomp, matrix2labels_kernels
import torch
import time
from ase.neighborlist import NeighborList
import random
import cupy as cp

class Fock_Targets:
    """
    Consists of two main components:
    1. Atomic graph and connectivity list for the input structure
    2. Fock target analysis components for a given atomic basis (this is the same for any structure sharing the same atomic basis)
    3. If fock_matrix input is not None, computes the fock matrix decomposition into orbital blocks (for each pair of atoms)
    Sets up inputs/targets for supervised (atomic_structure -> Fock matrix) training for an set of atoms.
    Input target shape to standardize across molecules with different elements
    """

    def __init__(self, atoms, cutoff, orbital_basis,
                fock_matrix=None,
                charge=0,
                spin_multiplicity=1,
                dtype=torch.float32,
                half_edges=False,
                compute_fock_eigenvalues=False,
                scale_shift_data=None,
                orbital_starts=None,
                orbital_template=None,
                basis_transformation=None,
                req_output_irreps=None):
        """
        atoms - ASE atoms object of the atomic structure
        neighbor_list - H2O: [[0, 0, 1, 1, 2, 2], [1, 2, 2, 0, 0, 1]]
        orbital_basis - H2O: {8: [0, 0, 0, 1, 1, 2], 1: [0, 0, 1]} (ex. dzvp)
        fock_matrix - Norb x Norb fock matrix (dense)
        half_edges - when True, only considers 'forward' edge blocks (backward edges are constructed as the transpose of the forward ones)
        """

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.atoms = atoms
        self.orbital_basis = orbital_basis
        self.dtype = dtype
        self.half_edges = half_edges
        self.spin_multiplicity = spin_multiplicity
        self.charge = charge

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
        if self.half_edges:
            self.forward_edge_mask = self.neighbour_list[0] < self.neighbour_list[1]
        else:
            self.forward_edge_mask = [True]*len(self.neighbour_list[0])                 # keep all edges

        # index of self.neighbour_list which contains the forward edge
        self.reverse_edge_map = [-1] * len(self.neighbour_list[0])  # Initialize with -1 for safety
        edge_dict = {(i.item(), j.item()): idx for idx, (i, j) in enumerate(zip(self.neighbour_list[0], self.neighbour_list[1]))}
        for ind, (i, j) in enumerate(zip(self.neighbour_list[0], self.neighbour_list[1])):
            if i < j:
                self.reverse_edge_map[ind] = ind
            else:
                self.reverse_edge_map[ind] = edge_dict.get((j.item(), i.item()), None)

        # --> Analyze structure of orbital interactions
        if orbital_template is None or basis_transformation is None or req_output_irreps is None:
            targets, self.req_output_irreps, self.simplified_out_irreps, ls_list, out_js_list, self.orbital_starts, full_orb_interaction_list = utils_tensor_decomp.make_output_irreps(self.orbital_basis)
            self.equivariant_blocks = utils_tensor_decomp.process_targets(self.orbital_basis, targets, ls_list, out_js_list, full_orb_interaction_list)
            self.orbital_template = matrix2labels_kernels.get_orbital_template(self.equivariant_blocks, self.orbital_starts)
            self.basis_transformation = utils_tensor_decomp.e3TensorDecomp(self.req_output_irreps,
                                                                out_js_list,
                                                                default_dtype_torch=dtype,
                                                                if_sort=False,
                                                                device_torch=self.device)
        else:
            self.orbital_template = orbital_template
            self.orbital_starts = orbital_starts
            self.basis_transformation = basis_transformation
            self.req_output_irreps = req_output_irreps

        # ls list will define the max basis needed (eg, for OMOL: tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 1, 2])
        ls_list = []
        for l in range(20): # large to account for possible diffuse functions which are incremented by 10
            counts = [torch.sum(torch.tensor(self.orbital_basis[el]) == l) for el in self.orbital_basis]
            max_count = max(counts).item()
            ls_list.append(torch.tensor(max_count * [l], dtype=torch.int))

        # Shift back all the diffuse orbitals (which were incremented by 10 in utils_tensor_decomp.py)
        for atom, orbitals in self.orbital_basis.items():
            self.orbital_basis[atom] = [orb % 10 for orb in orbitals]

        self.ls_list = torch.cat(ls_list) % 10       # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]

        # print(f'Required irreps to represent orbital interactions: {self.req_output_irreps}')
        # print(f'Simplified irreps: {self.simplified_out_irreps}')
        self.scale_shift_data = scale_shift_data

        # If the fock targets should be computed on-the-fly rather than loaded from the db:
        self.node_labels = None
        self.edge_labels = None
        if fock_matrix is not None:

            # self.fock_matrix = torch.from_numpy(fock_matrix).to(self.device)
            self.fock_matrix = fock_matrix

            # if compute_fock_eigenvalues:
            #     fock_matrix = torch.from_numpy(fock_matrix).to(self.device)
            #     eigenvalues, eigenvectors = torch.linalg.eigh(fock_matrix)
            #     self.eigenvalues = eigenvalues.to(self.device)                    # store eigenvalues
            #     self.eigenvectors = eigenvectors.to(self.device)                  # store eigenvectors

            self.target_len = None

            # Decompose the Fock matrix into orbital blocks and insert them into the targets
            self.make_targets()

            if self.scale_shift_data is not None:
                self.node_labels = self.scale_shift_node_blocks(self.node_labels)  # scale and shift the node labels (l=0 irreps) in the targets

    def make_targets(self):
        """
        Creates padded node/edge labels from the fock matrix
        """

        open_shell = True if len(self.fock_matrix) == 2 else False

        # each target should fit in a NxN matrix (to be flattened)
        self.target_len = self.get_target_len()
        num_atoms = len(self.atomic_numbers)
        num_edges = len(self.neighbour_list[0])

        # Augment neighbor list with node self-neighbors, because we will stack the nodes together with the edges
        src_idx, target_idx = self.neighbour_list[0], self.neighbour_list[1]
        src_idxes = np.concatenate([src_idx, np.arange(num_atoms)])
        target_idxes = np.concatenate([target_idx, np.arange(num_atoms)])
        fock_block_offsets = np.concatenate([np.array([0]), np.cumsum(self.orbitals_per_atom)])

        # initialize tensors for node and edge labels for training
        num_spins = 2 if open_shell else 1
        self.node_labels = torch.empty((num_spins, num_atoms, self.target_len), device=self.device)
        self.edge_labels = torch.empty((num_spins, num_edges, self.target_len), device=self.device)

        for spin in range(num_spins):

            labels = torch.empty((num_edges + num_atoms, self.target_len), device=self.device)
            matrix = self.fock_matrix[spin] if open_shell else self.fock_matrix
            matrix = torch.from_numpy(matrix).to(self.device)

            # Populate the matrix elements into the correct positions in the labels
            matrix2labels_kernels.numpy_single_matrix2label(
                                                                self.orbital_template,
                                                                fock_block_offsets,
                                                                self.atomic_numbers,
                                                                src_idxes,
                                                                target_idxes,
                                                                matrix,
                                                                labels,
                                                                forward=True
                                                            )
            # Basis transformation:
            labels = self.basis_transformation.get_net_out(labels)

            # ---------------------------------------------
            self.node_labels[spin] = labels[num_edges:, :]
            self.edge_labels[spin] = labels[:num_edges, :]
            # ----------------------------------------------

    def make_edge_vectors(self):

        # ---------------------------------------------------------------------------------------------
        indices0 = self.neighbour_list[0]  # First atom indices
        indices1 = self.neighbour_list[1]  # Second atom indices

        self.edge_dist = torch.zeros((len(indices0), 4), dtype=self.dtype)
        self.edge_dist[:, 1:4] = torch.from_numpy(self.atoms.get_distances(indices1, indices0, vector=True))    # Vector components
        self.edge_dist[:, 0] = torch.linalg.norm(self.edge_dist[:, 1:4], dim=-1, keepdim=False)                 # Scalar distances


    # def scale_shift_node_blocks(self, node_blocks, node_atomic_numbers=None):
    #     """
    #     Scale the l=0 values in the targets
    #     scales - a list of scaling factors for each l=0 irrep component
    #     shifts - a list of shifts for each l=0 irrep component
    #     scalar_indices - a list of indices in the node_labels that correspond to the l=0 irreps
    #     self.node_labels - the node labels that will be scaled
    #     NOTE: if an element does not have that scalar value, the corresponding mean is 0.0 and std is 1.0
    #     """

    #     if node_atomic_numbers is None:
    #         node_atomic_numbers = self.atomic_numbers

    #     means = self.scale_shift_data['element_scalar_means']
    #     stds = self.scale_shift_data['element_scalar_stds']
    #     scalar_indices = self.scale_shift_data['scalar_irrep_indices']

    #     # Process each node block
    #     for i, (node_block, z) in enumerate(zip(node_blocks, node_atomic_numbers)):
    #         z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)
    #         mean_vals = means[z]
    #         std_vals = stds[z]

    #         # Scale and shift the l=0 values in the node block
    #         for idx_offset, idx in enumerate(scalar_indices):
    #             node_block[idx] = (node_block[idx] - mean_vals[idx_offset]) / std_vals[idx_offset]
    #         # print("maximum value in node block for element", z, ":", torch.max(node_block).item(), flush=True)

    #     return node_blocks

    def scale_shift_node_blocks(self, node_blocks, node_atomic_numbers=None):
        """
        Scale the l=0 values in the targets, and optionally all irrep degrees if extended data is available
        scales - a list of scaling factors for each l=0 irrep component
        shifts - a list of shifts for each l=0 irrep component
        scalar_indices - a list of indices in the node_labels that correspond to the l=0 irreps
        self.node_labels - the node labels that will be scaled
        NOTE: if an element does not have that scalar value, the corresponding mean is 0.0 and std is 1.0
        """

        if node_atomic_numbers is None:
            node_atomic_numbers = self.atomic_numbers

        # Check if we have extended multi-degree scaling data
        if 'element_irrep_means' in self.scale_shift_data and 'irrep_indices_by_l' in self.scale_shift_data:
            # Use new multi-degree scaling
            element_means = self.scale_shift_data['element_irrep_means']
            element_stds = self.scale_shift_data['element_irrep_stds']
            irrep_indices_by_l = self.scale_shift_data['irrep_indices_by_l']
            lmax = self.scale_shift_data['lmax']

            # Process each node block
            for i, (node_block, z) in enumerate(zip(node_blocks, node_atomic_numbers)):
                z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)

                # Scale each irrep degree
                for l in range(lmax + 1):
                    if l not in irrep_indices_by_l or z not in element_means[l]:
                        continue

                    irrep_indices = irrep_indices_by_l[l]
                    mean_vals = element_means[l][z]
                    std_vals = element_stds[l][z]

                    if l == 0:
                        # For l=0 (scalars), scale components directly
                        for idx_offset, idx in enumerate(irrep_indices):
                            node_block[idx] = (node_block[idx] - mean_vals[idx_offset]) / std_vals[idx_offset]

                    else:
                        # For l>0, scale the norms while preserving directions
                        irrep_start_idx = 0
                        for irrep_idx in range(len(mean_vals)):
                            # Get indices for this irrep (2*l+1 components)
                            irrep_size = 2 * l + 1
                            start_idx = irrep_indices[irrep_start_idx]
                            end_idx = irrep_indices[irrep_start_idx + irrep_size - 1] + 1

                            # Extract irrep components
                            irrep_components = node_block[start_idx:end_idx]

                            # Compute current norm
                            current_norm = torch.norm(irrep_components)

                            # Scale the norm
                            if current_norm > 1e-10:  # Avoid division by zero
                                target_norm = (current_norm - mean_vals[irrep_idx]) / std_vals[irrep_idx]
                                scale_factor = target_norm / current_norm
                                node_block[start_idx:end_idx] = irrep_components * scale_factor
                            else:
                                # For zero norms, just apply the scaling to the mean
                                scaled_mean = -mean_vals[irrep_idx] / std_vals[irrep_idx]
                                node_block[start_idx:end_idx] = scaled_mean / torch.sqrt(torch.tensor(irrep_size, dtype=node_block.dtype))

                            irrep_start_idx += irrep_size

        else:
            # Fallback to original l=0 only scaling for backwards compatibility
            means = self.scale_shift_data['element_scalar_means']
            stds = self.scale_shift_data['element_scalar_stds']
            scalar_indices = self.scale_shift_data['scalar_irrep_indices']

            # Process each node block
            for i, (node_block, z) in enumerate(zip(node_blocks, node_atomic_numbers)):
                z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)
                mean_vals = means[z]
                std_vals = stds[z]

                # Scale and shift the l=0 values in the node block
                for idx_offset, idx in enumerate(scalar_indices):
                    node_block[idx] = (node_block[idx] - mean_vals[idx_offset]) / std_vals[idx_offset]

        return node_blocks

    def unscale_shift_node_blocks(self, node_blocks):
        """
        Undo the scaling and shifting applied to the targets (l=0 values and optionally all irrep degrees).
        """

        if self.scale_shift_data is None:
            print("Possible Error! No scale/shift data provided! Not unscaling")
            return node_blocks

        new_node_blocks = node_blocks.clone()  # Create a copy to avoid modifying the original list

        # Check if we have extended multi-degree scaling data
        if 'element_irrep_means' in self.scale_shift_data and 'irrep_indices_by_l' in self.scale_shift_data:
            print("Using extended multi-degree unscaling")

            # Use new multi-degree unscaling
            element_means = self.scale_shift_data['element_irrep_means']
            element_stds = self.scale_shift_data['element_irrep_stds']
            irrep_indices_by_l = self.scale_shift_data['irrep_indices_by_l']
            lmax = self.scale_shift_data['lmax']

            # Process each node block
            for i, (node_block, z) in enumerate(zip(node_blocks, self.atomic_numbers)):
                z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)

                # Unscale each irrep degree
                for l in range(lmax + 1):
                    if l not in irrep_indices_by_l or z not in element_means[l]:
                        continue

                    irrep_indices = irrep_indices_by_l[l]
                    mean_vals = element_means[l][z]
                    std_vals = element_stds[l][z]

                    if l == 0:
                        # For l=0 (scalars), unscale components directly
                        for idx_offset, idx in enumerate(irrep_indices):
                            new_node_blocks[i][idx] = node_block[idx] * std_vals[idx_offset] + mean_vals[idx_offset]

                    else:
                        # For l>0, unscale the norms while preserving directions
                        irrep_start_idx = 0
                        for irrep_idx in range(len(mean_vals)):
                            # Get indices for this irrep (2*l+1 components)
                            irrep_size = 2 * l + 1
                            start_idx = irrep_indices[irrep_start_idx]
                            end_idx = irrep_indices[irrep_start_idx + irrep_size - 1] + 1

                            # Extract irrep components
                            irrep_components = node_block[start_idx:end_idx]

                            # Compute current scaled norm
                            current_scaled_norm = torch.norm(irrep_components)

                            # Unscale the norm: scaled_norm = (original_norm - mean) / std
                            # So: original_norm = scaled_norm * std + mean
                            if current_scaled_norm > 1e-10:  # Avoid division by zero
                                original_norm = current_scaled_norm * std_vals[irrep_idx] + mean_vals[irrep_idx]
                                if original_norm > 1e-10:
                                    scale_factor = original_norm / current_scaled_norm
                                    new_node_blocks[i][start_idx:end_idx] = irrep_components * scale_factor
                                else:
                                    # If original norm would be zero, set components to zero
                                    new_node_blocks[i][start_idx:end_idx] = torch.zeros_like(irrep_components)
                            else:
                                # If scaled norm is zero, reconstruct from the mean
                                original_norm = mean_vals[irrep_idx]
                                if original_norm > 1e-10:
                                    # Set to uniform distribution with the target norm
                                    new_node_blocks[i][start_idx:end_idx] = original_norm / torch.sqrt(torch.tensor(irrep_size, dtype=node_block.dtype))
                                else:
                                    new_node_blocks[i][start_idx:end_idx] = torch.zeros_like(irrep_components)

                            irrep_start_idx += irrep_size

        else:
            # Fallback to original l=0 only unscaling for backwards compatibility
            print("Using l=0 only unscaling")
            means = self.scale_shift_data['element_scalar_means']
            stds = self.scale_shift_data['element_scalar_stds']
            scalar_indices = self.scale_shift_data['scalar_irrep_indices']

            for i, (node_block, z) in enumerate(zip(node_blocks, self.atomic_numbers)):
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

    def undo_scale_shift(self, node_blocks):

        # Unscale and shift the node blocks!
        if self.scale_shift_data is None:
            print("Possible Error! No scale/shift data provided! Not unscaling")
            return node_blocks
        else:
            print("Unscaling node blocks with scale/shift data")
            return self.unscale_shift_node_blocks(node_blocks)

    # def reconstruct_matrix(self, node_blocks, edge_blocks, symmetrize_matrix_if_needed=False):
    #     """
    #     Note: always returns a symmetric matrix (symmetrizes it if not already symmetric)
    #     """

    #     N = self.block_starts[-1]
    #     reconstructed_matrix = torch.zeros((N, N), dtype=self.dtype, device=self.device)

    #     # Insert node orbital blocks (diagonal blocks)
    #     for i, node in enumerate(node_blocks):
    #         starting_i, num_orbitals_i = self.locate_atom_in_matrix(i)
    #         reconstructed_matrix[starting_i:starting_i+num_orbitals_i, starting_i:starting_i+num_orbitals_i] = node_blocks[node]

    #     # Insert edge orbital blocks (off-diagonal blocks)
    #     edge_counter = 0
    #     for index_edge in range(len(self.neighbour_list[0])):

    #         # get just the forward edges if we are using a reduced set of edges
    #         if self.forward_edge_mask[index_edge]:

    #             i, j = self.neighbour_list[0][index_edge], self.neighbour_list[1][index_edge]
    #             starting_i, num_orbitals_i = self.locate_atom_in_matrix(i)
    #             starting_j, num_orbitals_j = self.locate_atom_in_matrix(j)

    #             # Insert forward edge
    #             edge = edge_blocks[(i, j)]
    #             reconstructed_matrix[starting_i:starting_i+num_orbitals_i, starting_j:starting_j+num_orbitals_j] = edge

    #             # If reflection symmetry is used, insert the backward edge as the transpose
    #             if self.half_edges:
    #                 reconstructed_matrix[starting_j:starting_j+num_orbitals_j, starting_i:starting_i+num_orbitals_i] = edge.T

    #             edge_counter += 1

    #     # Check if the matrix is symmetric and symmetrize if not
    #     if not torch.allclose(reconstructed_matrix, reconstructed_matrix.T, atol=1e-10) and symmetrize_matrix_if_needed:
    #         print("Matrix is not already symmetrix! Symmetrizing the matrix")
    #         reconstructed_matrix = (reconstructed_matrix + reconstructed_matrix.T) / 2

    #     return reconstructed_matrix
