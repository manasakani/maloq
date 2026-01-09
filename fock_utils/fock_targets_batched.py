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

        # self.atoms = atoms
        self.orbital_basis = orbital_basis
        self.dtype = dtype

        # Create structures and neighbor lists
        self.atomic_numbers_list = []
        self.atomic_positions_list = []
        self.neighbour_list_list = []
        self.edge_dist_list = []
        self.orbitals_per_atom_list = []
        self.block_starts_list = []
        self.make_structures(atomic_numbers, atomic_positions, cutoff)

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
        self.node_labels_list = []
        self.edge_labels_list = []
        if fock_matrices is not None:

            self.target_len = None

            # Decompose the Fock matrix into orbital blocks and insert them into the targets
            self.make_targets(fock_matrices)


    def make_structures(self, atomic_numbers, atomic_positions, cutoff):

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

    def make_targets(self, fock_matrices):
        """
        Creates padded node/edge labels from the fock matrix
        """

        open_shell = False

        # each target should fit in a NxN matrix (to be flattened)
        self.target_len = self.get_target_len()

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

        orbital_template_ptrs = cp.array(
            orbital_template_ptrs, dtype=cp.uintp
        )

        for i, fock_matrix in enumerate(fock_matrices):
            fock_matrix = torch.from_numpy(fock_matrix).to(self.device)

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

            for spin in range(num_spins):

                # Populate the matrix elements into the correct positions in the labels

                # labels = torch.zeros((num_edges + num_atoms, self.target_len), device=self.device)
                # matrix = fock_matrix[spin] if open_shell else fock_matrix
                # matrix = torch.from_numpy(matrix).to(self.device)
                # matrix2labels_kernels.numpy_single_matrix2label(
                #                                                     self.orbital_template,
                #                                                     fock_block_offsets,
                #                                                     mol_atomic_numbers,
                #                                                     src_idxes,
                #                                                     target_idxes,
                #                                                     matrix,
                #                                                     labels,
                #                                                     forward=True
                #                                                 )

                matrix = cp.array(fock_matrix[spin]) if open_shell else cp.array(fock_matrix)
                labels = cp.zeros((num_edges + num_atoms, self.target_len), dtype=self.torch_dtype_to_cupy_dtype(self.dtype))
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
                labels = labels.get()
                labels = torch.from_numpy(labels).to(self.device)

                # Basis transformation:
                labels = self.basis_transformation.get_net_out(labels)

                # ---------------------------------------------
                node_labels[spin] = labels[num_edges:, :]
                edge_labels[spin] = labels[:num_edges, :]
                # ----------------------------------------------

                # scale and shift the node labels (l=0 irreps) in the targets
                print("max and min before scaling: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item())
                if self.scale_shift_data is not None:
                    if open_shell:
                        node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers, spin_string=spin_strings[spin])
                    else:
                        node_labels[spin] = self.scale_shift_node_blocks(node_labels[spin], mol_atomic_numbers)
                print("max and min after scaling: ", torch.max(node_labels[spin]).item(), torch.min(node_labels[spin]).item())

            self.node_labels_list.append(node_labels)
            self.edge_labels_list.append(edge_labels)


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

    def unscale_shift_node_blocks(self, node_blocks, atomic_numbers=None):
        """
        Undo the scaling and shifting applied to the targets (l=0 values and optionally all irrep degrees).
        """

        if self.scale_shift_data is None:
            print("Possible Error! No scale/shift data provided! Not unscaling")
            return node_blocks

        new_node_blocks = node_blocks.clone()  # Create a copy to avoid modifying the original list

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

    def undo_scale_shift(self, node_blocks):

        # Unscale and shift the node blocks!
        if self.scale_shift_data is None:
            print("Possible Error! No scale/shift data provided! Not unscaling")
            return node_blocks
        else:
            print("Unscaling node blocks with scale/shift data")
            return self.unscale_shift_node_blocks(node_blocks)
    
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
