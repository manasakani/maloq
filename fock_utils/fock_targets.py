import numpy as np
import torch
from . import utils_tensor_decomp
import torch
import time
from ase.neighborlist import NeighborList

class Fock_Targets:
    """
    Sets up inputs/targets for supervised (atomic_structure -> Fock matrix) training for an set of atoms.
    Input target shape to standardize across molecules with different elements
    """

    def __init__(self, atoms, cutoff, orbital_basis, fock_matrix, target_len=0, dtype=torch.float32):
        """
        atoms - ASE atoms object of the atomic structure
        neighbor_list - H2O: [[0, 0, 1, 1, 2, 2], [1, 2, 2, 0, 0, 1]] 
        orbital_basis - H2O: {8: [0, 0, 0, 1, 1, 2], 1: [0, 0, 1]} (ex. dzvp)
        fock_matrix - Norb x Norb fock matrix (dense)
        """

        if torch.cuda.is_available():
            self.device = torch.device('cuda')         
        else:
            self.device = torch.device('cpu')
                
        self.atoms = atoms                          
        self.orbital_basis = orbital_basis          
        self.fock_matrix = torch.from_numpy(fock_matrix).to(self.device)
        self.dtype = dtype

        # Connectivity list:
        num_atoms = len(atoms)
        neighbours = NeighborList(np.ones(num_atoms)*cutoff, skin=0, self_interaction=False, bothways=True)
        neighbours.update(self.atoms)
        neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
        self.neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])

        self.NA = len(atoms)
        self.atomic_numbers = self.atoms.get_atomic_numbers()
        self.orbitals_per_atom = ([ sum([(2*l+1)    
                                         for l in orbital_basis[atom_number]]) 
                                         for atom_number in self.atomic_numbers ])
        
        ### Using a different target shape per molecule ###
        molecule_orbital_basis = {atom_number: self.orbital_basis[atom_number] for atom_number in self.atomic_numbers}
        print("Using a molecule-specific basis! only one atom type")
        ### Using a different target shape per molecule ###
        
        # Analyze structure of orbital interactions
        # targets, self.req_output_irreps, self.simplified_out_irreps = utils_tensor_decomp.make_output_irreps(self.orbital_basis)     # list of all possible irreps required to capture the orbital interactions
        targets, self.req_output_irreps, self.simplified_out_irreps = utils_tensor_decomp.make_output_irreps(molecule_orbital_basis)     # list of all possible irreps required to capture the orbital interactions
        equivariant_blocks, out_js_list, orbital_starts = utils_tensor_decomp.process_targets(self.orbital_basis, targets)
        self.basis_transformation = utils_tensor_decomp.e3TensorDecomp(self.req_output_irreps,
                                                            out_js_list,
                                                            default_dtype_torch=dtype,
                                                            if_sort=False,
                                                            device_torch=self.device)
        
        print(f'Required irreps to represent orbital interactions: {self.req_output_irreps}')
        print(f'Simplified irreps: {self.simplified_out_irreps}')

        self.block_starts = np.hstack([0, np.cumsum(self.orbitals_per_atom)])       # start index of atom i in the matrix (and block_starts[-1] is the matrix size)
        self.target_len = target_len if target_len != 0 else None                   

        self.node_labels = None
        self.edge_labels = None
        self.edge_dist = None

        # Decompose the Fock matrix into orbital blocks and insert them into the targets
        self.make_targets(equivariant_blocks, orbital_starts)

    def make_targets(self, equivariant_blocks, orbital_starts):

        self.target_len = self.get_target_len()                                 # each target should fit in a NxN matrix (to be flattened)

        # initialize torch tensors of size N for nodes and edges
        node_labels = torch.zeros(( len(self.atoms), self.target_len ), dtype=self.dtype, device=self.device)
        edge_labels = torch.zeros(( len(self.neighbour_list[0]), self.target_len ), dtype=self.dtype, device=self.device)

        # Extract blocks from fock matrix:
        self_edges = [list(range(self.NA)), list(range(self.NA))]
        node_orbital_blocks = self.get_orbital_blocks(self_edges)
        edge_orbital_blocks = self.get_orbital_blocks(self.neighbour_list)

        # // !!! do garbage cleanup for the fock matrix here !!! //
        
        flat_blocks = []
        for index_target, equivariant_block in enumerate(equivariant_blocks):
            for N_M_str, block_slice in equivariant_block.items():
                condition_numbers = tuple(map(int, N_M_str.split()))
                slice_row = slice(block_slice[0], block_slice[1])
                slice_col = slice(block_slice[2], block_slice[3])
                slice_out = slice(orbital_starts[index_target], orbital_starts[index_target + 1])
                flat_blocks.append((condition_numbers, slice_row, slice_col, slice_out))

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

                edge_labels[edge_idx, slice_out] += torch.squeeze(
                    edge_orbital_blocks[edge_idx][slice_row, slice_col].reshape(1, -1)
                )

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

        # dump the targets:
        # import os
        # output_dir = './fock_tensors_dumped'
        # file_path = os.path.join(output_dir, f'molecule_{0}.txt')
        # with open(file_path, 'w') as f:

        #     f.write("node_labels\n")
        #     f.write(f"{self.node_labels.shape}\n")
        #     f.write(' '.join(map(str, self.node_labels.flatten().tolist())) + "\n")
            
        #     f.write("edge_labels\n")
        #     f.write(f"{self.edge_labels.shape}\n")
        #     f.write(' '.join(map(str, self.edge_labels.flatten().tolist())) + "\n")
        # exit()

        # # for debug:
        # print("after basis transform: ", self.node_labels[0])
        # plt.imshow(np.log(np.abs(self.node_labels[0].reshape(H_size, H_size).detach().cpu())))
        # plt.savefig("self.node_labels_transformed[0].png", dpi=300, bbox_inches='tight')
        # plt.close()

        # self.node_labels = self.basis_transformation.get_H(self.node_labels)
        # self.edge_labels = self.basis_transformation.get_H(self.edge_labels)

        # print("first label_netout: ", self.node_labels[0])
        # plt.imshow(np.log(np.abs(self.node_labels[0].reshape(H_size, H_size).detach().cpu())))
        # plt.savefig("self.node_labels_back[0].png", dpi=300, bbox_inches='tight')
        # plt.close()
        # print("self.neighbour_list: ", self.neighbour_list)
        # exit()
        
        # Apply Rotation to rotate (1) the structure and (2) every block of H from [xyz] to [yzx] order: 
        # - not needed, ORCA is already in yzx after permutation
        # ---------------------------------------------------------------------------------------------
        # R_cart, R_sphere = self.get_cartesian_and_spherical_rotations_to_yzx()

        # transpose each position into a [Nx1], multiply it by the rotation, and then transpose it back to [1xN]
        # self.node_labels = torch.matmul(R_sphere, self.node_labels.permute(1, 0)).permute(1, 0)
        # self.edge_labels = torch.matmul(R_sphere, self.edge_labels.permute(1, 0)).permute(1, 0)
        # self.atoms.positions = torch.matmul(R_cart, torch.tensor(self.atoms.get_positions().transpose(), dtype=self.dtype)).numpy().transpose()

        # Make edge vectors
        # ---------------------------------------------------------------------------------------------
        # self.edge_dist = torch.zeros(( len(self.neighbour_list[0]), 4 ), dtype=self.dtype)
        # for i in range(len( self.neighbour_list[0]) ):
        #     # self.edge_dist[i, 0:3] = torch.from_numpy(self.atoms.get_distance(self.neighbour_list[1][i], self.neighbour_list[0][i], vector=True)) # < -- this one is wrong
        #     self.edge_dist[i, 1:4] = torch.from_numpy(self.atoms.get_distance(self.neighbour_list[0][i], self.neighbour_list[1][i], vector=True)) # < -- this one is correct
        #     self.edge_dist[i, 0] = self.atoms.get_distance(self.neighbour_list[1][i], self.neighbour_list[0][i], vector=False)

        # Get all pairs of atom indices from neighbor list
        indices0 = self.neighbour_list[0]  # First atom indices
        indices1 = self.neighbour_list[1]  # Second atom indices

        self.edge_dist = torch.zeros((len(indices0), 4), dtype=self.dtype)
        self.edge_dist[:, 1:4] = torch.from_numpy(self.atoms.get_distances(indices1, indices0, vector=True))    # Vector components
        self.edge_dist[:, 0] = torch.linalg.norm(self.edge_dist[:, 1:4], dim=-1, keepdim=False)                 # Scalar distances

    def get_cartesian_and_spherical_rotations_to_yzx(self):
        """
        Specifically gets the cartesian and spherical rotations for xyz -> yzx
        """
        
        # R_cart = torch.tensor([[0.0,  1.0, 0.0],
        #                        [ 0.0, 0.0, 1.0],
        #                        [ 1.0, 0.0, 0.0]])
        R_cart = torch.tensor([[0.0,  0.0, 1.0],
                               [ 1.0, 0.0, 0.0],
                               [ 0.0, 1.0, 0.0]])
        R_sphere = self.req_output_irreps.D_from_matrix(R_cart).to(self.device)
        return R_cart, R_sphere

    def get_target_len(self):
        """
        Returns the expected size of the targets which contain the maximum orbital interactions.
        This corresponds to max(Ns)x1 + max(Np)x3 + max(Nd)x5 + max(Nf)x7 + max(Ng)x9
        Searches for up to g-orbitals
        """

        N = 0
        for l in range(5):
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

            starting_i, num_orbitals_i = self.locate_atom_in_matrix(atom_i_index)
            starting_j, num_orbitals_j = self.locate_atom_in_matrix(atom_j_index)
            mat = self.fock_matrix[starting_i:starting_i+num_orbitals_i, starting_j:starting_j+num_orbitals_j]

            orbital_blocks[i] = mat

        return orbital_blocks
    
    def reconstruct_matrix():
        raise NotImplementedError