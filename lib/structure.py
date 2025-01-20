# Description: This file contains the class definition for the Structure class. 
# The class contains methods to initialize the atomic structure from an XYZ file, 
# initialize the electronic structure from Hamiltonian and overlap matrices, 
# and extract orbital blocks from the Hamiltonian matrix based on the edges between atoms. 

from utils import orbital_type_dict
import utils as utils
from scipy.sparse import csr_matrix, coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os

# Atomic Simulation Environment (ASE) package   
from ase.io import read
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.geometry import find_mic
from dscribe.descriptors import SOAP

# Graph partitioning packages
import networkx as nx
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial import KDTree
import pymetis

from mpi4py import MPI
import warnings

# A structure defines the atomic and electronic structure of collection of atoms
class Structure:
    def __init__(
        self,
        xyz_file,
        hamiltonian_file,
        overlap_file,
        pbc,
        orbital_basis,
        dataset="custom",
        database_props=None,
        self_interaction=True,
        bothways=False,
        make_soap=False,
        save_matrices=False,
        rcut=4.0,
        is_reorder=False,
        reorder_method = ''
    ):
        # input quantities
        self.xyz_file = xyz_file                            # XYZ file containing atomic positions              
        self.hamiltonian_file = hamiltonian_file            # File containing the Hamiltonian matrix
        self.overlap_file = overlap_file                    # File containing the overlap matrix
        self.database_props = database_props                # SchNet database
        self.periodic_cell = None                           # Periodic cell size

        # Structure properties
        self.hamiltonian = None                             # Hamiltonian matrix
        self.overlap = None                                 # Overlap matrix
        self.neighbour_list = None                          # Neighbor list for atomic structure
        self.edge_matrix = None                             # Edge matrix for atomic structure
        self.num_orbitals_per_atom = None                   # Number of orbitals per atom
        self.num_unique_orbitals = None                     # Number of unique orbitals in the system
        self.soap_features = None                           # SOAP descriptor features 
        self.basis = orbital_basis                          # Orbital basis for electronic structure
        self.atomic_species = None                          # Atomic species in the structure
        self.atomic_numbers = None                          # Atomic numbers in the structure

        # Reordering properties, if the structure gets reordered before use
        self.is_reorder = is_reorder                        # Reorder the atomic structure
        self.reorder_method = 'CUSTOM'                      # Method to reorder the atomic structure
        self.reorder_map = None                             # Reorder map for the atomic structure, maps old atom indices to new atom indices
        self.counts = None                                  # Number of atoms in each partition

        # parameters:
        self.rcut = rcut                                    # cutoff radius for neighbor list

        if dataset == 'schnet':

            if database_props is None:
                raise ValueError("Database properties must be provided for SchNet dataset.")

            # initialize atomic structure
            self.init_atomic_structure_schnet(self.database_props, pbc, self_interaction, bothways)

            # initialize electronic structure
            self.init_electronic_structure_schnet(self.database_props)

        else: 

            # initialize atomic structure
            self.init_atomic_structure(self.xyz_file, pbc, self_interaction, bothways)

            # initialize SOAP features
            if make_soap:
                self.make_soap_features(pbc)

            # initialize electronic structure
            self.init_electronic_structure(self.hamiltonian_file, self.overlap_file, save_matrices)

        # reorder the structure if specified
        if is_reorder:
            self.reorder(self.reorder_method)
        else:
            self.reorder_map = np.arange(len(self.atomic_numbers))

    def init_atomic_structure_schnet(self, database_props, pbc, self_interaction, bothways):

        # Extract the xyz coordinates and atomic numbers from the database properties
        positions = np.array(database_props['_positions'], dtype=np.float64)
        atomic_numbers = np.array(database_props['_atomic_numbers'], dtype=int)
        self.atomic_numbers = atomic_numbers
        
        # Create an ASE Atoms object
        self.atomic_structure = Atoms(numbers=atomic_numbers, positions=positions, pbc=pbc)
        self.atomic_species = self.atomic_structure.get_chemical_symbols()

        # neighbor list
        array_rcut = np.ones(len(self.atomic_structure))*self.rcut
        self.neighbour_list = NeighborList(array_rcut, skin=0, self_interaction=self_interaction, bothways=bothways)
        self.neighbour_list.update(self.atomic_structure)

        # adjacency matrix
        matrix = self.neighbour_list.get_connectivity_matrix(sparse=True)
        matrix = matrix.tocoo()
        edge_matrix_np = np.array([matrix.row, matrix.col], dtype=np.int64)
        edge_matrix = torch.tensor(edge_matrix_np, dtype=torch.long)
        self.edge_matrix = edge_matrix_np


    def init_atomic_structure(self, xyz_file, pbc, self_interaction, bothways):
        """
        Initialize the atomic structure from an XYZ file.
        """

        # atomic positions
        self.atomic_structure = read(xyz_file)

        # set the elements in the atomic structure:
        self.atomic_species = self.atomic_structure.get_chemical_symbols()
        self.atomic_numbers = torch.tensor([utils.periodic_table[i] for i in self.atomic_species])

        # lattice vectors (periodic box size)
        if pbc:
            print("Periodic boundary conditions are set.")
            last_three_values = list(self.atomic_structure.info.keys())[-3:]
            lattice_vector_components = [float(value.strip(',')) for value in last_three_values]
            a, b, c = lattice_vector_components
            self.atomic_structure.set_cell([a, b, c])
            self.atomic_structure.set_pbc([pbc, pbc, pbc])
            self.periodic_cell = np.array([a, b, c])

        # neighbor list
        array_rcut = np.ones(len(self.atomic_structure))*self.rcut
        self.neighbour_list = NeighborList(array_rcut, skin=0, self_interaction=self_interaction, bothways=bothways)
        self.neighbour_list.update(self.atomic_structure)

        # adjacency matrix
        matrix = self.neighbour_list.get_connectivity_matrix(sparse=True)
        matrix = matrix.tocoo()
        edge_matrix_np = np.array([matrix.row, matrix.col], dtype=np.int64)
        self.edge_matrix = edge_matrix_np


    def partition_graph(self, n_clusters, write_xyz=False):

        """
        KMEANS: Partition the graph into `n_clusters` using K-means clustering.
        """
        # Create a NetworkX graph from the edge matrix
        G = nx.Graph()
        G.add_edges_from(self.edge_matrix.T)

        # Convert the graph to an adjacency matrix
        adj_matrix = nx.to_numpy_array(G)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(adj_matrix)

        # Group nodes by their cluster
        partitions = {i: np.where(labels == i)[0] for i in range(n_clusters)}

        if write_xyz:
            for i, (cluster, subgraph_nodes) in enumerate(partitions.items()):
                filename = 'cluster_' + str(cluster) + '.xyz'
                utils.write_xyz_file(filename, self.atomic_structure.get_chemical_symbols(), self.atomic_structure.get_positions(), subgraph_nodes)

        return partitions
    

    def init_electronic_structure_schnet(self, database_props):

        # initialize atomic orbital data
        self.num_orbitals_per_atom = [np.sum(2 * np.array(orbital_type_dict[self.basis][species]) + 1) for species in self.atomic_structure.get_chemical_symbols()]    
        unique_atomic_species = set(self.atomic_structure.get_chemical_symbols())
        self.num_unique_orbitals = np.sum([np.sum(2*np.array(orbital_type_dict[self.basis][species])+1) for species in unique_atomic_species])

        hamiltonian = database_props['hamiltonian']
        overlap = database_props['overlap']

        # convert complex spherical harmonics to real spherical harmonics by permuting the order of p-orbitals
        hamiltonian = self.complex_to_real_SH(hamiltonian)

        hamiltonian_csr = csr_matrix(hamiltonian)  
        overlap_csr = csr_matrix(overlap)  

        # check if hamiltonian_csr is symmetric
        assert((hamiltonian_csr != hamiltonian_csr.T).nnz == 0)

        self.hamiltonian = self.csr_to_dict(hamiltonian_csr)
        self.overlap = self.csr_to_dict(overlap_csr)

        # print("First 5 elements of the Hamiltonian matrix: ", list(self.hamiltonian.items())[:5])
        # self.imagesc_dict(self.hamiltonian, log=True)


    def init_electronic_structure(self, hamiltonian_file, overlap_file, save_matrices):
        """
        Initialize the electronic structure from the Hamiltonian and overlap matrices.
        """

        hamiltonian_pickle = "hamiltonian.pkl"
        overlap_pickle = "overlap.pkl"

        # set up the Hamiltonian and overlap matrices (load from saved pickle if they exist)
        if os.path.exists(hamiltonian_pickle) and save_matrices==True:
            print("Unpickling hamiltonian matrix...")
            with open(hamiltonian_pickle, "rb") as f:
                self.hamiltonian = pickle.load(f)
        else:
            self.hamiltonian = self.read_sparse_matrix_csr(hamiltonian_file)
            if save_matrices:
                with open(hamiltonian_pickle, "wb") as f:
                    pickle.dump(self.hamiltonian, f)

        # In case we want the overlap matrix
        # if os.path.exists(overlap_pickle):
        #     print("Unpickling overlap matrix...")
        #     with open(overlap_pickle, "rb") as f:
        #         self.overlap = pickle.load(f)
        # else:
        #     self.overlap = self.read_sparse_matrix_csr(overlap_file)
        #     with open(overlap_pickle, "wb") as f:
        #         pickle.dump(self.overlap, f)

        # initialize atomic orbital data
        self.num_orbitals_per_atom = [np.sum(2 * np.array(orbital_type_dict[self.basis][species]) + 1) for species in self.atomic_structure.get_chemical_symbols()]

        unique_atomic_species = set(self.atomic_structure.get_chemical_symbols())
        self.num_unique_orbitals = np.sum([np.sum(2*np.array(orbital_type_dict[self.basis][species])+1) for species in unique_atomic_species])


    def get_adjacentcy_matrix(self, edges):
        """
        Get the adjacency matrix from the edge matrix.
        """

        n_nodes = np.max(edges) + 1
        # no self loops
        n_edges = len(edges[0,:]) + n_nodes 
        data = np.ones(n_edges)
        rows = np.concatenate((edges[0,:], np.arange(n_nodes)))
        cols = np.concatenate((edges[1,:], np.arange(n_nodes)))
        return coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    def sparse_matrix_to_adjlist(self, matrix):
        """
        Transform a sparse matrix to an adjacency list.
        """

        if not isinstance(matrix, coo_matrix):
            matrix = matrix.tocoo()

        n_nodes = matrix.shape[0]
        adjacency_list = [[] for _ in range(n_nodes)]

        rows = matrix.row
        cols = matrix.col

        for i, j in zip(rows, cols):
            if i != j:  # Exclude self-loops
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

        # Remove duplicates and convert to set to ensure unique neighbors
        adjacency_list = [list(set(neighbors)) for neighbors in adjacency_list]

        return adjacency_list

    def cut_domain(self, sub_domain_size, n, atom_pos, atom_degree, order, dims=[0, 1, 2], atom_indices=None, origin=None):
        """
        Recursively cut a domain into sub-domains such that each sub-division has an equal number of atoms.
        """
        if origin is None:
            origin = np.zeros_like(sub_domain_size, dtype=float)

        if atom_indices is None:
            atom_indices = np.arange(len(atom_pos))

        if n == 0:
            order.append(atom_indices)
            return

        # Find the largest dimension to cut
        largest_dim = np.argmax(sub_domain_size[dims])          # dims = which dimension(s) to consider for cutting

        # Sort atoms along the largest dimension and find the median position
        sorted_indices = atom_indices[np.argsort(atom_pos[atom_indices, largest_dim])]
        sorted_degrees = atom_degree[sorted_indices]

        # Calculate the cumulative sum of degrees
        cumulative_edges = np.cumsum(sorted_degrees)
        total_edges = cumulative_edges[-1]

        # Find the split index where cumulative edges are approximately balanced
        split_idx = np.searchsorted(cumulative_edges, total_edges / 2)

        # Split atoms into left and right groups
        left_indices = sorted_indices[:split_idx]
        right_indices = sorted_indices[split_idx:]

        # Update the sub-domain size for the next cut
        sub_domain_size = sub_domain_size.copy()
        sub_domain_size[largest_dim] /= 2

        # Define origins for left and right sub-domains
        origin_left = origin.copy()
        origin_right = origin.copy()
        origin_right[largest_dim] = origin[largest_dim] + sub_domain_size[largest_dim]

        # Recursively cut the domain
        self.cut_domain(sub_domain_size, n - 1, atom_pos, atom_degree, order, dims, atom_indices=left_indices, origin=origin_left)
        self.cut_domain(sub_domain_size, n - 1, atom_pos, atom_degree, order, dims, atom_indices=right_indices, origin=origin_right)

    def get_degree(self):
        """
        Get the degree of each atom in the atomic structure (like the degree of each node in the graph)
        """

        num_atoms = len(self.atomic_structure)
        degree = np.zeros(num_atoms)

        modified_atomic_structure = self.atomic_structure.copy()
        cell = modified_atomic_structure.get_cell()
        modified_atomic_structure.wrap()

        positions = modified_atomic_structure.get_positions()

        # Use a KDTree for neighbor searching
        tree = KDTree(positions, boxsize=cell.diagonal())  # Takes into account periodicity

        # Query neighbors for all atoms
        for i in range(num_atoms):
            
            # Find indices of neighbors within the cutoff radius
            neighbors = tree.query_ball_point(positions[i], self.rcut)
            
            # Exclude self-interaction
            neighbors = [j for j in neighbors if j != i]
            
            # Update degree
            degree[i] = len(neighbors)

        return degree

    def reorder(self, method):
        """
        Reorder the graph to create a mapping from the original atom indices to the new atom indices.
        """
        adj_matrix = self.get_adjacentcy_matrix(self.edge_matrix)
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        atomic_positions = self.atomic_structure.get_positions()

        print("Reordering the graph using method: ", method)

        if method == 'RCM':
            self.reorder_map = np.array(reverse_cuthill_mckee(adj_matrix), dtype=np.int64)

        elif method == 'METIS':
            G = self.sparse_matrix_to_adjlist(adj_matrix)
            (_, parts) = pymetis.part_graph(size, adjacency=G)
            self.reorder_map = np.argsort(parts)
            parts = np.array(parts)
            self.counts = np.array([ np.sum(parts == k)  for k in range(size)])

        elif method == 'CUSTOM':
            # assert size power of 2
            if size & (size - 1) != 0:
                raise ValueError("Number of partitions must be a power of 2.")
            n = np.log2(size)

            # with padding
            lx = np.max(atomic_positions[:,0]) - np.min(atomic_positions[:,0]) + 0.0001
            ly = np.max(atomic_positions[:,1]) - np.min(atomic_positions[:,1]) + 0.0001
            lz = np.max(atomic_positions[:,2]) - np.min(atomic_positions[:,2]) + 0.0001

            sub_domain_size = np.array([lx, ly, lz])
            dims = [0, 1, 2]    # which dimensions cutting is allowed

            # list of arrays with atom indices
            order = []
            origin = np.array([np.min(atomic_positions[:,i]) for i in range(3)])
            atomic_degree = self.get_degree()

            self.cut_domain(sub_domain_size, n, atomic_positions, atomic_degree, order, dims, origin=origin)
            self.reorder_map = np.concatenate([o.reshape(-1) for o in order], axis=-1)
            self.counts = np.array([len(o) for o in order])

        else:
            # Reorder is true, but no valid method is specified
            warnings.warn("No valid method specified for reordering the graph. Using the original order.")
            self.reorder_map = np.arange(len(self.atomic_numbers))
            total_num_nodes = atomic_positions.shape[0]
            local_num_nodes = total_num_nodes // size
            self.counts = np.array([local_num_nodes] * size, dtype=np.int32)
            for i in range(total_num_nodes % size):
                self.counts[i] += 1


        # # ### PLOTTING TEST
        # if_plot = True
        # if rank == 0 and if_plot:
        #     parts_per_rank = [count for count in self.counts]

        #     from ase.io import write
        #     cmap = plt.cm.get_cmap('turbo')
        #     points = np.linspace(0, 1, len(parts_per_rank))
        #     discrete_colormap = [cmap(point) for point in points]
        #     color_parts = []
        #     for i, p in enumerate(parts_per_rank):
        #         tmp = np.ones((p, 4))
        #         tmp[:,:] *= discrete_colormap[i]    
        #         color_parts.extend(tmp)

        #     rotated_structure = self.atomic_structure[self.reorder_map].copy()
        #     rotated_structure.rotate(10, 'x', center='COM')
        #     rotated_structure.rotate(45, 'y', center='COM')
        #     write('atomic_structure_' + method + '_size={}_.png'.format(size), rotated_structure, show_unit_cell=2, colors=color_parts)
        # exit()
        # # ### END PLOTTING TEST


        # reorder structure with new atom indices
        self.atomic_structure = self.atomic_structure[self.reorder_map]
        self.atomic_numbers = torch.tensor([self.atomic_numbers[i] for i in self.reorder_map])
        self.atomic_species = [self.atomic_species[i] for i in self.reorder_map]
        # NOTE: self.num_orbitals_per_atom always refers to the original order!

        # Redo the neighbor list
        array_rcut = np.ones(len(self.atomic_structure))*self.rcut
        self.neighbour_list = NeighborList(array_rcut, skin=0, self_interaction=False, bothways=True)
        self.neighbour_list.update(self.atomic_structure)
        matrix = self.neighbour_list.get_connectivity_matrix(sparse=True)
        matrix = matrix.tocoo()
        edge_matrix_np = np.array([matrix.row, matrix.col], dtype=np.int64)
        self.edge_matrix = edge_matrix_np

    def complex_to_real_SH(self, hamiltonian):
        """
        Convert the ORCA order to CP2K order (only p and d orbitals implemented)
        """

        # iterate over atoms in structure:
        for i in range(len(self.atomic_structure)):

            species = self.atomic_structure.get_chemical_symbols()[i]
            starting_index = int(np.sum(self.num_orbitals_per_atom[:i]))       
            orbital_shell = orbital_type_dict[self.basis][species]
            num_s_orbitals = orbital_shell.count(0)
            num_p_orbitals = orbital_shell.count(1)
            num_d_orbitals = orbital_shell.count(2)

            for p in range(num_p_orbitals):
                start_p_index = starting_index + 1*num_s_orbitals + 3*p

                # ORCA order -> CP2K order: [2, 0, 1]
                # [-1, 0, 1] -> [1, -1, 0]
                # swap(0, 1), swap(0, 2)
                hamiltonian = self.swap(hamiltonian, start_p_index+0, start_p_index+1)
                hamiltonian = self.swap(hamiltonian, start_p_index+0, start_p_index+2)

            for d in range(num_d_orbitals):

                # ORCA order -> CP2K order: [4, 2, 0, 1, 3] 
                # [0, 1, -1, 2, -2] -> [-2, -1, 0, 1, 2]
                # swap(0, 4), (1, 2), (2, 4), (3, 4) 
                start_d_index = starting_index + 1*num_s_orbitals + 3*num_p_orbitals + 5*d
                hamiltonian = self.swap(hamiltonian, start_d_index+0, start_d_index+4)
                hamiltonian = self.swap(hamiltonian, start_d_index+1, start_d_index+2)
                hamiltonian = self.swap(hamiltonian, start_d_index+2, start_d_index+4)
                hamiltonian = self.swap(hamiltonian, start_d_index+3, start_d_index+4)

        return hamiltonian

    def swap(self, matrix, i, j):
        
        matrix[[i, j]] = matrix[[j, i]]
        matrix[:, [i, j]] = matrix[:, [j, i]]
        
        return matrix

    def csr_to_dict(self, csr_matrix):
        """
        Convert a CSR matrix to a dictionary format - ONLY FOR SCHNET
        """

        # Extract CSR components
        indptr = csr_matrix.indptr
        indices = csr_matrix.indices
        data = csr_matrix.data
        
        # Initialize dictionary to store (row, col) -> value mappings
        dict_matrix = {}
        
        # Populate the dictionary
        for row in range(len(indptr) - 1):
            start_idx = indptr[row]
            end_idx = indptr[row + 1]
            for idx in range(start_idx, end_idx):
                col = indices[idx]
                value = data[idx]
                # Note: the SCHNET hamiltonians are zero-indexed so we add 1
                dict_matrix[(row+1, col+1)] = value  
                # dict_matrix[(row, col)] = value  

        return dict_matrix


    def imagesc_dict(self, dict_matrix, log=True):
        """
        Plot the Hamiltonian matrix as an imagesc plot.
        """
        
        # Extract all row and column indices
        rows, cols = zip(*dict_matrix.keys())
        n_rows = max(rows) + 1
        n_cols = max(cols) + 1
        full_matrix = np.zeros((n_rows, n_cols))

        # Populate the full matrix with the data from the sparse matrix
        for (i, j), value in dict_matrix.items():
            if log:
                full_matrix[i, j] = np.log(np.abs(value))
            else:
                full_matrix[i, j] = value

        # Plot the matrix using matplotlib
        plt.figure()
        plt.imshow(full_matrix, cmap='Blues')
        c = plt.colorbar()
        c.ax.yaxis.label.set_size(15)
        c.set_label(r"log(|$(H_{ij})_{\alpha \beta}^{GT}$|)", fontsize=15)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('hamiltonian_matrix.png', dpi=500)


    def make_soap_features(self, pbc):
        """
        Make SOAP features for the atomic structure.
        """

        # Set up the SOAP descriptor
        species = self.atomic_structure.get_chemical_symbols()
        soap = SOAP(
            species=species,
            r_cut=7.0,
            n_max=5,
            l_max=5,
            rbf="polynomial",
            periodic=pbc,
            sparse=False,
        )

        # Get SOAP features
        self.soap_features = soap.create(self.atomic_structure)
        print("size of SOAP feature matrix: ", np.shape(self.soap_features))


    def read_matrix(self, file_path):
        """
        Read a matrix file and return the matrix in a dictionary format.
        """
        matrix_data = {}

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_str = line.strip().split()
                if len(data_str) >= 3:
                    indices = (int(data_str[0]), int(data_str[1]))
                    value = float(data_str[2])
                    matrix_data[indices] = value
                    # Assuming the matrix is symmetric, also add the transpose value
                    matrix_data[(indices[1], indices[0])] = value

        return matrix_data
    

    def read_sparse_matrix_csr(self, file_path):
        """
        Read a sparse matrix in CSR format from a file and return the matrix in a dictionary format.
        """

        indptr = []
        indices = []
        data = []

        print("reading file: ", file_path)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_str = line.strip().split()
                if len(data_str) >= 3:
                    indices.append([int(data_str[0]),int(data_str[1])])
                    data.append(float(data_str[2]))
        csr_matrix = {}
        for i in range(len(indices)):
            csr_matrix[(indices[i][0],indices[i][1])] = data[i]
            csr_matrix[(indices[i][1],indices[i][0])] = data[i]

        return csr_matrix
    
    def get_max_interaction_radius(self, eps):
        """
        Return the maximum distance between two atoms, such that the Hamiltonian matrix has at 
        least one element with a magnitude greater than eps. Also saves the interaction distances 
        to a file and plots a histogram of them.
        Require rcut to be overestimated.
        """

        cell = self.atomic_structure.get_cell()
        interaction_distance_list = []

        # iterate over all the edges in the edge matrix
        # for i, edge in enumerate(self.edge_matrix.T):
        for i, edge in enumerate(self.edge_matrix.T):

            print(i+1, "/", len(self.edge_matrix.T))

            # edge is a 1D array with two elements: [atom_i_index, atom_j_index]
            atom_i_index = edge[0]
            atom_j_index = edge[1]
            orbital_block = self.get_orbital_blocks([[atom_i_index], [atom_j_index]])

            # check if any element in the orbital block has a magnitude greater than eps
            for key in orbital_block:
                if np.max(np.abs(orbital_block[key])) > eps:
                    atom_i_pos = self.atomic_structure.get_positions()[atom_i_index]
                    atom_j_pos = self.atomic_structure.get_positions()[atom_j_index]
                    distance = find_mic(atom_i_pos - atom_j_pos, cell)
                    interaction_distance_list.append(distance[1])

        # save the interaction distances to a file
        with open('interaction_distances.txt', 'w') as f:
            for item in interaction_distance_list:
                f.write("%s\n" % item)

        print("Max interaction distance: ", max(interaction_distance_list))

        # plot a histogram of the interaction distances
        fig, ax = plt.subplots()
        ax.hist(interaction_distance_list, bins=50)
        ax.set_xlabel('Distance between atoms (A)')
        ax.set_ylabel('Frequency')
        plt.savefig('interaction_distances.png', dpi=300)
        
        return max(interaction_distance_list)
    

    def map_atom_to_orbital(self, atom_index):
        """
        Map the atom index to the starting orbital index and the number of orbitals
        """
        atom_index = self.reorder_map[atom_index]                                   # convert the new atom index to the original atom index  
        starting_index = int(np.sum(self.num_orbitals_per_atom[:atom_index])+1)     # index where this atom's orbitals start in H and S
        num_orbitals = self.num_orbitals_per_atom[atom_index]                       # number of orbitals for this atom

        return starting_index, num_orbitals

    
    def get_orbital_blocks(self, edge_idx):
        """
        Given the edges between two atoms (as a tuple), extract and return the corresponding orbital blocks
        from the hamiltonian matrix. (add overlap)
        """

        orbital_blocks = {}

        try:

            # iterates over all the edges specified in the input edge_idx list
            for i in range(len(edge_idx[0])): 

                # atom pair
                atom_i_index = edge_idx[0][i]
                atom_j_index = edge_idx[1][i]
                key_str = (atom_i_index, atom_j_index)

                # initialize size of the orbital block using the # orbitals of the two atoms
                starting_i, num_orbitals_i = self.map_atom_to_orbital(atom_i_index)
                starting_j, num_orbitals_j = self.map_atom_to_orbital(atom_j_index)
                mat = np.zeros(shape=(num_orbitals_i, num_orbitals_j), dtype = float)
                
                # fill in the orbital block from the hamiltonian matrix
                for alpha in range(num_orbitals_i):
                    for beta in range(num_orbitals_j):

                        # extract the hamiltonian value from the csr matrix if it exists (is nonzero)
                        if(starting_i+alpha, starting_j+beta) in self.hamiltonian:
                            mat[alpha,beta] = self.hamiltonian[(starting_i+alpha, starting_j+beta)]

                orbital_blocks[key_str] = mat

        except TypeError as e:
            print("TypeError occurred: {}".format(e))
            print("!! The hamiltonian and overlap files were probably not loaded into the Structure. !!")

        return orbital_blocks


# class for multiple molecules merged together into one big structure to make data processing easier:
class Merged_Structure(Structure):
    def __init__(self, structures_to_merge, dataset='custom', self_interaction=False, bothways=False):

        reorder_map = structures_to_merge[0].reorder_map

        assert(not structures_to_merge[0].is_reorder)       # Do not merge reordered structures!!!

        # get basic properties from the first structure in the list to use for the Structure constructor
        super().__init__(
            xyz_file=structures_to_merge[0].xyz_file,
            hamiltonian_file=structures_to_merge[0].hamiltonian_file,
            overlap_file=structures_to_merge[0].overlap_file,
            pbc=structures_to_merge[0].periodic_cell,
            orbital_basis=structures_to_merge[0].basis,
            dataset=dataset,
            database_props=structures_to_merge[0].database_props,
            self_interaction=self_interaction,
            bothways=bothways,
            rcut=structures_to_merge[0].rcut,
            is_reorder=False
        )

        self.reorder_map = np.arange(np.sum(len(structure.reorder_map) for structure in structures_to_merge))
        self.structures_to_merge = structures_to_merge
        self.merge_structures(self_interaction, bothways)

    def merge_structures(self, self_interaction, bothways):
        """
        Merge the atomic structures of the structures in the structures_to_merge
        """

        # collapse the two for loops!
        
        # merge the atomic structures
        combined_atomic_numbers = []
        combined_positions = []
        for structure in self.structures_to_merge:
            atomic_structure = structure.atomic_structure
            combined_atomic_numbers.extend(atomic_structure.get_atomic_numbers())
            combined_positions.extend(atomic_structure.get_positions())
            combined_pbc = atomic_structure.get_pbc()

            self.num_orbitals_per_atom.extend(structure.num_orbitals_per_atom)

        self.atomic_structure = Atoms(numbers=combined_atomic_numbers, positions=combined_positions, pbc=combined_pbc)
        self.atomic_species = self.atomic_structure.get_chemical_symbols()
        self.atomic_numbers = torch.tensor([utils.periodic_table[i] for i in self.atomic_species])
        
        unique_atomic_species = set(self.atomic_structure.get_chemical_symbols())
        self.num_unique_orbitals = np.sum([np.sum(2*np.array(orbital_type_dict[self.basis][species])+1) for species in unique_atomic_species])
        self.basis = self.structures_to_merge[0].basis

        # Build combined neighbor list
        node_track = 0
        self.edge_matrix = []
        src_edges = []
        dst_edges = []
        for structure in self.structures_to_merge:
            structure_edge_matrix = structure.edge_matrix

            # update the edge matrix to reflect the new atom indices
            new_edge_matrix_0 = [i + node_track for i in structure_edge_matrix[0]]
            new_edge_matrix_1 = [i + node_track for i in structure_edge_matrix[1]]
            src_edges.extend(new_edge_matrix_0)
            dst_edges.extend(new_edge_matrix_1)
            node_track += len(structure.atomic_numbers)

        self.edge_matrix = np.array([src_edges, dst_edges], dtype=np.int64)

        # merge the hamiltonian dictionaries, while updating the new atom indices
        hamiltonian = {}
        overlap = {}
        Hsize = 0
        for structure in self.structures_to_merge:
            for key in structure.hamiltonian:
                new_key = (key[0] + Hsize, key[1] + Hsize)
                hamiltonian[new_key] = structure.hamiltonian[key]

            Hsize += max(self.hamiltonian.keys(), key=lambda x: x[0])[0]
            
        self.hamiltonian = hamiltonian
        self.structures_to_merge[0].imagesc_dict(hamiltonian, log=True)
