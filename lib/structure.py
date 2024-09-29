# Description: This file contains the class definition for the Structure class. 
# The class contains methods to initialize the atomic structure from an XYZ file, 
# initialize the electronic structure from Hamiltonian and overlap matrices, 
# and extract orbital blocks from the Hamiltonian matrix based on the edges between atoms. 

from utils import orbital_type_dict
import utils as utils
from scipy.sparse import csr_matrix
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

# A structure defines the atomic and electronic structure of collection of atoms
class Structure:
    def __init__(self, xyz_file, hamiltonian_file, overlap_file, pbc, orbital_basis, dataset='custom', database_props=None, self_interaction=True, bothways=False, make_soap=False, save_matrices=False, rcut=4.0):

        # input quantities
        self.xyz_file = xyz_file                            # XYZ file containing atomic positions              
        self.hamiltonian_file = hamiltonian_file            # File containing the Hamiltonian matrix
        self.overlap_file = overlap_file                    # File containing the overlap matrix
        self.database_props = database_props                # SchNet database
        self.periodic_cell = None                           # Periodic cell size

        self.hamiltonian = None                             # Hamiltonian matrix
        self.overlap = None                                 # Overlap matrix
        self.neighbour_list = None                          # Neighbor list for atomic structure
        self.edge_matrix = None                             # Edge matrix for atomic structure
        self.num_orbitals_per_atom = None                   # Number of orbitals per atom
        self.num_unique_orbitals = None                     # Number of unique orbitals in the system
        self.soap_features = None                           # SOAP descriptor features 
        self.basis = orbital_basis                          # Orbital basis for electronic structure
        self.rotate_dic = None                              # dictionary of rotation matrices for each edge
        self.atomic_species = None                          # Atomic species in the structure
        self.atomic_numbers = None                          # Atomic numbers in the structure

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


    def init_atomic_structure_schnet(self, database_props, pbc, self_interaction, bothways):

        # Extract the xyz coordinates and atomic numbers from the database properties
        positions = np.array(database_props['_positions'], dtype=np.float64)
        atomic_numbers = np.array(database_props['_atomic_numbers'], dtype=int)
        
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
        plt.imshow(full_matrix, cmap='viridis')
        plt.colorbar()
        plt.title('CSR Matrix Visualization')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.savefig('hamiltonian_matrix.png', dpi=300)


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
                key_str = (atom_i_index,atom_j_index)

                # initialize size of the orbital block using the # orbitals of the two atoms
                starting_i, num_orbitals_i = self.map_atom_to_orbital(atom_i_index)
                starting_j, num_orbitals_j = self.map_atom_to_orbital(atom_j_index)
                mat = np.zeros(shape=(num_orbitals_i,num_orbitals_j),dtype = float)
                
                # fill in the orbital block from the hamiltonian matrix
                for alpha in range(num_orbitals_i):
                    for beta in range(num_orbitals_j):

                        # extract the hamiltonian value from the csr matrix if it exists (is nonzero)
                        if(starting_i+alpha,starting_j+beta) in self.hamiltonian:
                            mat[alpha,beta] = self.hamiltonian[(starting_i+alpha,starting_j+beta)]

                orbital_blocks[key_str] = mat

        except TypeError as e:
            print("TypeError occurred: {}".format(e))
            print("!! The hamiltonian and overlap files were probably not loaded into the Structure. !!")

        return orbital_blocks