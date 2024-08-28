import numpy as np
import torch
from scipy.spatial.transform import Rotation
from e3nn.o3 import Irrep, Irreps, matrix_to_angles
from ase.neighborlist import NeighborList


periodic_table = {'Ac': 89, 'Ag': 47, 'Al': 13, 'Am': 95, 'Ar': 18, 'As': 33, 'At': 85, 'Au': 79, 'B': 5, 'Ba': 56,
                  'Be': 4, 'Bi': 83, 'Bk': 97, 'Br': 35, 'C': 6, 'Ca': 20, 'Cd': 48, 'Ce': 58, 'Cf': 98, 'Cl': 17,
                  'Cm': 96, 'Co': 27, 'Cr': 24, 'Cs': 55, 'Cu': 29, 'Dy': 66, 'Er': 68, 'Es': 99, 'Eu': 63, 'F': 9,
                  'Fe': 26, 'Fm': 100, 'Fr': 87, 'Ga': 31, 'Gd': 64, 'Ge': 32, 'H': 1, 'He': 2, 'Hf': 72, 'Hg': 80,
                  'Ho': 67, 'I': 53, 'In': 49, 'Ir': 77, 'K': 19, 'Kr': 36, 'La': 57, 'Li': 3, 'Lr': 103, 'Lu': 71,
                  'Md': 101, 'Mg': 12, 'Mn': 25, 'Mo': 42, 'N': 7, 'Na': 11, 'Nb': 41, 'Nd': 60, 'Ne': 10, 'Ni': 28,
                  'No': 102, 'Np': 93, 'O': 8, 'Os': 76, 'P': 15, 'Pa': 91, 'Pb': 82, 'Pd': 46, 'Pm': 61, 'Po': 84,
                  'Pr': 59, 'Pt': 78, 'Pu': 94, 'Ra': 88, 'Rb': 37, 'Re': 75, 'Rh': 45, 'Rn': 86, 'Ru': 44, 'S': 16,
                  'Sb': 51, 'Sc': 21, 'Se': 34, 'Si': 14, 'Sm': 62, 'Sn': 50, 'Sr': 38, 'Ta': 73, 'Tb': 65, 'Tc': 43,
                  'Te': 52, 'Th': 90, 'Ti': 22, 'Tl': 81, 'Tm': 69, 'U': 92, 'V': 23, 'W': 74, 'Xe': 54, 'Y': 39,
                  'Yb': 70, 'Zn': 30, 'Zr': 40, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
                  'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}

# for DZVP
orbital_type_dict_DZVP = {'H':[0,0,1],'O':[0,0,1,1,2]} 
                              #1s 2s 2p    1s 2s 2p 3s 3p 3d           s s p d 

num_orbitals_per_atom_DZVP = {'H':5, 'O':13}

orbital_type_dict_def2_SVP = {'H':[0,0,1],'O':[0, 0, 0, 1, 1, 2], 'N':[0, 0, 0, 1, 1, 2], 'C':[0, 0, 0, 1, 1, 2]}  # for Schnet

num_orbitals_per_atom_def2_SVP = {'H':5, 'O':14, 'N':14, 'C':14}


# for SZVP 
orbital_type_dict_SZV = {'H':[0],'O':[0,1],'Hf':[0,0,1,2], 'Cl': [0, 1], 'Na': [0, 0, 1]} #Hf includes 3s, 3p and 4d orbitals
                         #s       s p        s s p d          s  p          s s p

num_orbitals_per_atom_SZV = {'H':1, 'O':4, 'Hf':10, 'Cl':4, 'Na':5}

orbital_type_dict = {
    'DZVP': orbital_type_dict_DZVP,
    'def2_SVP': orbital_type_dict_def2_SVP,
    'SZV': orbital_type_dict_SZV
}

num_orbitals_per_atom = {
    'DZVP': num_orbitals_per_atom_DZVP,
    'def2_SVP': num_orbitals_per_atom_def2_SVP,
    'SZV': num_orbitals_per_atom_SZV
}

def element_statistics(numbers):
    index_to_Z, inverse_indices = torch.unique(numbers, sorted=True, return_inverse=True)
    Z_to_index = torch.full((100,), -1, dtype=torch.int64)
    Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))
    return index_to_Z, Z_to_index

def read_xyz_file(file_path):
    """
    Read an XYZ file and return the atomic species (string list) and coordinates (float list).
    """

    atomic_species = []
    coordinates = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[0])

        for line in lines[2:]:  # Skip the first two lines
            data = line.split()
            atomic_species.append(data[0])
            coordinates.append([float(coord) for coord in data[1:4]])

    return atomic_species, coordinates


def read_sparse_hamiltonian_csr(file_path):
    """
    Read a sparse Hamiltonian matrix in CSR format from a file and return the matrix in a dictionary format.
    """

    indptr = []
    indices = []
    data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data_str = line.strip().split()
            if len(data_str) >= 3:
                indices.append([int(data_str[0]),int(data_str[1])])
                data.append(float(data_str[2]))
    full_hamiltonian = {}
    for i in range(len(indices)):
        full_hamiltonian[(indices[i][0],indices[i][1])] = data[i]
        full_hamiltonian[(indices[i][1],indices[i][0])] = data[i]
    return full_hamiltonian

def compute_rotation(axis_a, axis_b):

    # Normalize vectors
    axis_a_norm = np.linalg.norm(axis_a)
    axis_b_norm = np.linalg.norm(axis_b)
    norm_axis_a = axis_a / axis_a_norm
    norm_axis_b = axis_b / axis_b_norm

    # Compute rotation axis
    rotation_axis = np.cross(norm_axis_a, norm_axis_b)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm != 0:
        rotation_axis /= rotation_axis_norm  # Normalize the rotation axis

    # Compute rotation angle
    dot_product = np.dot(norm_axis_a, norm_axis_b)
    rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to avoid numerical errors

    return rotation_axis, np.degrees(rotation_angle)

def create_rotation_matrix(initial_vector, final_vector):
    axis, angle = compute_rotation(initial_vector, final_vector)
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    rotated_x = rotate_vector(x, angle, axis)
    rotated_y = rotate_vector(y, angle, axis)
    rotated_z = rotate_vector(z, angle, axis)


    R = np.stack((rotated_x,rotated_y,rotated_z),axis=0) #each row vector represents a basis vector 
    R = torch.tensor(R)
    return R

def rotate_vector(vector, angle, axis):
    # Create a rotation object
    rotation = Rotation.from_rotvec(np.radians(angle) * axis)
    # Apply rotation to the vector
    rotated_vector = rotation.apply(vector)
    return rotated_vector

def rotate_ham(R, H, orbital_type_dict,atomic_species,edge):
    R = R.index_select(0, R.new_tensor([1, 2, 0]).int()).index_select(1, R.new_tensor([1, 2, 0]).int())
    l_lefts = orbital_type_dict[atomic_species[edge[0]]]
    l_right = orbital_type_dict[atomic_species[edge[1]]]

    irreps_left = Irreps([(1, (l, 1)) for l in l_lefts])
    U_left = irreps_left.D_from_matrix(R)
    irreps_right = Irreps([(1, (l, 1)) for l in l_right])
    U_right = irreps_right.D_from_matrix(R)

    H_rotated = (U_left.transpose(-1,-2)@H@U_right).numpy()
    
    return H_rotated


def create_rotation_dic(edge_indices,coordinates,structure):
    coordinates = np.array(coordinates)
    vectors = [(coordinates[edge_indices[1][i]]-coordinates[edge_indices[0][i]]) for i in range(len(edge_indices[0]))]
    rotate_dic = {}

    # print(vectors[0])

    array_rcut = np.ones(len(structure.atomic_structure))*structure.rcut

    neighbour_list = NeighborList(array_rcut, skin=0, self_interaction=False, bothways=True)
    neighbour_list.update(structure.atomic_structure)

    # indicies = neighbour_list.get_neighbors(0)
    # print(indices)

    for i in range(len(edge_indices[0])):

            key_str = (edge_indices[0][i].item(),edge_indices[1][i].item())

            if edge_indices[1][i] == edge_indices[0][i]:
                nearest_neighbours, offsets = neighbour_list.get_neighbors(edge_indices[0][i])
                nearest_vectors = [(coordinates[nearest_neighbours[k]]-coordinates[edge_indices[0][i]]) for k in range(len(nearest_neighbours))]
                norms = [np.linalg.norm(v) for v in nearest_vectors]
                vectors[i] = nearest_vectors[np.argmin(norms)]

            
            R = create_rotation_matrix(np.array([1,0,0]),np.array(vectors[i]))
            rotate_dic[key_str] = R

    return rotate_dic


def rotate_data(y,edge_indices,coordinates,orbital_type_dict,atomic_species,rotate_dic):
    
    coordinates = np.array(coordinates)
    rotated_y = y
    for i in range(len(y)):
        R = rotate_dic[(edge_indices[0][i].item(),edge_indices[1][i].item())]
        edge = [edge_indices[0][i],edge_indices[1][i]]
        rotated_y[i] = rotate_ham(R.T,y[i],orbital_type_dict,atomic_species,edge)
    
    return rotated_y


# def rotate_data_back(pred,y,edge_indices,coordinates,orbital_type_dict,atomic_species,rotate_dic,rotate_back):
def rotate_data_back(pred, y, edge_indices, rotate_dic, structures):

    rotated_pred = []
    reduced_y = []

    graph_num = 0
    num_edges = int( len(edge_indices[0]) / len(structures) )

    # iterates over the edges in the graph
    for i in range(len(y)):

        # Gets the structure for the edge
        if i > 0 and i % num_edges == 0:
            graph_num += 1

        structure = structures[graph_num]
        atomic_species = np.array(structure.atomic_structure.get_chemical_symbols())

        # Calculate the starting index for each atom type in the orbital block
        unique_elements = set(structure.atomic_structure.get_chemical_symbols())
        mat_block_start = {}
        block_start = 0
        for element in unique_elements:
            mat_block_start[element] = block_start
            block_start += num_orbitals_per_atom[structure.basis][element]

        atom_i_index = edge_indices[0][i] % len(atomic_species) # local atom index
        atom_j_index = edge_indices[1][i] % len(atomic_species) # local atom index

        atom_i_element = atomic_species[atom_i_index]
        atom_j_element = atomic_species[atom_j_index]

        atom_i_start = mat_block_start[atom_i_element]
        atom_j_start = mat_block_start[atom_j_element]
        
        # starting_index_i, num_orbitals_i = structure.map_atom_to_orbital(atom_i_index) #edge_indices[0][i])
        # starting_index_j, num_orbitals_j = structure.map_atom_to_orbital(atom_j_index) #edge_indices[1][i])
        
        # atom_i_end = atom_i_start+num_orbitals_i
        # atom_j_end = atom_j_start+num_orbitals_j

        atom_i_end = atom_i_start + structure.num_orbitals_per_atom[atom_i_index]
        atom_j_end = atom_j_start + structure.num_orbitals_per_atom[atom_j_index]

        prediction = pred[i][atom_i_start:atom_i_end,atom_j_start:atom_j_end].detach().numpy()
        reshaped_y = y[i][atom_i_start:atom_i_end,atom_j_start:atom_j_end].detach().numpy()
        
        R = rotate_dic[(atom_i_index.item(), atom_j_index.item())]

        edge = [atom_i_index, atom_j_index]

        # pad the blocks again
        blocksize = structure.num_unique_orbitals
        prediction_padded = np.zeros((blocksize, blocksize))
        prediction_padded[atom_i_start:atom_i_end,atom_j_start:atom_j_end] = rotate_ham(R,prediction,orbital_type_dict[structure.basis],atomic_species,edge)
        rotated_pred.append(prediction_padded)
        
        reshaped_y_padded = np.zeros((blocksize, blocksize))
        reshaped_y_padded[atom_i_start:atom_i_end,atom_j_start:atom_j_end] = rotate_ham(R,reshaped_y,orbital_type_dict[structure.basis],atomic_species,edge)
        reduced_y.append(reshaped_y_padded)

        # rotated_pred.append(rotate_ham(R,prediction,orbital_type_dict[structure.basis],atomic_species,edge))
    
    rotated_pred = torch.tensor(np.array(rotated_pred))
    reduced_y = torch.tensor(np.array(reduced_y))
    
    return rotated_pred, reduced_y

def atom_dist(atom1_pos, atom2_pos, lattice):
    '''
    Calculates the distance between two atoms. If pbc = 1 below, the distance
    accounts for periodic boundary conditions in the y and z directions.

    Args:
        1. [x y z] position of atom1
        2. [x y z] position of atom2
        3. [x y z] dimensions of the orthogonal unit cell
        4. pbc: 0 or 1 (periodic boundary conditions)
    '''
    
    if lattice is None:
        return np.sqrt((atom2_pos[0]-atom1_pos[0])**2 + (atom2_pos[1]-atom1_pos[1])**2 + (atom2_pos[2]-atom1_pos[2])**2)  
    else:

        # Find shortest distance between atom1 and the periodic images of atom2
        distance_frac = np.divide(atom1_pos, lattice) - np.divide(atom2_pos, lattice)
        distance_frac = distance_frac - [int(distance_frac[0]*1 + 0.5), int(distance_frac[1]*1 + 0.5), int(distance_frac[2]*1 + 0.5)]
        dist_xyz = distance_frac * lattice
        
        # Return the norm of the xyz distance:
        return np.sqrt(dist_xyz[0]**2 + dist_xyz[1]**2 + dist_xyz[2]**2)



def unflatten(H_pred, numbers, edge_index, equivariant_blocks, atom_orbitals, out_slices):  

    H_prev = {}

    for index_edge in range(edge_index.shape[1]):


        i = edge_index[0][index_edge].item() #atom index 
        j = edge_index[1][index_edge].item()

        key_term = (i,j)#edge key term 

        num_orbitals_i = np.sum(2*np.array(atom_orbitals[str(numbers[i].item())])+1)
        num_orbitals_j = np.sum(2*np.array(atom_orbitals[str(numbers[j].item())])+1)


        
        fill = 0 
        init = torch.full((num_orbitals_i, num_orbitals_j), fill, dtype=float)
        H_prev[key_term] = init

        for index_target, equivariant_block in enumerate(equivariant_blocks):
            for N_M_str, block_slice in equivariant_block.items():
                slice_row = slice(block_slice[0], block_slice[1])
                slice_col = slice(block_slice[2], block_slice[3])
                len_row = block_slice[1] - block_slice[0]
                len_col = block_slice[3] - block_slice[2]
                slice_out = slice(out_slices[index_target], out_slices[index_target + 1])

                # print(H_prev)
                # print(H_prev[key_term][slice_row, slice_col])
                # print(H_pred[index_edge][slice_out].reshape(len_row, len_col))

                condition_atomic_number_i, condition_atomic_number_j = N_M_str.split()

                if (numbers[edge_index[0][index_edge]].item() == int(condition_atomic_number_i) and numbers[edge_index[1][index_edge]].item() == int(condition_atomic_number_j)):
                    H_prev[key_term][slice_row, slice_col] = H_pred[index_edge][slice_out].reshape(len_row, len_col)

    return H_prev