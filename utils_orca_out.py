import numpy as np
import re

### Utility functions to extract information from the orca output files

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

def get_fock_size(elements, basis):
    """
    Compute size of fock matrix, given the elements in the structure and their orbital basis
    """
    
    N = 0
    for element in elements:
        N += np.sum([2*l+1 for l in basis[element]])                        # 2*l+1 gets the size of a spherical tensor with degree l

    return N

def read_orca_out(orca_file):
    """
    Get structure information (elements, coordinates, basis) and fock matrix from orca output file
    """
    elements = []
    coordinates = []
    basis = {}                                                              # basis in the form {atomic_number : [degrees]}
    N = 0                                                                   # size of Fock matrix
    ls = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}                           # l conversion from string to degree #
    
    orca_out = open(orca_file).readlines()
    for idx, line in enumerate(orca_out):

        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            for line in orca_out[idx+2:]:
                if len(line.split()) == 0:
                    break
                elements.append(periodic_table[str(line.split()[0])])       # atomic element numbers
                coordinates.append([float(x) for x in line.split()[1:]])

        if 'BASIS SET' in line:
            for line in orca_out[idx:]:

                if 'Group' in line:
                    element = periodic_table[str(line.split()[3])]          # atomic element number
                    basis[element] = []
                    basis_string = re.split('(\d+)', line.split()[8])[1:]

                    for i, orb in enumerate(basis_string[1::2]):
                        basis[element].extend([ls[orb]]*int(basis_string[2*i]))
                
                if 'AUXILIARY/J BASIS SET INFORMATION' in line:
                    break
        
            assert(basis)
            N = get_fock_size(elements, basis)

        # note: matrix cols are spread across multiple lines, we re-parse cols when new idx are found
        if 'FOCK' in line:
            assert(N != 0)
            fock_matrix = np.zeros((N, N))

            # get column size:
            cols = [int(x) for x in orca_out[idx+2:][0].split()]    
            cols_per_line = len(cols) 

            for line in orca_out[idx+3:]:
                if len(line.split()) == 0:
                    break

                # new columns in file:
                if len(line.split()) != cols_per_line + 1:
                    cols = [int(x) for x in line.split()]   
                    cols_per_line = len(cols) 

                # matrix entries:
                else:
                    row = int(line.split()[0])
                    vals = [float(x) for x in line.split()[1:]]
                    fock_matrix[row, cols] = vals

    return fock_matrix, elements, coordinates, basis


def sort_by_m(hamiltonian, orbital_basis, atomic_numbers):
    """
    Converts hamiltonian matrix m-components from ORCA order to the one 
    expected by e3nn
    
    l = 0: m = [0] -> [0]
    l = 1: m = [0 +1 -1] -> [-1 0 1]
    ...
    """

    num_cols = hamiltonian.shape[0]
    m_to_m_conversion = {0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3], 3: [6, 4, 2, 0, 1, 3, 5], 4: [8, 6, 4, 2, 0, 1, 3, 5, 7]}
    permutation = np.arange(0, num_cols)

    full_orb_list = np.hstack([orbital_basis[atomic_numbers[i]] for i in range(len(atomic_numbers))])

    # Create permutation list, one block at a time
    block_start = 0
    for l in full_orb_list:
        numel = 2*l + 1
        block_end = block_start + numel
        orbital_perm = m_to_m_conversion[l]
        permutation[block_start:block_end] = [x+block_start for x in orbital_perm]
        block_start += numel

    permuted_hamiltonian = hamiltonian[permutation, :]
    permuted_hamiltonian = permuted_hamiltonian[:, permutation]
    return permuted_hamiltonian