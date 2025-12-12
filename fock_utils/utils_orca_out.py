import numpy as np
import re
from cclib.parser.orcaparser import ORCA
from cclib.parser.nboparser import NBO
from typing import Any
from pathlib import Path
import logging
from e3nn.o3 import Irreps

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

periodic_table_number = {value: key for key, value in periodic_table.items()}

# Manual extraction of total energy fro orca.out 6.0 if cclib didn't find it
def extract_total_energy_manual(file_path):
    """Extract total energy manually from ORCA output file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "TOTAL SCF ENERGY" in line:
                # Look for the "Total Energy" line in the next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    if "Total Energy" in lines[j] and "Eh" in lines[j]:
                        # Extract the energy value
                        # Format: "Total Energy       :      -2436.18793915151446 Eh          -66292.04405 eV"
                        parts = lines[j].split()
                        for k, part in enumerate(parts):
                            if part == "Eh" and k > 0:
                                return float(parts[k-1])
        return None
    except Exception as e:
        print(f"Error extracting total energy manually: {e}")
        return None

def get_finalms(file_path):
    """
    Extracts the value of 'finalms' from an ORCA input file.
    Returns a float if found, otherwise None.
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = re.search(r'finalms\s+([-\d.]+)', line, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        return None
    except Exception as e:
        print(f"Error extracting finalms: {e}")
        return None

def extract_multiplicity_from_output(orca_output_path: Path) -> int:
    """
    Reads the ORCA output file and extracts the Multiplicity value (M).
    An ORCA output file usually contains a section echoing the input, e.g.:
    Multiplicity: 2
    """
    with open(orca_output_path, 'r') as f:
        for line in f:
            # Matches "Multiplicity: X" where X is a digit
            match = re.search(r'Multiplicity:\s+(\d+)', line)
            if match:
                return int(match.group(1))
    return 1  # Default to 1 (Singlet) if not found

def extract_charge_and_spin_from_path(orca_output_path: Path) -> tuple[int | None, int | None]:
    """
    Extracts charge and spin from the parent directory name, which is assumed
    to be in the format: ..._charge_spin (e.g., hexahydroantimonate_mol527_-1_1 -> charge=-1, spin=1)
    """
    # 1. Get the name of the informative directory: 'hexahydroantimonate_mol527_-1_1'
    # The parent of the .out file is 'stepX'. The parent of 'stepX' is the one we want.
    informative_folder_name = orca_output_path.parent.parent.name
    
    # 2. Use regex to find the last two integer components separated by underscores.
    # FIXED PATTERN: Allows an optional minus sign (-) before the first group of digits.
    # (-?) matches zero or one minus sign.
    match = re.search(r'_(-?\d+)_(\d+)$', informative_folder_name)
    
    if match:
        # Group 1 is the charge (now handles negative values)
        charge = int(match.group(1))
        # Group 2 is the multiplicity (should always be positive)
        multiplicity = int(match.group(2))
        return charge, multiplicity
    else:
        print(f"Warning: Could not reliably parse charge/spin from path: {informative_folder_name}")
        return None, None

def manually_parse_output(
    orca_output_path: Path,
    source: str,
) -> dict[str, Any]:
    """
    Reads the Orca output file at the input path and returns a dictionary
    of the important fields extracted from it. Manual version.
    """
    desired_data = {}

    total_energy = extract_total_energy_manual(orca_output_path)
    print(f"Manually extracted total energy: {total_energy} Eh")    

    desired_data["total_energy [Eh]"] = total_energy

    # Check for open-shell calculations
    multiplicity = extract_multiplicity_from_output(orca_output_path)
    desired_data["unrestricted"] = multiplicity > 1

    charge, multiplicity_path = extract_charge_and_spin_from_path(orca_output_path)
    desired_data["total_charge"] = charge
    desired_data["spin_multiplicity"] = multiplicity_path

    print("Extracted data: open_shell =", desired_data["unrestricted"], " charge =", desired_data["total_charge"], " multiplicity =", desired_data["spin_multiplicity"])

    return desired_data

# from https://github.com/facebookexternal/ocp-modeling-dev/blob/master/foundation_models/data/omol/process/orca_parsing.py#L295
def parse_output(
    orca_output_path: Path,
    source: str,
) -> dict[str, Any]:
    """
    Reads the Orca output file at the input path and returns a dictionary
    of the important fields extracted from it.
    """

    # Try to parse the main properties. Raise if this fails because there
    # isn't much to do without this data.
    orca_props = ORCA(str(orca_output_path)).parse()
    orca_props.listify()

    # Try to parse NBO data. We can proceed if this fails for some reason.
    # Just assume the data doesn't exist.
    # try:
    #     nbo_props = NBO(str(orca_output_path)).parse()
    #     nbo_props.listify()
    # except Exception:
    #     logging.exception(f"Failed to parse nbo data from {source}")
    nbo_props = None

    # Extract important data into a dictionary
    desired_data = {}
    desired_data["source"] = source
    desired_data["job_input"] = orca_props.metadata.get("input_file_contents")
    desired_data["job_keywords"] = orca_props.metadata.get("keywords")
    desired_data["atom_numbers"] = orca_props.atomnos
    desired_data["coords [A]"] = orca_props.atomcoords
    desired_data["total_charge"] = orca_props.charge
    desired_data["total_spin"] = orca_props.mult
    desired_data["n_cores"] = orca_props.metadata.get("num_cpu")
    desired_data["n_atoms"] = orca_props.natom
    desired_data["n_basis"] = orca_props.nbasis
    desired_data["n_ecp_electrons"] = int(sum(orca_props.coreelectrons))
    desired_data["n_electrons"] = (
        int(orca_props.nelectrons) - desired_data["n_ecp_electrons"]
    )
    if (total_energy := orca_props.metadata.get("total_energy")) is not None:
        total_energy = float(total_energy)
    else:
        # manual extraction
        total_energy = extract_total_energy_manual(orca_output_path)
        if total_energy is not None:
            print(f"Manually extracted total energy: {total_energy} Eh")
        else:
            print(f"Warning: Could not extract total energy for file at {orca_output_path}")

    desired_data["total_energy [Eh]"] = total_energy
    desired_data["gradient [Eh/bohr]"] = orca_props.grads[0]
    desired_data["s_squared"] = getattr(orca_props, "s_squared", 0.0)
    desired_data["s_squared_dev"] = getattr(orca_props, "s_squared_dev", 0.0)

    # Check for open-shell calculations
    is_open_shell = orca_props.mult > 1
    is_unrestricted = hasattr(orca_props, "s_squared")
    desired_data["unrestricted"] = is_unrestricted or is_open_shell

    charges = {}
    charges["mulliken"] = orca_props.atomcharges.get("mulliken")
    charges["lowdin"] = orca_props.atomcharges.get("lowdin")
    if (atomcharges := getattr(nbo_props, "atomcharges", None)) is not None:
        charges["nbo"] = atomcharges.get("nbo")
    else:
        charges["nbo"] = None
    desired_data["charges"] = charges

    if desired_data["unrestricted"]:
        spins = {}
        spins["mulliken"] = orca_props.atomspins.get("mulliken")
        spins["lowdin"] = orca_props.atomspins["lowdin"]
        if (atomspins := getattr(nbo_props, "atomspins", None)) is not None:
            spins["nbo"] = atomspins.get("nbo")
        else:
            spins["nbo"] = None
        desired_data["spins"] = spins

        # Handle spin inputs
        finalms = get_finalms(orca_output_path)
        if finalms is None:
            desired_data["spin_multiplicity"] = desired_data["total_spin"]
        else:
            desired_data["spin_multiplicity"] = 2 * abs(finalms) + 1

    else:
        desired_data["finalms"] = 0.0
        desired_data["spin_multiplicity"] = 1.0

    # print("Spin multiplicity: ", desired_data["spin_multiplicity"])
    desired_data["n_scf_steps"] = len(orca_props.scfvalues[0]) - 1

    if (core_hours := orca_props.metadata.get("cpu_time")) is not None:
        core_hours = core_hours[0].total_seconds() / 3600
    desired_data["core_hours"] = core_hours

    if (wall_hours := orca_props.metadata.get("wall_time")) is not None:
        wall_hours = wall_hours[0].total_seconds() / 3600
    desired_data["wall_hours"] = wall_hours

    desired_data["warnings"] = orca_props.metadata.get("warnings")
    desired_data["integrated_densities"] = orca_props.metadata.get("integrated_density")
    desired_data["nl_energy [Eh]"] = orca_props.metadata.get("nl_energy")
    desired_data["orbital_energies [Eh]"] = orca_props.moenergies
    desired_data["multipoles"] = orca_props.moments
    homo_es, lumo_es = get_homo_lumo_energies(desired_data)
    desired_data["homo_energy [Eh]"] = homo_es
    desired_data["homo-lumo_gap [Eh]"] = [
        lumo_e - homo_e for lumo_e, homo_e in zip(lumo_es, homo_es)
    ]

    return desired_data

def get_homo_lumo_energies(data):
    lumo = (data["n_electrons"] + data["total_spin"] - 1) // 2
    mo_energies = data["orbital_energies [Eh]"][0]
    lumo_e = [mo_energies[lumo]]
    homo_e = [mo_energies[lumo - 1]]
    if data["unrestricted"]:
        lumo_beta = (data["n_electrons"] - data["total_spin"] + 1) // 2
        mo_energies_beta = data["orbital_energies [Eh]"][1]
        lumo_e.append(mo_energies_beta[lumo_beta])
        homo_e.append(mo_energies_beta[lumo_beta - 1])

    return homo_e, lumo_e

def get_fock_size(elements, basis):
    """
    Compute size of fock matrix, given the elements in the structure and their orbital basis
    """

    N = 0
    for element in elements:
        N += np.sum([2*l+1 for l in basis[element]])                        # 2*l+1 gets the size of a spherical tensor with degree l

    return N

def read_orca_out(orca_file, unrestricted=False):
    """
    Get structure information (elements, coordinates, basis) and fock matrix from orca output file
    If unrestricted=True, returns {'alpha': ..., 'beta': ...} for the Fock matrices.
    """
    elements = []
    coordinates = []
    basis = {}                                                              # basis in the form {atomic_number : [degrees]}
    N = 0                                                                   # size of Fock matrix
    ls = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}                           # l conversion from string to degree #

    fock_matrices = []
    fock_matrix = None
    found_fock = 0

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
        # if 'FOCK' in line:
        if line.strip() == 'FOCK':
            assert(N != 0)
            fock_matrix = np.zeros((N, N))

            # get column size:
            cols = [int(x) for x in orca_out[idx+2:][0].split()]
            cols_per_line = len(cols)

            line_idx = idx + 3
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

                line_idx += 1

            fock_matrices.append(fock_matrix)

            # If unrestricted, parse the second Fock matrix (beta)
            if unrestricted:

                # Skip empty lines after first matrix
                while line_idx < len(orca_out) and len(orca_out[line_idx].split()) == 0:
                    line_idx += 1

                # The next non-empty line should be column headers for second matrix
                if line_idx < len(orca_out):
                    fock_matrix_beta = np.zeros((N, N))

                    # get column size for beta matrix:
                    cols = [int(x) for x in orca_out[line_idx].split()]
                    cols_per_line = len(cols)

                    line_idx += 1
                    for line in orca_out[line_idx:]:
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
                            fock_matrix_beta[row, cols] = vals

                    fock_matrices.append(fock_matrix_beta)

            break  # Found and parsed all needed matrices

    if unrestricted:
        if len(fock_matrices) != 2:
            raise ValueError(f"Expected 2 Fock matrices for unrestricted calculation, but found {len(fock_matrices)}")
        return {'alpha': fock_matrices[0], 'beta': fock_matrices[1]}, elements, coordinates, basis
    else:
        if len(fock_matrices) == 0:
            raise ValueError("No Fock matrix found in ORCA output file")
        return fock_matrices[0], elements, coordinates, basis

def sort_by_m(hamiltonian, orbital_basis, atomic_numbers, direction="orca_to_e3nn"):
    """
    Converts hamiltonian matrix m-components from ORCA order to the one
    expected by e3nn (m=0 is in the middle)

    l = 0: m = [0] -> [0]
    l = 1: m = [0 +1 -1] -> [-1 0 1]
    ...
    """

    num_cols = hamiltonian.shape[0]

    m_to_m_conversion = []
    if direction == "orca_to_e3nn" or direction == "e3nn_to_orca": # this one works
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3], 3: [6, 4, 2, 0, 1, 3, 5], 4: [8, 6, 4, 2, 0, 1, 3, 5, 7]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3], 3: [6, 4, 2, 0, 1, 3, 5], 4: [8, 6, 4, 2, 0, 1, 3, 5, 7]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3], 3: [6, 4, 2, 0, 1, 3, 5], 4: [8, 6, 4, 2, 0, 1, 3, 5, 7]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3], 3: [6, 4, 2, 0, 1, 3, 5], 4: [8, 6, 4, 2, 0, 1, 3, 5, 7]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3], 3: [6, 4, 2, 0, 1, 3, 5], 4: [8, 6, 4, 2, 0, 1, 3, 5, 7]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3], 3: [6, 4, 2, 0, 1, 3, 5], 4: [8, 6, 4, 2, 0, 1, 3, 5, 7]})
    if direction == "e3nn_to_pyscf" or direction == "pyscf_to_e3nn":
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8]})
        m_to_m_conversion.append({0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8]})

    reflection = {0: [1], 1: [1, 1, 1], 2: [1, 1, 1, 1, 1], 3: [-1, 1, 1, 1, 1, 1, -1], 4: [-1, -1, 1, 1, 1, 1, 1, -1, -1]} # reflection for each l

    permutation = np.arange(0, num_cols)
    full_orb_list = np.hstack([orbital_basis[atomic_numbers[i]] for i in range(len(atomic_numbers))])
    permuted_hamiltonian = hamiltonian.copy()

    # Create permutation list, one block at a time
    block_start = 0
    l_prev = 0
    principle_quantum_number = 0
    for l in full_orb_list:

        l = l % 10 # convention for diffuse functions, if needed

        if l != l_prev:
            principle_quantum_number = 0
        # print("principle_quantum_number: ", principle_quantum_number)

        numel = 2*l + 1
        block_end = block_start + numel

        this_m_to_m_conversion = m_to_m_conversion[principle_quantum_number]
        orbital_perm = this_m_to_m_conversion[l]
        refl = reflection[l]

        P = np.zeros((numel, numel))
        for i, j in enumerate(orbital_perm):
            P[i, j] = 1

        R = np.diag(refl)

        if direction == "orca_to_e3nn" or direction == "e3nn_to_pyscf":
            block = permuted_hamiltonian[block_start:block_end, :]
            block = R @ P @ block
            permuted_hamiltonian[block_start:block_end, :] = block
            block = permuted_hamiltonian[:, block_start:block_end]
            block = block @ (R @ P).T
            permuted_hamiltonian[:, block_start:block_end] = block
            block_start += numel
        elif direction == "e3nn_to_orca" or direction == "pyscf_to_e3nn":
            block = permuted_hamiltonian[block_start:block_end, :]
            block = P.T @ R @ block
            permuted_hamiltonian[block_start:block_end, :] = block
            block = permuted_hamiltonian[:, block_start:block_end]
            block = block @ (P.T @ R).T
            permuted_hamiltonian[:, block_start:block_end] = block
            block_start += numel
        else:
            raise NotImplementedError(f"Direction {direction} not implemented")

        principle_quantum_number += 1
        l_prev = l

    # permuted_hamiltonian = hamiltonian[permutation, :]
    # permuted_hamiltonian = permuted_hamiltonian[:, permutation]
    return permuted_hamiltonian

def sort_by_l(hamiltonian, orbital_basis, atomic_numbers):
    """
    Sorts the basis into l-major order within each atom block.
    """
    permutation = []
    current_index = 0

    # Iterate over each atom
    for atom in atomic_numbers:
        these_orbitals = orbital_basis[atom]

        # print(f"Atom {atom} orbitals: {these_orbitals}", flush=True)
        # Create a list of tuples (l, start_index, block_size) for the current atom
        orbital_blocks = []
        for l in these_orbitals:
            block_size = 2 * l + 1
            orbital_blocks.append((l, current_index, block_size))
            current_index += block_size

        # Sort the orbital blocks by l for the current atom
        orbital_blocks.sort(key=lambda x: x[0])

        # Create the permutation for the current atom based on sorted blocks
        for _, start_index, block_size in orbital_blocks:
            permutation.extend(range(start_index, start_index + block_size))

    # Apply the permutation to the Hamiltonian matrix
    permuted_hamiltonian = hamiltonian[permutation, :]
    permuted_hamiltonian = permuted_hamiltonian[:, permutation]
    return permuted_hamiltonian


def delete_rows_and_columns(matrix, indices):
    """
    Delete specified rows and columns from a matrix.
    Parameters:
    - matrix: The input matrix (2D NumPy array).
    - indices: A list of row/column indices to delete.
    Returns:
    - A new matrix with the specified rows and columns removed.
    """
    # Convert the list of indices to a NumPy array
    indices = np.array(indices)
    matrix_reduced = np.delete(matrix, indices, axis=0)
    matrix_reduced = np.delete(matrix_reduced, indices, axis=1)
    return matrix_reduced
