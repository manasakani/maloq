import re
import ase
from collections import Counter
import matplotlib.pyplot as plt
from fock_utils import utils_orca_out, utils_tensor_decomp, fock_targets, basis_sets
import torch
import numpy as np

# 1. Atomic structure analysis
# --------------------------------------------

# Regular expression to extract the symbols from the output file
symbols_regex = r"Structure:  Atoms\(symbols='([^']*)', pbc=False\)"
# Initialize an empty list to store the atomic elements
elements = []
num_atoms_per_structure = []
# Open the output file and read it line by line
with open("out-makeomol.out", "r") as f:
    for line in f:
        # Use regular expression to extract the symbols from the line
        match = re.search(symbols_regex, line)
        if match:
            symbols = match.group(1)
            # Create an ASE Atoms object from the symbols
            atoms = ase.Atoms(symbols=symbols)
            # Extract the atomic elements from the Atoms object
            elements.extend(atoms.get_chemical_symbols())
            # Store the number of atoms in the structure
            num_atoms_per_structure.append(len(atoms))

# Count the occurrences of each element
element_counts = Counter(elements)
# Filter out elements that don't exist in the dataset
existing_elements = [element for element in ase.data.chemical_symbols if element in element_counts]
# Sort the existing elements by atomic number
sorted_elements = sorted(existing_elements, key=lambda x: ase.data.atomic_numbers[x])

# Create a bar chart showing the distribution of elements
plt.figure(figsize=(15, 3))
plt.grid(True, linestyle='--', alpha=0.5, which='minor')
plt.bar(sorted_elements, [element_counts[element] for element in sorted_elements], color=plt.cm.summer(range(len(sorted_elements))), alpha=0.8)
# plt.xlabel("Element")
plt.ylabel("Count")
plt.yscale('log')  # Use logarithmic scale for better visibility
plt.xlabel("Element")
# plt.xticks(rotation=90)  # rotate x-axis labels for better readability
plt.tight_layout()  # adjust layout to fit rotated labels
plt.savefig("omol_elements.png", dpi=500, bbox_inches='tight')
plt.close()

# Create a histogram of the number of atoms per structure
plt.figure(figsize=(5, 3))
plt.hist(num_atoms_per_structure, bins=range(min(num_atoms_per_structure), max(num_atoms_per_structure) + 1), color='royalblue', alpha=0.5, rwidth=1.0)
plt.xlabel("# Atoms per Structure")
plt.ylabel("Count")
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.5, which='minor')
plt.savefig("omol_num_atoms_per_structure.png", dpi=500, bbox_inches='tight')
plt.close()

# # --> Interatomic distances
# # load omol_interatomic_distances.txt file
# with open('omol_interatomic_distances.txt', 'r') as f:
#     all_distances = list(map(float, f.read().strip().split()))

# # plot histogram of distances:
# plt.figure(figsize=(5, 2))
# plt.hist(all_distances, bins=100, color='blue', alpha=0.4, rwidth=1.0)
# plt.xlabel("r ($\AA$)")
# plt.ylabel("Count")
# plt.yscale('log')
# plt.grid(True, linestyle='--', alpha=0.4, which='minor')
# plt.savefig("omol_interatomic_distances_histogram.png", dpi=500, bbox_inches='tight')
# plt.close()

# --> Element interaction matrix
with open('omol_element_interaction_matrix.txt', 'r') as f:
    lines = f.readlines()
    element_interaction_matrix = []
    for line in lines:
        row = list(map(int, line.strip().split()))
        element_interaction_matrix.append(row)
element_interaction_matrix = torch.tensor(element_interaction_matrix)

# remove any row and column for which an atomic number is not in the sorted_elements list:
sorted_atomic_numbers = [ase.data.atomic_numbers[el]-1 for el in sorted_elements]
element_interaction_matrix = element_interaction_matrix[sorted_atomic_numbers][:, sorted_atomic_numbers]

# plot element interaction matrix as a heatmap:
plt.figure(figsize=(10, 10))
plt.imshow(np.log(np.abs(element_interaction_matrix)), cmap='Blues', interpolation='nearest')
plt.colorbar(label='Interaction Count')

plt.xticks(ticks=range(len(sorted_elements)), labels=sorted_elements, rotation=90)
plt.yticks(ticks=range(len(sorted_elements)), labels=sorted_elements)

# put xaxis on top:
plt.gca().xaxis.set_ticks_position('top')
plt.tight_layout()
plt.savefig("omol_element_interaction_matrix.png", dpi=500, bbox_inches='tight')
plt.close()

# ##############################

# 2. Orbital distribution analysis
# --------------------------------------------

# --> plot orbitals
orbital_basis = basis_sets.def2_tzvpd
orbital_basis = {k: sorted(v) for k, v in orbital_basis.items()} # The basis must be in l-major!!!
orbital_basis = {k: torch.tensor(v) for k, v in orbital_basis.items()}

# Find the maximum basis
ls_list = []
for l in range(5): # searching for up to g orbitals
    counts = [torch.sum(orbital_basis[el] == l) for el in orbital_basis]
    ls_list.append(torch.tensor(max(counts) * [l], dtype=torch.int))
ls_list = torch.cat(ls_list)        # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
ls_list = ls_list.tolist()
print("ls_list: ", ls_list)

elements = []
# Open the output file and read it line by line
with open("out-makeomol.out", "r") as f:
    for line in f:
        # Use regular expression to extract the symbols from the line
        match = re.search(symbols_regex, line)
        if match:
            symbols = match.group(1)
            # Create an ASE Atoms object from the symbols
            atoms = ase.Atoms(symbols=symbols)
            # Extract the atomic elements from the Atoms object
            elements.extend(atoms.get_chemical_symbols())

# Count the occurrences of each element
element_counts = Counter(elements)
# Initialize an empty dictionary to store the orbital counts
orbital_counts = [0] * len(ls_list)
# Loop over each element and its count
for element, count in element_counts.items():
    # Get the orbital basis for the current element
    basis = orbital_basis[element]

    # print(f"Element: {element}, Basis: {basis}, Count: {count}")
    num_l0 = torch.sum(basis == 0).item()
    num_l1 = torch.sum(basis == 1).item()
    num_l2 = torch.sum(basis == 2).item()
    num_l3 = torch.sum(basis == 3).item()
    
    idx_start = ls_list.index(0)
    orbital_counts[idx_start: idx_start + num_l0] = [x + count for x in orbital_counts[idx_start: idx_start + num_l0]]

    idx_start = ls_list.index(1)
    orbital_counts[idx_start: idx_start + num_l1] = [x + count for x in orbital_counts[idx_start: idx_start + num_l1]]

    idx_start = ls_list.index(2)
    orbital_counts[idx_start: idx_start + num_l2] = [x + count for x in orbital_counts[idx_start: idx_start + num_l2]]

    idx_start = ls_list.index(3)
    orbital_counts[idx_start: idx_start + num_l3] = [x + count for x in orbital_counts[idx_start: idx_start + num_l3]]
    print(f"Orbital counts after {element}: {orbital_counts}")

print("Orbital Counts:", orbital_counts)

# Create a list of x-axis labels
labels = []
s_idx, p_idx, d_idx, f_idx = 0, 0, 0, 0
for i, orbital in enumerate(ls_list):
    if orbital == 0:
        labels.append(f"s$_{{{s_idx}}}$")
        s_idx += 1
    elif orbital == 1:
        labels.append(f"p$_{{{p_idx}}}$")
        p_idx += 1
    elif orbital == 2:
        labels.append(f"d$_{{{d_idx}}}$")
        d_idx += 1
    elif orbital == 3:
        labels.append(f"f$_{{{f_idx}}}$")
        f_idx += 1
print("Labels: ", labels)

# Create a color map
color_map = {0: '#4567b7', 1: '#ff8c00', 2: '#8bc34a', 3: '#e74c3c'}

# Create the bar plot
plt.figure(figsize=(5, 1))
for i, orbital in enumerate(ls_list):
    plt.bar(labels[i], orbital_counts[i], color=color_map[orbital], alpha=0.8)
plt.yscale('log')  # Use logarithmic scale for better visibility
plt.xlabel("Orbital")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5, which='minor')

plt.savefig("omol_orbitals.png", dpi=500, bbox_inches='tight')


# 3. Orbital interaction analysis
# --------------------------------------------

# # get minimal basis required to represent all atomic interaction

# orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
# orbital_basis = {k: sorted(v) for k, v in orbital_basis.items()} # The basis must be in l-major!!!
# orbital_basis = {k: torch.tensor(v) for k, v in orbital_basis.items()}

# ls_list = []
# for l in range(5): # searching for up to g orbitals
#     counts = [torch.sum(orbital_basis[el] == l) for el in orbital_basis]
#     ls_list.append(torch.tensor(max(counts) * [l], dtype=torch.int))

# ls_list = torch.cat(ls_list)        # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
# print("ls_list: ", ls_list)         # [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]

# elements = []
# # Open the output file and read it line by line
# with open("out-makeomol_unscaled_orderedbasis.out", "r") as f:
#     for line in f:
#         # Use regular expression to extract the symbols from the line
#         match = re.search(symbols_regex, line)
#         if match:
#             symbols = match.group(1)
#             # Create an ASE Atoms object from the symbols
#             atoms = ase.Atoms(symbols=symbols)
#             # print atomic positions:
#             print("Atomic positions: ", atoms.get_positions())
#             # Extract the atomic elements from the Atoms object
#             elements.extend(atoms.get_chemical_symbols())

# # # Count the occurrences of each element
# # element_counts = Counter(elements)
# # # Initialize an empty dictionary to store the orbital counts
# # orbital_counts = [0] * len(ls_list)
# # # Loop over each element and its count
# # for element, count in element_counts.items():
# #     # Get the orbital basis for the current element
# #     basis = orbital_basis[element]