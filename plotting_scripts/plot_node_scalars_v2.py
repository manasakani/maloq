import torch
import matplotlib.pyplot as plt
import numpy as np

# Periodic table mappings
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

# Load the data from both datasets
try:
    omol_data = torch.load('../fock_datasets/element_scale_shifts_omol.pt')
    print("Loaded omol data successfully")
except FileNotFoundError:
    print("Warning: element_scale_shifts_omol.pt not found")
    omol_data = None

try:
    nabla_data = torch.load('../fock_datasets/element_scale_shifts_nablaDFT.pt')
    print("Loaded nablaDFT data successfully")
except FileNotFoundError:
    print("Warning: element_scale_shifts_nablaDFT.pt not found")
    nabla_data = None

# Create the plot
plt.figure(figsize=(12, 3))

# Plot omol data
if omol_data is not None:
    omol_means = omol_data['element_scalar_means']
    elements_omol = []
    scalar_values_omol = []
    colors_omol = []
    
    for atomic_num, values in omol_means.items():
        element_symbol = periodic_table_number[atomic_num]
        num_scalars = len(values)
        # Add each scalar value for this element with color based on sequence
        for i, value in enumerate(values):
            elements_omol.append(element_symbol)
            scalar_values_omol.append(value)
            # Color from blue (0) to yellow (1) based on position in sequence
            color_ratio = i / max(1, num_scalars - 1) if num_scalars > 1 else 0
            colors_omol.append(color_ratio)
    
    scatter_omol = plt.scatter(elements_omol, scalar_values_omol, alpha=0.3, s=25, 
                              c=colors_omol, cmap='coolwarm', marker='o', 
                              label='OMol_CSH_58k')

# Plot nablaDFT data
if nabla_data is not None:
    nabla_means = nabla_data['element_scalar_means']
    elements_nabla = []
    scalar_values_nabla = []
    colors_nabla = []
    
    for atomic_num, values in nabla_means.items():
        element_symbol = periodic_table_number[atomic_num]
        num_scalars = len(values)
        # Add each scalar value for this element with color based on sequence
        for i, value in enumerate(values):
            elements_nabla.append(element_symbol)
            scalar_values_nabla.append(value)
            # Color from blue (0) to yellow (1) based on position in sequence
            color_ratio = i / max(1, num_scalars - 1) if num_scalars > 1 else 0
            colors_nabla.append(color_ratio)
    
    scatter_nabla = plt.scatter(elements_nabla, scalar_values_nabla, alpha=0.8, s=20, 
                               color='black', marker='.', 
                               label=r'$\nabla^2$DFT')

# Customize the plot
plt.xlabel('Atomic Element', fontsize=12)
plt.ylabel(r'$\langle l = 0 \rangle$ ($E_h$)', fontsize=12)
# plt.title('Element Scalar Means (Averaged) - omol vs nablaDFT', fontsize=14, fontweight='bold')

# Add colorbar to show the sequence mapping
if omol_data is not None or nabla_data is not None:
    cbar = plt.colorbar(scatter_omol if omol_data is not None else scatter_nabla, ax=plt.gca())
    cbar.set_label(r'ordered $\langle l=0 \rangle$ (Blue→Red)', fontsize=10)
    # remove the labels on the colorbar ticks:
    cbar.set_ticks([])  # This removes the tick marks and labels

# Add legend with updated labels
legend_elements = []
if omol_data is not None:
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='gray', markersize=8, 
                                    label='OMol_CSH_58k'))
if nabla_data is not None:
    legend_elements.append(plt.Line2D([0], [0], marker='.', color='w', 
                                    markerfacecolor='black', markersize=8, 
                                    label=r'$\nabla^2$DFT'))

if legend_elements:
    plt.legend(handles=legend_elements, fontsize=10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
# plt.show()

# Optional: Save the plot
plt.savefig('element_scalar_means_comparison.png', dpi=300, bbox_inches='tight')

# Print statistics
if omol_data is not None:
    print(f"omol dataset - Total data points: {len(scalar_values_omol)}, Unique elements: {len(set(elements_omol))}")

if nabla_data is not None:
    print(f"nablaDFT dataset - Total data points: {len(scalar_values_nabla)}, Unique elements: {len(set(elements_nabla))}")

if omol_data is not None and nabla_data is not None:
    # Find common elements
    common_elements = set(elements_omol) & set(elements_nabla)
    print(f"Common elements between datasets: {sorted(common_elements)}")
    print(f"Elements only in omol: {sorted(set(elements_omol) - set(elements_nabla))}")
    print(f"Elements only in nablaDFT: {sorted(set(elements_nabla) - set(elements_omol))}")