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

# Load the data
try:
    data = torch.load('./element_scale_shifts_omol_all_l.pt')
    print("Loaded element_scale_shifts_omol_all_l.pt successfully")
    print(f"Data type: {type(data)}")
    print(f"Keys in data: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
    
    # Examine the structure of the data
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"\n--- {key} ---")
            print(f"Type: {type(value)}")
            if isinstance(value, dict):
                print(f"Elements (atomic numbers): {list(value.keys())}")
                # Look at first element to understand structure
                first_element = list(value.keys())[0]
                first_value = value[first_element]
                print(f"First element ({first_element}): {periodic_table_number.get(first_element, 'Unknown')}")
                print(f"Type of values for first element: {type(first_value)}")
                if isinstance(first_value, list) and len(first_value) > 0:
                    print(f"Length of list: {len(first_value)}")
                    print(f"Type of first item in list: {type(first_value[0])}")
                    if hasattr(first_value[0], 'shape'):
                        print(f"Shape of first item: {first_value[0].shape}")
                    print(f"First few values: {first_value[:3] if len(first_value) >= 3 else first_value}")
                elif isinstance(first_value, (list, np.ndarray, torch.Tensor)):
                    print(f"Shape/Length: {getattr(first_value, 'shape', len(first_value))}")
                    print(f"Sample values: {first_value}")
            elif isinstance(value, list):
                print(f"Length: {len(value)}")
                print(f"Sample values: {value[:10] if len(value) > 10 else value}")
    
    # If there are angular momentum components, analyze them
    if 'element_all_l_means' in data or 'element_means' in data:
        element_means_key = 'element_all_l_means' if 'element_all_l_means' in data else 'element_means'
        element_means = data[element_means_key]
        
        print(f"\n=== Analysis of {element_means_key} ===")
        
        # Analyze the structure for different elements
        for atomic_num in sorted(list(element_means.keys())[:5]):  # Look at first 5 elements
            element_symbol = periodic_table_number.get(atomic_num, f"Z={atomic_num}")
            values = element_means[atomic_num]
            print(f"\nElement {element_symbol} (Z={atomic_num}):")
            if isinstance(values, list):
                print(f"  Number of components: {len(values)}")
                for i, component in enumerate(values[:5]):  # Show first 5 components
                    if hasattr(component, 'shape'):
                        print(f"  Component {i}: shape {component.shape}")
                    else:
                        print(f"  Component {i}: {type(component)} - {component}")
            else:
                print(f"  Type: {type(values)}, Value: {values}")
        
        # Try to identify angular momentum components
        print(f"\n=== Angular Momentum Component Analysis ===")
        if len(element_means) > 0:
            sample_element = list(element_means.keys())[0]
            sample_values = element_means[sample_element]
            
            print(f"Sample element: {periodic_table_number.get(sample_element, f'Z={sample_element}')}")
            print(f"Number of components: {len(sample_values) if isinstance(sample_values, list) else 'Not a list'}")
            
            if isinstance(sample_values, list):
                total_components = 0
                component_sizes = []
                for i, component in enumerate(sample_values):
                    if hasattr(component, 'shape'):
                        size = component.shape[0] if len(component.shape) > 0 else 1
                    elif isinstance(component, (list, tuple)):
                        size = len(component)
                    else:
                        size = 1
                    component_sizes.append(size)
                    total_components += size
                    
                    # Guess angular momentum based on component size
                    if size == 1:
                        l_guess = "l=0 (scalar)"
                    elif size == 3:
                        l_guess = "l=1 (vector)"
                    elif size == 5:
                        l_guess = "l=2 (rank-2 tensor)"
                    elif size == 7:
                        l_guess = "l=3"
                    elif size == 9:
                        l_guess = "l=4"
                    else:
                        l_guess = f"unknown (size {size})"
                    
                    print(f"  Component {i}: size {size} -> likely {l_guess}")
                
                print(f"Total components across all l: {total_components}")
                print(f"Component sizes: {component_sizes}")

except FileNotFoundError:
    print("Error: element_scale_shifts_omol_all_l.pt not found in ../fock_datasets/")
    print("Please check the file path and ensure the file exists.")
except Exception as e:
    print(f"Error loading file: {e}")
    data = None

# Now create plots for each angular momentum component (l=0 to l=6)
if 'element_irrep_means' in data and 'irrep_indices_by_l' in data:
    element_irrep_means = data['element_irrep_means']
    irrep_indices_by_l = data['irrep_indices_by_l']
    
    print(f"\n=== Creating plots for l=0 to l=6 ===")
    
    # Create subplots for all l values
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Define colors for different l values
    l_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for l in range(7):  # l=0 to l=6
        if l in irrep_indices_by_l:
            indices_for_l = irrep_indices_by_l[l]
            
            print(f"\nProcessing l={l}")
            print(f"Number of components for l={l}: {len(indices_for_l)}")
            
            # Collect data for this l value
            elements_l = []
            values_l = []
            colors_l = []
            
            for atomic_num in sorted(element_irrep_means.keys()):
                element_symbol = periodic_table_number.get(atomic_num, f"Z={atomic_num}")
                irrep_means = element_irrep_means[atomic_num]
                
                if l in irrep_means:
                    l_components = irrep_means[l]
                    if isinstance(l_components, (list, tuple, torch.Tensor, np.ndarray)):
                        # Handle multiple components for this l
                        for i, component_value in enumerate(l_components):
                            elements_l.append(element_symbol)
                            if isinstance(component_value, torch.Tensor):
                                values_l.append(component_value.item())
                            else:
                                values_l.append(float(component_value))
                            
                            # Color based on component index within this l
                            color_ratio = i / max(1, len(l_components) - 1) if len(l_components) > 1 else 0
                            colors_l.append(color_ratio)
                    else:
                        # Single component for this l
                        elements_l.append(element_symbol)
                        if isinstance(l_components, torch.Tensor):
                            values_l.append(l_components.item())
                        else:
                            values_l.append(float(l_components))
                        colors_l.append(0.0)
            
            # Create the plot for this l value
            ax = axes[l]
            if len(values_l) > 0:
                scatter = ax.scatter(elements_l, values_l, alpha=0.6, s=25, 
                                   c=colors_l, cmap='coolwarm', marker='o')
                
                ax.set_xlabel('Atomic Element', fontsize=10)
                ax.set_ylabel(rf'$\langle l = {l} \rangle$ ($E_h$)', fontsize=10)
                ax.set_title(f'l={l} Components', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add colorbar if there are multiple components
                max_colors = max(colors_l) if colors_l else 0
                if max_colors > 0:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(f'Component Index', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No data for l={l}', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'l={l} Components (No Data)', fontsize=12)
            
            print(f"l={l}: {len(values_l)} data points across {len(set(elements_l))} elements")
        else:
            # No data for this l value
            ax = axes[l]
            ax.text(0.5, 0.5, f'No data for l={l}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'l={l} Components (No Data)', fontsize=12)
    
    # Remove the last subplot (we have 7 plots in 2x4 grid)
    axes[7].remove()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('element_irrep_means_all_l.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to element_irrep_means_all_l.png")
    
    # Show the plot
    # plt.show()
    
    # Create individual plots for each l value for better detail
    for l in range(7):
        if l in irrep_indices_by_l:
            plt.figure(figsize=(14, 6))
            
            elements_l = []
            values_l = []
            colors_l = []
            
            for atomic_num in sorted(element_irrep_means.keys()):
                element_symbol = periodic_table_number.get(atomic_num, f"Z={atomic_num}")
                irrep_means = element_irrep_means[atomic_num]
                
                if l in irrep_means:
                    l_components = irrep_means[l]
                    if isinstance(l_components, (list, tuple, torch.Tensor, np.ndarray)):
                        for i, component_value in enumerate(l_components):
                            elements_l.append(element_symbol)
                            if isinstance(component_value, torch.Tensor):
                                values_l.append(component_value.item())
                            else:
                                values_l.append(float(component_value))
                            
                            color_ratio = i / max(1, len(l_components) - 1) if len(l_components) > 1 else 0
                            colors_l.append(color_ratio)
                    else:
                        elements_l.append(element_symbol)
                        if isinstance(l_components, torch.Tensor):
                            values_l.append(l_components.item())
                        else:
                            values_l.append(float(l_components))
                        colors_l.append(0.0)
            
            if len(values_l) > 0:
                scatter = plt.scatter(elements_l, values_l, alpha=0.6, s=40, 
                                    c=colors_l, cmap='coolwarm', marker='o')
                
                plt.xlabel('Atomic Element', fontsize=12)
                plt.ylabel(rf'$\langle l = {l} \rangle$ ($E_h$)', fontsize=12)
                plt.title(f'Element Averages for l={l} Components', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add colorbar if there are multiple components
                max_colors = max(colors_l) if colors_l else 0
                if max_colors > 0:
                    cbar = plt.colorbar(scatter)
                    cbar.set_label(f'l={l} Component Index (Blue→Red)', fontsize=10)
                
                plt.tight_layout()
                plt.savefig(f'element_irrep_means_l_{l}.png', dpi=300, bbox_inches='tight')
                print(f"Saved individual plot for l={l}")
                # plt.show()