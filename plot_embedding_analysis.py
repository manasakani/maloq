import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import binned_statistic
from ase import Atoms
from ase.visualize import view
from ase.io import write

def load_embeddings_from_npy(output_folder, molecule_index=0):
    """Load embeddings and outputs/targets from .npy files for a specific molecule"""
    try:
        node_embeddings = np.load(os.path.join(output_folder, f'node_embeddings_{molecule_index}.npy'))
        edge_embeddings = np.load(os.path.join(output_folder, f'edge_embeddings_{molecule_index}.npy'))
        edge_distances = np.load(os.path.join(output_folder, f'edge_distances_{molecule_index}.npy'))
        positions = np.load(os.path.join(output_folder, f'positions_{molecule_index}.npy'))
        atomic_numbers = np.load(os.path.join(output_folder, f'atomic_numbers_{molecule_index}.npy'))
        
        # Try to load outputs and targets
        try:
            node_outputs = np.load(os.path.join(output_folder, f'node_outputs_{molecule_index}.npy'))
            edge_outputs = np.load(os.path.join(output_folder, f'edge_outputs_{molecule_index}.npy'))
            node_targets = np.load(os.path.join(output_folder, f'node_targets_{molecule_index}.npy'))
            edge_targets = np.load(os.path.join(output_folder, f'edge_targets_{molecule_index}.npy'))
            
            # Load irreps information
            try:
                with open(os.path.join(output_folder, 'irreps.txt'), 'r') as f:
                    irreps_line = f.readline().strip()
                    irreps_str = irreps_line.split(': ')[1]  # Extract irreps string after "Head irreps: "
            except:
                irreps_str = None
                
            return (node_embeddings, edge_embeddings, edge_distances, positions, atomic_numbers,
                   node_outputs, edge_outputs, node_targets, edge_targets, irreps_str)
        except FileNotFoundError:
            print("Outputs/targets not found, analyzing embeddings only")
            return (node_embeddings, edge_embeddings, edge_distances, positions, atomic_numbers,
                   None, None, None, None, None)
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return tuple([None] * 10)

def parse_e3nn_irreps(irreps_str):
    """
    Parse e3nn irreps string like "1x0e+1x0e+1x1e+2x2e+..." 
    Returns list of (multiplicity, l, parity) tuples and their indices
    """
    if irreps_str is None:
        return None, None
    
    irreps = []
    current_idx = 0
    l_indices = {}
    
    # Split by '+' to get individual irrep terms
    terms = irreps_str.split('+')
    
    for term in terms:
        # Parse term like "1x0e" or "2x2o"
        if 'x' in term:
            mult_str, l_parity = term.split('x')
            multiplicity = int(mult_str)
        else:
            multiplicity = 1
            l_parity = term
            
        # Extract l and parity
        if l_parity[-1] in ['e', 'o']:
            l = int(l_parity[:-1])
            parity = l_parity[-1]
        else:
            l = int(l_parity)
            parity = 'e' if l % 2 == 0 else 'o'
            
        # Calculate dimension of this irrep
        irrep_dim = multiplicity * (2 * l + 1)
        
        # Store irrep info
        irreps.append((multiplicity, l, parity))
        
        # Store indices for each l value
        if l not in l_indices:
            l_indices[l] = []
        l_indices[l].extend(range(current_idx, current_idx + irrep_dim))
        
        current_idx += irrep_dim
    
    return irreps, l_indices

def compute_l_component_magnitudes_from_irreps(data, l_indices):
    """
    Compute the magnitude of each l component from irrep decomposition
    data shape: [num_items, irrep_size]
    Returns: dict with l values as keys and magnitudes as values
    """
    if l_indices is None:
        return None
        
    l_magnitudes = {}
    
    for l, indices in l_indices.items():
        if len(indices) > 0:
            # Extract the l component: [num_items, num_features_for_l]
            l_component = data[:, indices]
            
            # Compute magnitude across the feature dimension
            l_magnitudes[l] = np.linalg.norm(l_component, axis=1)
        else:
            l_magnitudes[l] = np.zeros(data.shape[0])
    
    return l_magnitudes

def create_ase_atoms(positions, atomic_numbers):
    """Create ASE Atoms object from positions and atomic numbers"""
    return Atoms(numbers=atomic_numbers, positions=positions)

def visualize_molecule(atoms, output_folder, molecule_index=0, save_image=True):
    """Visualize molecule using ASE and optionally save as image"""
    print(f"Molecule {molecule_index} composition: {atoms.get_chemical_formula()}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Atomic numbers: {atoms.get_atomic_numbers()}")
    
    if save_image:
        # Save as various formats
        write(os.path.join(output_folder, f'molecule_{molecule_index}.xyz'), atoms)
        write(os.path.join(output_folder, f'molecule_{molecule_index}.pdb'), atoms)
        print(f"Saved molecule structure to {output_folder}/molecule_{molecule_index}.[xyz,pdb]")
        
        # Create a matplotlib visualization of the molecule
        from ase.data import covalent_radii, atomic_names
        from ase.data.colors import jmol_colors
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        
        # Plot atoms as spheres
        for i, (pos, num) in enumerate(zip(positions, atomic_numbers)):
            color = jmol_colors[num]
            radius = covalent_radii[num] * 200  # Scale for visibility
            ax.scatter(pos[0], pos[1], pos[2], 
                      c=[color], s=radius, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Draw bonds (simple distance-based)
        bond_threshold = 16.0  # Angstroms
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dist = atoms.get_distance(i, j)
                if dist < bond_threshold:
                    pos1, pos2 = positions[i], positions[j]
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                           'k-', alpha=0.1, linewidth=2)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'3D Structure: {atoms.get_chemical_formula()} ({len(atoms)} atoms)', fontsize=16)
        
        # Remove all axes, labels, and background for clean atom-only view
        ax.set_axis_off()  # Remove axes
        ax.grid(False)     # Remove grid
        
        # Remove background and make it transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Set background color to transparent
        fig.patch.set_facecolor('none')
        ax.patch.set_facecolor('none')
        
        plt.tight_layout()

        # Save the molecular structure plot
        molecule_png_path = os.path.join(output_folder, f'molecule_{molecule_index}_structure.png')
        plt.savefig(molecule_png_path, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
        print(f"Saved molecular structure plot to {molecule_png_path}")
        
        plt.show()  # This will display in Jupyter or save if using non-interactive backend
        plt.close()  # Clean up memory
        
        # Create a summary info plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Create summary text
        summary_text = f"""
Molecule {molecule_index} Analysis Summary
=====================================

Chemical Formula: {atoms.get_chemical_formula()}
Number of Atoms: {len(atoms)}

Atomic Composition:
"""
        
        # Count each element type
        from collections import Counter
        element_counts = Counter([atomic_names[num] for num in atomic_numbers])
        for element, count in sorted(element_counts.items()):
            summary_text += f"  {element}: {count} atoms\n"
        
        # Add distance statistics
        distances = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                distances.append(atoms.get_distance(i, j))
        
        if distances:
            summary_text += f"\nInteratomic Distances:\n"
            summary_text += f"  Min: {min(distances):.3f} Å\n"
            summary_text += f"  Max: {max(distances):.3f} Å\n"
            summary_text += f"  Mean: {sum(distances)/len(distances):.3f} Å\n"
        
        # Add position statistics
        summary_text += f"\nPosition Statistics:\n"
        summary_text += f"  X range: {positions[:, 0].min():.3f} to {positions[:, 0].max():.3f} Å\n"
        summary_text += f"  Y range: {positions[:, 1].min():.3f} to {positions[:, 1].max():.3f} Å\n"
        summary_text += f"  Z range: {positions[:, 2].min():.3f} to {positions[:, 2].max():.3f} Å\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.title(f'Molecule {molecule_index} Summary', fontsize=14, fontweight='bold')
        
        # Save the summary plot
        summary_png_path = os.path.join(output_folder, f'molecule_{molecule_index}_summary.png')
        plt.savefig(summary_png_path, dpi=300, bbox_inches='tight')
        print(f"Saved molecular summary to {summary_png_path}")

    # Skip interactive viewing on cluster
    print("Interactive viewing skipped (running on cluster)")

def get_spherical_harmonic_indices(lmax):
    """
    Get the indices for each l component in the spherical harmonic decomposition
    For lmax=6: l=0 (1 index), l=1 (3 indices), l=2 (5 indices), ..., l=6 (13 indices)
    Total: 1+3+5+7+9+11+13 = 49 indices
    """
    l_indices = {}
    current_idx = 0
    
    for l in range(lmax + 1):
        num_m = 2 * l + 1  # Number of m values for this l
        l_indices[l] = list(range(current_idx, current_idx + num_m))
        current_idx += num_m
    
    return l_indices

def compute_l_component_magnitudes(embeddings, l_indices):
    """
    Compute the magnitude of each l component across all channels
    embeddings shape: [num_embeddings, 49, num_channels]
    Returns: dict with l values as keys and magnitudes as values
    """
    l_magnitudes = {}
    
    for l, indices in l_indices.items():
        # Extract the l component: [num_embeddings, num_m_for_l, num_channels]
        l_component = embeddings[:, indices, :]
        
        # Compute magnitude across m and channel dimensions
        # This gives the total magnitude for this l component
        l_magnitudes[l] = np.linalg.norm(l_component, axis=(1, 2))
    
    return l_magnitudes

def plot_comprehensive_analysis(output_folder, lmax=6, molecule_index=0):
    """
    Comprehensive analysis including embeddings, outputs, and targets
    """
    # Load all data
    data = load_embeddings_from_npy(output_folder, molecule_index)
    (node_embeddings, edge_embeddings, edge_distances, positions, atomic_numbers,
     node_outputs, edge_outputs, node_targets, edge_targets, irreps_str) = data
    
    if edge_embeddings is None:
        print("No embeddings files found!")
        return
    
    print(f"Successfully loaded data for molecule {molecule_index}!")
    print(f"Node embeddings shape: {node_embeddings.shape}")
    print(f"Edge embeddings shape: {edge_embeddings.shape}")
    
    has_outputs = node_outputs is not None
    if has_outputs:
        print(f"Node outputs shape: {node_outputs.shape}")
        print(f"Edge outputs shape: {edge_outputs.shape}")
        print(f"Node targets shape: {node_targets.shape}")
        print(f"Edge targets shape: {edge_targets.shape}")
        print(f"Head irreps: {irreps_str}")
    
    # Create and visualize ASE atoms object
    atoms = create_ase_atoms(positions, atomic_numbers)
    visualize_molecule(atoms, output_folder, molecule_index)
    
    # Extract distances
    if len(edge_distances.shape) == 2 and edge_distances.shape[1] == 4:
        edge_dist_magnitudes = edge_distances[:, 0]
    else:
        if len(edge_distances.shape) == 2 and edge_distances.shape[1] == 3:
            edge_dist_magnitudes = np.linalg.norm(edge_distances, axis=1)
        else:
            edge_dist_magnitudes = edge_distances.flatten()
    
    # Analyze embeddings (spherical harmonics)
    embedding_l_indices = get_spherical_harmonic_indices(lmax)
    edge_embedding_l_mags = compute_l_component_magnitudes(edge_embeddings, embedding_l_indices)
    node_embedding_l_mags = compute_l_component_magnitudes(node_embeddings, embedding_l_indices)
    
    # Analyze outputs/targets (irreps) if available
    output_l_mags = {}
    target_l_mags = {}
    if has_outputs:
        irreps_info, irreps_l_indices = parse_e3nn_irreps(irreps_str)
        if irreps_l_indices is not None:
            print(f"Parsed irreps structure:")
            for l, indices in irreps_l_indices.items():
                print(f"  l={l}: {len(indices)} features at indices {indices[:3]}...{indices[-3:] if len(indices) > 3 else indices}")
            
            # Remove padding: set outputs to 0.0 where targets are 0.0
            node_outputs_cleaned = node_outputs.copy()
            edge_outputs_cleaned = edge_outputs.copy()
            
            # Create masks for zero targets
            node_zero_mask = (node_targets == 0.0)
            edge_zero_mask = (edge_targets == 0.0)
            
            # Apply masks to outputs
            node_outputs_cleaned[node_zero_mask] = 0.0
            edge_outputs_cleaned[edge_zero_mask] = 0.0
            
            print(f"Removed padding: {node_zero_mask.sum()} node elements, {edge_zero_mask.sum()} edge elements set to 0.0")
            
            output_l_mags['node'] = compute_l_component_magnitudes_from_irreps(node_outputs_cleaned, irreps_l_indices)
            output_l_mags['edge'] = compute_l_component_magnitudes_from_irreps(edge_outputs_cleaned, irreps_l_indices)
            target_l_mags['node'] = compute_l_component_magnitudes_from_irreps(node_targets, irreps_l_indices)
            target_l_mags['edge'] = compute_l_component_magnitudes_from_irreps(edge_targets, irreps_l_indices)
        else:
            print("Could not parse irreps structure")
    
    # Create comprehensive plots
    if has_outputs and irreps_l_indices is not None:
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.viridis(np.linspace(0, 1, max(lmax + 1, 7)))
    
    # 1. Embedding analysis (same as your version)
    ax = axes[0, 0]
    num_nodes = node_embeddings.shape[0]
    
    # Plot edges first, then nodes
    for l in range(lmax + 1):
        ax.scatter(edge_dist_magnitudes, edge_embedding_l_mags[l], 
                  alpha=0.2, s=2, color=colors[l], marker='.', zorder=1)
        ax.scatter(np.zeros(num_nodes), node_embedding_l_mags[l], 
                  alpha=0.7, s=40, color=colors[l], marker='o', 
                  edgecolors='black', linewidths=0.3, zorder=2)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[l], markersize=8, 
                                 markeredgecolor='black', markeredgewidth=0.5,
                                 label=f'l={l}', linestyle='None') for l in range(lmax + 1)]
    
    ax.set_xlabel('Inter-atomic Distance (Å)')
    ax.set_ylabel('Embedding Magnitude')
    ax.set_title('Embeddings: Magnitude vs Distance (by l)')
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. Binned averages for embeddings
    ax = axes[0, 1]
    combined_distances = np.concatenate([np.zeros(num_nodes), edge_dist_magnitudes])
    
    bins_near_zero = np.linspace(0, 0.5, 10)
    bins_far = np.linspace(0.5, combined_distances.max(), 40)
    distance_bins = np.concatenate([bins_near_zero[:-1], bins_far])
    
    for l in range(lmax + 1):
        combined_mags = np.concatenate([node_embedding_l_mags[l], edge_embedding_l_mags[l]])
        bin_means, bin_edges, _ = binned_statistic(combined_distances, combined_mags, 
                                                  statistic='mean', bins=distance_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid_mask = ~np.isnan(bin_means)
        ax.plot(bin_centers[valid_mask], bin_means[valid_mask], 
               color=colors[l], linewidth=2, label=f'l={l}')
    
    ax.set_xlabel('Inter-atomic Distance (Å)')
    ax.set_ylabel('Mean Embedding Magnitude')
    ax.set_title('Embeddings: Binned Averages (by l)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 3. Distance histogram
    ax = axes[1, 0]
    ax.hist(combined_distances, bins=51, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Inter-atomic Distance (Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Distances')
    ax.grid(True, alpha=0.3)
    
    # 4. Node embeddings by element type (your style)
    ax = axes[1, 1]
    unique_elements = np.unique(atomic_numbers)
    colors_elements = plt.cm.viridis(np.linspace(0, 1, len(unique_elements)))
    
    from ase.data import atomic_names
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_name = atomic_names[element]
        
        for l in range(lmax + 1):
            y_values = node_embedding_l_mags[l][mask]
            x_offset = 0.1 * (i - len(unique_elements)/2)
            x_values = np.ones(len(y_values)) * l + x_offset
            
            ax.scatter(x_values, y_values, alpha=0.7, s=30, 
                      color=colors_elements[i], 
                      label=f'{element_name} (Z={element})' if l == 0 else "")
    
    ax.set_xlabel('l value')
    ax.set_ylabel('Node Embedding Magnitude')
    ax.set_title(f'Node Embeddings by Element Type')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xticks(range(lmax + 1))
    ax.set_xlim(-0.5, lmax + 0.5)
    
    # 5. & 6. Outputs/Targets analysis (if available)
    if has_outputs and irreps_l_indices is not None:
        # Available l values in irreps
        available_l_values = sorted(irreps_l_indices.keys())
        irrep_colors = plt.cm.viridis(np.linspace(0, 1, len(available_l_values)))
        
        # 5. Outputs vs targets magnitude by l (edges + nodes)
        ax = axes[2, 0]
        
        # Plot outputs and targets for each l value
        for i, l in enumerate(available_l_values):
            # Edge outputs with circle markers
            ax.scatter(edge_dist_magnitudes, output_l_mags['edge'][l], 
                      alpha=0.2, s=50, color=irrep_colors[i], marker='o', 
                      edgecolors='none', label=f'Pred. $l$={l}')
            # Edge targets with x markers
            ax.scatter(edge_dist_magnitudes, target_l_mags['edge'][l], 
                      alpha=0.7, s=10, color=irrep_colors[i], marker='x', 
                      label=f'Ref. $l$={l}')
            
            # Node outputs at distance 0 with same circle markers
            ax.scatter(np.zeros(len(output_l_mags['node'][l])), output_l_mags['node'][l], 
                      alpha=0.1, s=50, color=irrep_colors[i], marker='o', 
                      edgecolors='none')
            # Node targets at distance 0 with same x markers
            ax.scatter(np.zeros(len(target_l_mags['node'][l])), target_l_mags['node'][l], 
                      alpha=0.7, s=10, color=irrep_colors[i], marker='x')
        
        ax.set_xlabel('Inter-atomic Distance (Å)')
        ax.set_ylabel('$H_{ij}$ ($E_h$)')
        # xaxis limits from -1 to 20:
        # ax.set_xlim(-1, 20)

        ax.set_title('Outputs vs Targets (by l)\nOutputs=circles, Targets=x')
        ax.legend(fontsize=7, ncol=1, frameon=False)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 6. Node outputs vs targets by element and l
        ax = axes[2, 1]
        for i, element in enumerate(unique_elements):
            mask = atomic_numbers == element
            element_name = atomic_names[element]
            
            for j, l in enumerate(available_l_values[:7]):  # Extended to show all l values up to 6
                # Node outputs
                y_values_out = output_l_mags['node'][l][mask]
                # Node targets  
                y_values_tgt = target_l_mags['node'][l][mask]
                
                x_offset = 0.1 * (i - len(unique_elements)/2)
                x_values = np.ones(len(y_values_out)) * l + x_offset
                x_values_tgt = np.ones(len(y_values_tgt)) * l + x_offset + 0.05
                
                ax.scatter(x_values, y_values_out, alpha=0.7, s=20, 
                          color=colors_elements[i], marker='o',
                          label=f'{element_name} Output' if l == available_l_values[0] else "")
                ax.scatter(x_values_tgt, y_values_tgt, alpha=0.7, s=20,
                          color=colors_elements[i], marker='x',
                          label=f'{element_name} Target' if l == available_l_values[0] else "")
        
        ax.set_xlabel('l value')
        ax.set_ylabel('Node Output/Target Magnitude')
        ax.set_title('Node Outputs vs Targets by Element')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xticks(available_l_values[:7])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'comprehensive_analysis_molecule_{molecule_index}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print correlation analysis
    print("\nEmbedding correlations:")
    for l in range(lmax + 1):
        combined_mags = np.concatenate([node_embedding_l_mags[l], edge_embedding_l_mags[l]])
        correlation = np.corrcoef(combined_distances, combined_mags)[0, 1]
        print(f"  Embedding l={l}: correlation = {correlation:.4f}")
    
    if has_outputs and irreps_l_indices is not None:
        print("\nOutput/Target correlations and statistics:")
        for l in available_l_values:
            # Edge correlation (using cleaned outputs)
            edge_corr_out = np.corrcoef(edge_dist_magnitudes, output_l_mags['edge'][l])[0, 1]
            edge_corr_tgt = np.corrcoef(edge_dist_magnitudes, target_l_mags['edge'][l])[0, 1]
            
            # Value ranges for debugging (using cleaned outputs)
            out_min, out_max = output_l_mags['edge'][l].min(), output_l_mags['edge'][l].max()
            tgt_min, tgt_max = target_l_mags['edge'][l].min(), target_l_mags['edge'][l].max()
            
            # Count non-zero values after cleaning
            nonzero_out = np.count_nonzero(output_l_mags['edge'][l])
            nonzero_tgt = np.count_nonzero(target_l_mags['edge'][l])
            total_vals = len(output_l_mags['edge'][l])
            
            print(f"  Edge l={l}: output_corr = {edge_corr_out:.4f}, target_corr = {edge_corr_tgt:.4f}")
            print(f"            output_range = [{out_min:.6f}, {out_max:.6f}], target_range = [{tgt_min:.6f}, {tgt_max:.6f}]")
            print(f"            non-zero: outputs={nonzero_out}/{total_vals} ({100*nonzero_out/total_vals:.1f}%), targets={nonzero_tgt}/{total_vals} ({100*nonzero_tgt/total_vals:.1f}%)")

def plot_individual_l_components_with_nodes(output_folder, lmax=6, molecule_index=0):
    """
    Create separate plots for each l component including nodes at distance 0
    """
    # Load embeddings
    data = load_embeddings_from_npy(output_folder, molecule_index)
    node_embeddings, edge_embeddings, edge_distances, positions, atomic_numbers = data[:5]
    
    if edge_embeddings is None:
        return
    
    # Extract scalar distances
    if len(edge_distances.shape) == 2 and edge_distances.shape[1] == 4:
        edge_dist_magnitudes = edge_distances[:, 0]  # First column is scalar distance
    else:
        if len(edge_distances.shape) == 2 and edge_distances.shape[1] == 3:
            edge_dist_magnitudes = np.linalg.norm(edge_distances, axis=1)
        else:
            edge_dist_magnitudes = edge_distances.flatten()
    
    # Get l component indices and magnitudes
    l_indices = get_spherical_harmonic_indices(lmax)
    edge_l_magnitudes = compute_l_component_magnitudes(edge_embeddings, l_indices)
    
    # Check if we can include nodes
    include_nodes = (node_embeddings.shape[1] == edge_embeddings.shape[1])
    if include_nodes:
        node_l_magnitudes = compute_l_component_magnitudes(node_embeddings, l_indices)
        num_nodes = node_embeddings.shape[0]
    
    # Create individual plots for each l - keeping your 1x7 layout
    fig, axes = plt.subplots(1, 7, figsize=(20, 3))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, lmax + 1))
    
    for l in range(lmax + 1):
        if l < len(axes):
            # Combine node and edge data
            if include_nodes:
                combined_distances = np.concatenate([np.zeros(num_nodes), edge_dist_magnitudes])
                combined_magnitudes = np.concatenate([node_l_magnitudes[l], edge_l_magnitudes[l]])
            else:
                combined_distances = edge_dist_magnitudes
                combined_magnitudes = edge_l_magnitudes[l]
            
            # Plot edge scatter points first (background)
            axes[l].scatter(edge_dist_magnitudes, edge_l_magnitudes[l], 
                           alpha=0.2, s=3, color=colors[l], marker='.', zorder=1)
            
            # Binned average
            if include_nodes:
                bins_near_zero = np.linspace(0, 0.5, 6)
                bins_far = np.linspace(0.5, combined_distances.max(), 25)
                distance_bins = np.concatenate([bins_near_zero[:-1], bins_far])
            else:
                distance_bins = np.linspace(combined_distances.min(), combined_distances.max(), 30)
                
            bin_means, bin_edges, _ = binned_statistic(combined_distances, combined_magnitudes, 
                                                      statistic='mean', bins=distance_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            valid_mask = ~np.isnan(bin_means)
            axes[l].plot(bin_centers[valid_mask], bin_means[valid_mask], 
                        color=colors[l], linewidth=2, zorder=3)
            
            # Add scatter points for nodes if available (on top)
            if include_nodes:
                axes[l].scatter(np.zeros(num_nodes), node_l_magnitudes[l], 
                               alpha=0.7, s=25, color=colors[l], marker='o', 
                               edgecolors='black', linewidths=0.3, zorder=2)
            
            axes[l].set_xlabel('Distance (Å)')
            axes[l].set_ylabel('Magnitude')
            axes[l].set_title(f'l={l} Component')
            axes[l].grid(True, alpha=0.3)
            axes[l].set_yscale('log')
    
    # Hide unused subplots
    for i in range(lmax + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'individual_l_components_molecule_{molecule_index}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    # output_folder = "outputs_omol_closedshell_25k_scaled_pt2"
    # output_folder = './iclr_2025_data/outputs_nablaDFT_tiny_scaled_rcut10_10k'
    # output_folder = 'outputs_nablaDFT_tiny_scaled_rcut10_10k'
    output_folder = 'outputs_QM7_water'
    molecule_index = 2 # Which molecule to analyze (0 for first saved molecule)
    lmax = 4
    
    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} does not exist!")
    else:
        print(f"Analyzing embeddings for molecule {molecule_index} in {output_folder}")
        plot_comprehensive_analysis(output_folder, lmax=lmax, molecule_index=molecule_index)
        plot_individual_l_components_with_nodes(output_folder, lmax=lmax, molecule_index=molecule_index)