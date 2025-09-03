import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from ase.data import atomic_names, covalent_radii
from ase.data.colors import jmol_colors
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

def get_spherical_harmonic_indices(lmax):
    """
    Get the indices for each l component in the spherical harmonic decomposition
    For lmax=4: l=0 (1 index), l=1 (3 indices), l=2 (5 indices), l=3 (7 indices), l=4 (9 indices)
    """
    l_indices = {}
    current_idx = 0
    
    for l in range(lmax + 1):
        num_m = 2 * l + 1  # Number of m values for this l
        l_indices[l] = list(range(current_idx, current_idx + num_m))
        current_idx += num_m
    
    return l_indices

def compute_l_component_magnitudes(embeddings, l_indices, lmax):
    """
    Compute the magnitude of each l component across all channels
    embeddings shape: [num_atoms, l_dimension, E] where l_dimension = sum(2*l+1) for l=0 to lmax
    Returns: dict with l values as keys and magnitudes as values
    """
    l_magnitudes = {}
    
    for l in range(lmax + 1):
        indices = l_indices[l]
        # Extract the l component: [num_atoms, num_m_for_l, E]
        l_component = embeddings[:, indices, :]
        
        # Compute magnitude across m and channel dimensions
        # This gives the total magnitude for this l component
        l_magnitudes[l] = np.linalg.norm(l_component, axis=(1, 2))
    
    return l_magnitudes

def load_node_embeddings_from_npy(output_folder, molecule_index=0):
    """Load node embeddings and atomic information for a specific molecule"""
    try:
        node_embeddings = np.load(os.path.join(output_folder, f'node_embeddings_{molecule_index}.npy'))
        positions = np.load(os.path.join(output_folder, f'positions_{molecule_index}.npy'))
        atomic_numbers = np.load(os.path.join(output_folder, f'atomic_numbers_{molecule_index}.npy'))
        
        return node_embeddings, positions, atomic_numbers
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None, None, None

def load_multiple_molecules(output_folder, max_molecules=10):
    """Load node embeddings from multiple molecules"""
    all_embeddings = []
    all_atomic_numbers = []
    all_positions = []
    molecule_indices = []
    
    for mol_idx in range(max_molecules):
        node_embeddings, positions, atomic_numbers = load_node_embeddings_from_npy(output_folder, mol_idx)
        
        if node_embeddings is not None:
            # Keep embeddings as 3D if they are [num_atoms, l_dimension, E]
            # Don't flatten - we need the l structure!
            
            all_embeddings.append(node_embeddings)
            all_atomic_numbers.append(atomic_numbers)
            all_positions.append(positions)
            molecule_indices.extend([mol_idx] * len(atomic_numbers))
            
            print(f"Loaded molecule {mol_idx}: {len(atomic_numbers)} atoms, embedding shape: {node_embeddings.shape}")
        else:
            print(f"Could not load molecule {mol_idx}")
            break
    
    if all_embeddings:
        combined_embeddings = np.vstack(all_embeddings)
        combined_atomic_numbers = np.concatenate(all_atomic_numbers)
        combined_positions = np.vstack(all_positions)
        molecule_indices = np.array(molecule_indices)
        
        print(f"\nTotal loaded: {len(all_embeddings)} molecules, {len(combined_atomic_numbers)} atoms")
        print(f"Combined embeddings shape: {combined_embeddings.shape}")
        
        return combined_embeddings, combined_atomic_numbers, combined_positions, molecule_indices
    else:
        return None, None, None, None

def plot_l_component_analysis(embeddings, atomic_numbers, molecule_indices, output_folder, lmax):
    """Analyze embeddings by spherical harmonic l components"""
    print(f"Running l-component analysis with lmax={lmax}...")
    
    # Get l component indices and magnitudes
    l_indices = get_spherical_harmonic_indices(lmax)
    l_magnitudes = compute_l_component_magnitudes(embeddings, l_indices, lmax)
    
    # Get unique elements and colors
    unique_elements = np.unique(atomic_numbers)
    colors = [jmol_colors[z] for z in unique_elements]
    l_colors = plt.cm.viridis(np.linspace(0, 1, lmax + 1))
    
    # Create comprehensive l-component analysis (removed by molecule plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: L-component magnitudes by element
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_name = atomic_names[element]
        
        for l in range(lmax + 1):
            y_values = l_magnitudes[l][mask]
            x_offset = 0.1 * (i - len(unique_elements)/2)
            x_values = np.ones(len(y_values)) * l + x_offset;
            
            axes[0, 0].scatter(x_values, y_values, alpha=0.7, s=20, 
                              color=colors[i], 
                              label=f'{element_name} (Z={element})' if l == 0 else "")
    
    axes[0, 0].set_xlabel('l value')
    axes[0, 0].set_ylabel('Embedding Magnitude')
    axes[0, 0].set_title('L-Component Magnitudes by Element')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xticks(range(lmax + 1))
    axes[0, 0].set_xlim(-0.5, lmax + 0.5)
    
    # Plot 2: Distribution of magnitudes for each l
    l_magnitude_data = [l_magnitudes[l] for l in range(lmax + 1)]
    box_plot = axes[0, 1].boxplot(l_magnitude_data, labels=[f'l={l}' for l in range(lmax + 1)], 
                                  patch_artist=True)
    
    # Color the boxplots
    for patch, color in zip(box_plot['boxes'], l_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0, 1].set_xlabel('l value')
    axes[0, 1].set_ylabel('Embedding Magnitude')
    axes[0, 1].set_title('Distribution of L-Component Magnitudes')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Mean magnitude by l and element
    mean_mags_by_element = np.zeros((len(unique_elements), lmax + 1))
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        for l in range(lmax + 1):
            mean_mags_by_element[i, l] = np.mean(l_magnitudes[l][mask])
    
    im = axes[0, 2].imshow(mean_mags_by_element, cmap='viridis', aspect='auto')
    axes[0, 2].set_xlabel('l value')
    axes[0, 2].set_ylabel('Element')
    axes[0, 2].set_title('Mean L-Component Magnitudes by Element')
    axes[0, 2].set_xticks(range(lmax + 1))
    axes[0, 2].set_yticks(range(len(unique_elements)))
    axes[0, 2].set_yticklabels([atomic_names[z] for z in unique_elements])
    plt.colorbar(im, ax=axes[0, 2], label='Mean Magnitude')
    
    # Plot 4: Correlation between different l components
    l_corr_matrix = np.zeros((lmax + 1, lmax + 1))
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            l_corr_matrix[l1, l2] = np.corrcoef(l_magnitudes[l1], l_magnitudes[l2])[0, 1]
    
    im2 = axes[1, 0].imshow(l_corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 0].set_xlabel('l value')
    axes[1, 0].set_ylabel('l value')
    axes[1, 0].set_title('Correlation Between L-Components')
    axes[1, 0].set_xticks(range(lmax + 1))
    axes[1, 0].set_yticks(range(lmax + 1))
    plt.colorbar(im2, ax=axes[1, 0], label='Correlation')
    
    # Add correlation values as text
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            axes[1, 0].text(l2, l1, f'{l_corr_matrix[l1, l2]:.2f}', 
                           ha='center', va='center', fontsize=8)
    
    # Plot 5: Variance of each l component across all atoms
    l_variances = [np.var(l_magnitudes[l]) for l in range(lmax + 1)]
    bars = axes[1, 1].bar(range(lmax + 1), l_variances, color=l_colors, alpha=0.7)
    axes[1, 1].set_xlabel('l value')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_title('Variance of L-Component Magnitudes')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xticks(range(lmax + 1))
    
    # Add variance values as text on bars
    for bar, var in zip(bars, l_variances):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{var:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Element statistics summary
    axes[1, 2].axis('off')
    text_content = "L-Component Statistics:\n\n"
    text_content += f"Max L: {lmax}\n"
    text_content += f"Total dimension: {sum(len(get_spherical_harmonic_indices(lmax)[l]) for l in range(lmax + 1))}\n\n"
    text_content += "Dimension per L:\n"
    for l in range(lmax + 1):
        dim = 2 * l + 1
        text_content += f"  l={l}: {dim} components\n"
    
    text_content += "\nElement counts:\n"
    for element in unique_elements:
        count = np.sum(atomic_numbers == element)
        text_content += f"  {atomic_names[element]}: {count} atoms\n"
    
    axes[1, 2].text(0.1, 0.9, text_content, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'l_component_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print l-component statistics
    print(f"\nL-Component Analysis Results:")
    print(f"  L-component dimensions: {[len(l_indices[l]) for l in range(lmax + 1)]}")
    print(f"  Total embedding dimension: {sum(len(l_indices[l]) for l in range(lmax + 1))}")
    
    print(f"\nMean magnitudes by l-component:")
    for l in range(lmax + 1):
        mean_mag = np.mean(l_magnitudes[l])
        std_mag = np.std(l_magnitudes[l])
        print(f"  l={l}: mean={mean_mag:.4f}, std={std_mag:.4f}, var={l_variances[l]:.4e}")
    
    print(f"\nMean magnitudes by element and l-component:")
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_name = atomic_names[element]
        print(f"  {element_name} (Z={element}):")
        for l in range(lmax + 1):
            element_mean = np.mean(l_magnitudes[l][mask])
            element_std = np.std(l_magnitudes[l][mask])
            print(f"    l={l}: mean={element_mean:.4f}, std={element_std:.4f}")
    
    return l_magnitudes, l_indices

def plot_tsne_by_element(embeddings, atomic_numbers, molecule_indices, output_folder, lmax, perplexity=30, n_iter=1000):
    """Create t-SNE visualization decomposed by spherical harmonic components"""
    print(f"Running t-SNE with perplexity={perplexity}...")
    
    # Flatten embeddings for t-SNE
    if len(embeddings.shape) == 3:
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_flat = embeddings
    
    # Run t-SNE on flattened embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings_flat)
    
    # Get l-component indices and compute individual spherical harmonic magnitudes
    l_indices = get_spherical_harmonic_indices(lmax)
    
    # Get unique elements and their colors
    unique_elements = np.unique(atomic_numbers)
    colors = [jmol_colors[z] for z in unique_elements]
    
    # Create separate plot for node embeddings t-SNE by element type
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_name = atomic_names[element]
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], s=30, alpha=0.7, 
                   label=f'{element_name} (Z={element})', edgecolors='black', linewidth=0.3)
    
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.set_title('Node Embeddings t-SNE (by Element Type)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'tsne_node_embeddings_by_element.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create decomposed t-SNE analysis: decompose embeddings into l-components
    # Each scatter point will represent one l-component from one node
    print("Decomposing embeddings into spherical harmonic l-components...")
    
    l_component_data = []
    l_component_labels = []
    element_labels_expanded = []
    molecule_labels_expanded = []
    
    # Extract l-components and create expanded datasets
    for l in range(min(lmax + 1, 5)):  # Limit to l=0 through l=4
        if l not in l_indices:
            continue
            
        m_indices = l_indices[l]  # Get indices for this l value
        
        # Extract the l-component for all atoms: [num_atoms, num_m_for_l, E]
        l_component = embeddings[:, m_indices, :]
        
        # Collapse the m dimension by taking norm: [num_atoms, E]
        l_component_collapsed = np.linalg.norm(l_component, axis=1)
        
        l_component_data.append(l_component_collapsed)
        l_component_labels.extend([l] * len(l_component_collapsed))
        element_labels_expanded.extend(atomic_numbers)
        molecule_labels_expanded.extend(molecule_indices)
    
    if not l_component_data:
        print("No l-component data found!")
        return embeddings_2d
    
    # Combine all l-components into one dataset
    combined_l_data = np.vstack(l_component_data)  # Shape: [num_atoms * num_l_components, E]
    l_component_labels = np.array(l_component_labels)
    element_labels_expanded = np.array(element_labels_expanded)
    molecule_labels_expanded = np.array(molecule_labels_expanded)
    
    print(f"Created decomposed dataset: {combined_l_data.shape}")
    print(f"Total points: {len(combined_l_data)} (from {len(embeddings)} atoms × {len(np.unique(l_component_labels))} l-components)")
    
    # Run t-SNE on the decomposed l-component data
    print("Running t-SNE on decomposed l-component data...")
    tsne_decomposed = TSNE(n_components=2, perplexity=min(30, len(combined_l_data)//4), 
                          n_iter=n_iter, random_state=42, verbose=1)
    l_components_2d = tsne_decomposed.fit_transform(combined_l_data)
    
    # Create plots for the decomposed data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Colored by l-component
    l_colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(l_component_labels))))
    for i, l in enumerate(np.unique(l_component_labels)):
        mask = l_component_labels == l
        axes[0].scatter(l_components_2d[mask, 0], l_components_2d[mask, 1], 
                       c=[l_colors[i]], s=20, alpha=0.7, 
                       label=f'l={l}', edgecolors='black', linewidth=0.2)
    
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].set_title('Decomposed t-SNE: L-Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Colored by element type
    unique_elements_expanded = np.unique(element_labels_expanded)
    element_colors = [jmol_colors[z] for z in unique_elements_expanded]
    
    for i, element in enumerate(unique_elements_expanded):
        mask = element_labels_expanded == element
        element_name = atomic_names[element]
        axes[1].scatter(l_components_2d[mask, 0], l_components_2d[mask, 1], 
                       c=[element_colors[i]], s=20, alpha=0.7, 
                       label=f'{element_name} (Z={element})', 
                       edgecolors='black', linewidth=0.2)
    
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].set_title('Decomposed t-SNE: By Element Type')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Colored by magnitude
    l_component_magnitudes = np.linalg.norm(combined_l_data, axis=1)
    scatter = axes[2].scatter(l_components_2d[:, 0], l_components_2d[:, 1], 
                             c=l_component_magnitudes, cmap='Blues', s=20, alpha=0.7,
                             edgecolors='black', linewidth=0.2)
    axes[2].set_xlabel('t-SNE Component 1')
    axes[2].set_ylabel('t-SNE Component 2')
    axes[2].set_title('Decomposed t-SNE: By Magnitude')
    plt.colorbar(scatter, ax=axes[2], label='L-Component Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'tsne_decomposed_l_components.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved decomposed l-component t-SNE analysis")
    
    # Create individual plots for each l-component
    fig, axes = plt.subplots(1, len(np.unique(l_component_labels)), figsize=(5*len(np.unique(l_component_labels)), 5))
    if len(np.unique(l_component_labels)) == 1:
        axes = [axes]
    
    for i, l in enumerate(np.unique(l_component_labels)):
        mask = l_component_labels == l
        
        # Color by element for this l-component
        for j, element in enumerate(unique_elements_expanded):
            element_mask = (element_labels_expanded == element) & mask
            if np.sum(element_mask) > 0:
                element_name = atomic_names[element]
                axes[i].scatter(l_components_2d[element_mask, 0], l_components_2d[element_mask, 1], 
                               c=[element_colors[j]], s=30, alpha=0.7, 
                               label=f'{element_name}', 
                               edgecolors='black', linewidth=0.3)
        
        axes[i].set_xlabel('t-SNE Component 1')
        axes[i].set_ylabel('t-SNE Component 2')
        axes[i].set_title(f'l={l} Components by Element')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'tsne_individual_l_components.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual l-component plots")
    
    # Also create a summary plot showing l-component magnitudes for comparison
    l_magnitudes = compute_l_component_magnitudes(embeddings, l_indices, lmax)
    
    fig2, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for l in range(5):  # l=0, 1, 2, 3, 4
        if l <= lmax:
            scatter = axes[l].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=l_magnitudes[l], cmap='Blues', s=30, alpha=0.7,
                                    edgecolors='black', linewidth=0.3)
            axes[l].set_xlabel('t-SNE Component 1')
            axes[l].set_ylabel('t-SNE Component 2')
            axes[l].set_title(f't-SNE Colored by l={l} Magnitude')
            plt.colorbar(scatter, ax=axes[l], label=f'l={l} Magnitude')
            axes[l].grid(True, alpha=0.3)
        else:
            axes[l].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'tsne_l_components_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return embeddings_2d

def plot_umap_analysis(embeddings, atomic_numbers, molecule_indices, output_folder, lmax, n_neighbors=15, min_dist=0.1):
    """Create UMAP visualization and analysis"""
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping UMAP analysis...")
        return None
    
    print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    
    # Flatten embeddings for UMAP
    if len(embeddings.shape) == 3:
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_flat = embeddings
    
    # Run UMAP on flattened embeddings
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, verbose=True)
    embeddings_2d = reducer.fit_transform(embeddings_flat)
    
    # Get l-component indices
    l_indices = get_spherical_harmonic_indices(lmax)
    
    # Get unique elements and their colors
    unique_elements = np.unique(atomic_numbers)
    colors = [jmol_colors[z] for z in unique_elements]
    
    # Create separate plot for node embeddings UMAP by element type
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 6))
    
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_name = atomic_names[element]
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], s=30, alpha=0.7, 
                   label=f'{element_name} (Z={element})', edgecolors='black', linewidth=0.3)
    
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')
    ax1.set_title('Node Embeddings UMAP (by Element Type)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'umap_node_embeddings_by_element.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create decomposed UMAP analysis: decompose embeddings into l-components
    print("Decomposing embeddings into spherical harmonic l-components for UMAP...")
    
    l_component_data = []
    l_component_labels = []
    element_labels_expanded = []
    molecule_labels_expanded = []
    
    # Extract l-components and create expanded datasets
    for l in range(min(lmax + 1, 5)):  # Limit to l=0 through l=4
        if l not in l_indices:
            continue
            
        m_indices = l_indices[l]  # Get indices for this l value
        
        # Extract the l-component for all atoms: [num_atoms, num_m_for_l, E]
        l_component = embeddings[:, m_indices, :]
        
        # Collapse the m dimension by taking norm: [num_atoms, E]
        l_component_collapsed = np.linalg.norm(l_component, axis=1)
        
        l_component_data.append(l_component_collapsed)
        l_component_labels.extend([l] * len(l_component_collapsed))
        element_labels_expanded.extend(atomic_numbers)
        molecule_labels_expanded.extend(molecule_indices)
    
    if not l_component_data:
        print("No l-component data found!")
        return embeddings_2d
    
    # Combine all l-components into one dataset
    combined_l_data = np.vstack(l_component_data)  # Shape: [num_atoms * num_l_components, E]
    l_component_labels = np.array(l_component_labels)
    element_labels_expanded = np.array(element_labels_expanded)
    molecule_labels_expanded = np.array(molecule_labels_expanded)
    
    print(f"Created decomposed dataset: {combined_l_data.shape}")
    print(f"Total points: {len(combined_l_data)} (from {len(embeddings)} atoms × {len(np.unique(l_component_labels))} l-components)")
    
    # Run UMAP on the decomposed l-component data
    print("Running UMAP on decomposed l-component data...")
    reducer_decomposed = umap.UMAP(n_neighbors=min(n_neighbors, len(combined_l_data)//4), 
                                   min_dist=min_dist, random_state=42, verbose=True)
    l_components_2d = reducer_decomposed.fit_transform(combined_l_data)
    
    # Create plots for the decomposed data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Colored by l-component
    l_colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(l_component_labels))))
    for i, l in enumerate(np.unique(l_component_labels)):
        mask = l_component_labels == l
        axes[0].scatter(l_components_2d[mask, 0], l_components_2d[mask, 1], 
                       c=[l_colors[i]], s=20, alpha=0.7, 
                       label=f'l={l}', edgecolors='black', linewidth=0.2)
    
    axes[0].set_xlabel('UMAP Component 1')
    axes[0].set_ylabel('UMAP Component 2')
    axes[0].set_title('Decomposed UMAP: L-Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Colored by element type
    unique_elements_expanded = np.unique(element_labels_expanded)
    element_colors = [jmol_colors[z] for z in unique_elements_expanded]
    
    for i, element in enumerate(unique_elements_expanded):
        mask = element_labels_expanded == element
        element_name = atomic_names[element]
        axes[1].scatter(l_components_2d[mask, 0], l_components_2d[mask, 1], 
                       c=[element_colors[i]], s=20, alpha=0.7, 
                       label=f'{element_name} (Z={element})', 
                       edgecolors='black', linewidth=0.2)
    
    axes[1].set_xlabel('UMAP Component 1')
    axes[1].set_ylabel('UMAP Component 2')
    axes[1].set_title('Decomposed UMAP: By Element Type')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Colored by magnitude
    l_component_magnitudes = np.linalg.norm(combined_l_data, axis=1)
    scatter = axes[2].scatter(l_components_2d[:, 0], l_components_2d[:, 1], 
                             c=l_component_magnitudes, cmap='Blues', s=20, alpha=0.7,
                             edgecolors='black', linewidth=0.2)
    axes[2].set_xlabel('UMAP Component 1')
    axes[2].set_ylabel('UMAP Component 2')
    axes[2].set_title('Decomposed UMAP: By Magnitude')
    plt.colorbar(scatter, ax=axes[2], label='L-Component Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'umap_decomposed_l_components.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved decomposed l-component UMAP analysis")
    
    # Create individual plots for each l-component
    fig, axes = plt.subplots(1, len(np.unique(l_component_labels)), figsize=(5*len(np.unique(l_component_labels)), 5))
    if len(np.unique(l_component_labels)) == 1:
        axes = [axes]
    
    for i, l in enumerate(np.unique(l_component_labels)):
        mask = l_component_labels == l
        
        # Color by element for this l-component
        for j, element in enumerate(unique_elements_expanded):
            element_mask = (element_labels_expanded == element) & mask
            if np.sum(element_mask) > 0:
                element_name = atomic_names[element]
                axes[i].scatter(l_components_2d[element_mask, 0], l_components_2d[element_mask, 1], 
                               c=[element_colors[j]], s=30, alpha=0.7, 
                               label=f'{element_name}', 
                               edgecolors='black', linewidth=0.3)
        
        axes[i].set_xlabel('UMAP Component 1')
        axes[i].set_ylabel('UMAP Component 2')
        axes[i].set_title(f'l={l} Components by Element (UMAP)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'umap_individual_l_components.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual l-component UMAP plots")
    
    # Also create a summary plot showing l-component magnitudes for comparison
    l_magnitudes = compute_l_component_magnitudes(embeddings, l_indices, lmax)
    
    fig2, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for l in range(5):  # l=0, 1, 2, 3, 4
        if l <= lmax:
            scatter = axes[l].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=l_magnitudes[l], cmap='Blues', s=30, alpha=0.7,
                                    edgecolors='black', linewidth=0.3)
            axes[l].set_xlabel('UMAP Component 1')
            axes[l].set_ylabel('UMAP Component 2')
            axes[l].set_title(f'UMAP Colored by l={l} Magnitude')
            plt.colorbar(scatter, ax=axes[l], label=f'l={l} Magnitude')
            axes[l].grid(True, alpha=0.3)
        else:
            axes[l].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'umap_l_components_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return embeddings_2d

def plot_pca_analysis(embeddings, atomic_numbers, molecule_indices, output_folder, lmax, n_components=10):
    """Perform PCA analysis and create visualizations"""
    print(f"Running PCA with {n_components} components...")
    
    # Flatten embeddings for PCA
    if len(embeddings.shape) == 3:
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_flat = embeddings
    
    # Run PCA
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_flat)
    
    # Get unique elements and their colors
    unique_elements = np.unique(atomic_numbers)
    colors = [jmol_colors[z] for z in unique_elements]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Explained variance
    axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_ * 100)
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance (%)')
    axes[0, 0].set_title('PCA Explained Variance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_ * 100)
    axes[0, 1].plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
    axes[0, 1].axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90%')
    axes[0, 1].axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95%')
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Cumulative Explained Variance (%)')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: First two PCs colored by element
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_name = atomic_names[element]
        axes[1, 0].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                          c=[colors[i]], s=30, alpha=0.7, 
                          label=f'{element_name} (Z={element})', 
                          edgecolors='black', linewidth=0.3)
    
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1, 0].set_title('First Two Principal Components (by Element)')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: First two PCs colored by total magnitude
    total_magnitude = np.linalg.norm(embeddings_flat, axis=1)
    scatter = axes[1, 1].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                                c=total_magnitude, cmap='viridis', s=30, alpha=0.7,
                                edgecolors='black', linewidth=0.3)
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1, 1].set_title('First Two Principal Components (by Total Magnitude)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Total Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_node_embeddings.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA Results:")
    print(f"  Total explained variance by first 2 components: {cumsum[1]:.1f}%")
    print(f"  Total explained variance by first 5 components: {cumsum[4] if len(cumsum) > 4 else cumsum[-1]:.1f}%")
    
    return embeddings_pca, pca

def plot_clustering_analysis(embeddings, atomic_numbers, molecule_indices, output_folder, lmax, n_clusters=8):
    """Perform clustering analysis on embeddings"""
    print(f"Running K-means clustering with {n_clusters} clusters...")
    
    # Flatten embeddings for clustering
    if len(embeddings.shape) == 3:
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_flat = embeddings
    
    # Standardize embeddings for clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_flat)
    
    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    # Run PCA for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_scaled)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Clusters in PCA space
    scatter = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=cluster_labels, cmap='tab10', s=30, alpha=0.7,
                                edgecolors='black', linewidth=0.3)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0, 0].set_title(f'K-means Clusters (k={n_clusters})')
    plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Clusters colored by element type
    unique_elements = np.unique(atomic_numbers)
    colors = [jmol_colors[z] for z in unique_elements]
    
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_name = atomic_names[element]
        axes[0, 1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=[colors[i]], s=30, alpha=0.7, 
                          label=f'{element_name} (Z={element})', 
                          edgecolors='black', linewidth=0.3)
    
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0, 1].set_title('Embeddings by Element Type')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cluster composition by element
    cluster_element_counts = np.zeros((n_clusters, len(unique_elements)))
    for cluster in range(n_clusters):
        cluster_mask = cluster_labels == cluster
        for i, element in enumerate(unique_elements):
            element_mask = atomic_numbers == element
            cluster_element_counts[cluster, i] = np.sum(cluster_mask & element_mask)
    
    # Normalize by cluster size
    cluster_sizes = cluster_element_counts.sum(axis=1)
    cluster_element_fractions = cluster_element_counts / cluster_sizes[:, np.newaxis]
    
    im = axes[1, 0].imshow(cluster_element_fractions.T, cmap='viridis', aspect='auto')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Element')
    axes[1, 0].set_title('Cluster Composition (Fraction by Element)')
    axes[1, 0].set_xticks(range(n_clusters))
    axes[1, 0].set_yticks(range(len(unique_elements)))
    axes[1, 0].set_yticklabels([atomic_names[z] for z in unique_elements])
    plt.colorbar(im, ax=axes[1, 0], label='Fraction')
    
    # Plot 4: Cluster sizes
    cluster_sizes = np.bincount(cluster_labels)
    axes[1, 1].bar(range(n_clusters), cluster_sizes)
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of Atoms')
    axes[1, 1].set_title('Cluster Sizes')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'clustering_node_embeddings.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print cluster analysis
    print(f"\nClustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Silhouette score: {calculate_silhouette_score(embeddings_scaled, cluster_labels):.3f}")
    
    print("\nCluster composition:")
    for cluster in range(n_clusters):
        cluster_mask = cluster_labels == cluster
        cluster_elements = atomic_numbers[cluster_mask]
        unique_cluster_elements, counts = np.unique(cluster_elements, return_counts=True)
        
        element_info = []
        for elem, count in zip(unique_cluster_elements, counts):
            element_info.append(f"{atomic_names[elem]}:{count}")
        
        print(f"  Cluster {cluster}: {cluster_sizes[cluster]} atoms - {', '.join(element_info)}")
    
    return cluster_labels, kmeans

def calculate_silhouette_score(embeddings, labels):
    """Calculate silhouette score for clustering"""
    try:
        from sklearn.metrics import silhouette_score
        return silhouette_score(embeddings, labels)
    except:
        return 0.0

def plot_embedding_statistics(embeddings, atomic_numbers, molecule_indices, output_folder, lmax):
    """Plot various statistics about the embeddings"""
    
    # Flatten embeddings for general statistics
    if len(embeddings.shape) == 3:
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_flat = embeddings
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Embedding magnitude distribution
    embedding_norms = np.linalg.norm(embeddings_flat, axis=1)
    axes[0, 0].hist(embedding_norms, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Embedding Magnitude')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Embedding Magnitudes')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Embedding magnitude by element
    unique_elements = np.unique(atomic_numbers)
    colors = [jmol_colors[z] for z in unique_elements]
    
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_name = atomic_names[element]
        element_norms = embedding_norms[mask]
        
        axes[0, 1].scatter(np.ones(len(element_norms)) * i, element_norms,
                          c=[colors[i]], s=20, alpha=0.7, 
                          label=f'{element_name} (Z={element})',
                          edgecolors='black', linewidth=0.3)
    
    axes[0, 1].set_xlabel('Element')
    axes[0, 1].set_ylabel('Embedding Magnitude')
    axes[0, 1].set_title('Embedding Magnitudes by Element')
    axes[0, 1].set_xticks(range(len(unique_elements)))
    axes[0, 1].set_xticklabels([atomic_names[z] for z in unique_elements])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Feature variance
    feature_vars = np.var(embeddings_flat, axis=0)
    axes[0, 2].plot(feature_vars)
    axes[0, 2].set_xlabel('Feature Index')
    axes[0, 2].set_ylabel('Variance')
    axes[0, 2].set_title('Feature Variance Across All Atoms')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # Plot 4: L-dimension correlation matrix
    if len(embeddings.shape) == 3:
        # Compute correlation between different l-dimensions across all atoms and channels
        l_indices = get_spherical_harmonic_indices(lmax)
        l_dim = embeddings.shape[1]
        
        # Average over all atoms and channels to get one value per l-dimension index
        l_dim_values = np.mean(embeddings, axis=(0, 2))  # Shape: [l_dimension]
        
        # Create correlation matrix for l-dimensions
        # For better visualization, we'll compute correlations between l-dimension features
        # by reshaping to [num_atoms * num_channels, l_dimension]
        reshaped_for_corr = embeddings.transpose(0, 2, 1).reshape(-1, l_dim)
        l_corr_matrix = np.corrcoef(reshaped_for_corr.T)
        
        im = axes[1, 0].imshow(l_corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 0].set_xlabel('L-dimension Index')
        axes[1, 0].set_ylabel('L-dimension Index')
        axes[1, 0].set_title(f'L-dimension Correlation Matrix\n({l_dim} l-dimensions)')
        
        # Add l-value labels on the axes
        l_boundaries = []
        current_idx = 0
        for l in range(lmax + 1):
            l_boundaries.append(current_idx)
            current_idx += 2 * l + 1
        l_boundaries.append(current_idx)
        
        # Add grid lines to separate different l values
        for boundary in l_boundaries[1:-1]:
            axes[1, 0].axhline(boundary - 0.5, color='white', linewidth=1, alpha=0.7)
            axes[1, 0].axvline(boundary - 0.5, color='white', linewidth=1, alpha=0.7)
        
        # Add l-value annotations
        for l in range(lmax + 1):
            start_idx = l_boundaries[l]
            end_idx = l_boundaries[l + 1]
            center_idx = (start_idx + end_idx) / 2
            axes[1, 0].text(center_idx, -2, f'l={l}', ha='center', va='top', fontsize=8, fontweight='bold')
            axes[1, 0].text(-2, center_idx, f'l={l}', ha='right', va='center', fontsize=8, fontweight='bold')
        
    else:
        # Fallback to feature correlation for flattened embeddings
        n_features_to_show = min(50, embeddings_flat.shape[1])
        sample_indices = np.linspace(0, embeddings_flat.shape[1]-1, n_features_to_show, dtype=int)
        corr_matrix = np.corrcoef(embeddings_flat[:, sample_indices].T)
        
        im = axes[1, 0].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 0].set_xlabel('Feature Index (sampled)')
        axes[1, 0].set_ylabel('Feature Index (sampled)')
        axes[1, 0].set_title(f'Feature Correlation Matrix\n(sample of {n_features_to_show} features)')
    
    plt.colorbar(im, ax=axes[1, 0], label='Correlation')
    
    # Plot 2: Embedding magnitude distribution by element
    unique_elements = np.unique(atomic_numbers)
    colors = [jmol_colors[z] for z in unique_elements]
    
    for i, element in enumerate(unique_elements):
        mask = atomic_numbers == element
        element_norms = embedding_norms[mask]
        element_name = atomic_names[element]
        
        axes[1, 1].scatter(np.ones(len(element_norms)) * i, element_norms,
                          c=[colors[i]], s=20, alpha=0.7, 
                          label=f'{element_name} (Z={element})',
                          edgecolors='black', linewidth=0.3)
    
    axes[1, 1].set_xlabel('Element Type')
    axes[1, 1].set_ylabel('Embedding Magnitude')
    axes[1, 1].set_title('Embedding Magnitudes by Element')
    axes[1, 1].set_xticks(range(len(unique_elements)))
    axes[1, 1].set_xticklabels([atomic_names[z] for z in unique_elements], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # Plot 6: Mean embedding by element
    mean_embeddings = []
    element_labels = []
    
    for element in unique_elements:
        mask = atomic_numbers == element
        if np.sum(mask) > 0:
            mean_emb = np.mean(embeddings_flat[mask], axis=0)
            mean_embeddings.append(mean_emb)
            element_labels.append(atomic_names[element])
    
    if len(mean_embeddings) > 1:
        mean_embeddings = np.array(mean_embeddings)
        # PCA on mean embeddings
        pca_mean = PCA(n_components=2)
        mean_emb_2d = pca_mean.fit_transform(mean_embeddings)
        
        for i, (element, color) in enumerate(zip(unique_elements, colors)):
            if i < len(mean_emb_2d):
                axes[1, 2].scatter(mean_emb_2d[i, 0], mean_emb_2d[i, 1],
                                  c=[color], s=100, 
                                  label=f'{atomic_names[element]} (Z={element})',
                                  edgecolors='black', linewidth=1)
                axes[1, 2].annotate(atomic_names[element], 
                                   (mean_emb_2d[i, 0], mean_emb_2d[i, 1]),
                                   xytext=(5, 5), textcoords='offset points')
        
        axes[1, 2].set_xlabel(f'PC1 ({pca_mean.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1, 2].set_ylabel(f'PC2 ({pca_mean.explained_variance_ratio_[1]*100:.1f}%)')
        axes[1, 2].set_title('Mean Embeddings by Element (PCA)')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Not enough elements\nfor comparison', 
                       transform=axes[1, 2].transAxes, ha='center', va='center')
        axes[1, 2].set_title('Mean Embeddings by Element')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'embedding_statistics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nEmbedding Statistics:")
    print(f"  Original embedding shape: {embeddings.shape}")
    print(f"  Flattened embedding shape: {embeddings_flat.shape}")
    print(f"  Mean magnitude: {np.mean(embedding_norms):.4f}")
    print(f"  Std magnitude: {np.std(embedding_norms):.4f}")
    print(f"  Min magnitude: {np.min(embedding_norms):.4f}")
    print(f"  Max magnitude: {np.max(embedding_norms):.4f}")
    
    print(f"\nMagnitude by element:")
    for element in unique_elements:
        mask = atomic_numbers == element
        element_norms = embedding_norms[mask]
        element_name = atomic_names[element]
        print(f"  {element_name}: mean={np.mean(element_norms):.4f}, std={np.std(element_norms):.4f}, count={len(element_norms)}")

def main(output_folder=None, max_molecules=10, start_molecule=0, lmax=4):
    """Main function to run all embedding analyses"""
    if output_folder is None:
        output_folder = 'outputs_nablaDFT_focktrained_medium'  # Default folder
    
    print(f"Loading embeddings from {output_folder}...")
    print(f"Loading {max_molecules} molecules starting from molecule {start_molecule}")
    print(f"Using lmax={lmax} for spherical harmonic analysis")
    
    # Load embeddings from multiple molecules
    all_embeddings = []
    all_atomic_numbers = []
    all_positions = []
    molecule_indices = []
    
    for mol_idx in range(start_molecule, start_molecule + max_molecules):
        node_embeddings, positions, atomic_numbers = load_node_embeddings_from_npy(output_folder, mol_idx)
        
        if node_embeddings is not None:
            # Keep embeddings as 3D if they are [num_atoms, l_dimension, E]
            # Don't flatten - we need the l structure!
            
            all_embeddings.append(node_embeddings)
            all_atomic_numbers.append(atomic_numbers)
            all_positions.append(positions)
            molecule_indices.extend([mol_idx] * len(atomic_numbers))
            
            print(f"Loaded molecule {mol_idx}: {len(atomic_numbers)} atoms, embedding shape: {node_embeddings.shape}")
        else:
            print(f"Could not load molecule {mol_idx}")
            if mol_idx == start_molecule:
                print(f"Could not load starting molecule {start_molecule}!")
                return
    
    if not all_embeddings:
        print("No embeddings found!")
        return
    
    embeddings = np.vstack(all_embeddings)
    atomic_numbers = np.concatenate(all_atomic_numbers)
    positions = np.vstack(all_positions)
    molecule_indices = np.array(molecule_indices)
    
    print(f"\nTotal loaded: {len(all_embeddings)} molecules, {len(atomic_numbers)} atoms")
    print(f"Combined embeddings shape: {embeddings.shape}")
    print(f"Unique elements: {sorted(np.unique(atomic_numbers))}")
    print(f"Element names: {[atomic_names[z] for z in sorted(np.unique(atomic_numbers))]}")
    
    # Validate embedding dimensions for lmax
    expected_l_dim = sum(2*l + 1 for l in range(lmax + 1))
    if len(embeddings.shape) == 3 and embeddings.shape[1] != expected_l_dim:
        print(f"Warning: Expected l-dimension {expected_l_dim} for lmax={lmax}, but got {embeddings.shape[1]}")
        print(f"Adjusting lmax to match embedding dimensions...")
        # Calculate actual lmax from embedding dimensions
        l_dim = embeddings.shape[1]
        actual_lmax = 0
        total_dim = 1  # l=0 contributes 1
        while total_dim < l_dim:
            actual_lmax += 1
            total_dim += 2 * actual_lmax + 1
        if total_dim == l_dim:
            lmax = actual_lmax
            print(f"Adjusted lmax to {lmax}")
        else:
            print(f"Could not determine lmax from embedding dimensions. Using lmax={lmax} anyway.")
    
    # Create output directory for plots
    analysis_folder = os.path.join(output_folder, 'embedding_analysis')
    os.makedirs(analysis_folder, exist_ok=True)
    
    # Run analyses
    print("\n" + "="*50)
    print("RUNNING EMBEDDING ANALYSES")
    print("="*50)
    
    # 1. L-component analysis (new!)
    print("\n1. Computing l-component analysis...")
    l_magnitudes, l_indices = plot_l_component_analysis(embeddings, atomic_numbers, molecule_indices, analysis_folder, lmax)
    
    # 2. Basic statistics
    print("\n2. Computing embedding statistics...")
    plot_embedding_statistics(embeddings, atomic_numbers, molecule_indices, analysis_folder, lmax)
    
    # 3. PCA analysis
    print("\n3. Running PCA analysis...")
    embeddings_pca, pca = plot_pca_analysis(embeddings, atomic_numbers, molecule_indices, analysis_folder, lmax)
    
    # 4. t-SNE analysis
    print("\n4. Running t-SNE analysis...")
    embeddings_tsne = plot_tsne_by_element(embeddings, atomic_numbers, molecule_indices, analysis_folder, lmax)
    
    # 5. Clustering analysis
    print("\n5. Running clustering analysis...")
    cluster_labels, kmeans = plot_clustering_analysis(embeddings, atomic_numbers, molecule_indices, analysis_folder, lmax)
    
    # 6. UMAP analysis (new!)
    print("\n6. Running UMAP analysis...")
    embeddings_umap = plot_umap_analysis(embeddings, atomic_numbers, molecule_indices, analysis_folder, lmax)
    
    print(f"\nAnalysis complete! Results saved to {analysis_folder}")
    print("\nGenerated files:")
    print("  - l_component_analysis.png")
    print("  - embedding_statistics.png")
    print("  - pca_node_embeddings.png") 
    print("  - tsne_node_embeddings_by_element.png")
    print("  - tsne_decomposed_l_components.png")
    print("  - tsne_individual_l_components.png")
    print("  - tsne_l_components_summary.png")
    print("  - umap_node_embeddings_by_element.png")
    print("  - umap_decomposed_l_components.png")
    print("  - umap_individual_l_components.png")
    print("  - umap_l_components_summary.png")
    print("  - clustering_node_embeddings.png")

if __name__ == "__main__":
    # Configuration - modify these values directly
    output_folder = 'outputs_nablaDFT_energytrained_tiny/embeddings'  # Change this to your output folder
    max_molecules = 100  # Number of molecules to load
    start_molecule = 0  # Starting molecule index
    lmax = 4  # Maximum l value for spherical harmonic analysis
    
    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} does not exist!")
    else:
        print(f"Analyzing embeddings from {output_folder}")
        print(f"Loading {max_molecules} molecules starting from molecule {start_molecule}")
        print(f"Using lmax={lmax} for spherical harmonic analysis")
        main(output_folder, max_molecules, start_molecule, lmax)
