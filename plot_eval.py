import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np

# Define the different evaluation datasets
# eval_datasets = ['eval_2k', 'eval_5k', 'eval_10k']
eval_datasets = ['./']
# base_path = './outputs_nablaDFT_tiny_scaled_rcut10_10k'
base_path = './outputs_QM7_uracil'

largest_width = 7000

# Prepare data for plotting
all_data = []
width_ratios = {}
for dataset in eval_datasets:
    file_pattern = f'{base_path}/{dataset}/model_eval_*.txt'
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"Warning: No files found for {dataset}")
        width_ratios[dataset] = 0.0 / largest_width
        continue
    
    dataframes = []
    for file in files:
        rank_data = pd.read_csv(file, sep='\t', header=None)
        rank_data.columns = ['edges', 'nodes', 'total', 'eigs']
        # rank_data.columns = ['total', 'eigs']
        rank_data *= 1e6  # convert to uEh
        dataframes.append(rank_data)
    
    data = pd.concat(dataframes, ignore_index=True)

    print("number of points:", data.shape[0])
    width_ratios[dataset] = data.shape[0] / largest_width

    # Melt the data for seaborn plotting
    melted_data = data.melt(var_name='metric', value_name='error')
    melted_data['dataset'] = dataset
    all_data.append(melted_data)

# Combine all datasets
combined_data = pd.concat(all_data, ignore_index=True)

print("\nWidth ratios for each dataset:")
for dataset, ratio in width_ratios.items():
    print(f"{dataset}: {ratio:.4f}")

# Print means for each dataset
for dataset in eval_datasets:
    dataset_data = combined_data[combined_data['dataset'] == dataset]
    means = dataset_data.groupby('metric')['error'].mean()
    print(f"\nMeans for {dataset}:")
    print(means)

# Create the grouped violin plot
plt.figure(figsize=(4, 4))

# use only the last two entires (total and eigenvalues):
combined_data = combined_data[combined_data['metric'].isin(['total', 'eigs'])]

# Use seaborn's violin plot with hue for grouping
ax = sns.violinplot(
    data=combined_data, 
    x='metric', 
    y='error', 
    hue='dataset',
    bw=0.05, 
    cut=0,
    scale='width',
    width=0.6,
    alpha=0.7,
    gridsize=1000, 
)

# Scale violin widths based on data count
for i, collection in enumerate(ax.collections):
    
    # Get the dataset for this violin
    dataset_idx = i % len(eval_datasets)
    dataset = eval_datasets[dataset_idx]
    scale_factor = width_ratios[dataset]
    
    # Scale the violin width
    paths = collection.get_paths()
    for path in paths:
        vertices = path.vertices
        # Scale x-coordinates (width) around the center
        center_x = vertices[:, 0].mean()
        vertices[:, 0] = center_x + (vertices[:, 0] - center_x) * scale_factor
        path.vertices = vertices


# Add min/max range lines for each violin
metrics = ['total', 'eigs']
datasets = eval_datasets

# Get actual violin positions from the plot
# Seaborn creates violins in order: all datasets for metric 1, then all datasets for metric 2
violin_positions = []

# For each metric
for metric_idx, metric in enumerate(metrics):
    # For each dataset within that metric
    for dataset_idx, dataset in enumerate(datasets):
        # Calculate the collection index
        collection_idx = metric_idx * len(datasets) + dataset_idx
        
        if collection_idx < len(ax.collections):
            collection = ax.collections[collection_idx]
            paths = collection.get_paths()
            
            if paths:
                # Get the actual center x-coordinate
                vertices = paths[0].vertices
                center_x = vertices[:, 0].mean()
                violin_positions.append((metric, dataset, center_x))

# Add vertical lines showing min/max range
for metric, dataset, x_pos in violin_positions:
    # Get data for this specific violin
    mask = (combined_data['metric'] == metric) & (combined_data['dataset'] == dataset)
    violin_data = combined_data[mask]['error']
    
    if len(violin_data) > 0:
        min_val = violin_data.min()
        max_val = violin_data.max()
        
        # Draw vertical line from min to max
        ax.plot([x_pos, x_pos], [min_val, max_val], 
                color='dimgrey', linewidth=1.5, alpha=0.7)
        
        # Add caps at min and max
        cap_width = 0.02
        ax.plot([x_pos - cap_width, x_pos + cap_width], [min_val, min_val], 
                color='dimgrey', linewidth=1, alpha=0.7)
        ax.plot([x_pos - cap_width, x_pos + cap_width], [max_val, max_val], 
                color='dimgrey', linewidth=1, alpha=0.7)



# Set labels and formatting
plt.xticks(ticks=[0, 1], labels=['H$_{ij}$', '$\\lambda$'])
plt.ylabel('Error ($\mu$E$_h$)')
plt.yscale('log')
plt.xlabel('Metric')

# y limits:
plt.ylim(2e1, 2e4)

# Improve legend
plt.legend(title='Test split', frameon=False)

# Adjust layout and save
plt.tight_layout()
plt.savefig('model_eval_comparison.png', dpi=300, bbox_inches='tight')