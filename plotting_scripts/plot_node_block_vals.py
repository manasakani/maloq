import numpy as np
import matplotlib.pyplot as plt

from fock_utils import utils_orca_out, fock_targets
from train_utils import loss, utils_compute, splittrainer
from dataset_utils import get_loader, dataset_analysis, get_scale_shift
from dataset_utils.ASEDataset import ASEAtomsData, ASEDataset
from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
from torch_geometric.loader import DataLoader
import gc


# Load the data from the npy file
data = np.load('./plotting_scripts/omol_element_node_block_values_unscaled.npy', allow_pickle=True).item()

# Get all atomic numbers first
atomic_numbers = list(data.keys())
element_labels = [utils_orca_out.periodic_table_number[key] for key in atomic_numbers]

# Create figure
fig, ax = plt.subplots(figsize=(5, 4))

# Don't concatenate all data at once - sample from each chunk
for index_track, atomic_number in enumerate(atomic_numbers):
    print(f"Processing element {utils_orca_out.periodic_table_number[atomic_number]} with atomic number {atomic_number}", flush=True)
    
    # Process in chunks instead of concatenating everything
    all_values = []
    for chunk in data[atomic_number]:
        # Sample only a subset from each chunk to control memory
        if len(chunk) > 1000:
            indices = np.random.choice(len(chunk), 1000, replace=False)
            all_values.extend(chunk[indices])
        else:
            all_values.extend(chunk)
        
        # Limit total number of values
        if len(all_values) > 500000:
            print("Reached 500,000 values, breaking to save memory.")
            break
    
    node_block_values = np.array(all_values)
    
    # Plot violin
    ax.violinplot(node_block_values, positions=[index_track], widths=1.0, showmeans=False, showmedians=True, bw_method=0.1)
    
    # Free memory
    del data[atomic_number]
    del node_block_values
    del all_values
    gc.collect()

# # Create a list of element labels
# element_labels = []
# for key in atomic_numbers:
#     element_name = utils_orca_out.periodic_table_number[key]
#     element_labels.append(element_name)

# Create a figure and axis object
# fig, ax = plt.subplots(figsize=(5, 4))

# # Iterate over the data and create a scatter plot for each element
# index_track = 0
# for atomic_number, node_blocks in data.items():
#     print(f"Processing element {utils_orca_out.periodic_table_number[atomic_number]} with atomic number {atomic_number}", flush=True)
#     node_block_values = np.concatenate(node_blocks)

#     # make a boxplot:
#     # ax.boxplot(node_block_values, positions=[index_track], widths=0.3, notch=True, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='red'))

#     # plot a violin:
#     ax.violinplot(node_block_values, positions=[index_track], widths=0.3, showmeans=False, showmedians=True, bw_method=0.1)

#     # # remove all the node_block_values between -0.1 and 0.1:
#     # node_block_values = node_block_values[np.abs(node_block_values) > 0.25]
#     # print("Number of scatter points to plot: ", len(node_block_values), flush=True)

#     # ax.scatter([index_track] * len(node_block_values), node_block_values, s=5.0, alpha=0.50)
#     index_track += 1

#     # free the memory for this element's node blocks in data:
#     del data[atomic_number]

# Set the x-axis ticks and labels
ax.set_xticks(range(len(element_labels)))
ax.set_xticklabels(element_labels)

# Set the title and labels
ax.set_xlabel('Element')
ax.set_ylabel('Node Block Values')

# Show the grid
ax.grid(True)

# Save the plot to a file
plt.savefig('omol_node_block_values.png', bbox_inches='tight', dpi=500)