import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matrix2labels_kernels


save_path = Path("/capstor/store/cscs/pasc/c33/amaeder/harmonic/limited_example")
graph_targets = np.load(save_path / "graph_targets_0.npy", allow_pickle=True).item()

edge_labels_ref = torch.clone(graph_targets.edge_labels)
node_labels_ref = torch.clone(graph_targets.node_labels)

# will overwrite edge_labels and node_labels
graph_targets.make_targets()

# assert torch.allclose(edge_labels_ref, graph_targets.edge_labels)
# assert torch.allclose(node_labels_ref, graph_targets.node_labels)

# plot graph_targets.fock_matrix

plt.imshow(np.log(np.abs(graph_targets.fock_matrix.cpu().detach().numpy())), cmap='viridis')
plt.savefig("fock_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
    

fig, ax = plt.subplots(figsize=(10, 8))

# visualize edge labels
ax.matshow(np.log(np.abs(graph_targets.edge_labels.cpu().detach().numpy())), cmap='viridis')
plt.savefig("edge_labels.png", dpi=300, bbox_inches='tight')
plt.close()
    
fig, ax = plt.subplots(figsize=(10, 8))

ax.matshow(np.log(np.abs(graph_targets.node_labels.cpu().detach().numpy())), cmap='viridis')
plt.savefig("node_labels.png", dpi=300, bbox_inches='tight')
plt.close()
    

new_target = torch.zeros(( len(graph_targets.neighbour_list[0]) + len(graph_targets.atoms), graph_targets.target_len ), dtype=graph_targets.dtype, device=graph_targets.device)

# get rows and cols for edges
src_idx, target_idx = graph_targets.neighbour_list[0], graph_targets.neighbour_list[1]

# nodes will be added at the end
num_atoms = len(graph_targets.atoms)
src_idxes = np.concatenate([src_idx, np.arange(num_atoms)])
target_idxes = np.concatenate([target_idx, np.arange(num_atoms)])

# calculate the fock_block offsets
fock_block_offsets = np.concatenate([np.array([0]), np.cumsum(graph_targets.orbitals_per_atom)])

# make flat blocks a dictionary with the atom interactionas as keys
flat_blocks_dict = {}
for index_target, equivariant_block in enumerate(graph_targets.equivariant_blocks):
    for N_M_str, block_slice in equivariant_block.items():

        condition_numbers = tuple(map(int, N_M_str.split()))

        if condition_numbers not in flat_blocks_dict:
            flat_blocks_dict[condition_numbers] = []

        slice_out = slice(graph_targets.orbital_starts[index_target], graph_targets.orbital_starts[index_target + 1])

        slice_row = slice(block_slice[0], block_slice[1])
        slice_col = slice(block_slice[2], block_slice[3])
        flat_blocks_dict[condition_numbers].append((slice_row, slice_col, slice_out))

# print(flat_blocks_dict[(8,8)])

idx_to_atomic_number = graph_targets.atomic_numbers

idx_range = np.arange(len(graph_targets.atomic_numbers))

max_elements = matrix2labels_kernels.max_elements

element_idx = np.arange(max_elements)

orbital_template = [[] for _ in range(max_elements**2)]

for index_target, equivariant_block in enumerate(graph_targets.equivariant_blocks):
    for N_M_str, block_slice in equivariant_block.items():

        condition_numbers = tuple(map(int, N_M_str.split()))

        if condition_numbers not in flat_blocks_dict:
            flat_blocks_dict[condition_numbers] = []

        slice_out = slice(graph_targets.orbital_starts[index_target], graph_targets.orbital_starts[index_target + 1])

        slice_row = slice(block_slice[0], block_slice[1])
        slice_col = slice(block_slice[2], block_slice[3])

        orbital_template[
            condition_numbers[0]*max_elements + condition_numbers[1]
        ].append((slice_row, slice_col, slice_out))


# dictionary to list of list


matrix2labels_kernels.single_matrix2label(
    orbital_template,
    fock_block_offsets,
    idx_to_atomic_number,
    src_idxes,
    target_idxes,
    graph_targets.fock_matrix,
    new_target,
    forward=True
)

new_target_edge_labels = new_target[:len(graph_targets.neighbour_list[0])]
new_target_node_labels = new_target[len(graph_targets.neighbour_list[0]):]

assert torch.allclose(new_target_edge_labels, graph_targets.edge_labels)
assert torch.allclose(new_target_node_labels, graph_targets.node_labels)


new_matrix = torch.zeros_like(graph_targets.fock_matrix)

matrix2labels_kernels.single_matrix2label(
    orbital_template,
    fock_block_offsets,
    idx_to_atomic_number,
    src_idxes,
    target_idxes,
    new_matrix,
    new_target,
    forward=False
)

assert torch.allclose(new_matrix, graph_targets.fock_matrix)
