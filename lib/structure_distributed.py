import dgl
import torch
import numpy as np
from lib import utils
from torch.utils.data import Dataset
from ase.geometry import find_mic

# DGLGraphDataset class, inherit from torch.utils.data.Dataset
class DGLGraphDataset(Dataset):
    def __init__(self, structures, equivariant_blocks, out_slices, construct_kernel, device, dtype=torch.float32):
        """
        Args:
            structures (list of Structure): List of structures to convert to DGL graphs.
            equivariant_blocks: Equivariant blocks used for creating labels.
            out_slices: Output slices for the labels.
            construct_kernel: Kernel for converting labels (used to rotate the input H)
        """
        self.dtype = dtype
        self.graphs = []
        self.labels = []
        self._create_dgl_graphs(structures, equivariant_blocks, out_slices, construct_kernel)

    def _create_dgl_graphs(self, structures, equivariant_blocks, out_slices, construct_kernel):

        for structure in structures:

            # Node features: atomic numbers 
            node_features = torch.tensor( [utils.periodic_table[i] for i in structure.atomic_species], 
                                            dtype=torch.int64, 
                                        )

            # Edge list (needs to be in COO format)
            edge_src, edge_dst = structure.edge_matrix
            edge_index = torch.tensor(np.array([edge_src, edge_dst]), dtype=torch.int64)

            # DGL graph object
            g = dgl.graph((edge_src, edge_dst))

            # Generate edge and node labels
            edge_fea, edge_labels, node_labels = self._create_labels(structure, 
                                                           edge_index, 
                                                           equivariant_blocks, 
                                                           out_slices, 
                                                           construct_kernel)

            g.ndata['feat'] = node_features
            g.edata['edge_attr'] = edge_fea
            g.edata['label'] = edge_labels
            g.ndata['node_label'] = node_labels

            self.graphs.append((g, edge_labels, node_labels))

    def _create_labels(self, structure, edge_index, equivariant_blocks, out_slices, construct_kernel):
        
        """
        Args:
            structure: The structure for which labels are created.
            edge_index: Edge index tensor.
            equivariant_blocks: Equivariant blocks used for creating labels.
            out_slices: Output slices for the labels.
            construct_kernel: Kernel for converting labels.

        Returns:
            edge_labels, node_labels: Tensors containing the labels for edges and nodes.
        """
        print("Creating labels...", flush=True)

        numbers = torch.tensor(
            [utils.periodic_table[i] for i in structure.atomic_species],
            dtype=torch.int64,
        )
        coordinates = torch.tensor(
            structure.atomic_structure.get_positions(), 
            dtype=self.dtype,
        )
        cell = structure.atomic_structure.get_cell()

        # Get Hamiltonian blocks for edges
        edge_hams = structure.get_orbital_blocks(edge_index.numpy())
        H_blocks_edge = [
                            torch.tensor(edge_hams[(edge_index[0, i].item(), edge_index[1, i].item())])
                            for i in range(edge_index.size(1))
                        ]
        print("Got Hamiltonian blocks for edges...", flush=True)

        # Diagonal orbital blocks (onsite Hamiltonian)
        onsite_edge_index = np.array([np.arange(len(numbers)), np.arange(len(numbers))])
        onsite_hams = structure.get_orbital_blocks(onsite_edge_index)
        onsite = [
                    torch.tensor(onsite_hams[(onsite_edge_index[0][i].item(), onsite_edge_index[1][i].item())])
                    for i in range(len(numbers))
                ]
        print("Got Hamiltonian blocks for nodes...", flush=True)

        # Create edge features (distance)

        edge_fea = torch.empty((len(edge_index[0]),4), dtype=self.dtype) 
        for i in range(len(edge_index[0])):
            distance_vector, distance = find_mic(coordinates[edge_index[1][i]] - coordinates[edge_index[0][i]], cell)
            edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))
        print("Created edge features...", flush=True)

        # Create edge labels
        edge_labels = []
        for i in range(len(edge_index[0])):
            label = torch.zeros(out_slices[-1], dtype=self.dtype)
            for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])
                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()

                    if (numbers[edge_index[0][i]].item() == int(condition_number_i) and numbers[edge_index[1][i]].item() == int(condition_number_j)):
                        label[slice_out] += H_blocks_edge[i][slice_row, slice_col].reshape(-1)

            edge_labels.append(label)
        print("Created edge labels...", flush=True)

        # Create node labels
        node_labels = []
        for i in range(len(onsite_edge_index[0])):
            label = torch.zeros(out_slices[-1], dtype=self.dtype)
            for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])
                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()

                    if (numbers[onsite_edge_index[0][i]].item() == int(condition_number_i) and numbers[onsite_edge_index[1][i]].item() == int(condition_number_j)):
                        label[slice_out] += onsite[i][slice_row, slice_col].reshape(-1)

            node_labels.append(label)
        print("Created node labels...", flush=True)

        edge_labels = torch.stack(edge_labels)
        node_labels = torch.stack(node_labels)

        # Convert Hamiltonian labels from uncoupled space to coupled space (to avoid conversion during training)
        print("Rotating labels...", flush=True)
        y = construct_kernel.get_net_out(edge_labels)
        node_y = construct_kernel.get_net_out(node_labels)
        print("Labels created.", flush=True)

        return edge_fea, y, node_y
           
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        Returns a single graph and its labels.

        Args:
            idx (int): Index of the structure to return.

        Returns:
            (DGLGraph, edge_labels, node_labels)
        """
        graph, _, _ = self.graphs[idx]
        return graph