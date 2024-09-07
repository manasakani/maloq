import dgl
import torch
import numpy as np
from lib import utils
from dgl.data import DGLDataset
from ase.geometry import find_mic

# DGLGraphDataset class, inherit from DGLDataset, makes graph required for DGL
class DGLGraphDataset(DGLDataset):
    def __init__(self, structure, equivariant_blocks, out_slices, construct_kernel, device, dtype=torch.float32):
        """
        Args:
            structure (Structure): Structure to convert to DGL graph.
            equivariant_blocks: Equivariant blocks used for creating labels.
            out_slices: Output slices for the labels.
            construct_kernel: Kernel for converting labels (used to rotate the input H)
        """

        self.dtype = dtype
        self.structure = structure
        self.equivariant_blocks = equivariant_blocks
        self.out_slices = out_slices
        self.construct_kernel = construct_kernel

        # 'name' parameter is probably just metadata
        super().__init__(name="amorphous_gnns")


    # def _create_dgl_graphs(self, structures, equivariant_blocks, out_slices, construct_kernel):
    def process(self):
        
        structure = self.structure
        equivariant_blocks = self.equivariant_blocks
        out_slices = self.out_slices
        construct_kernel = self.construct_kernel

        # Node features: atomic numbers 
        node_features = torch.tensor( [utils.periodic_table[i] for i in structure.atomic_species], 
                                        dtype=torch.int64  )

        # Edge list (needs to be in COO format)
        edge_src, edge_dst = structure.edge_matrix
        edge_index = np.array([edge_src, edge_dst])

        # DGL graph object
        self.graph = dgl.graph((edge_src, edge_dst))
        print(self.graph.edges(), flush=True)
        print("^ Global Edge list", flush=True)
        
        # Generate edge and node labels
        edge_fea, edge_labels, node_labels = self._create_labels(structure, 
                                                                edge_index, 
                                                                equivariant_blocks, 
                                                                out_slices, 
                                                                construct_kernel)

        # Add node and edge features and labels to the graph
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['node_label'] = node_labels
        self.graph.edata['edge_attr'] = edge_fea
        self.graph.edata['label'] = edge_labels

        print("Node types in graph:", self.graph.ntypes)
        print("Edge types in graph:", self.graph.etypes)

    def flatten_data(self, H_blocks, edge_matrix, numbers, equivariant_blocks, out_slices):
        """
        Flattens the Hamiltonian blocks H_blocks into a 1D tensor for each edge in the slice sub-structure/graph
        """

        number_of_edges = len(edge_matrix[0])

        labels = []

        # for each edge in the graph...
        for i in range(number_of_edges):
            label = np.zeros(out_slices[-1])

            # for each target subblock in the edge...
            for index_target, equivariant_block in enumerate(equivariant_blocks):

                # for each block in the Hamiltonian...
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])

                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()
                        
                    # insert the flattened block into the indices specified by slice_out in the label tensor
                    if (numbers[edge_matrix[0][i]].item() == int(condition_number_i) and numbers[edge_matrix[1][i]].item() == int(condition_number_j)):
                        # slice_out should match with the flattened slice_row x slice_col
                        label[slice_out] += np.squeeze(H_blocks[i][slice_row, slice_col].reshape(1,-1)) 

            labels.append(label)    

        return labels

    def _create_labels(self, structure, edge_index, equivariant_blocks, out_slices, construct_kernel):
        
        """
        Args:
            structure: The structure for which labels are created.
            edge_index: Edge index tensor (numpy array).
            equivariant_blocks: Equivariant blocks used for creating labels.
            out_slices: Output slices for the labels.
            construct_kernel: Kernel for converting labels.

        Returns:
            edge_labels, node_labels: Tensors containing the labels for edges and nodes.
        """
        print("Creating labels...", flush=True)

        # Get atomic numbers, coordinates and cell
        numbers = structure.atomic_numbers
        numbers = torch.tensor(numbers, dtype=torch.int64)

        coordinates = torch.tensor(
            structure.atomic_structure.get_positions(), 
            dtype=self.dtype,
        )
        cell = structure.atomic_structure.get_cell()

        number_of_edges = len(edge_index[0])
        number_of_nodes = len(numbers)
        onsite_edge_index = np.array([np.arange(len(numbers)), np.arange(len(numbers))])

        print("number_of_edges", number_of_edges, flush=True)
        print("number_of_nodes", number_of_nodes, flush=True)
        # print("onsite_edge_index", onsite_edge_index, flush=True)
        # print("edge_index", edge_index, flush=True)
        # print("numbers", numbers, flush=True)

        # Get Hamiltonian blocks for edges
        edge_hams = structure.get_orbital_blocks(edge_index)   
        H_blocks_edge = [edge_hams[(edge_index[0, i].item(), edge_index[1, i].item())] 
                        for i in range(number_of_edges)]                                       
        print("Got Hamiltonian blocks for edges...", flush=True)

        # Get Hamiltonian blocks for nodes
        onsite_hams = structure.get_orbital_blocks(onsite_edge_index)
        H_blocks_node = [onsite_hams[(onsite_edge_index[0][i].item(), onsite_edge_index[1][i].item())] 
                        for i in range(number_of_nodes)]
        print("Got Hamiltonian blocks for nodes...", flush=True)

        # Create edge features (edge edge is defined by a 1x4 vector of [scalar distance, vector distance])
        edge_fea = torch.empty((number_of_edges, 4), dtype=self.dtype)  
        for i in range(number_of_edges):
            distance_vector, distance = find_mic(coordinates[edge_index[1][i]] - coordinates[edge_index[0][i]], cell)
            edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))
        print("Created edge features...", flush=True)

        # Create edge labels
        H_blocks_edge = np.array(H_blocks_edge, dtype=object) 
        edge_labels = self.flatten_data(H_blocks_edge, edge_index, numbers, equivariant_blocks, out_slices)
        edge_labels_np = np.array(edge_labels)  
        edge_labels = torch.tensor(edge_labels_np, dtype=self.dtype)
        print("Created edge labels...", flush=True)

        # Create node labels
        H_blocks_node = np.array(H_blocks_node, dtype=object)
        node_labels = self.flatten_data(H_blocks_node, onsite_edge_index, numbers, equivariant_blocks, out_slices)
        node_labels_np = np.array(node_labels) 
        node_labels = torch.tensor(node_labels_np, dtype=self.dtype)
        print("Created node labels...", flush=True)

        # Convert Hamiltonian labels from uncoupled space to coupled space (to avoid conversion during training)
        print("Converting labels...", flush=True)
        y = construct_kernel.get_net_out(edge_labels)
        node_y = construct_kernel.get_net_out(node_labels)

        print("Labels created.", flush=True)
        
        y = edge_labels
        node_y = node_labels

        return edge_fea, y, node_y
    
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    # def __getitem__(self, idx):
    #     """
    #     Returns a single graph and its labels.

    #     Args:
    #         idx (int): Index of the structure to return.

    #     Returns:
    #         (DGLGraph, edge_labels, node_labels)
    #     """
    #     graph, _, _ = self.graphs[idx]
    #     return graph