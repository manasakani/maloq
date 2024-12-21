# This file contains the functions to process the data, create the input data object for the GNN, and batch the data for training

import torch
import numpy as np
import utils

from torch_geometric.data import Data as gnnData
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
from ase.geometry import find_mic
import torch.distributed as dist
from mpi4py import MPI

# Custom dataset class for the GNN
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def custom_collate_fn(batch):
    return Batch.from_data_list(batch)


def split_data_indices(num_train, num_validate, num_test, num_total, offset=0):
    """
    Splits the data indices into training, validation, and test sets
    """
    indices = np.arange(offset, num_total)
    np.random.shuffle(indices)

    train_indices = indices[:num_train]
    validate_indices = indices[num_train:num_train+num_validate]
    test_indices = indices[num_train+num_validate:num_train+num_validate+num_test]

    return train_indices, validate_indices, test_indices


def create_input_data_molecules(structure, partition, equivariant_blocks, out_slices, construct_kernel, device, dtype):
    """
    Adds the structure data to the data list
    Need to keep track of the batch size, since the start and end nodes index the full data list and not just the current structure.
    """

    comm = partition.comm
    
    start_node = partition.start_node
    end_node = partition.end_node
    start_edge = partition.start_edge
    end_edge = partition.end_edge
    local_edge_index = partition.local_edge_index
    global_edge_index = partition.global_edge_index

    # Note: for SO2 network, edge_index has two-way edges, and does not include self-connections 
    global_atomic_numbers = torch.tensor([utils.periodic_table[i] for i in structure.atomic_species])
    local_atomic_numbers = global_atomic_numbers[start_node:end_node]

    global_coordinates = structure.atomic_structure.get_positions()
    cell = structure.atomic_structure.get_cell()

    # off-diagonal orbital blocks for each edge (bothways)
    edge_hams = structure.get_orbital_blocks(local_edge_index)
    local_edge_index = torch.tensor(local_edge_index)
    H_blocks_edge = [edge_hams[(local_edge_index[0][i].item(), local_edge_index[1][i].item())] for i in range(len(local_edge_index[0]))]

    # The following does the equivalent of [H_blocks_edge = np.array(H_blocks_edge, dtype=object)] while circumventing some of numpy's size checks for objects:
    H_blocks_edge = [np.array(block) for block in H_blocks_edge]
    H_blocks_edge_object = np.empty(len(H_blocks_edge), dtype=object)
    for i, block in enumerate(H_blocks_edge):
        H_blocks_edge_object[i] = block
    H_blocks_edge = H_blocks_edge_object

    # diagonal orbital blocks (onsite Hamiltonian)
    local_onsite_edge_index = np.array([np.arange(start_node, end_node), np.arange(start_node, end_node)])
    onsite_hams = structure.get_orbital_blocks(local_onsite_edge_index)
    H_blocks_node = [onsite_hams[(local_onsite_edge_index[0][i].item(), local_onsite_edge_index[1][i].item())] for i in range(len(local_atomic_numbers))]  

    # The following does the equivalent of [H_blocks_node = np.array(H_blocks_node, dtype=object)] while circumventing some of numpy's size checks for objects:
    H_blocks_node = [np.array(block) for block in H_blocks_node]
    H_blocks_node_object = np.empty(len(H_blocks_node), dtype=object)
    for i, block in enumerate(H_blocks_node):
        H_blocks_node_object[i] = block
    H_blocks_node = H_blocks_node_object

    # Prepare the off-diagonal orbital blocks --> Edge labels
    edge_labels = []
    for i in range(len(local_edge_index[0])):
        # print("Working on edge ", i, " of ", len(local_edge_index[0]))
        label = np.zeros(out_slices[-1])
        for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])

                    # len_row = block_slice[1] - block_slice[0]
                    # len_col = block_slice[3] - block_slice[2]
                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()

                    if (global_atomic_numbers[local_edge_index[0][i]].item() == int(condition_number_i) 
                        and global_atomic_numbers[local_edge_index[1][i]].item() == int(condition_number_j)):

                        label[slice_out] += np.squeeze(H_blocks_edge[i][slice_row, slice_col].reshape(1,-1))

        edge_labels.append(label)


    # Prepare the diagonal orbital blocks --> Node labels
    node_labels = []
    for i in range(len(local_onsite_edge_index[0])):
        # print("Working on node ", i, " of ", len(local_onsite_edge_index[0]))
        label = np.zeros(out_slices[-1])
        for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])
                    # len_row = block_slice[1] - block_slice[0]
                    # len_col = block_slice[3] - block_slice[2]
                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()

                    if (global_atomic_numbers[local_onsite_edge_index[0][i]].item() == int(condition_number_i) 
                        and global_atomic_numbers[local_onsite_edge_index[1][i]].item() == int(condition_number_j)):

                        label[slice_out] += np.squeeze(H_blocks_node[i][slice_row, slice_col].reshape(1,-1))

        node_labels.append(label)
    dist.barrier()
    
    # Edge distances --> edge features
    edge_fea = torch.empty((len(local_edge_index[0]),4))
    global_coordinates = torch.tensor(global_coordinates)
    for i in range(len(local_edge_index[0])):
        # print("Working on edge feature ", i, " of ", len(local_edge_index[0]))
        distance_vector, distance = find_mic(global_coordinates[local_edge_index[1][i]] - global_coordinates[local_edge_index[0][i]], cell)
        edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))

    edge_fea = torch.tensor(edge_fea, dtype=dtype)

    # --> allgatherv the global edge distance vector, because this is needed for deterministic rotation matrices
    dist.barrier()
    edge_distance_vec = edge_fea[:, [2, 3, 1]]

    flattened_edge_distance_vec = edge_distance_vec.cpu().detach().numpy().reshape(-1) # gloo
    local_edge_dist_vec_size = len(flattened_edge_distance_vec)
    # flattened_edge_distance_vec = edge_distance_vec.detach().reshape(-1).contiguous()  # nccl
    # local_edge_dist_vec_size = flattened_edge_distance_vec.numel()
    
    all_counts = comm.allgather(local_edge_dist_vec_size)
    displacements = np.cumsum([0] + all_counts[:-1])

    total_edge_dist_vec_size = sum(all_counts)
    global_edge_distance_vec = torch.empty(total_edge_dist_vec_size, dtype=dtype)
    
    ###
    comm.Allgatherv(flattened_edge_distance_vec, [global_edge_distance_vec, all_counts, displacements, MPI.DOUBLE])
    ###
    # gathered_tensors = comm.allgather(flattened_edge_distance_vec) # trying regular allgather with 3 ranks
    # dist.barrier()
    # print("!!! Replace this allgather with Allgatherv !!!")
    # global_edge_distance_vec = torch.cat(gathered_tensors)
    ###
    global_edge_distance_vec = global_edge_distance_vec.reshape(-1, 3)
    dist.barrier()

    # fill in globals in the partition object that are needed during the forward pass
    partition.global_edge_distance_vec = torch.tensor(global_edge_distance_vec, device=device)
    partition.global_atomic_numbers = torch.tensor(global_atomic_numbers, device=device)

    # Atomic numbers --> node features
    local_atomic_numbers = local_atomic_numbers.numpy()
    x = torch.tensor(local_atomic_numbers)

    # convert Hamiltonian labels from uncoupled space to coupled space (to avoid conversion during training)
    edge_labels = torch.tensor(np.array(edge_labels), dtype=dtype, device=device)
    node_labels = torch.tensor(np.array(node_labels), dtype=dtype, device=device)
    y = construct_kernel.get_net_out(edge_labels) 
    node_y = construct_kernel.get_net_out(node_labels)

    data = gnnData(x=x, edge_index=local_edge_index, edge_attr=edge_fea, y=y, node_y=node_y)

    return data


def slice_criteria(atom, cutoff, location, pos, cell):
    
    distance_vector, distance = find_mic(pos[atom]-pos[location], cell)
    if abs(distance_vector[0]) < cutoff:
        return True 
    else:
        return False

def create_slice_graph(atom_index, edge_matrix, add_virtual = True, two_way = False):

    """
    Generates required data to locate atoms and edges belonging to the slice sub-structure/graph

    Note: Virtual atoms are always at the end of the atom index list.

    Inputs: atom_index: list of atom indices that are part of the slice
           edge_matrix: edge indices of the full structure 
           add_virtual: if True, virtual atoms are added to the slice atom index list and their edges are included in the slice edge index list    

    Outputs: slice_graph: dictionary containing the following keys: 
            full_atom_index: atom indices of the slice sub-structure/graph, including virtual nodes 
            full_mapped_edge_index: edge indices of the slice sub-structure/graph, follows the order of the atom index list
            full_edge_positions: numbers indicating the positions of selected edge indices within the full edge index list
            node_degree: number of edges connected to each node
            reduced_node_degree: number of non-virtual edges connected to each node
            real_node_size: number of non-virtual atoms in the full atom index list, used to separate the virtual atoms from the labelled atoms
            real_edge_size: number of non-virtual edges in the full edge index list, used to separate the virtual edges from the labelled edges
    """
    
    virtual_atom_index = [] #atom indices of the virtual atoms
    edge_positions = [] #numbers indicating the positions of selected edge indices within the full edge index list 

    mapped_edge_index = [] #edge indices of the slice sub-structure/graph, follows the order of the atom index list  e.g. if atom index list is [25, 26, 40 ...], then atom 25 is atom 0 in the sub-structure/graph
    node_degree = [] #number of edges connected to each node
    reduced_node_degree = [] #number of non-virtual edges connected to each node

    slice_graph = {}

    for i in range(len(atom_index)):
        edge_position = np.squeeze(np.where(edge_matrix[0] == atom_index[i])) #locate the positions of all edges connected to that particular atom
        node_degree.append(len(edge_position))
        count = 0
        for j in range(len(edge_position)):
            if edge_matrix[1][edge_position[j]] in atom_index:
                atom_source_index = atom_index.index(edge_matrix[0][edge_position[j]]) #find the positions of the source and target atoms that are part of the slice (to create the edge indices for the data objects)
                atom_target_index = atom_index.index(edge_matrix[1][edge_position[j]])
                mapped_edge_index.append([atom_source_index,atom_target_index])
                edge_positions.append(edge_position[j])
                count = count + 1
            else:
                if edge_matrix[1][edge_position[j]] not in virtual_atom_index: #if the target atom is not part of the slice, add it to the virtual atom index list. Avoid duplicates
                    virtual_atom_index.append(edge_matrix[1][edge_position[j]].item())
                    
        reduced_node_degree.append(count)


    if (add_virtual == True):
        full_atom_index = atom_index + virtual_atom_index #add the indices of the virtual atoms to the original slice atom index list
        virtual_edge_positions = []
        virtual_mapped_edge_index = []

        for i in range(len(virtual_atom_index)): 
            virtual_edge_position = np.squeeze(np.where(edge_matrix[0] == virtual_atom_index[i])) #find the virtual edges connected to the virtual atoms
            for j in range(len(virtual_edge_position)):
                if edge_matrix[1][virtual_edge_position[j]] in atom_index:
                    atom_i_index = full_atom_index.index(edge_matrix[0][virtual_edge_position[j]]) #only include one way edges where the source atom is a virtual atom and the target atom is part of the slice
                    atom_j_index = full_atom_index.index(edge_matrix[1][virtual_edge_position[j]])
                    virtual_mapped_edge_index.append([atom_i_index,atom_j_index])
                    virtual_edge_positions.append(virtual_edge_position[j])

        full_mapped_edge_index = mapped_edge_index + virtual_mapped_edge_index #mapped edge indices of the full graph including virtual nodes 
        full_edge_positions = edge_positions + virtual_edge_positions
        
        if (two_way == True):
            print('Using two-way edges for virtual nodes')
            for i in range(len(atom_index)): 
                virtual_edge_position = np.squeeze(np.where(edge_matrix[0] == atom_index[i])) #find the virtual edges connected to the real atoms (source is now the real atom, target is the virtual atom)
                for j in range(len(virtual_edge_position)):
                    if edge_matrix[1][virtual_edge_position[j]] in virtual_atom_index:
                        atom_i_index = full_atom_index.index(edge_matrix[0][virtual_edge_position[j]]) 
                        atom_j_index = full_atom_index.index(edge_matrix[1][virtual_edge_position[j]])
                        virtual_mapped_edge_index.append([atom_i_index,atom_j_index])
                        virtual_edge_positions.append(virtual_edge_position[j])

    else:
        full_atom_index = atom_index
        full_mapped_edge_index = mapped_edge_index
        full_edge_positions = edge_positions

    slice_graph['full_atom_index'] = torch.tensor(full_atom_index)
    slice_graph['full_mapped_edge_index'] = torch.tensor(full_mapped_edge_index).T
    slice_graph['full_edge_positions'] = torch.tensor(full_edge_positions)
    slice_graph['node_degree'] = node_degree
    slice_graph['reduced_node_degree'] = reduced_node_degree
    slice_graph['real_node_size'] = len(atom_index) #index of the labelled atoms that are part of the slice 
    slice_graph['real_edge_size'] = len(edge_positions) #index of the labelled edges that are part of the slice
    
    return slice_graph


def flatten_data(H_blocks, edge_matrix, numbers, equivariant_blocks, out_slices):
    """
    Flattens the Hamiltonian blocks H_blocks into a 1D tensor for each edge in the slice sub-structure/graph
    """

    labels = []
    for i in range(len(edge_matrix[0])):
        label = np.zeros(out_slices[-1])
        for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])
                    # len_row = block_slice[1] - block_slice[0]
                    # len_col = block_slice[3] - block_slice[2]
                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()
                    if (numbers[edge_matrix[0][i]].item() == int(condition_number_i) and numbers[edge_matrix[1][i]].item() == int(condition_number_j)):
                        label[slice_out] += np.squeeze(H_blocks[i][slice_row, slice_col].reshape(1,-1)) #slice_out should match with slice_row x slice_row when flattened

        labels.append(label)    

    return labels

def createdata_graphpartition(structure, subgraph_nodes, equivariant_blocks, out_slices, construct_kernel, dtype=torch.float64):

    # call create_subgraph_dict
    pos = structure.atomic_structure.get_positions()
    cell = structure.atomic_structure.get_cell()
    edge_matrix = structure.edge_matrix
    numbers = structure.atomic_numbers

    # the subgraph nodes should be a list, not a numpy array
    slice_graph = create_slice_graph(subgraph_nodes.tolist(), edge_matrix)

    full_mapped_edge_index = slice_graph['full_mapped_edge_index']
    full_edge_positions = slice_graph['full_edge_positions']
    full_atom_index = slice_graph['full_atom_index']

    edge_matrix = torch.tensor(edge_matrix)

    # find the off-diagonal Hamiltonian blocks of all edges that are part of the graph
    edge_index = edge_matrix.T[full_edge_positions].numpy() 
    edge_index = edge_index.T
    offsite_ham = structure.get_orbital_blocks(edge_index)
    H_blocks_edge = []
    for i in range(len(edge_index[0])):
        H_blocks_edge.append(offsite_ham[(edge_index[0][i].item(), edge_index[1][i].item())])

    H_blocks_edge = np.array(H_blocks_edge, dtype=object) 
    edge_labels = flatten_data(H_blocks_edge, edge_index, numbers, equivariant_blocks, out_slices)

    # find the onsite Hamiltonian blocks for all atoms that are part of the graph
    onsite_edge_index = np.array([np.array(full_atom_index),np.array(full_atom_index)])
    onsite_ham = structure.get_orbital_blocks(onsite_edge_index)
    H_blocks_node = []
    for i in range(len(onsite_edge_index[0])):
         H_blocks_node.append(onsite_ham[(onsite_edge_index[0][i].item(),onsite_edge_index[1][i].item())])
    H_blocks_node = np.array(H_blocks_node, dtype=object) 
    node_labels = flatten_data(H_blocks_node, onsite_edge_index, numbers, equivariant_blocks, out_slices)

    # edge features are the interatomic distances - include periodic boundary conditions
    edge_fea = torch.empty((len(edge_index[0]),4))
    for i in range(len(edge_index[0])):
        distance_vector, distance = find_mic(pos[edge_index[1][i]] - pos[edge_index[0][i]], cell)
        edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))

    edge_fea = torch.tensor(edge_fea, dtype = dtype)

    # create the node features, which are the atomic numbers of the atoms in the slice
    atomic_numbers = numbers[full_atom_index] 
    x = torch.tensor(atomic_numbers)

    edge_labels_np = np.array(edge_labels)  # Convert list of numpy arrays to a single numpy ndarray
    edge_labels = torch.tensor(edge_labels_np,dtype = dtype)

    # convert Hamiltonian labels from uncoupled space to coupled space (to avoid conversion during training)
    y = construct_kernel.get_net_out(edge_labels) 
    node_labels_np = np.array(node_labels)  # Convert list of numpy arrays to a single numpy ndarray
    node_labels = torch.tensor(node_labels_np, dtype = dtype)
    node_y = construct_kernel.get_net_out(node_labels)

    atom_indices = torch.tensor(full_atom_index)
    atom_coordinates = torch.tensor(pos[atom_indices])

    # create the data object
    data = Data(x=x, 
                edge_index=full_mapped_edge_index, 
                edge_attr=edge_fea, 
                y=y, 
                node_y=node_y, 
                labelled_edge_size=slice_graph['real_edge_size'],
                labelled_node_size=slice_graph['real_node_size'], 
                node_degree=slice_graph['node_degree'], 
                reduced_node_degree=slice_graph['reduced_node_degree'], 
                atom_indices=atom_indices, 
                atom_coordinates=atom_coordinates)    

    return data


# create a data object for a subgraph of the input Structure specified by slice_center
def createdata_subgraph(structure, slice_center, cutoff, equivariant_blocks, out_slices, construct_kernel, dtype=torch.float64):
    
    pos = structure.atomic_structure.get_positions()
    cell = structure.atomic_structure.get_cell()
    edge_matrix = structure.edge_matrix
    numbers = structure.atomic_numbers

    atom_index = []
    for i in range(len(numbers)):
        if slice_criteria(i,cutoff, slice_center, pos, cell):
            atom_index.append(i)

    slice_graph = create_slice_graph(atom_index, edge_matrix)

    full_mapped_edge_index = slice_graph['full_mapped_edge_index']
    full_edge_positions = slice_graph['full_edge_positions']
    full_atom_index = slice_graph['full_atom_index']

    edge_matrix = torch.tensor(edge_matrix)

    # find the off-diagonal Hamiltonian blocks of all edges that are part of the graph
    edge_index = edge_matrix.T[full_edge_positions].numpy() 
    edge_index = edge_index.T
    offsite_ham = structure.get_orbital_blocks(edge_index)

    
    H_blocks_edge = []
    for i in range(len(edge_index[0])):
        H_blocks_edge.append(offsite_ham[(edge_index[0][i].item(), edge_index[1][i].item())])

    H_blocks_edge = np.array(H_blocks_edge, dtype=object) 
    edge_labels = flatten_data(H_blocks_edge, edge_index, numbers, equivariant_blocks, out_slices)

    # find the onsite Hamiltonian blocks for all atoms that are part of the graph
    onsite_edge_index = np.array([np.array(full_atom_index),np.array(full_atom_index)])
    onsite_ham = structure.get_orbital_blocks(onsite_edge_index)


    H_blocks_node = []
    for i in range(len(onsite_edge_index[0])):
         H_blocks_node.append(onsite_ham[(onsite_edge_index[0][i].item(),onsite_edge_index[1][i].item())])
    H_blocks_node = np.array(H_blocks_node, dtype=object) 
    node_labels = flatten_data(H_blocks_node, onsite_edge_index, numbers, equivariant_blocks, out_slices)

    # edge features are the interatomic distances - include periodic boundary conditions
    edge_fea = torch.empty((len(edge_index[0]),4))
    for i in range(len(edge_index[0])):
        distance_vector, distance = find_mic(pos[edge_index[1][i]] - pos[edge_index[0][i]], cell)
        edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))

    edge_fea = torch.tensor(edge_fea, dtype = dtype)

    # create the node features, which are the atomic numbers of the atoms in the slice
    atomic_numbers = numbers[full_atom_index] 
    x = torch.tensor(atomic_numbers)

    edge_labels_np = np.array(edge_labels)  # Convert list of numpy arrays to a single numpy ndarray
    edge_labels = torch.tensor(edge_labels_np,dtype = dtype)

    # convert Hamiltonian labels from uncoupled space to coupled space (to avoid conversion during training)
    y = construct_kernel.get_net_out(edge_labels) 
    node_labels_np = np.array(node_labels)  # Convert list of numpy arrays to a single numpy ndarray
    node_labels = torch.tensor(node_labels_np, dtype = dtype)
    node_y = construct_kernel.get_net_out(node_labels)

    atom_indices = torch.tensor(full_atom_index)
    atom_coordinates = torch.tensor(pos[atom_indices])

    # create the data object
    data = Data(x=x, 
                edge_index=full_mapped_edge_index, 
                edge_attr=edge_fea, 
                y=y, 
                node_y=node_y, 
                labelled_edge_size=slice_graph['real_edge_size'],
                labelled_node_size=slice_graph['real_node_size'], 
                node_degree=slice_graph['node_degree'], 
                reduced_node_degree=slice_graph['reduced_node_degree'], 
                atom_indices=atom_indices, 
                atom_coordinates=atom_coordinates)    

    return data



def slice_cartesian(atom_pos,start,length):
    if atom_pos[0] >= start and atom_pos[0] < start + length:
        return True
    else:
        return False

def createdata_subgraph_cartesian(structure, start, length, equivariant_blocks, out_slices, construct_kernel, dtype=torch.float64, add_virtual = True, two_way = False):
    
    pos = structure.atomic_structure.get_positions()
    cell = structure.atomic_structure.get_cell()
    edge_matrix = structure.edge_matrix
    numbers = structure.atomic_numbers

    atom_index = []

    for i in range(len(numbers)):
        if slice_cartesian(pos[i],start,length):
            atom_index.append(i)

    slice_graph = create_slice_graph(atom_index, edge_matrix, add_virtual, two_way)

    full_mapped_edge_index = slice_graph['full_mapped_edge_index']
    full_edge_positions = slice_graph['full_edge_positions']
    full_atom_index = slice_graph['full_atom_index']

    edge_matrix = torch.tensor(edge_matrix)

    # find the off-diagonal Hamiltonian blocks of all edges that are part of the graph
    edge_index = edge_matrix.T[full_edge_positions].numpy() 
    edge_index = edge_index.T
    offsite_ham = structure.get_orbital_blocks(edge_index)

    
    H_blocks_edge = []
    for i in range(len(edge_index[0])):
        H_blocks_edge.append(offsite_ham[(edge_index[0][i].item(), edge_index[1][i].item())])

    H_blocks_edge = np.array(H_blocks_edge, dtype=object) 
    edge_labels = flatten_data(H_blocks_edge, edge_index, numbers, equivariant_blocks, out_slices)

    # find the onsite Hamiltonian blocks for all atoms that are part of the graph
    onsite_edge_index = np.array([np.array(full_atom_index),np.array(full_atom_index)])
    onsite_ham = structure.get_orbital_blocks(onsite_edge_index)


    H_blocks_node = []
    for i in range(len(onsite_edge_index[0])):
         H_blocks_node.append(onsite_ham[(onsite_edge_index[0][i].item(),onsite_edge_index[1][i].item())])
    H_blocks_node = np.array(H_blocks_node, dtype=object) 
    node_labels = flatten_data(H_blocks_node, onsite_edge_index, numbers, equivariant_blocks, out_slices)

    # edge features are the interatomic distances - include periodic boundary conditions
    edge_fea = torch.empty((len(edge_index[0]),4))
    for i in range(len(edge_index[0])):
        distance_vector, distance = find_mic(pos[edge_index[1][i]] - pos[edge_index[0][i]], cell)
        edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))

    edge_fea = torch.tensor(edge_fea, dtype = dtype)

    # create the node features, which are the atomic numbers of the atoms in the slice
    atomic_numbers = numbers[full_atom_index] 
    x = torch.tensor(atomic_numbers)

    edge_labels_np = np.array(edge_labels)  # Convert list of numpy arrays to a single numpy ndarray
    edge_labels = torch.tensor(edge_labels_np,dtype = dtype)

    # convert Hamiltonian labels from uncoupled space to coupled space (to avoid conversion during training)
    y = construct_kernel.get_net_out(edge_labels) 
    node_labels_np = np.array(node_labels)  # Convert list of numpy arrays to a single numpy ndarray
    node_labels = torch.tensor(node_labels_np, dtype = dtype)
    node_y = construct_kernel.get_net_out(node_labels)

    atom_indices = torch.tensor(full_atom_index)
    atom_coordinates = torch.tensor(pos[atom_indices])

    # create the data object
    data = Data(x=x, 
                edge_index=full_mapped_edge_index, 
                edge_attr=edge_fea, 
                y=y, 
                node_y=node_y, 
                labelled_edge_size=slice_graph['real_edge_size'],
                labelled_node_size=slice_graph['real_node_size'], 
                node_degree=slice_graph['node_degree'], 
                reduced_node_degree=slice_graph['reduced_node_degree'], 
                atom_indices=atom_indices, 
                atom_coordinates=atom_coordinates)    

    return data


# Creates a dataloader for a dataset with a list of molecules
def batch_data_molecules(structures, partition, device, num_graph=1, batch_size=1, equivariant_blocks=None, out_slices=None, construct_kernel=None, dtype=torch.float64):

    data_list = []

    for i in range(num_graph):

        data = create_input_data_molecules(structures[i], partition, equivariant_blocks, out_slices, construct_kernel, device, dtype=dtype)
        data_list.append(data)
    
    dataset = CustomDataset(data_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

    print("*** Batch properties:")
    for batch in loader:
        print("Node Features (x):", batch.x.size())
        print("Edge Index:", batch.edge_index.size())
        print("Edge Features (edge_attr):", batch.edge_attr.size())    

    return loader
    

# Subgraphs without periodic boundary conditions
def batch_data_subgraph(graph, slice_list, cutoff=2, equivariant_blocks=None, out_slices=None, construct_kernel=None, dtype=torch.float64):
    """
    structures: list of Structure objects
    slice_list: list of indices which define the center of each subgraph
    cutoff: cutoff boundary of the slice used for training 
    equivariant_blocks: dictionary containing the start and end indices of the equivariant blocks in i and j direction for each target in targets
    out_slices: marks the start and end of indices belonging to a certain target. Slice 1 (0 to 1) corresponds to the first target in equivariant blocks
    construct_kernel: SO2.e3TensorDecomp object
    """

    data_list = []

    for i in range(len(slice_list)):
        train_data = createdata_subgraph(graph, slice_list[i], cutoff ,equivariant_blocks, out_slices, construct_kernel, dtype=dtype)
        data_list.append(train_data)

    dataset = CustomDataset(data_list)

    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    print("*** Batch properties:")
    for batch in loader:
        print("--> Batch: ")
        print("Node Features (x):", batch.x.size())
        print("Edge Index:", batch.edge_index.size())
        print("Edge Features (edge_attr):", batch.edge_attr.size())    

    return loader


# used in structures/materials/a-HfO2/
def batch_data_HfO2_cartesian(graph, start, total_length, num_slices, test_list = None, save_file = 'None', cutoff = 2, equivariant_blocks = None, out_slices = None, construct_kernel=None, dtype = torch.float32, add_virtual = True, two_way = False):

    data_list = []

    start = start
    length = total_length/num_slices
    num_atoms = 0
    num_edges = 0

    print("length of each slice (minus remainder): ", length)

    for i in range(num_slices):
        train_data = createdata_subgraph_cartesian(graph, start, length ,equivariant_blocks, out_slices, construct_kernel, dtype, add_virtual, two_way)
        data_list.append(train_data)
        start = start + length
        num_atoms += train_data.labelled_node_size
        num_edges += train_data.labelled_edge_size
        print("Number of atoms in slice ", i, ":", train_data.labelled_node_size)
        print("Number of edges in slice ", i, ":", train_data.labelled_edge_size)
              
    print("----------------------")
    print("Total Number of Atoms: ", num_atoms)
    print("Total Number of Edges: ", num_edges)
            
    dataset = CustomDataset(data_list)

    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    print("*** Batch properties:")
    for batch in loader:
        print("--> Batch: ")
        print("Node Features (x):", batch.x.size())
        print("Edge Index:", batch.edge_index.size())
        print("Edge Features (edge_attr):", batch.edge_attr.size())
        print("Average Node Degree:", np.mean(np.array(batch.node_degree)))
        print("Average Reduced Node Degree", np.mean(np.array(batch.reduced_node_degree)))     

    return loader

def batch_data_graphpartition(graph, num_subgraph, num_batch, equivariant_blocks=None, out_slices=None, construct_kernel=None, dtype=torch.float64):

    # Partition the large input Structure into smaller subgraphs for training using spectral clustering
    # print("*** Partitioning the graph into " + str(num_subgraph) + " subgraphs, batch size: " + str(num_batch))
    partitions = graph.partition_graph(num_subgraph)

    data_list = []

    for i, (cluster, subgraph_nodes) in enumerate(partitions.items()):
        print(f"Number of nodes in cluster {cluster}: {len(subgraph_nodes)}")
        train_data = createdata_graphpartition(graph, 
                                                subgraph_nodes, 
                                                equivariant_blocks, 
                                                out_slices, 
                                                construct_kernel, 
                                                dtype=dtype)
        data_list.append(train_data)
        if len(data_list) == num_batch:
            break

    dataset = CustomDataset(data_list)
    
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, 
                            sampler=sampler, 
                            batch_size=1, 
                            shuffle=False, 
                            collate_fn=custom_collate_fn)
    else:
        loader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            collate_fn=custom_collate_fn)

    print("*** Batch properties:")
    for batch in loader:
        print("--> Batch: ")
        print("Node Features (x):", batch.x.size())
        print("Edge Index:", batch.edge_index.size())
        print("Edge Features (edge_attr):", batch.edge_attr.size())    

    return loader