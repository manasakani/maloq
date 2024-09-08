import argparse
print("Importing numpy...", flush=True)
import numpy as np
print("Importing torch...", flush=True)
import torch #, torch_scatter
print("Importing e3nn.o3 Irreps...", flush=True)
from e3nn.o3 import Irreps
print("Importing pickle...", flush=True)
import pickle
print("Importing random...", flush=True)
import random
print("Importing os...", flush=True)
import os
print("Importing dgl...", flush=True)
import dgl
# import dgl.distributed as dist_dgl

# from dgl.dataloading import enable_cpu_affinity 
# ^^^ only works with newer versions of DGL

print("Importing lib.training...", flush=True)
import lib.training as training
print("Importing lib.structure...", flush=True)
import lib.structure as structure
print("Importing lib_equiformer.SO2...", flush=True)
import lib_equiformer.SO2 as SO2
print("Importing lib_equiformer.so2_model...", flush=True)
import lib.so2_model as so2_model
print("Importing DGLGraphDataset...", flush=True)
from lib.structure_distributed import DGLGraphDataset
print("Importing lib_equiformer.SO3...", flush=True)
import lib_equiformer.SO3 as SO3

print("Importing torch.distributed...", flush=True)
import torch.distributed as dist

# reference for distribution with DGL: 
# https://docs.dgl.ai/en/1.1.x/tutorials/dist/1_node_classification.html#sphx-glr-tutorials-dist-1-node-classification-py

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in state_dict for distributed restart."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# # Function to load node and edge features for a partition
# def load_graph_partitions(graph_dir, part_id):

#     graph_path = f"{graph_dir}/part{part_id}/graph.dgl"
#     node_feats_path = f"{graph_dir}/part{part_id}/node_feat.dgl"
#     edge_feats_path = f"{graph_dir}/part{part_id}/edge_feat.dgl"

#     # Load partitioned graph (this loads the subgraph for this partition)
#     graphs, _ = dgl.load_graphs(graph_path)
#     g = graphs[0]  

#     num_nodes_in_partition = g.num_nodes()
#     num_edges_in_partition = g.number_of_edges()
#     # print(f"Number of nodes in partition {part_id}: {num_nodes_in_partition}")
#     # print(f"Number of edges in partition {part_id}: {num_edges_in_partition}")

#     # Get the local node mask (real nodes inside the partition)
#     local_nodes_mask = g.ndata['inner_node'].bool()
#     num_local_nodes = local_nodes_mask.sum().item()

#     local_edges_mask = g.edata['inner_edge'].bool()
#     num_local_edges = local_edges_mask.sum().item()

#     # print("local_nodes_mask: ", local_nodes_mask)
#     # print("local_edges_mask: ", local_edges_mask)
#     # print("num_nodes_per_rank: ", num_local_nodes)
#     # print("num_edges_per_rank: ", num_local_edges)

#     # Add the node features/labels to the graph's nodes
#     node_feats = dgl.data.load_tensors(node_feats_path)

#     for key, feat in node_feats.items():
#         # print(f"Processing node feature '{key}' with shape {feat.shape}")
#         if len(feat.shape) == 1:
#             # node feature are scalars
#             g.ndata[key] = torch.zeros(num_nodes_in_partition, dtype=feat.dtype)
#             g.ndata[key][local_nodes_mask] = feat[:num_local_nodes]
#         else:
#             # node labels are multi-dimensional
#             g.ndata[key] = torch.zeros((num_nodes_in_partition,) + feat.shape[1:], dtype=feat.dtype)
#             g.ndata[key][local_nodes_mask] = feat[:num_local_nodes]

#     # Add the edge features/labels to the graph's edges
#     edge_feats = dgl.data.load_tensors(edge_feats_path)
#     for key, feat in edge_feats.items():
#         # print(f"Processing edge feature '{key}' with shape {feat.shape}")
#         if len(feat.shape) == 1:
#             g.edata[key] = torch.zeros(num_edges_in_partition, dtype=feat.dtype)
#             g.edata[key][local_edges_mask] = feat[:num_local_edges]
#         else:
#             # Edge features and labels are multi-dimensional
#             g.edata[key] = torch.zeros((num_edges_in_partition,) + feat.shape[1:], dtype=feat.dtype)
#             g.edata[key][local_edges_mask] = feat[:num_local_edges]

#     return g
    
# Function to load node and edge features for a partition
def load_graph_partitions(graph_dir, part_id, num_partitions):
    """
    Does two things:
        (1) Load the partitioned graph for the current worker 
        (2) Adds our custom node and edge features back into the graph
                Node features: atomic numbers, Node labels: diagonal Hamiltonian blocks
                Edge features: [bond distances, bond vector], Edge labels: off-diagonal Hamiltonian blocks

        Each partition has some local nodes and edges, and some halo nodes and edges.
        First we add the core node and edge features and labels to the graph (corresponding to the local nodes and edges), which
        are stored in the node_feat.dgl and edge_feat.dgl files of the local partitition.
        Then we add the halo node and edge features and labels to the graph (corresponding to the halo nodes and edges), which 
        are stored in the node_feat.dgl and edge_feat.dgl files of the other partitions, so we iterate through the other partitions' folders, 
        figure out if they have any halo nodes of the current parition, and then populate the current's partition's halo nodes/edges
        with the features from the other partition.
    """

    graph_path = f"{graph_dir}/part{part_id}/graph.dgl"
    node_feats_path = f"{graph_dir}/part{part_id}/node_feat.dgl"
    edge_feats_path = f"{graph_dir}/part{part_id}/edge_feat.dgl"

    graphs, _ = dgl.load_graphs(graph_path)                                         # Load the partitioned graph (this loads the subgraph for this partition)  
    g = graphs[0]  

    num_nodes_in_partition = g.num_nodes()                                          # total number of nodes in the partition (local and halo nodes)
    num_edges_in_partition = g.number_of_edges()                                    # total number of edges in the partition (local and halo edges)
    local_nodes_mask = g.ndata['inner_node'].bool()                                 # T/F mask of local nodes = inner nodes, or the nodes that belong to this partition
    local_edges_mask = g.edata['inner_edge'].bool()                                 # T/F mask of local edges = inner edges, or the edges that belong to this partition
    num_local_nodes = local_nodes_mask.sum().item()                                 # number of local nodes in the partition
    num_local_edges = local_edges_mask.sum().item()                                 # number of local edges in the partition

    # print("local_nodes_mask: ", local_nodes_mask)
    # print("local_edges_mask: ", local_edges_mask)
    # print("num_nodes_per_rank: ", num_local_nodes)
    # print("num_edges_per_rank: ", num_local_edges)

    # --> 1/4: Add core node features and labels 
    node_feats = dgl.data.load_tensors(node_feats_path)
    for key, feat in node_feats.items():
        print(f"Processing node feature '{key}' with shape {feat.shape}")
        if len(feat.shape) == 1:                                                    # case: node feature are scalars
            g.ndata[key] = torch.zeros(num_nodes_in_partition, dtype=feat.dtype)    # Initialize node feature tensor with zeros
            g.ndata[key][local_nodes_mask] = feat[:num_local_nodes]                 # Add local node features to the tensor
        else:                                                                       # case: node labels are multi-dimensional
            g.ndata[key] = torch.zeros((num_nodes_in_partition,) + feat.shape[1:], dtype=feat.dtype)
            g.ndata[key][local_nodes_mask] = feat[:num_local_nodes]
        
    
    # --> 2/4: Add halo node features and labels
    halo_nodes_mask = ~local_nodes_mask                                             # T/F mask of halo nodes = outer nodes, or the nodes that do not belong to this partition
    halo_node_ids = g.ndata.get(dgl.NID, g.ndata.get('_ID'))[halo_nodes_mask]  
    # print(f"Rank {part_id} - Halo node IDs: ", halo_node_ids)

    # Iterate through other partitions to add halo node features
    for other_part_id in range(num_partitions):

        # skip the current partition
        if other_part_id == part_id:
            continue  

        # Load other partition's graph and node features
        other_node_feats_path = f"{graph_dir}/part{other_part_id}/node_feat.dgl"
        other_graph_path = f"{graph_dir}/part{other_part_id}/graph.dgl"
        other_graphs, _ = dgl.load_graphs(other_graph_path)
        other_g = other_graphs[0]
        other_node_feats = dgl.data.load_tensors(other_node_feats_path)                         # Load node features from the other partition
        num_other_local_nodes = other_g.ndata['inner_node'].sum().item()                        # Number of local nodes in the other partition          

        # Get other partition's global node IDs (use the same key check as before)
        other_global_node_ids = other_g.ndata.get(dgl.NID, other_g.ndata.get('_ID'))
        other_global_node_ids = other_global_node_ids[:num_other_local_nodes]                   # Only consider local nodes in the other partition, since they are the ones which have features

        # print(f"Rank {part_id} - other_global_node_ids from part {other_part_id}: ", other_global_node_ids)

        # Identify which halo nodes belong to this other partition by comparing global IDs
        halo_in_other_part_mask = halo_node_ids[:, None] == other_global_node_ids               # Broadcasted comparison
        halo_in_other_part = halo_in_other_part_mask.any(dim=1)
        # print(f"Rank {part_id} - halo_in_other_part mask for part {other_part_id}: ", halo_in_other_part)

        if halo_in_other_part.any():
            matching_halo_indices_in_other_part = halo_in_other_part_mask.nonzero(as_tuple=False)[:, 1]
            # print(f"Rank {part_id} - matching_halo_indices_in_other_part: ", matching_halo_indices_in_other_part)

            # For each feature key, update halo node features using the features from the other partition
            for key, feat in other_node_feats.items():
                halo_feat_mask = halo_nodes_mask.nonzero(as_tuple=True)[0][halo_in_other_part]
                # print(f"Rank {part_id} - halo_feat_mask for part {other_part_id}: ", halo_feat_mask)
                # print(f"Rank {part_id} - Processing node feature '{key}' with shape {feat.shape} and entries: ", feat)

                # # scalar node features
                # if len(feat.shape) == 1 and feat.shape[0] == 1:
                #     g.ndata[key][halo_feat_mask] = feat[0]
                # else:
                #     # g.ndata[key][halo_feat_mask] = feat[matching_halo_indices_in_other_part]
                #     g.ndata[key][halo_nodes_mask][halo_in_other_part] = feat[matching_halo_indices_in_other_part].clone().detach()
                #     print(f"Rank {part_id} - adding node feature '{key}' with shape {feat.shape} and entries: ", feat[matching_halo_indices_in_other_part])
                #     print(f"Rank {part_id} - after add node g.ndata[key][halo_nodes_mask][halo_in_other_part]: ", g.ndata[key][halo_nodes_mask][halo_in_other_part])
                updated_node_feats = g.ndata[key].clone().detach()
                updated_node_feats[halo_nodes_mask] = torch.index_select(feat, 0, matching_halo_indices_in_other_part).clone().detach()
                g.ndata[key] = updated_node_feats

                # print(f"Rank {part_id} - before add node g.ndata[key][halo_nodes_mask][halo_in_other_part]: ", g.ndata[key][halo_nodes_mask][halo_in_other_part])
                # print(f"Rank {part_id} - feat[matching_halo_indices_in_other_part]: ", feat[matching_halo_indices_in_other_part])
                # print(f"Rank {part_id} - after add node g.ndata[key][halo_nodes_mask][halo_in_other_part]: ", g.ndata[key][halo_nodes_mask][halo_in_other_part])



    # --> 3/4: Add core edge features and labels
    edge_feats = dgl.data.load_tensors(edge_feats_path)
    for key, feat in edge_feats.items():
        print(f"Processing edge feature '{key}' with shape {feat.shape}")
        if len(feat.shape) == 1:
            g.edata[key] = torch.zeros(num_edges_in_partition, dtype=feat.dtype)
            g.edata[key][local_edges_mask] = feat[:num_local_edges]
        else:
            # Edge features and labels are multi-dimensional
            g.edata[key] = torch.zeros((num_edges_in_partition,) + feat.shape[1:], dtype=feat.dtype)
            g.edata[key][local_edges_mask] = feat[:num_local_edges]


    # --> 4/4: Add halo edge features and labels
    halo_edges_mask = ~local_edges_mask
    local_edge_ids = g.edata.get(dgl.EID, g.edata.get('_ID'))[local_edges_mask]
    halo_edge_ids = g.edata.get(dgl.EID, g.edata.get('_ID'))[halo_edges_mask]
    # print(f"Rank {part_id} - Halo edge IDs: ", halo_edge_ids)

    for other_part_id in range(num_partitions):
            
            # skip the current partition
            if other_part_id == part_id:
                continue
    
            # Load other partition's graph and edge features
            other_edge_feats_path = f"{graph_dir}/part{other_part_id}/edge_feat.dgl"
            other_graph_path = f"{graph_dir}/part{other_part_id}/graph.dgl"
            other_graphs, _ = dgl.load_graphs(other_graph_path)
            other_g = other_graphs[0]
            other_edge_feats = dgl.data.load_tensors(other_edge_feats_path)                         # Load edge features from the other partition
            num_other_local_edges = other_g.edata['inner_edge'].sum().item()                        # Number of local edges in the other partition          
    
            # Get other partition's global edge IDs (use the same key check as before)
            other_global_edge_ids = other_g.edata.get(dgl.EID, other_g.edata.get('_ID'))
            other_global_edge_ids = other_global_edge_ids[:num_other_local_edges]                   # Only consider local edges in the other partition, since they are the ones which have features
    
            # print(f"Rank {part_id} - other_global_edge_ids from part {other_part_id}: ", other_global_edge_ids)
    
            # Identify which halo edges belong to this other partition by comparing global IDs
            halo_in_other_part_mask = halo_edge_ids[:, None] == other_global_edge_ids
            halo_in_other_part = halo_in_other_part_mask.any(dim=1)

            if halo_in_other_part.any():
                matching_halo_indices_in_other_part = halo_in_other_part_mask.nonzero(as_tuple=False)[:, 1]
                # print(f"Rank {part_id} - edge matching_halo_indices_in_other_part: ", matching_halo_indices_in_other_part)
    
                # For each feature key, update halo edge features using the features from the other partition
                for key, feat in other_edge_feats.items():
                    halo_feat_mask = halo_edges_mask.nonzero(as_tuple=True)[0][halo_in_other_part]
                    # print(f"Rank {part_id} - edge halo_feat_mask for part {other_part_id}: ", halo_feat_mask)
    
                    g.edata[key][halo_feat_mask] = feat[matching_halo_indices_in_other_part]
                    g.edata[key][halo_edges_mask][halo_in_other_part] = feat[matching_halo_indices_in_other_part].clone().detach()
                    # print(f"Rank {part_id} - after add edge g.edata[key][halo_edges_mask][halo_in_other_part]: ", g.edata[key][halo_edges_mask][halo_in_other_part])

    print(f"Rank {part_id} - Final node features: ", g.ndata)
    print(f"Rank {part_id} - Final edge features: ", g.edata)

    return g

def main(folder, ip_config):

    if not torch.cuda.is_available():
        raise RuntimeError("No GPUs are available!")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Distributed training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'SLURM_PROCID' in os.environ:  
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        backend = 'gloo'  # Use NCCL to enable rdma

        print("Initializing process group...")
        dgl.distributed.initialize(ip_config=ip_config)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
        print("Process group initialized")
        print("main: torch.cuda.device_count(): ", torch.cuda.device_count())
        print("main: torch.cuda.current_device(): ", torch.cuda.current_device())

    else:  
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        backend = 'gloo'  # Use Gloo for attelas (single GPU)

    if dist.is_initialized() and dist.get_rank() == 0:  
        print("Torch version: ", torch.__version__, flush=True)
        print("DGL version: ", dgl.__version__, flush=True)
        print("DGL backend (cuda?): ", dgl.backend.device_type('cuda'), flush=True)
        print("CUDA Available: ", torch.cuda.is_available())
        print("CUDA_VISIBLE_DEVICES: ", os.environ.get('CUDA_VISIBLE_DEVICES'))
        print("CUDA Device Count: ", torch.cuda.device_count())
        print("CUDA Device Name: ", torch.cuda.get_device_name(0))
        print(f"RANK: {rank}, WORLD_SIZE: {world_size}, LOCAL_RANK: {local_rank}", flush=True)

    # ************************************************************
    # Input parameters and for the HfO2 dataset
    # ************************************************************

    data_folder = os.path.join(folder, 'datasets/H2O/H2O_DZVP_1')
    xyz_file = os.path.join(data_folder, 'snapshot.xyz')
    hamiltonian_file = os.path.join(data_folder, 'H.csr')
    overlap_file = os.path.join(data_folder, 'S.csr')
    DGL_pickle_file_path = None                                 # Path to the DGLGraphDataset pickle file, used for save/load if exists

    # Material parameters:
    pbc = False
    orbital_basis = 'DZVP'
    rcut = 100.0       
    lmax_list = [4]                 
    mmax_list = [lmax_list[0]]

    # Parameters:
    restart_file = None
    save_file = 'model_H2O_'+str(world_size)+'_DGL.pth'  
    train_or_test = 'train'                                          
    num_MP_layers = 1                                                               # Number of message passing layers 
    num_epochs = 100                                                
    learning_rate = 1e-3
    loss_tol = 0                                                    
    dtype = torch.float32

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 16
    num_heads = 2
    attn_hidden_channels = 64  # increase to 128
    attn_alpha_channels = 32
    attn_value_channels = 32
    ffn_hidden_channels = 64

    # Define irreducible representations for the SO2 model
    irreps_in = Irreps([(sphere_channels, (0, 1)), (sphere_channels, (1, 1)), (sphere_channels, (2, 1)), (sphere_channels, (3, 1)), (sphere_channels, (4, 1))])
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]  

    # ************************************************************
    # Create the dataset
    # ************************************************************

    # *** Initialize the domain and electronic structure matrices:
    H2O = structure.Structure(xyz_file, 
                            hamiltonian_file, 
                            overlap_file, 
                            pbc, 
                            orbital_basis, 
                            make_soap=False, 
                            save_matrices=False,
                            self_interaction=False,
                            bothways=True, 
                            rcut=rcut)

    # *** Perform orbital analysis:
    atom_orbitals = {'1': [0, 0, 1],'8':[0, 0, 1, 1, 2]}                                  # Orbital types of each atom in the structure
    numbers = H2O.atomic_numbers                                                          # Atomic numbers of each atom in the structure
    no_parity = True                                                                      # No parity symmetry          
    orbital_types = [[0,0,1],[0, 0, 1, 1, 2]]                                             # orbital types of each atom in the structure 

    targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    index_to_Z, inverse_indices = torch.unique(numbers, sorted=True, return_inverse=True)
    equivariant_blocks, out_js_list, out_slices = SO2.process_targets(orbital_types, index_to_Z, targets)
    
    print("Orbital analysis completed", flush=True)                                       # equivariant_blocks: start and end indices of the equivariant blocks in i and j direction for each target in targets
                                                                                          # out_js_list: ll the l1 l2 interactions needed 
                                                                                          # out_slices: marks the start and end of indices belonging to a certain target. Slice 1 (0 to 1) corresponds to the first target in equivariant blocks 

    construct_kernel = SO2.e3TensorDecomp(net_out_irreps, 
                                          out_js_list, 
                                          default_dtype_torch= torch.float32, 
                                          spinful=False,
                                          no_parity=no_parity, 
                                          if_sort=False, 
                                          device_torch='cpu')                             # the data is created on cpu, so the construct_kernel must be on cpu 

    # *** Create/Load the DGLGraphDataset:
    if DGL_pickle_file_path is not None:
        print("Unpickling dataset...", flush=True)
        with open(DGL_pickle_file_path, 'rb') as f:
            H2O_DGL = pickle.load(f)
    else:
        print("Creating DGLGraphDataset (this takes a while)...", flush=True)
        H2O_DGL = DGLGraphDataset(H2O, 
                                    equivariant_blocks, 
                                    out_slices, 
                                    construct_kernel, 
                                    device=device, 
                                    dtype=dtype)
        with open('dgl_graph_dataset.pkl', 'wb') as f:
            pickle.dump(H2O_DGL, f)
    
    print("DGLGraphDataset created", flush=True)
    graph = H2O_DGL[0]
    num_nodes = H2O.atomic_numbers.shape[0]

    # print node labels:
    # print("Node labels (graph): ", graph.ndata['node_label'])
    # print("Edge labels (graph): ", graph.edata['label'])

    # ************************************************************
    # Partition the graph and create the DataLoader for DGL
    # ************************************************************
    rank = dist.get_rank()  
    
    # make sure all workers have loaded the data before starting training
    dist.barrier()

    graph_name = 'H2O'                                  # Name for the graph (just used for saving/loading)
    part_dir = './graph_partitions'                     # Directory to store the partitions

    # 1. Partition the graph, and save the partitioned graph to disk
    dgl.distributed.partition_graph(graph, 
                                    graph_name, 
                                    num_parts=world_size, 
                                    out_path=part_dir, 
                                    num_hops=num_MP_layers, 
                                    part_method='metis')
    print("Graph partitioned and saved to: ", part_dir)

    # make sure all workers have loaded the data before starting training
    dist.barrier()
    print("Barrier 2 for rank ", rank, " passed")

    # 2. Load the partitioned graph for the current worker
    # graph_partition, node_feat, edge_feat, gpb, graph_name, node_types, edge_types  = dgl.distributed.load_partition(part_dir + '/' + graph_name + '.json', rank)
    graph_partition = load_graph_partitions(part_dir, rank, world_size)
    print(f"Loaded partition for rank {rank} with {graph_partition.num_nodes()} nodes")
    print("graph_partition.ndata.keys(): ", graph_partition.ndata.keys())
    print("graph_partition.edata.keys(): ", graph_partition.edata.keys())

    # # Add node features and labels back into the graph_partition of the current worker
    # for key, feat in node_feat.items():
    #     graph_partition.ndata[key] = feat
    # # Add edge features and labels back into the graph_partition of the current worker
    # for key, feat in edge_feat.items():
    #     graph_partition.edata[key] = feat

    # Verify features have been added
    print("graph_partition.ndata.keys(): ", graph_partition.ndata.keys())
    print("graph_partition.edata.keys(): ", graph_partition.edata.keys())

    # for key in graph_partition.ndata.keys():
    #     print("graph_partition.ndata[{}]: ".format(key), graph_partition.ndata[key])
    # for key in graph_partition.edata.keys():
    #     print("graph_partition.edata[{}]: ".format(key), graph_partition.edata[key])

    # 3. Sample the graph (using MultiLayerFullNeighborSampler for full-batch training with no sampling)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_MP_layers)

    # make sure all workers have loaded the data before starting training
    dist.barrier()
    print("Barrier 3 for rank ", rank, " passed")

    # 4. Create the DataLoader with the sampler (enable_cpu_affinity() only works with newer versions of DGL)
    train_nids = graph_partition.nodes()
    num_nodes_in_partition = len(train_nids)
    data_loader = dgl.dataloading.DistNodeDataLoader(
        graph_partition,                                # Partitioned graph loaded for the current worker
        train_nids,                                     # Training node IDs
        sampler,                                        # MultiLayerFullNeighborSampler
        batch_size=num_nodes_in_partition,              # this is the number of nodes in each batch, since we are doing full-batch training
        shuffle=False,
        drop_last=False
    )
    print(f"Distributed data loader created with {len(train_nids)} nodes.", flush=True)

    # make sure all workers have loaded the data before starting training
    dist.barrier()
    print("Barrier 4 for rank ", rank, " passed")


    for input_nodes, output_nodes, blocks in data_loader:
        print("Subgraph Node Types in dataloder:", blocks[0].ntypes)
        print("Subgraph Edge Types in dataloder:", blocks[0].etypes)
    
    # make sure all workers have loaded the data before starting training
    dist.barrier()
    
    # ************************************************************
    # Initialize the SO2 model
    # ************************************************************

    mappingReduced = SO3.CoefficientMappingModule(lmax_list, mmax_list)
    irreps_out = net_out_irreps
    model = so2_model.SO2Net(num_MP_layers, 
                            lmax_list, 
                            mmax_list, 
                            mappingReduced, 
                            sphere_channels, 
                            edge_channels_list, 
                            attn_hidden_channels, 
                            num_heads, 
                            attn_alpha_channels, 
                            attn_value_channels, 
                            ffn_hidden_channels, 
                            irreps_in, 
                            irreps_out)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if restart_file is not None:
        print("Restarting training from a saved model and optimizer state...")
        checkpoint = torch.load(restart_file)
        state_dict = checkpoint['model_state_dict']

        if dist.is_available() and dist.is_initialized():
            # If the model was saved with DDP, remove the 'module' prefix that it might have (just in case)
            if 'module.' in next(iter(checkpoint['model_state_dict'].keys())):
                prefix = 'module.'
                state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            # with the current training setup, the module prefix is already removed
            model.load_state_dict(state_dict)
        else:
            state_dict = remove_module_prefix(checkpoint['model_state_dict'])
            model.load_state_dict(state_dict)

    print("Model initialized")
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    # ************************************************************
    # Train the model and evaluate based on training/testing mode
    # ************************************************************

    if train_or_test == 'train':
        
        print("training...")
        training.train_model_DGL_full(model, optimizer, data_loader, num_nodes, num_epochs, loss_tol, save_file=save_file, dtype=dtype)
        print("Model trained")

    else:
        print("evaluating in test mode...")
    
    training.evaluate_model_DGL(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amorphous GNNs --- HfO2")
    parser.add_argument("-f", "--folder", default="", required=False)
    parser.add_argument("--ip_config", type=str, required=True, help="Path to IP configuration file for distributed training.")
    args = parser.parse_args()

    print(f"Starting main ... dataset folder is '{args.folder}'", flush=True)

    main(args.folder, args.ip_config)