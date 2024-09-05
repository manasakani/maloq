import lib.data as data
import lib.training as training
import lib.structure as structure
import lib_equiformer.SO2 as SO2
import lib.so2_model as so2_model
from lib.structure_distributed import DGLGraphDataset
import lib_equiformer.SO3 as SO3
from e3nn.o3 import Irreps
import matplotlib.pyplot as plt
import numpy as np
import torch, torch_scatter
import pickle
import random
import os
import dgl
import torch.multiprocessing as mp

import torch.distributed as dist
import torch.multiprocessing as mp

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in state_dict for distributed restart."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():

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
        backend = 'gloo'  # Use NCCL for multi-GPU on Piz Daint (edit: uses RDMA, switching to gloo)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    else:  
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        backend = 'gloo'  # Use Gloo for attelas (single GPU)

    if dist.is_initialized() and dist.get_rank() == 0:  
        print("Torch version: ", torch.__version__)
        print("DGL version: ", dgl.__version__)
        print("DGL backend (cuda?): ", dgl.backend.device_type('cuda'))
        print("Torch scatter version: ", torch_scatter.__version__)
        print("CUDA Available: ", torch.cuda.is_available())
        print("CUDA Device Count: ", torch.cuda.device_count())
        print("CUDA Device Name: ", torch.cuda.get_device_name(0))
        print(f"RANK: {rank}, WORLD_SIZE: {world_size}, LOCAL_RANK: {local_rank}")

    # ************************************************************
    # Input parameters and for the HfO2 dataset
    # ************************************************************

    data_folder = './datasets/a-HfO2/'
    xyz_file = data_folder + 'structure.xyz'
    hamiltonian_file = data_folder + 'H.csr'
    overlap_file = data_folder + 'S.csr'
    DGL_pickle_file_path = 'dgl_graph_dataset_4rcut.pkl'                                 # Path to the DGLGraphDataset pickle file, used for save/load if exists

    # Material parameters:
    pbc = True
    orbital_basis = 'SZV'
    rcut = 4.0       
    lmax_list = [4]     
    mmax_list = [lmax_list[0]]

    # Parameters:
    restart_file = 'model_HfO2_4_DGL.pth'
    save_file = 'model_HfO2_'+str(world_size)+'_DGL.pth'  
    train_or_test = 'train'                                          
    num_MP_layers = 2                                                               # Number of message passing layers 
    num_epochs = 500                                                
    learning_rate = 1e-4
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
    a_HfO2 = structure.Structure(xyz_file, 
                                    hamiltonian_file, 
                                    overlap_file, 
                                    pbc, 
                                    orbital_basis, 
                                    make_soap=False, 
                                    save_matrices=True,
                                    self_interaction=False,
                                    bothways=True, 
                                    rcut = rcut)

     # *** Perform orbital analysis:
    atom_orbitals = {'8':[0,1], '72':[0,0,1,2]}                                           # Orbital types of each atom in the structure
    numbers = a_HfO2.atomic_numbers                                                       # Atomic numbers of each atom in the structure
    no_parity = True                                                                      # No parity symmetry          
    orbital_types = [[0,1],[0,0,1,2]]                                                     # basis rank of each atom in the structure 

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
        print("Loading dataset from pickle file...", flush=True)
        with open(DGL_pickle_file_path, 'rb') as f:
            a_HfO2_DGL = pickle.load(f)
    else:
        print("Creating DGLGraphDataset...", flush=True)
        a_HfO2_DGL = DGLGraphDataset([a_HfO2], 
                                    equivariant_blocks, 
                                    out_slices, 
                                    construct_kernel, 
                                    device=device, 
                                    dtype=dtype)
        with open('dgl_graph_dataset.pkl', 'wb') as f:
            pickle.dump(a_HfO2_DGL, f)
    
    print("DGLGraphDataset created", flush=True)
    graph = a_HfO2_DGL[0]

    num_nodes = a_HfO2.atomic_numbers.shape[0]
    train_nids = torch.arange(num_nodes)

    # --> Using MultiLayerFullNeighborSampler
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_MP_layers)
    batch_size = 1 #int(3000/world_size)               # each gpu takes 1 batch

    # load the dataloader from a pickle file:
    # with open('dataloader.pkl', 'rb') as f:
    #     data_loader = pickle.load(f)
    # print("Data loader loaded from pickle", flush=True)

    # Create the DataLoader with the sampler
    data_loader = dgl.dataloading.DataLoader(
        graph,
        train_nids,
        sampler,
        batch_size=batch_size,  # i think this is the number of nodes in each batch
        shuffle=False,
        drop_last=False,
        num_workers=0,  # try setting this to 0 to disable multi-worker loading
    )
    print("Data loader created", flush=True)
    
    # dump the dataloader to a pickle file
    # with open('dataloader.pkl', 'wb') as f:
    #     pickle.dump(data_loader, f)
    # print("Data loader for rcut of 5 dumped to pickle file", flush=True)

    # --> Using ClusterGCNSampler (neighbour sampling) instead of MultiLayerFullNeighborSampler
    # num_partitions = 30
    # batch_size = 5
    # sampler = dgl.dataloading.ClusterGCNSampler( 
    #     graph,
    #     num_partitions
    # )

    # # DataLoader for generic dataloading with a graph, a set of indices (can use any indices, like
    # # partition IDs here), and a graph sampler.
    # indices = torch.arange(num_partitions)
    # data_loader = dgl.dataloading.DataLoader(
    #     graph,
    #     indices,
    #     sampler,
    #     device="cuda",
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=world_size,         # test num_workers set to the number of GPUs
    #     use_uva=False,                  # This uses unified virtual address space for shared memory access, check if its okay to use
    # )
    # print("DataLoader created", flush=True)

    if dist.is_initialized():
        dist.barrier()
    # print("Structure created")
    
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

    if train_or_test == 'train':
        
        print("training...")
        training.train_model_DGL_full(model, optimizer, data_loader, num_epochs, loss_tol, save_file=save_file, dtype=dtype)
        print("Model trained")

    else:
        print("evaluating in test mode...")

    training.evaluate_model_DGL(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)

if __name__ == "__main__":
    main()