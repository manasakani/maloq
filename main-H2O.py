import argparse
import lib.data as data
import lib.training as training
import lib.structure as structure
import lib_equiformer.SO2 as SO2
import lib.so2_model as so2_model
import lib_equiformer.SO3 as SO3
from e3nn.o3 import Irreps
import numpy as np
import torch
import os
import random
from lib import so2_model

import torch.distributed as dist
# import torch.multiprocessing as mp

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in state_dict."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main(folder):

    if not torch.cuda.is_available():
        raise RuntimeError("No GPUs are available!")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ************************************************************
    # Distributed training setup (if running on multiple GPUs)
    # ************************************************************

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, flush=True)

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
        print(f"RANK: {rank}", flush=True)
        print(f"WORLD_SIZE: {world_size}", flush=True)
        print(f"LOCAL_RANK: {local_rank}", flush=True)

    # ************************************************************
    # Input parameters and for the HfO2 dataset
    # ************************************************************

    data_folder = os.path.join(folder, 'datasets/H2O/H2O_DZVP_1')
    xyz_file = os.path.join(data_folder, 'snapshot.xyz')
    hamiltonian_file = os.path.join(data_folder, 'H.csr')
    overlap_file = os.path.join(data_folder, 'S.csr')

    # Material parameters:
    pbc = False
    orbital_basis = 'DZVP'
    rcut = 4.0          
    lmax_list = [4]     
    mmax_list = [4] # changed to 4 directly

    # Graph partitioning methods (the first three are for the slice option, the last two are for the graph partitioning option):
    partitioning = 'graph'                                                          # 'slice' or 'graph' partitioning
    
    # slice parameters:
    slice_list = [0]                                                             # slice boundaries for partitioning the structure into subgraphs                
    cutoff = 300000                                                                     # cutoff boundary of the slice used for training (interaction radius = 2*cutoff)
    
    # graph partitioning parameters:
    num_subgraph = 1                                                                # min 10 for P100 GPU memory with attn_hidden_channels=64
    num_batch = 1                                                                   # number of subgraphs which will actually be added to the dataset for training,
                                                                                    # after dividing the graph into 'num_subgraph' subgraphs
    # Parameters:
    restart_file = None
    save_file = 'model_HfO2_'+str(world_size)+'_subgraph.pth'  
    train_or_test = 'train'                                          
    num_MP_layers = 2                                                               # Number of message passing layers 
    num_epochs = 500                                                
    learning_rate = 1e-3                                                            # Initial Learning rate                 
    loss_tol = 1e-7                                                                    # Loss tolerance for early stopping
    dtype = torch.float32

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 16
    num_heads = 2
    attn_hidden_channels = 128  
    attn_alpha_channels = 32
    attn_value_channels = 32
    ffn_hidden_channels = 64

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
                                    rcut = rcut)
    if dist.is_initialized():
        dist.barrier()
    print("Structure created", flush=True)
    
    # ************************************************************
    # Initialize the SO2 model
    # ************************************************************

    # *** Define irreducible representations
    irreps_in = Irreps([(sphere_channels, (0, 1)), (sphere_channels, (1, 1)), (sphere_channels, (2, 1)), (sphere_channels, (3, 1)), (sphere_channels, (4, 1))])
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]  

    # *** Perform orbital analysis:
    atom_orbitals = {'1': [0, 0, 1], '8':[0, 0, 1, 1, 2]}                                  # Orbital types of each atom in the structure
    numbers = H2O.atomic_numbers                                                           # Atomic numbers of each atom in the structure    
    no_parity = True                                                                       # No parity symmetry          
    orbital_types = [[0,0,1], [0, 0, 1, 1, 2]]                                             # basis rank of each atom in the structure 

    targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    index_to_Z, inverse_indices = torch.unique(numbers, sorted=True, return_inverse=True)    
    equivariant_blocks, out_js_list, out_slices = SO2.process_targets(orbital_types, index_to_Z, targets)
    # equivariant_blocks: start and end indices of the equivariant blocks in i and j direction for each target in targets
    # out_js_list: ll the l1 l2 interactions needed 
    # out_slices: marks the start and end of indices belonging to a certain target. Slice 1 (0 to 1) corresponds to the first target in equivariant blocks 

    # *** Construct the kernel used to transform the orbital blocks
    construct_kernel = SO2.e3TensorDecomp(net_out_irreps, 
                                          out_js_list, 
                                          default_dtype_torch=torch.float32, 
                                          spinful=False,
                                          no_parity=no_parity, 
                                          if_sort=False, 
                                          device_torch='cpu') #the data is created on cpu, so the construct_kernel must be on cpu 
    print("Orbital analysis completed", flush=True)

    # *** Create the input dataloader:
    if partitioning == 'slice':
        data_loader = data.batch_data_subgraph(H2O, slice_list, cutoff, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
        print("Data loader created - using " + str(len(slice_list)) + " slices", flush=True)

    else:
        data_loader = data.batch_data_graphpartition(H2O, num_subgraph, num_batch, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
        print("Data loader created - using " + str(num_subgraph) + " subgraphs", flush=True)
        
    if dist.is_initialized():
        dist.barrier()

    # *** Initialize the model:
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
        print("Restarting training from a saved model and optimizer state...", flush=True)
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

    print("Model initialized", flush=True)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()), flush=True)

    print("memory: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB", flush=True)
    if dist.is_initialized():
        dist.barrier()

    # ************************************************************
    # Training and testing the model
    # ************************************************************

    if train_or_test == 'train':
        
        # *** Train the model parameters:
        print("training...", flush=True)
        print("using train_model_subgraph without labelling", flush=True)
        training.train_model_subgraph(model, optimizer, data_loader, num_epochs, loss_tol, save_file=save_file, dtype=dtype)
        # training.train_and_validate_model_SO2(model, optimizer, data_loader, data_loader, num_epochs, loss_tol, save_file=save_file, dtype=dtype)

        print("Model trained, plotting fit to training data", flush=True)
        training.evaluate_model(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)

    # use with a restarted model, to test the model
    elif train_or_test == 'test':
        print("testing on unseen data...", flush=True)
        training.evaluate_model(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amorphous GNNs --- H2O")
    parser.add_argument("-f", "--folder", default="", required=False)
    args = parser.parse_args()

    print(f"Starting main ... dataset folder is '{args.folder}'", flush=True)

    main(args.folder)
