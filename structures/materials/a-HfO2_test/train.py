import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
lib_root = os.path.join(project_root, 'lib')
lib_equiformer_root = os.path.join(project_root, 'lib_equiformer')
sys.path.append(lib_root)
sys.path.append(lib_equiformer_root)
print(f"Added {lib_root} to the path", flush=True)
print(f"Added {lib_equiformer_root} to the path", flush=True)

import argparse
import numpy as np
import torch.distributed as dist
import torch
import random

import data as data
import training as training
import structure as structure
import SO2 as SO2
import network as network
import SO3 as SO3
import compute_env as env
import utils as utils
from e3nn.o3 import Irreps
print("Imported libraries", flush=True)

def main(folder):

    if not torch.cuda.is_available():
        raise RuntimeError("No GPUs are available!")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    print(f"Folder: {folder}", flush=True)

    # ************************************************************
    # Distributed training setup (if running on multiple GPUs)
    # ************************************************************

    device, world_size = env.initialize_compute_env()
    print("Device: ", device, ", World size: ", world_size, flush=True)

    # ************************************************************
    # Input parameters and for the HfO2 dataset
    # ************************************************************

    train_data_folder = os.path.join(folder, 'datasets/HfO2_2')
    val_data_folder = os.path.join(folder, 'datasets/HfO2_1')
    show_fit_for = "train"                                                          # Show fit for the training (train) or validation (val) data
    tag = 'HfO2'                                                                    # String tag for the output files

    # Graph partitioning:
    slice_start_train = 25                                                          # Start index of the slice for training
    slice_length_train = 6                                                          # Length of the slice for training                    
    num_slice_train = 1                                                             # Number of slices for training  
    slice_start_val = 25                                                            # Start index of the slice for validation       
    slice_length_val = 3                                                            # Length of the slice for validation        
    num_slice_val = 1                                                               # Number of slices for validation  
    
    # Restart calculations:
    restart_file = None
    save_file = 'model'

    if not os.path.exists('results_' + tag):
        os.makedirs('results_' + tag)
    save_file = 'results_' + tag + '/' + save_file
    
    # Network training:
    num_MP_layers = 1                                                               # Number of message passing layers 
    num_epochs = 50000                                                              # Number of epochs                                                
    learning_rate = 1e-4                                                            # Initial Learning rate                 
    loss_tol = 0                                                                    # Loss tolerance for early stopping
    patience = 500
    threshold = 1e-3
    dtype = torch.float32

    # Material parameters:
    pbc = True
    orbital_basis = 'SZV'
    rcut = 4.0                                                                      # Interaction radius (1/2*rcut) in Angstroms
    lmax = 4     
    mmax = lmax

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 128
    num_heads = 2
    attn_hidden_channels = 128 
    attn_alpha_channels = 16
    attn_value_channels = 16
    ffn_hidden_channels = 64

    # ************************************************************
    # Create the dataset
    # ************************************************************

    # *** Initialize the domain and electronic structure matrices:
    a_HfO2_train = structure.Structure(os.path.join(train_data_folder, 'structure.xyz'), 
                                    os.path.join(train_data_folder, 'H.csr'), 
                                    os.path.join(train_data_folder, 'S.csr'), 
                                    pbc, 
                                    orbital_basis, 
                                    self_interaction=False,
                                    bothways=True, 
                                    rcut = rcut)
    print("Training structure created", flush=True)

    a_HfO2_val = structure.Structure(os.path.join(val_data_folder, 'structure.xyz'),
                                        os.path.join(val_data_folder, 'H.csr'),
                                        os.path.join(val_data_folder, 'S.csr'),
                                        pbc, 
                                        orbital_basis, 
                                        self_interaction=False,
                                        bothways=True, 
                                        rcut = rcut)
    print("Validation structure created", flush=True)

    # make sure all ranks have created the structures before proceeding
    if dist.is_initialized():
        dist.barrier()
    
    # ************************************************************
    # Initialize the SO2 model
    # ************************************************************

    # *** Define irreducible representations
    irreps_in = Irreps([(sphere_channels, (0, 1)), 
                        (sphere_channels, (1, 1)), 
                        (sphere_channels, (2, 1)), 
                        (sphere_channels, (3, 1)), 
                        (sphere_channels, (4, 1))])
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]  

    # *** Perform orbital analysis:
    atom_orbitals = {'8': [0,1], '72': [0,0,1,2]}                                         # Orbital types of each atom in the structure
    numbers = a_HfO2_train.atomic_numbers                                                 # Atomic numbers of each atom in the structure
    no_parity = True                                                                      # No parity symmetry          
    orbital_types = [[0,1], [0,0,1,2]]                                                    # basis rank of each atom in the structure 

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

    # *** Initialize the model:
    mappingReduced = SO3.CoefficientMappingModule(lmax, mmax)
    irreps_out = net_out_irreps
    model = network.SO2Net(num_MP_layers, 
                            lmax, 
                            mmax, 
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
        model, optimizer = env.dist_restart('results_' + tag + '/' + restart_file + '.pt', model, optimizer)

    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {number_of_parameters} parameters", flush=True)
    print("memory allocated for the model: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB", flush=True)
    if dist.is_initialized():
        dist.barrier()

    # ************************************************************
    # Training the model
    # ************************************************************

    # *** Create the input dataloader: slice_length partitioning
    train_data_loader = data.batch_data_HfO2_cartesian(a_HfO2_train, slice_start_train, slice_length_train, num_slice_train, 
                                                        equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel,
                                                        dtype=torch.float32)

    validation_loader = data.batch_data_HfO2_cartesian(a_HfO2_val, slice_start_val, slice_length_val, num_slice_val, 
                                                        equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, 
                                                        dtype=torch.float32)
    print("data loaders created")

    print("training...", flush=True)
    training.train_and_validate_model_subgraph(model, optimizer, train_data_loader, validation_loader, num_epochs, loss_tol, patience, threshold, save_file=save_file, schedule=True, dtype=dtype)
    print("Training completed", flush=True)

    if show_fit_for == "train":
        print("Plotting fit to training data", flush=True)
        training.evaluate_model(model, train_data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file=save_file)
    else:
        print("Plotting fit to validation data...", flush=True)
        training.evaluate_model(model, validation_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file=save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amorphous GNNs --- HfO2")
    parser.add_argument("-f", "--folder", default="", required=False)
    args = parser.parse_args()

    print(f"Starting main ... dataset folder is '{args.folder}'", flush=True)

    main(args.folder)
