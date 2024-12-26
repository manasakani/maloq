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

import data, training, structure, SO2, network, SO3, compute_env as env, utils
import warnings
from mpi4py import MPI
from e3nn.o3 import Irreps

# ************************************************************
# Distributed training setup (if running on multiple GPUs)
# ************************************************************

device, world_size = env.initialize_compute_env()
print("Device: ", device, ", World size: ", world_size, flush=True)

env.rank_zero_print(f"Added {lib_root} to the path", flush=True)
env.rank_zero_print(f"Added {lib_equiformer_root} to the path", flush=True)
env.rank_zero_print("Imported libraries", flush=True)
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")

def main(folder):

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    print(f"Folder: {folder}", flush=True)

    # ************************************************************
    # Input parameters for the HfO2 dataset
    # ************************************************************

    train_data_folder = os.path.join(folder, 'datasets/HfO2_2')
    val_data_folder = os.path.join(folder, 'datasets/HfO2_1')
    show_fit_for = "train"                                                          # Show fit for the training (train) or validation (val) data
    tag = 'HfO2'                                                                    # String tag for the output files

    # Dataset parameters:
    num_train = 1                                                                   # Number of training samples
    num_validate = 1                                                                # Number of validation samples             
    num_test = 1     
    batch_size = 1                                                                  # Single graph batch size 
    show_fit_for = "val"                                                            # Show fit for the training (train) or validation (val) data

    # Restart calculations:
    restart_file = None
    save_file = 'model'

    @env.only_rank_zero
    def make_output_folder(save_file, tag): 
        if not os.path.exists('results_' + tag):
            os.makedirs('results_' + tag)
        save_file = 'results_' + tag + '/' + save_file
        return save_file
    save_file = make_output_folder(save_file, tag)

    if dist.is_initialized():
        rank = dist.get_rank()
        size = dist.get_world_size()
        comm = MPI.COMM_WORLD
        save_file = comm.bcast(save_file, root=0)

    # Network training:
    num_MP_layers = 1                                                               # Number of message passing layers 
    num_epochs = 100# 50000                                                              # Number of epochs                                                
    learning_rate = 1e-4                                                            # Initial Learning rate                 
    loss_tol = 0                                                                    # Loss tolerance for early stopping
    patience = 500
    threshold = 1e-3
    dtype = torch.float64
    torch.set_default_dtype(torch.float64)

    # Material parameters:
    pbc = True
    orbital_basis = 'SZV'
    rcut = 4.0                                                                        # Interaction radius (1/2*rcut) in Angstroms
    lmax = 4     
    mmax = 4

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 64
    num_heads = 2
    attn_hidden_channels = 64 
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

    assert(num_train % batch_size == 0) # batch size should divide the number of training samples for current distribution
    partition = {}
    partition['train'] = env.Domain_Decomp(a_HfO2_train, device)
    partition['validate'] = env.Domain_Decomp(a_HfO2_val, device)
    dist.barrier()

    partition['train'].print_info()
    partition['validate'].print_info()
    dist.barrier()

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
                                          device_torch=device) #the data is created on cpu, so the construct_kernel must be on cpu 
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
    print("Creating training data loader...", flush=True)
    training_data_loader = data.batch_data_molecules([a_HfO2_train], partition['train'], device, num_train, batch_size, equivariant_blocks, out_slices, construct_kernel, dtype)
    print("Creating training data loader...", flush=True)
    validation_data_loader = data.batch_data_molecules([a_HfO2_val], partition['validate'], device, num_validate, batch_size, equivariant_blocks, out_slices, construct_kernel, dtype)
    print("Data loaders created")

    print("Training model...", flush=True)
    training.train_and_validate_model_subgraph(model, 
                                                optimizer, 
                                                partition,
                                                training_data_loader, 
                                                validation_data_loader, 
                                                num_epochs, 
                                                loss_tol, 
                                                patience, 
                                                threshold, 
                                                min_lr=1e-10,
                                                save_file=save_file, 
                                                dtype=dtype,
                                                unflatten=False,
                                                construct_kernel=construct_kernel,
                                                equivariant_blocks=equivariant_blocks, 
                                                atom_orbitals=atom_orbitals, 
                                                out_slices=out_slices)
    print("Model trained")

    # create new construct_kernel for the training, this time on the cpu
    construct_kernel = SO2.e3TensorDecomp(net_out_irreps, 
                                        out_js_list, 
                                        default_dtype_torch=dtype, 
                                        spinful=False,
                                        no_parity=no_parity, 
                                        if_sort=False, 
                                        device_torch='cpu')

    if show_fit_for == "train":
        print("Plotting fit to training data", flush=True)
        training.evaluate_model(model, partition['train'], training_data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file=save_file)
    else:
        print("Plotting fit to validation data...", flush=True)
        training.evaluate_model(model, partition['validate'], validation_data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file=save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amorphous GNNs --- HfO2")
    parser.add_argument("-f", "--folder", default="", required=False)
    args = parser.parse_args()

    print(f"Starting main ... dataset folder is '{args.folder}'", flush=True)

    main(args.folder)
