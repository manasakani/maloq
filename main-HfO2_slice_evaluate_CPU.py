import argparse
import lib.data as data
import lib.training as training
import lib.structure as structure
import lib_equiformer.SO2 as SO2
import lib.so2_model_local as so2_model
import lib_equiformer.SO3 as SO3
from e3nn.o3 import Irreps
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os

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

    # if not torch.cuda.is_available():
    #     raise RuntimeError("No GPUs are available!")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_folder = os.path.join(folder, '/usr/scratch/mont-fort26/chexia/ML_kevin/HamiltonianFitting__HfOx_dataset_and_model/pristine_KS_S_1/')
    xyz_file = '/usr/scratch/mont-fort26/chexia/ML_kevin/HamiltonianFitting__HfOx_dataset_and_model/HfO2.xyz'
    hamiltonian_file = os.path.join(data_folder, 'memrstors-KS_SPIN_1-1_0.csr')
    overlap_file = os.path.join(data_folder, 'memrstors-S_SPIN_1-1_0.csr')

    # Material parameters:
    pbc = True
    orbital_basis = 'SZV'
    rcut = 4.0          
    lmax_list = [4]     
    mmax_list = [lmax_list[0]]

    # Graph partitioning methods (the first three are for the slice option, the last two are for the graph partitioning option):
    partitioning = 'slice'                                                          # 'slice' or 'graph' partitioning
    
    # slice parameters:
    slice_list = [1500]                                                             # slice boundaries for partitioning the structure into subgraphs                
    cutoff = 1.5                                                                     # cutoff boundary of the slice used for training (interaction radius = 2*cutoff)
    
    # graph partitioning parameters:
    num_subgraph = 18                                                                # min 10 for P100 GPU memory with attn_hidden_channels=64
    num_batch = 1                                                                   # number of subgraphs which will actually be added to the dataset for training,
                                                                                    # after dividing the graph into 'num_subgraph' subgraphs
    # Parameters:
    # restart_file = 'model_HfO2_1_subgraph_CPU_state_dic.pt'
    restart_file = 'model_HfO2_18_cartesian_test_state_dic.pt'
    save_file = 'model_HfO2_'+str(num_subgraph)+'_subgraph_CPU'  
    num_MP_layers = 1                                                               # Number of message passing layers 
    num_epochs = 10000                                                               # Number of epochs                                                
    learning_rate = 1e-3                                                            # Initial Learning rate                 
    loss_tol = 0                                                                    # Loss tolerance for early stopping
    dtype = torch.float32
    test_data = torch.load('test_data_structures/full_data_structure_1.pt')


    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 64
    num_heads = 2
    attn_hidden_channels = 64  
    attn_alpha_channels = 32
    attn_value_channels = 32
    ffn_hidden_channels = 64

    # ************************************************************
    # Create the dataset
    # ************************************************************

    # *** Initialize the domain and electronic structure matrices:

    
    if test_data == None: 
        a_HfO2 = structure.Structure(xyz_file, 
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
    atom_orbitals = {'8':[0,1], '72':[0,0,1,2]}                                           # Orbital types of each atom in the structure
    # numbers = a_HfO2.atomic_numbers                                                       # Atomic numbers of each atom in the structure
    no_parity = True                                                                      # No parity symmetry          
    orbital_types = [[0,1],[0,0,1,2]]                                                     # basis rank of each atom in the structure 

    targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    index_to_Z, inverse_indices = torch.unique(torch.tensor([8,72]), sorted=True, return_inverse=True)
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
    if test_data == None:
        if partitioning == 'slice':
            data_loader = data.batch_data_subgraph(a_HfO2, slice_list, cutoff, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
            print("Data loader created - using " + str(len(slice_list)) + " slices", flush=True)
        else:
            data_loader = data.batch_data_graphpartition(a_HfO2, num_subgraph, num_batch, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
            print("Data loader created - using " + str(num_subgraph) + " subgraphs", flush=True)
        if dist.is_initialized():
            dist.barrier()

    # *** Initialize the model:
    mappingReduced = SO3.CoefficientMappingModule(lmax_list, mmax_list)
    irreps_out = net_out_irreps
    model = so2_model.SO2Net_local(num_MP_layers, 
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
        # checkpoint = torch.load(restart_file)
        # state_dict = checkpoint['model_state_dict']
        # state_dict = remove_module_prefix(checkpoint['model_state_dict'])
        # model.load_state_dict(state_dict)

        state_dict = torch.load(restart_file, map_location=torch.device('cpu'))
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        # torch.save(state_dict, save_file+'_state_dic.pt')

    print("Model initialized", flush=True)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()), flush=True)

    print("memory: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB", flush=True)
    if dist.is_initialized():
        dist.barrier()

    # ************************************************************
    # Training and testing the model
    # ************************************************************

    print("testing on unseen data...", flush=True)
    if test_data is not None:
        test_data_loader = data.batch_data_load(test_data, equivariant_blocks, out_slices, construct_kernel, dtype=torch.float32)
        # data_loader = data.batch_data_subgraph(a_HfO2, slice_list, cutoff, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
        # training.evaluate_model(model, test_data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)
        training.analyze_model(model, test_data, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file=save_file)

    else:
        training.evaluate_model(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amorphous GNNs --- HfO2")
    parser.add_argument("-f", "--folder", default="", required=False)
    args = parser.parse_args()

    print(f"Starting main ... dataset folder is '{args.folder}'", flush=True)

    main(args.folder)
