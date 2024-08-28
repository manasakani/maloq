import lib.data as data
import lib.models as models
import lib.training as training
import lib.structure as structure
import lib.utils as utils
import lib.SO2 as SO2
import lib.so2_model as so2_model
import lib.SO3 as SO3
from e3nn.o3 import Irreps
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random


def main():

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    data_folder = '/usr/scratch/mont-fort26/chexia/ML_kevin/HamiltonianFitting__HfOx_dataset_and_model/pristine_KS_S_1/'
    # Inputs:
    a_HfO2s = []
    xyz_file = '/usr/scratch/mont-fort26/chexia/ML_kevin/HamiltonianFitting__HfOx_dataset_and_model/HfO2.xyz'
    hamiltonian_file = data_folder + 'memrstors-KS_SPIN_1-1_0.csr'
    overlap_file = data_folder + 'memrstors-S_SPIN_1-1_0.csr'
    pbc = True
    orbital_basis = 'SZV'

    a_HfO2s.append(structure.Structure(xyz_file, hamiltonian_file, overlap_file, pbc, orbital_basis, make_soap=True, save_matrices=True,self_interaction=False,bothways=True, rcut = 4))

    # Parameters:
    # restart_file = 'model_H2O.pth'                                  # Restart training from a saved model
    restart_file = None
    save_file = 'model_test_GPU_batch_validate'
    node_embedding_type = 'SOAP'                                    # Node embeddings
    num_MP_layers = 2                                               # Number of message passing layers - note that for SO2 there is already 1 layer in the model
    batch_size = 1                                                 # Batch size for training
    num_epochs = 10000                                                # Number of epochs for training (minimum of 1)
    learning_rate = 1e-4
    num_graph = 1                                 # Number of graphs to create (each input structure is one graph)
    dtype = torch.float32
    lmax_list = [4]
    mmax_list = [4]

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 64
    num_heads = 2
    attn_hidden_channels = 64
    attn_alpha_channels = 32
    attn_value_channels = 32
    ffn_hidden_channels = 64

    # irreps_in = Irreps('64x0e+64x1e+64x2e+64x3e+64x4e') 
    irreps_in = Irreps([(sphere_channels, (0, 1)), (sphere_channels, (1, 1)), (sphere_channels, (2, 1)), (sphere_channels, (3, 1)), (sphere_channels, (4, 1))])

    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]         # @kevin 128 is hardcoded in the model, please remove the hardcode so it can be changed
    loss_tol = 0 #does not finish prematurely 

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Device: ", device)

    # *** Initialize the domain and electronic structure matrices:

    # *** Preform orbital analysis:
    atom_orbitals = {'8':[0,1],'72':[0,0,1,2]}                                            # Orbital types of each atom in the structure
    numbers = a_HfO2s[0].atomic_numbers     # Atomic numbers of each atom in the structure
    no_parity = True                                                                            # No parity symmetry          
    orbital_types = [[0,1],[0,0,1,2]]                                                       # orbital types of each atom in the structure 

    targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    # index_to_Z, Z_to_index  = utils.element_statistics(numbers)
    index_to_Z,inverse_indices = torch.unique(numbers, sorted=True, return_inverse=True)
    equivariant_blocks, out_js_list, out_slices = SO2.process_targets(orbital_types, index_to_Z, targets)

    # equivariant_blocks: start and end indices of the equivariant blocks in i and j direction for each target in targets
    # out_js_list: ll the l1 l2 interactions needed 
    # out_slices: marks the start and end of indices belonging to a certain target. Slice 1 (0 to 1) corresponds to the first target in equivariant blocks 

    construct_kernel = SO2.e3TensorDecomp(net_out_irreps, 
                                          out_js_list, 
                                          default_dtype_torch= torch.float32, 
                                          spinful=False,
                                          no_parity=no_parity, 
                                          if_sort=False, 
                                          device_torch='cpu') #the data is created on cpu, so the construct_kernel must be on cpu 

    # *** Create the input dataloader:

    slice_list = [1000,1200,1400]
    test_list = None
    cutoff = 1.5 #cutoff boundary of the slice used for training 
    data_loader = data.batch_data_HfO2(a_HfO2s, slice_list, test_list, save_file, cutoff, equivariant_blocks = equivariant_blocks, out_slices = out_slices, construct_kernel=construct_kernel, dtype = torch.float32)


    print("Data loader created")

    # *** Initialize the model:
    if restart_file is None:
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
                

        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
        torch.save(model, save_file+'_cpu.pt')
        model = model.to(device)

    else:
        print("Restarting training from a saved model...")
        model = torch.load(restart_file)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    print("Model initialized")

    # *** Train the model parameters:
    print("training model...")
    test_batch = torch.load('model_test_GPU_validate_structure_0_training_2000_1.5.pt')
    # training.train_model_HfO2(model, data_loader, node_embedding_type, num_epochs, learning_rate, loss_tol, save_file=save_file, dtype=dtype)
    training.train_and_validate_model_HfO2(model, data_loader, test_batch, node_embedding_type, num_epochs, learning_rate, loss_tol, save_file, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, dtype=torch.float32)

    print("Model trained")

if __name__ == "__main__":
    main()