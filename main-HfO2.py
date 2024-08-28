import lib.data as data
import lib.training as training
import lib.structure as structure
import lib.SO2 as SO2
import lib.so2_model as so2_model
import lib.SO3 as SO3
from e3nn.o3 import Irreps
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

def main():

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    # Dataset  
    data_folder = './datasets/a-HfO2/'
    xyz_file = data_folder + 'structure.xyz'
    hamiltonian_file = data_folder + 'H.csr'
    overlap_file = data_folder + 'S.csr'

    # Material parameters:
    pbc = True
    orbital_basis = 'SZV'
    rcut = 4.0
    lmax_list = [4]
    mmax_list = [4]

    # Graph partitioning parameters:
    slice_list = [1000,1200,1400,]
    cutoff = 1.5 # cutoff boundary of the slice used for training 

    # Parameters:
    restart_file = None
    save_file = 'model_HfO2.pth'  
    num_MP_layers = 2                                               # Number of message passing layers 
    num_epochs = 1                                                
    learning_rate = 1e-4
    loss_tol = 0                                                    
    dtype = torch.float32

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 64
    num_heads = 2
    attn_hidden_channels = 64
    attn_alpha_channels = 32
    attn_value_channels = 32
    ffn_hidden_channels = 64

    # Define irreducible representations for the SO2 model
    irreps_in = Irreps([(sphere_channels, (0, 1)), (sphere_channels, (1, 1)), (sphere_channels, (2, 1)), (sphere_channels, (3, 1)), (sphere_channels, (4, 1))])
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]  

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
    print("Structure created")

    # *** Perform orbital analysis:
    atom_orbitals = {'8':[0,1], '72':[0,0,1,2]}                                           # Orbital types of each atom in the structure
    numbers = a_HfO2.atomic_numbers                                                       # Atomic numbers of each atom in the structure
    no_parity = True                                                                      # No parity symmetry          
    orbital_types = [[0,1],[0,0,1,2]]                                                     # basis rank of each atom in the structure 

    targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    index_to_Z, inverse_indices = torch.unique(numbers, sorted=True, return_inverse=True)
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
    print("Orbital analysis completed")

    # *** Create the input dataloader:
    data_loader = data.batch_data_subgraph(a_HfO2, slice_list, cutoff, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
    print("Data loader created")

    # *** Initialize the model:
    # if restart_file is None:
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
        checkpoint = torch.load(save_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("Model initialized")
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    # *** Train the model parameters:
    print("training...")
    training.train_model_subgraph(model, optimizer, data_loader, num_epochs, loss_tol, save_file=save_file, dtype=dtype)
    print("Model trained")

if __name__ == "__main__":
    main()