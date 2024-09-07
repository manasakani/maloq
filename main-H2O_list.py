import lib.data as data
import lib.models as models
import lib.training as training
import lib.structure as structure
import lib.utils as utils
import lib_equiformer.SO2 as SO2
import lib_equiformer.SO3 as SO3
from e3nn.o3 import Irreps
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
from lib import so2_model

def main():

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Inputs:    
    # base_folder = './datasets/H2O_3D/'
    # all_folders = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    # data_folders = [os.path.join(base_folder, folder) for folder in all_folders]
    data_folders = ['./datasets/H2O/H2O_DZVP_1', './datasets/H2O/H2O_DZVP_2']


    # Dataset parameters:
    num_train = 1                                                  # Number of training samples
    num_validate = 1                                               # Number of validation samples             
    num_test = 0                                                   # Number of testing samples     

    restart_file = None                                             # Restart training from a saved model
    save_file = 'model_H2O.pth'
    train_or_test = 'train'                                         # Train or test the model    
    num_epochs = 300                                                # Number of epochs for training (minimum of 1)
    batch_size = len(data_folders)                                  # Batch size for training (currently all data at once)
    loss_tol = 1e-7                                                 # Loss tolerance for stopping training           
    lr = 1e-3                                                       # Learning rate     

    # Structure and Network parameters:
    pbc = False
    bothways = True
    orbital_basis = 'DZVP'                                          # Orbital basis set (see utils for options)
    num_MP_layers = 2                                               # Number of message passing layers - note that for SO2 there is already 1 layer in the model
    dtype = torch.float32
    lmax_list = [4]                                                 # maximum rank of tensor product 
    mmax_list = [4]                                                 # maximum order for expansion of spherical harmonics

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 16
    num_heads = 2
    attn_hidden_channels = 128  
    attn_alpha_channels = 32
    attn_value_channels = 32
    ffn_hidden_channels = 64
    irreps_in = Irreps([(sphere_channels, (0, 1)), (sphere_channels, (1, 1)), (sphere_channels, (2, 1)), (sphere_channels, (3, 1)), (sphere_channels, (4, 1))])
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]  

    # *** Initialize the domain and electronic structure matrices:
    training_molecules = []
    validation_molecules = []
    for i in range(num_train):
        training_molecules.append(structure.Structure(data_folders[i] + '/snapshot.xyz', 
                                                    data_folders[i] + '/H.csr',  #'/memristors-KS_SPIN_1-1_0.csr', 
                                                    data_folders[i] + '/s.csr',  #'/memristors-S_SPIN_1-1_0.csr', 
                                                    pbc, orbital_basis, 
                                                    self_interaction=False, 
                                                    bothways=bothways, 
                                                    make_soap=False))
        
    for i in range(num_train, num_train + num_validate):
        validation_molecules.append(structure.Structure(data_folders[i] + '/snapshot.xyz', 
                                                data_folders[i] + '/H.csr',
                                                data_folders[i] + '/s.csr',
                                                pbc, orbital_basis, 
                                                self_interaction=False, 
                                                bothways=bothways, 
                                                make_soap=False))

    # *** Preform orbital analysis:
    atom_orbitals = {'1': [0, 0, 1],'8':[0,0,1,1,2]}                                                    # Orbital types of each atom in the structure
    numbers = torch.tensor([utils.periodic_table[i] for i in training_molecules[0].atomic_species])     # Atomic numbers of each atom in the structure
    no_parity = True                                                                                    # No parity symmetry          
    orbital_types = [[0,0,1],[0, 0, 1, 1, 2]]                                                           # basis rank of each atom in the structure 

    targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    index_to_Z, Z_to_index  = utils.element_statistics(numbers)
    equivariant_blocks, out_js_list, out_slices = SO2.process_targets(orbital_types, index_to_Z, targets)
    print("out_slices: ", out_slices)

    # equivariant_blocks: start and end indices of the equivariant blocks in i and j direction for each target in targets
    # out_js_list: ll the l1 l2 interactions needed 
    # out_slices: marks the start and end of indices belonging to a certain target. Slice 1 (0 to 1) corresponds to the first target in equivariant blocks 

    construct_kernel = SO2.e3TensorDecomp(net_out_irreps, 
                                          out_js_list, 
                                          default_dtype_torch= torch.float32, 
                                          spinful=False,
                                          no_parity=no_parity, 
                                          if_sort=False, 
                                          device_torch=device)

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
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    else:
        print("Restarting training from a saved model...")
        model = torch.load(restart_file)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    print("Model initialized")

    # *** Create the input dataloaders:
    training_data_loader = data.batch_data_SO2(training_molecules, device, num_train, batch_size, equivariant_blocks, out_slices, construct_kernel, dtype)
    validation_data_loader = data.batch_data_SO2(validation_molecules, device, num_validate, batch_size, equivariant_blocks, out_slices, construct_kernel, dtype)
    print("Data loaders created for training and validation")

    # *** Test the model on the training data, or on the testing data:
    if train_or_test == 'train':

        # *** Train the model parameters:
        print("training model...")
        training.train_and_validate_model_SO2(model, optimizer, training_data_loader, validation_data_loader, num_epochs, loss_tol, save_file=save_file, dtype=dtype)
        print("Model trained")

        # use training samples
        test_list = []
        test_structures = []
        for i in range(num_train):
            train_data = data.create_input_data_SO2(training_molecules[i], equivariant_blocks, out_slices, construct_kernel, device, dtype = dtype)
            test_list.append(train_data)
            test_structures.append(training_molecules[i])
        data_loader = training_data_loader

    else:
        # use unseen sample:
        testing_molecules = []

        for i in range(num_train + num_validate, num_train + num_validate + num_test):
            testing_molecules.append(structure.Structure(data_folders[i] + '/snapshot.xyz', 
                                                         data_folders[i] + '/H.csr',
                                                         data_folders[i] + '/S.csr',
                                                         pbc, orbital_basis, 
                                                         self_interaction=False, 
                                                         bothways=bothways, 
                                                         make_soap=True))

        testing_data_loader = data.batch_data_SO2(testing_molecules, num_test, batch_size, equivariant_blocks, out_slices, construct_kernel, dtype)

        test_list = []
        test_structures = []
        for i in range(num_test):
            test_data = data.create_input_data_SO2(testing_molecules[i], equivariant_blocks, out_slices, construct_kernel, dtype = dtype)
            test_list.append(test_data)
            test_structures.append(testing_molecules[i])
        data_loader = testing_data_loader

    test_batch = data.custom_collate_fn(test_list)
    training.test_model_SO2(construct_kernel, model, test_batch, dtype)


if __name__ == "__main__":
    main()