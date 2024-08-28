import lib.data as data
import lib.models as models
import lib.training as training
import lib.structure as structure
import lib.utils as utils
import lib.SO2 as SO2
import lib.SO3 as SO3
import lib.so2_model as so2_model
from e3nn.o3 import Irreps
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

# SchNetPack package for database handling
from schnetpack.data import ASEAtomsData

# Adding units to the dataset
# spkconvert --distunit Angstrom --propunit energy:Hartree,forces:Hartree/Angstrom,hamiltonian:Hartree,overlap:dimensionless /Users/manasakani/Documents/ETH/Repos/ham_predict/datasets/schnorb_hamiltonian_water.db
# Supplementary materials of oaper with comparison: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-12875-2/MediaObjects/41467_2019_12875_MOESM1_ESM.pdf

def get_number_orbitals(database):
        basis_def = database.metadata['basisdef']
        basis_def = np.array(basis_def)
        n_orbitals = np.zeros(basis_def.shape[0], dtype=int)

        for i in range(basis_def.shape[0]):
            n_orbitals[i] = int(np.count_nonzero(basis_def[i, :, 2]))

        return n_orbitals

def main():

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Inputs:    
    db_path = './datasets/schnorb_hamiltonian_water.db'
    database = ASEAtomsData(db_path)
    print("Number of Molecules in the database: ", len(database))

    norbs = get_number_orbitals(database)
    print("Number of orbitals: ", norbs)

    # Dataset parameters:
    num_train = 100                                                             # Number of training samples
    num_validate = 10                                                           # Number of validation samples             
    num_test = 40     

    restart_file = None                                                         
    save_file = 'model_H2O.pth'
    train_or_test = 'train'                                                     
    num_epochs = 100                                                           
    batch_size = 20                                                               
    loss_tol = 1e-8
    lr = 5e-5
    rcut = 1000.0

    # Structure and Network parameters:
    pbc = False
    bothways = True
    orbital_basis = 'def2_SVP' 
    num_MP_layers = 1                                                           # Number of message passing layers - note that for SO2 there is already 1 layer in the model
    dtype = torch.float32 # trying float64
    lmax_list = [4] 
    mmax_list = [4]

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 64
    num_heads = 2
    attn_hidden_channels = 64
    attn_alpha_channels = 32
    attn_value_channels = 32
    ffn_hidden_channels = 64

    # Create the training/validation/testing datasets
    training_data_indices, validation_data_indices, testing_data_indices = data.split_data_indices(num_train, num_validate, num_test, len(database))

    # Define irreducible representations for the SO2 model
    irreps_in = Irreps([(sphere_channels, (0, 1)), (sphere_channels, (1, 1)), (sphere_channels, (2, 1)), (sphere_channels, (3, 1)), (sphere_channels, (4, 1))])
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]  

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    # *** Initialize the domain and electronic structure matrices:
    training_molecules = []
    validation_molecules = []
    for i in range(num_train):
        molecule_index = int(training_data_indices[i])
        training_molecules.append(structure.Structure(None, None, None,
                                              pbc, 
                                              orbital_basis, 
                                              dataset='schnet', 
                                              database_props=database.__getitem__(molecule_index), 
                                              self_interaction=False, bothways=bothways, rcut=rcut))
    
    for i in range(num_validate):
        molecule_index = int(validation_data_indices[i])
        validation_molecules.append(structure.Structure(None, None, None,
                                                pbc, 
                                                orbital_basis, 
                                                dataset='schnet', 
                                                database_props=database.__getitem__(molecule_index), 
                                                self_interaction=False, bothways=bothways, rcut=rcut))
    print("Structures initialized")

    # *** Preform orbital analysis:
    atom_orbitals = {'1': [0, 0, 1],'8':[0, 0, 0, 1, 1, 2]}                                             # Orbital types of each atom in the structure
    numbers = torch.tensor([utils.periodic_table[i] for i in training_molecules[0].atomic_species])     # Atomic numbers of each atom in the structure
    no_parity = True                                                                                    # No parity symmetry          
    orbital_types = [[0,0,1],[0, 0, 0, 1, 1, 2]]                                                        # orbital types of each atom in the structure 
    required_block_type = 'a'                                                                         

    # targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, required_block_type, targets=None, no_parity=no_parity)
    # index_to_Z, Z_to_index  = utils.element_statistics(numbers)
    # equivariant_blocks, out_js_list, out_slices = SO2.process_targets(orbital_types, index_to_Z, targets)
    # # equivariant_blocks: start and end indices of the equivariant blocks in i and j direction for each target in targets
    # # out_js_list: ll the l1 l2 interactions needed 
    # # out_slices: marks the start and end of indices belonging to a certain target. 
    # # Slice 1 (0 to 1) corresponds to the first target in equivariant blocks 

    targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    index_to_Z, inverse_indices = torch.unique(numbers, sorted=True, return_inverse=True)
    equivariant_blocks, out_js_list, out_slices = SO2.process_targets(orbital_types, index_to_Z, targets)
    # equivariant_blocks: start and end indices of the equivariant blocks in i and j direction for each target in targets
    # out_js_list: ll the l1 l2 interactions needed 
    # out_slices: marks the start and end of indices belonging to a certain target. Slice 1 (0 to 1) corresponds to the first target in equivariant blocks 

    construct_kernel = SO2.e3TensorDecomp(net_out_irreps, 
                                          out_js_list, 
                                          default_dtype_torch=dtype, 
                                          spinful=False,
                                          no_parity=no_parity, 
                                          if_sort=False, 
                                          device_torch=device)

    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      
    if restart_file is not None:
        print("Restarting training from a saved model and optimizer state...")
        checkpoint = torch.load(save_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    print("Model initialized")


    # *** Create the input dataloader:
    training_data_loader = data.batch_data_SO2(training_molecules, device, num_train, batch_size, equivariant_blocks, out_slices, construct_kernel, dtype)
    validation_data_loader = data.batch_data_SO2(validation_molecules, device, num_validate, batch_size, equivariant_blocks, out_slices, construct_kernel, dtype)
    print("Data loaders created for training and validation")

    if train_or_test == 'train':

        # *** Train the model parameters:
        if train_or_test == 'train':
            print("training model...")
            training.train_and_validate_model_SO2(model, optimizer, training_data_loader, validation_data_loader, num_epochs, loss_tol, save_file=save_file, dtype=dtype)
            print("Model trained")

            # Testing the model on the input graphs:
            test_list = []
            test_structures = []
            for i in range(num_train):
                test_data = data.create_input_data_SO2(training_molecules[i], equivariant_blocks, out_slices, construct_kernel, dtype=dtype)
                test_list.append(test_data)
                test_structures.append(training_molecules[i])
            data_loader = training_data_loader

    else:

        # use unseen sample:
        testing_molecules = []
        for i in range(num_test):
            molecule_index = int(testing_data_indices[i])
            testing_molecules.append(structure.Structure(None, None, None,
                                                pbc, 
                                                orbital_basis, 
                                                dataset='schnet', 
                                                database_props=database.__getitem__(molecule_index), 
                                                self_interaction=False, bothways=True, make_soap=False))
        
        testing_data_loader = data.batch_data_SO2(testing_molecules, device, num_test, batch_size, equivariant_blocks, out_slices, construct_kernel, dtype)

        test_list = []
        test_structures = []
        for i in range(num_test):
            test_data = data.create_input_data_SO2(testing_molecules[i], equivariant_blocks, out_slices, construct_kernel, dtype = dtype)
            test_list.append(test_data)
            test_structures.append(testing_molecules[i])
        data_loader = testing_data_loader

    test_batch = data.custom_collate_fn(test_list)
    training.test_model_SO2(test_structures, construct_kernel, model, data_loader, test_batch, dtype)

if __name__ == "__main__":
    main()