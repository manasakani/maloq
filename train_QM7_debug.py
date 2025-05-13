import time
import_start = time.perf_counter()
import os, sys
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from fock_utils import utils_orca_out, fock_targets, utils_training

import torch
import torch.distributed as dist

from ASEDataset import ASEDataset, ASEAtomsData, sampleDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import random

from equiformer.network import SO2Net
from equiformer.SO3 import CoefficientMappingModule

from esen.esen import eSEN_Backbone    
from e3nn.o3 import Irreps

import_end = time.perf_counter()
print("Time to do imports: ", import_end - import_start)

def delete_rows_and_columns(matrix, indices):
    """
    Delete specified rows and columns from a matrix.
    Parameters:
    - matrix: The input matrix (2D NumPy array).
    - indices: A list of row/column indices to delete.
    Returns:
    - A new matrix with the specified rows and columns removed.
    """
    # Convert the list of indices to a NumPy array
    indices = np.array(indices)
    matrix_reduced = np.delete(matrix, indices, axis=0)
    matrix_reduced = np.delete(matrix_reduced, indices, axis=1)
    return matrix_reduced

# def custom_collate_fn(batch):
#     return Batch.from_data_list(batch)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Fix this later:
sys.path.append('/home/manasakani/fairchem/src/')

# Settings (just dumping everything here for now)
# -----------------------------------------------
dbpath = 'fock_datasets/schnorb_hamiltonian_water.db'
database = ASEAtomsData(dbpath)
print("Targets available: ", database.available_properties)

# -> Model settings:
l_embedding_dim = 128                   # sphere channels
num_distance_basis = 128                # number of gaussian basis functions used to expand the edge distance
hidden_dim = 128
cutoff = 6.0*2                          # Cutoff used for edge distance embedding
num_mp_layers = 1 
model_name = 'esen'
restart = False
output_folder = 'outputs_QM7_debug'
model_filename = 'model.pt.pt'

# -> Training settings:
train_or_eval = "train"
num_val = 1                           # Number of validation structures
num_train = 1
num_epochs = 5000
lr_init = 1e-3
dtype = torch.float32
batch_size = 1         # 1 for eval, 10 for train
loss_target = 'fock_matrix'
patience = 100                          # for scheduler
threshold = 1e-5                        # for scheduler
# loss_fxn = utils_training.l1_unpadded_loss
loss_fxn = utils_training.mse_padded_loss

# --> Compute env
device = torch.device('cuda')         
world_size = int(os.environ['SLURM_NTASKS'])
rank = int(os.environ['SLURM_PROCID'])
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
torch.cuda.set_device(0) # visibility is restricted to 0 in .sh file

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Prepare data
# --------------------------------------------
data_load_start = time.perf_counter()
max_mol = 5000 
num_molecules = num_val + num_train
# random_indices = random.sample(range(num_molecules), min(max_mol, num_molecules))
random_indices = [0, 0] 

datalist = []
for i in random_indices:
    mol = database.__getitem__(i)

    mol_atoms = Atoms(symbols=mol['_atomic_numbers'].numpy(), positions=mol['_positions'].numpy())
    rcut = 100.0                                            # connectivity cutoff
    num_atoms = len(mol['_positions'])
    energy = mol['energy']
    forces = mol['forces']

    # Electronic structure matrix:
    hamiltonian = mol['hamiltonian'].numpy()   
    orbital_basis = {8: [0, 0, 0, 1, 1, 2], 1: [0, 0, 1]}
    atomic_numbers = mol['_atomic_numbers'].numpy()
    hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)  

    ###
    # --> delete duplicate orbitals for l > 0:
    # full_orb_list = np.hstack([orbital_basis[atomic_numbers[i]] for i in range(len(atomic_numbers))])
    # orbital_starts = np.hstack([0, np.cumsum([2*l + 1 for l in full_orb_list])[:-1]])
    # indices_to_delete = (   list(range(9, 14)) #+         # oxygen d orbitals
    #                         # list(range(9, 18)) +       # oxygen p orbitals
    #                         # list(range(23, 33)) +      # oxygen d orbitals
    #                         # list(range(46, 49)) +      # hydrogen p orbital
    #                         # list(range(55, 58)) +       # hydrogen p orbital
    #                         # list(range(41, 43)) +
    #                         # list(range(50, 52))
    #                     )
    # hamiltonian = delete_rows_and_columns(hamiltonian, indices_to_delete)

    # # basis = {8: [0, 0, 0, 0, 0, 0, 1, 2, 3], 1: [0, 0, 0, 1]}
    # orbital_basis = {8: [0, 0, 0, 1, 1], 1: [0, 0, 1]}

    # full_orb_list = np.hstack([orbital_basis[atomic_numbers[i]] for i in range(len(atomic_numbers))])
    # expected_matrix_size = sum([2*l + 1 for l in full_orb_list])
    # orbital_starts = np.hstack([0, np.cumsum([2*l + 1 for l in full_orb_list])[:-1]])

    import matplotlib.pyplot as plt
    plt.imshow(hamiltonian)
    plt.colorbar()
    plt.savefig("qm7_fock.png", dpi=300, bbox_inches='tight')
    exit()
    # --------------------------------------------------------------------------------------
    ###

    time_start = time.perf_counter()
    graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian)
    time_end = time.perf_counter()
    print("time to make targets: ", time_end - time_start)

    data = gnnData(
                    pos=torch.tensor(graph_targets.atoms.positions, dtype=torch.float),
                    x=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long), 
                    edge_index=torch.tensor(graph_targets.neighbour_list).to(device),
                    edge_attr=graph_targets.edge_dist.to(device),
                    y=graph_targets.edge_labels,
                    node_y=graph_targets.node_labels,
                    atomic_numbers=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long).cpu(),  
                    energies=torch.tensor(energy, dtype=dtype),
                    forces=torch.tensor(forces, dtype=dtype),                                      # Hartree/Angstrom
                )
    datalist.append(data)

required_irreps = graph_targets.req_output_irreps
print("required irreps: ", required_irreps)

train_size = len(datalist) - num_val
train_datalist, val_datalist = torch.utils.data.random_split(datalist, [train_size, num_val])
train_dataset = sampleDataset(train_datalist)
val_dataset = sampleDataset(val_datalist)

# Check use of batch size higher than 1!!
train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

data_load_end = time.perf_counter()
print("Time to load dataset: ", data_load_end - data_load_start)

# Prepare model
# --------------------------------------------
model_setup_start = time.perf_counter()
if model_name == 'equiformer':
    irreps_in = Irreps([(l_embedding_dim, (l, 1)) for l in range(required_irreps.lmax + 1)]) 
    mappingReduced = CoefficientMappingModule(required_irreps.lmax, required_irreps.lmax)
    edge_channels_list = [l_embedding_dim, l_embedding_dim, l_embedding_dim]

    attn_hidden_channels = 128 
    attn_alpha_channels = 32
    attn_value_channels = 32 
    ffn_hidden_channels = 64 
    num_heads=2

    model = SO2Net(num_mp_layers, 
                    required_irreps.lmax, 
                    required_irreps.lmax, 
                    mappingReduced, 
                    l_embedding_dim, 
                    edge_channels_list, 
                    attn_hidden_channels, 
                    num_heads, 
                    attn_alpha_channels, 
                    attn_value_channels, 
                    ffn_hidden_channels, 
                    irreps_in, 
                    required_irreps)
elif model_name == 'esen':
    model = eSEN_Backbone(
                    required_irreps,
                    sphere_channels=l_embedding_dim,
                    hidden_channels=hidden_dim,
                    lmax=required_irreps.lmax,
                    mmax=required_irreps.lmax,
                    use_pbc=False,
                    cutoff=cutoff,
                    edge_channels=l_embedding_dim,
                    num_layers=num_mp_layers,
                    act_type='gate',
                    mlp_type = 'spectral',
                    num_distance_basis=num_distance_basis
                )

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
model_setup_end = time.perf_counter()
print("Time to setup model: ", model_setup_end - model_setup_start)

if restart:
    restart_file = output_folder + '/' + model_filename
    checkpoint = torch.load(restart_file)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False) # 'strict' should take care of the module prefix, but watch out


# Training or Evaluation
# --------------------------------------------

if train_or_eval == "train":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
    utils_training.train_model(model, 
                            optimizer, 
                            loss_fxn, 
                            loss_target, 
                            num_epochs, 
                            train_loader, 
                            val_loader, 
                            scheduler, 
                            device, 
                            output_folder)
if train_or_eval == "eval":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
    utils_training.eval_model(model, 
                            optimizer, 
                            loss_fxn, 
                            loss_target, 
                            num_epochs, 
                            train_loader, 
                            val_loader, 
                            scheduler, 
                            device, 
                            output_folder)