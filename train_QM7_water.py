import time
import_start = time.perf_counter()
import os, sys
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
import utils_orca_out, fock_targets

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
import utils_training

import_end = time.perf_counter()
print("Time to do imports: ", import_end - import_start)

def custom_collate_fn(batch):
    return Batch.from_data_list(batch)

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
num_mp_layers = 3
model_name = 'esen'
restart = False
output_folder = 'outputs_QM7'

# -> Training settings:
num_val = 1                           # Number of validation structures
num_train = 1
num_epochs = 20000
lr_init = 5e-5
dtype = torch.float32
batch_size = 10
loss_target = 'fock_matrix'
patience = 200                          # for scheduler
threshold = 1e-4                        # for scheduler
loss_fxn = utils_training.l1_padded_loss

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
random_indices = random.sample(range(num_molecules), min(max_mol, num_molecules))

datalist = []
for i in range(num_molecules):  # deterministic
# for i in random_indices:
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
                    forces=torch.tensor(forces, dtype=dtype),
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
    restart_file = output_folder + "/model.pt.pt"
    checkpoint = torch.load(restart_file)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)


# Training
# --------------------------------------------
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
