import time
import_start = time.perf_counter()
import os, sys
import numpy as np
import random

from ase import Atoms
from ase.neighborlist import NeighborList
from fock_utils import utils_orca_out, fock_targets, utils_training

import torch
import torch.distributed as dist
from ASEDataset import ASEDataset, ASEAtomsData, sampleDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
from nablaDFT_dataset_utils import HamiltonianDatabase, transform

from equiformer.network import SO2Net
from equiformer.SO3 import CoefficientMappingModule

from esen.esen import eSEN_Backbone
from e3nn.o3 import Irreps

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

# -------------------------------------------
# --> Settings (dump everything here for now)
# -------------------------------------------
database = HamiltonianDatabase("./fock_datasets/nabla2_DFT/train_2k.db")

# -> Model settings:
l_embedding_dim = 128                   # sphere channels
num_distance_basis = 128                # number of gaussian basis functions used to expand the edge distance
hidden_dim = 128
cutoff = 6.0*2                          # Cutoff used for edge distance embedding
num_mp_layers = 2
model_name = 'esen'
restart = True
output_folder = 'outputs_nablaDFT'
model_filename = 'model.pt.pt'

rcut = 5.0                                            # connectivity cutoff

# -> Training settings:
train_or_eval = "eval"
num_val = 1#0                           # Number of validation structures per GPU
num_train = 1000                       # Number of training structures per GPU
num_epochs = 10000
lr_init = 5e-4
dtype = torch.float32
batch_size = 1#50 
loss_target = 'fock_matrix'
patience = 100                          # for scheduler
threshold = 1e-5                        # for scheduler
loss_fxn = utils_training.l1_padded_loss
# loss_fxn = utils_training.mse_padded_loss

# ----------------------------
# --> Initialize compute setup
# ----------------------------
if 'SLURM_JOB_ID' in os.environ:
    print("Initializing with slurm...")
    device = torch.device('cuda')         
    world_size = int(os.environ['SLURM_NTASKS'])
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    gpu_id = 0                              # visibility is restricted to 0 in .sh file
    torch.cuda.set_device(gpu_id)


if rank == 0 and not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ----------------------------
# --> Prepare data
# --------------------------------------------
max_mol = 500000    # maximum number of molecules in the dataset (change this later)

# sample randomly from the dataset:
total_num_molecules = (num_val + num_train)*world_size

if rank == 0:
    random_indices = random.sample(range(total_num_molecules), min(max_mol, total_num_molecules))
    random_indices_tensor = torch.tensor(random_indices, dtype=torch.long)
else:
    random_indices_tensor = torch.empty(min(max_mol, total_num_molecules), dtype=torch.long)
dist.broadcast(random_indices_tensor, src=0)
random_indices = random_indices_tensor.tolist()

print("random_indices: ", random_indices)

local_num_molecules = total_num_molecules//world_size
counts = np.array([local_num_molecules]*world_size, dtype=np.int32)
for i in range(total_num_molecules % world_size):
    counts[i] += 1

displacements = np.zeros_like(counts)
for i in range(1, len(counts)):
    displacements[i] = displacements[i-1] + counts[i-1]

molecule_start_idx = displacements[rank]
molecule_end_idx = displacements[rank] + counts[rank]
local_num_molecules = counts[rank]
print(f"Processing {total_num_molecules} structures between {world_size} GPUs")
print(f"Rank {rank} does indicies {molecule_start_idx} to {molecule_end_idx}")

orbital_basis = {35: [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], 
                 17: [0, 0, 0, 0, 1, 1, 1, 2], 
                 16: [0, 0, 0, 0, 1, 1, 1, 2], 
                 9: [0, 0, 0, 1, 1, 2], 
                 8: [0, 0, 0, 1, 1, 2], 
                 7: [0, 0, 0, 1, 1, 2], 
                 6: [0, 0, 0, 1, 1, 2], 
                 1: [0, 0, 1]}
# print([database.get_orbitals(x) for x in orbital_basis.keys()])

# check if this is needed (i think can remove)
target_len = 0
for l in range(5):
    max_l_multiplicity = np.max([orbital_basis[el].count(l) for el in orbital_basis])
    target_len += (2*l + 1) * max_l_multiplicity

data_load_start = time.perf_counter()
datalist = []
for i in random_indices[molecule_start_idx:molecule_end_idx]:

    print(f"Rank {rank} of {world_size} is working on molecule {i}", flush=True)

    # atoms numbers, atoms positions, energy, forces, core hamiltonian, overlap matrix, coefficients matrix,
    # moses_id, conformation_id
    Z, R, E, F, H, S, C, moses_id, conformation_id = database[i]

    mol_atoms = Atoms(symbols=Z, positions=R)
    num_atoms = len(Z)
    energy = E
    forces = F

    # Electronic structure matrix:
    hamiltonian = H                                       # note that this Hamiltonian is already rotated into the complex spherical harmonic basis
    atomic_numbers = Z
    element_strings = mol_atoms.get_chemical_symbols()
    # hamiltonian = transform(hamiltonian_og, element_strings, convention="psi4")

    time_start = time.perf_counter()
    graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian, target_len=target_len)
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

required_irreps = graph_targets.req_output_irreps                                                   # all the graphs have the same required Irreps
print("required irreps: ", required_irreps)

# train_size = len(datalist) - num_val
print("Rank ", rank, "datalist size ", len(datalist))
train_datalist, val_datalist = torch.utils.data.random_split(datalist, [num_train, num_val])

train_dataset = sampleDataset(train_datalist)
val_dataset = sampleDataset(val_datalist)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

data_load_end = time.perf_counter()
print("Total time to load dataset: ", data_load_end - data_load_start)

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
    print("Restarting model from :", restart_file)
    checkpoint = torch.load(restart_file)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # DDP saves it with a module prefix
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)


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


dist.destroy_process_group()