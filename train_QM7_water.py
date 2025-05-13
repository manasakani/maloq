import time
import_start = time.perf_counter()
import os, sys, random
import numpy as np
import torch

from fock_utils import utils_orca_out, fock_targets
from train_utils import utils_training, utils_compute
from dataset_utils import get_loader
from dataset_utils.ASEDataset import ASEAtomsData

# Models
from equiformer.network import SO2Net
from equiformer.SO3 import CoefficientMappingModule
from esen.esen import eSEN_Backbone                         # NO EDGES OPTION: .esen_noedges
from e3nn.o3 import Irreps

import_end = time.perf_counter()
print("Time to do imports: ", import_end - import_start)

# Fix random seeds for testing
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------------------------------
# Settings (just dumping everything here for now)
# -----------------------------------------------
dbpath = 'fock_datasets/schnorb_hamiltonian_water.db'
database = ASEAtomsData(dbpath)
dataset_name = 'QM7'

# -> Model settings:
l_embedding_dim = 128                   # sphere channels
num_distance_basis = 128                # number of gaussian basis functions used to expand the edge distance
hidden_dim = 128
cutoff = 6.0*2                          # Cutoff used for edge distance embedding
num_mp_layers = 2 
model_name = 'esen'
restart = False
output_folder = 'outputs_QM7'
model_filename = 'model.pt.pt'

# -> Training settings:
train_or_eval = "train"
num_val = 10 #500                       # Number of validation structures
num_train = 10 #500
num_epochs = 5000
batch_size = 10                         # 1 for eval, 10 for train
rcut = 5.0                              # connectivity cutoff (=2xrcut)

dtype = torch.float32
lr_init = 1e-4
patience = 100                          # for scheduler
threshold = 1e-5                        # for scheduler

loss_target = 'fock_matrix'
loss_fxn = utils_training.mse_padded_loss

# --------------------------------------------
# Initialize compute environment 
# --------------------------------------------

rank = int(os.environ['SLURM_PROCID'])
world_size = int(os.environ['SLURM_NTASKS'])

print("I'm rank ", rank, "of ", world_size)
device = utils_compute.setup_env(rank, world_size)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --------------------------------------------
# Prepare data
# --------------------------------------------
data_load_start = time.perf_counter()

train_start_mol, train_end_mol, train_local_num_mol = utils_compute.split_indices(rank, world_size, num_train)
val_start_mol, val_end_mol, val_local_num_mol  = utils_compute.split_indices(rank, world_size, num_val)

val_start_mol += num_train  # the validation molecules start after training ones
val_end_mol += num_train

train_loader, required_irreps = get_loader.get_loader(database, train_start_mol, train_end_mol, dataset_name, rcut, batch_size)
val_loader, _ = get_loader.get_loader(database, val_start_mol, val_end_mol, dataset_name, rcut, batch_size)

data_load_end = time.perf_counter()
print("Time to load dataset: ", data_load_end - data_load_start)

# --------------------------------------------
# Get model
# --------------------------------------------
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

# --------------------------------------------
# Run Training or Evaluation
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