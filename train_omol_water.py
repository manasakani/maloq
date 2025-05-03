import os, sys
import numpy as np
import random
import matplotlib.pyplot as plt
import ase.db

import torch
import torch.nn as nn
import torch.distributed as dist
import time

from ASEDataset import ASEDataset
from torch_geometric.loader import DataLoader
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

from equiformer.network import SO2Net
from equiformer.SO3 import CoefficientMappingModule
from esen.esen import eSEN_Backbone
from e3nn.o3 import Irreps
import utils_training

# Fix this later:
sys.path.append('/home/manasakani/fairchem/src/')
torch.manual_seed(42)                   # fixing random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Settings (just dump everything here for now)
# --------------------------------------------

# -> Model settings:
dataset_folder = './fock_datasets/water_clusters_small_flexible_x8.db'
l_embedding_dim = 64
num_distance_basis = 64                # number of gaussian basis functions used to expand the edge distance
hidden_dim = 64
cutoff = 6.0*2                         # Cutoff used for edge distance embedding
num_mp_layers = 2
model_name = "esen"
output_folder = 'outputs_omol'

# -> Training settings:
num_epochs = 5
lr_init = 1e-3
dtype = torch.float32
num_train = 7                           # Number of training structures
num_val = 1                             # Number of validation structures
batch_size = 1
loss_target = 'energy'
patience = 100                          # for scheduler
threshold = 1e-4                        # for scheduler
loss_fxn = utils_training.unpadded_loss

# --> Compute env
device = torch.device('cuda')         
world_size = int(os.environ['SLURM_NTASKS'])
rank = int(os.environ['SLURM_PROCID'])
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
torch.cuda.set_device(0) # visibility needs to be restricted to 0 in .sh file!

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Prepare dataset
# --------------------------------------------

data_load_start = time.perf_counter()
dataset = ASEDataset(dataset_folder, dtype=dtype)
required_irreps = Irreps(dataset[0].required_irreps)

assert len(dataset) >= num_train+num_val
subset_indices = np.random.choice(len(dataset), size=num_train+num_val, replace=False)
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
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
