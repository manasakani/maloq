import os, sys
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import ase.db

import torch
import torch.nn as nn
import torch.distributed as dist
import time

from ASEDataset import ASEDataset
from torch_geometric.loader import DataLoader
import torch.distributed as dist

from esen.esen import eSEN_Backbone
from e3nn.o3 import Irreps
import utils_training

# read orca output, ase gives you the energy and forces!

# Fix this later:
sys.path.append('/home/manasakani/fairchem/src/')

# Settings (just dump everything here for now)
# --------------------------------------------

# -> Model settings:
dataset_folder = './fock_datasets/water_clusters_rcut_6.0_16x.db'
l_embedding_dim = 128
lmax = required_irreps.lmax           
cutoff = 6.0*2                          # Cutoff used for edge distance embedding
is_pbc = False
num_mp_layers = 2
num_distance_basis = 128                # number of gaussian basis functions used to expand the edge distance

# -> Things to get from the database instead of doing this:
required_irreps = Irreps("62x0e+104x1e+100x2e+60x3e+24x4e+7x5e+1x6e")

# -> Training settings:
num_epochs = 1000
lr_init = 1e-3
dtype = torch.float32
num_val = 1  # Number of validation structures

# --> Compute env
if torch.cuda.is_available():
    device = torch.device('cuda')         
else:
    device = torch.device('cpu')
world_size = int(os.environ['SLURM_NTASKS'])
rank = int(os.environ['SLURM_PROCID'])
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
torch.cuda.set_device(0) # visibility needs to be restricted to 0 in .sh file!


# Prepare data and model
# --------------------------------------------

data_load_start = time.perf_counter()
dataset = ASEDataset(dataset_folder, dtype=dtype)
# train_size = len(dataset) - num_val
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, num_val])

train_dataset = torch.utils.data.Subset(dataset, [0])
val_dataset = torch.utils.data.Subset(dataset, [1])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
data_load_end = time.perf_counter()

print("Time to load dataset: ", data_load_end - data_load_start)

# --> Set up model:
model_setup_start = time.perf_counter()
model = eSEN_Backbone(
    required_irreps,
    sphere_channels=l_embedding_dim,
    lmax=lmax,
    mmax=lmax,
    use_pbc=is_pbc,
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
loss_fxn = nn.MSELoss(reduction='mean')

# // wrap model in distributed data parallel

track_loss_node = []
track_loss_node_val = []

# --> Do training loop
for epoch in range(num_epochs):
    epoch_start = time.perf_counter()

    model.train()  
    for batch in train_loader:

        optimizer.zero_grad()

        # Forward pass
        batch = batch.to(device)
        data_dict = {
            "pos": batch.pos,
            "atomic_numbers": batch.atomic_numbers,
            "edge_index": batch.edge_index,
            "x": batch.x,
            "edge_attr": batch.edge_attr,
            "edge_dist": batch.edge_dist,
            "fock_matrix": batch.fock_matrix,
            "atomic_numbers": batch.atomic_numbers,
            "nedges": batch.nedges,
            "natoms": batch.natoms,
        }
        node_output = model(data_dict)  # Add edge output!!!

        # Loss
        loss_node = loss_fxn(node_output['node_embedding'], batch.x)
        
        # Backwards
        loss_node.backward()
        optimizer.step()
        
    track_loss_node.append(loss_node.cpu().detach().numpy() / len(batch))
    
    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            data_dict = {
                "pos": batch.pos,
                "atomic_numbers": batch.atomic_numbers,
                "edge_index": batch.edge_index,
                "x": batch.x,
                "edge_attr": batch.edge_attr,
                "edge_dist": batch.edge_dist,
                "fock_matrix": batch.fock_matrix,
                "nedges": batch.nedges,
                "natoms": batch.natoms,
            }
            node_output = model(data_dict)
            loss_node = loss_fxn(node_output['node_embedding'], batch.x)
            val_loss += loss_node.item()
    track_loss_node_val.append(loss_node.cpu().detach().numpy() / len(batch))

    print(f"Epoch {epoch+1}, Train Loss: {track_loss_node[-1]}")
    print(f"Epoch {epoch+1}, Val Loss: {track_loss_node_val[-1]}")

    epoch_end = time.perf_counter()
    print("Time per epoch: ", epoch_end - epoch_start)
    
    # Plot the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        utils_training.save_training_state(model, optimizer, track_loss_node, track_loss_node_val, 'model.pt')