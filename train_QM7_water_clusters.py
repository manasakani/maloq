import time
import_start = time.perf_counter()
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase import Atoms
from ase.neighborlist import NeighborList
import ase.db
import utils_orca_out, fock_targets

import torch
import torch.nn as nn
import torch.distributed as dist

from ASEDataset import ASEDataset, ASEAtomsData, sampleDataset, sample_collate_fn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset, DataLoader

from esen.esen import eSEN_Backbone
from e3nn.o3 import Irreps
import utils_training
import_end = time.perf_counter()
print("Time to do imports: ", import_end - import_start)

# read orca output, ase gives you the energy and forces!

# Fix this later:
sys.path.append('/home/manasakani/fairchem/src/')

# Settings (just dumping everything here for now)
# -----------------------------------------------
dbpath = 'schnorb_hamiltonian_water.db'
database = ASEAtomsData(dbpath)

# -> Model settings:
dataset_folder = './fock_datasets/water_clusters_rcut_6.0_16x.db'
l_embedding_dim = 32
num_distance_basis = 32                # number of gaussian basis functions used to expand the edge distance
hidden_dim = 32
cutoff = 6.0*2                         # Cutoff used for edge distance embedding
is_pbc = False
num_mp_layers = 2

# -> Training settings:
num_epochs = 1000
lr_init = 1e-3
dtype = torch.float32
num_val = 10  # Number of validation structures

# --> Compute env
if torch.cuda.is_available():
    device = torch.device('cuda')         
else:
    device = torch.device('cpu')
world_size = int(os.environ['SLURM_NTASKS'])
rank = int(os.environ['SLURM_PROCID'])
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
torch.cuda.set_device(0) # visibility is restricted to 0 in .sh file

output_folder = 'outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Prepare model
# --------------------------------------------

data_load_start = time.perf_counter()
num_molecules = 2*num_val
datalist = []
atom_count = 0

for i in range(num_molecules):
    mol = database.__getitem__(i)
    mol_atoms = Atoms(symbols=mol['_atomic_numbers'].numpy(), positions=mol['_positions'].numpy())
    rcut = 6.0                                            # connectivity cutoff
    num_atoms = len(mol['_positions'])

    # Connectivity list:
    neighbours = NeighborList(np.ones(num_atoms)*rcut, skin=0, self_interaction=False, bothways=True)
    neighbours.update(mol_atoms)
    neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
    neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])

    # Electronic structure matrix:
    hamiltonian = mol['hamiltonian'].numpy()                
    orbital_basis = {8: [0, 0, 0, 1, 1, 2], 1: [0, 0, 1]}
    atomic_numbers = mol['_atomic_numbers'].numpy()
    hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)

    time_start = time.perf_counter()
    graph_targets = fock_targets.Fock_Targets(mol_atoms, neighbour_list, orbital_basis, hamiltonian)
    time_end = time.perf_counter()
    print("time to make targets: ", time_end - time_start)

    # Create PyTorch Geometric Data object - Note that fock_matrix has shape [num_mol*N, N] insted of [num_mol, N, N]
    data = gnnData(
        pos=torch.tensor(graph_targets.atoms.positions, dtype=torch.float),
        edge_index=neighbour_list,
        x=graph_targets.node_labels.cpu(),
        edge_attr=graph_targets.edge_labels.cpu(),
        edge_dist=graph_targets.edge_dist.cpu(),
        atomic_numbers=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long).cpu(),  
        nedges=len(graph_targets.neighbour_list[0]), 
        natoms=len(atomic_numbers),  
    )
    
    datalist.append(data)
    atom_count += len(atomic_numbers)

required_irreps = graph_targets.req_output_irreps
lmax = required_irreps.lmax    
print("required irreps: ", required_irreps)

train_size = len(datalist) - num_val
train_datalist, val_datalist = torch.utils.data.random_split(datalist, [train_size, num_val])
train_dataset = sampleDataset(train_datalist)
val_dataset = sampleDataset(val_datalist)

# Do not use a batch size higher than 1!!
train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=sample_collate_fn, shuffle=False, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=sample_collate_fn, shuffle=False, num_workers=1)
data_load_end = time.perf_counter()

print("Time to load dataset: ", data_load_end - data_load_start)

# --> Set up model:
model_setup_start = time.perf_counter()
model = eSEN_Backbone(
    required_irreps,
    sphere_channels=l_embedding_dim,
    hidden_channels=hidden_dim,
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
track_loss_edge = []
track_loss_node_val = []
track_loss_edge_val = []

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, threshold=1e-4, verbose=True)

# --> Do training loop
for epoch in range(num_epochs):
    epoch_start = time.perf_counter()

    loss = 0

    model.train()  
    for batch in train_loader:

        optimizer.zero_grad()

        # print(torch.tensor(batch.edge_index, dtype=torch.long, device=device).squeeze(0).reshape(2, -1).shape)
        # exit()

        # Forward pass
        batch = batch.to(device)
        data_dict = {
            "pos": batch.pos,
            "atomic_numbers": batch.atomic_numbers,
            "edge_index": torch.tensor(batch.edge_index, dtype=torch.long, device=device).squeeze(0).reshape(2, -1),
            "x": batch.x,
            "edge_attr": batch.edge_attr,
            "edge_dist": batch.edge_dist,
            "atomic_numbers": batch.atomic_numbers,
            "nedges": sum(batch.nedges),
            "natoms": sum(batch.natoms),
        }
 
        output = model(data_dict) 

        # Loss
        loss_node = loss_fxn(output['node_embedding'], batch.x)
        loss_edge = loss_fxn(output['edge_embedding'], batch.edge_attr)
        loss = loss_node + loss_edge
        
    # Backwards
    loss.backward()
    optimizer.step()
        
    track_loss_node.append(loss_node.cpu().detach().numpy() / len(batch))
    track_loss_edge.append(loss_edge.cpu().detach().numpy() / len(batch))
    
    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            data_dict = {
                "pos": batch.pos,
                "atomic_numbers": batch.atomic_numbers,
                "edge_index": torch.tensor(batch.edge_index, dtype=torch.long, device=device).squeeze(0),
                "x": batch.x,
                "edge_attr": batch.edge_attr,
                "edge_dist": batch.edge_dist,
                "nedges": sum(batch.nedges),
                "natoms": sum(batch.natoms),
            }
            output = model(data_dict)
            loss_node = loss_fxn(output['node_embedding'], batch.x)
            loss_edge = loss_fxn(output['edge_embedding'], batch.edge_attr)
            loss = loss_node + loss_edge
            val_loss += loss.item()

    track_loss_node_val.append(loss_node.cpu().detach().numpy() / len(batch))
    track_loss_edge_val.append(loss_edge.cpu().detach().numpy() / len(batch))
    
    scheduler.step(loss)
    current_lr = optimizer.param_groups[0]['lr']
    print("current Lr: ", current_lr)
    print(f"Epoch {epoch+1}, Train Loss: {track_loss_node[-1]}")
    print(f"Epoch {epoch+1}, Val Loss: {track_loss_node_val[-1]}")

    epoch_end = time.perf_counter()
    print("Time per epoch: ", epoch_end - epoch_start)
    
    # Plot the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        utils_training.save_training_state(model, optimizer, track_loss_edge, track_loss_node, track_loss_edge_val, track_loss_node_val, 'model.pt')