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
# from torch.cuda.amp import autocast, GradScaler

import torch
import torch.nn as nn
import torch.distributed as dist

from ASEDataset import ASEDataset, ASEAtomsData, sampleDataset, sample_collate_fn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import random

from esen.esen import eSEN_Backbone
from e3nn.o3 import Irreps, rand_matrix
import utils_training
import_end = time.perf_counter()
print("Time to do imports: ", import_end - import_start)

# read orca output, ase gives you the energy and forces!

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

# -> Model settings:
l_embedding_dim = 128
num_distance_basis = 128               # number of gaussian basis functions used to expand the edge distance
hidden_dim = 128
cutoff = 6.0*2                         # Cutoff used for edge distance embedding
is_pbc = False
num_mp_layers = 2  

# -> Training settings:
num_val = 1  # Number of validation structures

# --> Compute env
if torch.cuda.is_available():
    device = torch.device('cuda')         
else:
    device = torch.device('cpu')
world_size = int(os.environ['SLURM_NTASKS'])
rank = int(os.environ['SLURM_PROCID'])
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
torch.cuda.set_device(0) # visibility is restricted to 0 in .sh file

# Prepare model
# --------------------------------------------

data_load_start = time.perf_counter()
num_molecules = num_val*2
datalist = []
max_mol = 5000 

for i in range(num_molecules):
    mol = database.__getitem__(i)
    mol_atoms = Atoms(symbols=mol['_atomic_numbers'].numpy(), positions=mol['_positions'].numpy())
    rcut = 100.0                                            # connectivity cutoff
    num_atoms = len(mol['_positions'])

    # Connectivity list:
    neighbours = NeighborList(np.ones(num_atoms)*rcut, skin=0, self_interaction=False, bothways=True)
    neighbours.update(mol_atoms)
    neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
    neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])

    # Electronic structure matrix:
    hamiltonian = mol['hamiltonian'].numpy()                
    # orbital_basis = {8: [0, 0, 0, 1, 1, 2], 1: [0, 0, 1]}
    orbital_basis = {1: [0, 0, 1], 8: [0, 0, 0, 1, 1, 2]}
    atomic_numbers = mol['_atomic_numbers'].numpy()
    hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)  

    time_start = time.perf_counter()
    graph_targets = fock_targets.Fock_Targets(mol_atoms, neighbour_list, orbital_basis, hamiltonian)
    time_end = time.perf_counter()
    print("time to make targets: ", time_end - time_start)

    # Create PyTorch Geometric Data object - Note that fock_matrix has shape [num_mol*N, N] insted of [num_mol, N, N]
    # data = gnnData(
    #     pos=torch.tensor(graph_targets.atoms.positions, dtype=torch.float),
    #     edge_index=neighbour_list,
    #     x=graph_targets.node_labels.cpu(),
    #     edge_attr=graph_targets.edge_labels.cpu(),
    #     edge_dist=graph_targets.edge_dist.cpu(),
    #     atomic_numbers=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long).cpu(),  
    #     nedges=len(graph_targets.neighbour_list[0]), 
    #     natoms=len(atomic_numbers),  
    # )
    # datalist.append(data)
    data = gnnData(
                    pos=torch.tensor(graph_targets.atoms.positions, dtype=torch.float),
                    x=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long), 
                    edge_index=torch.tensor(neighbour_list).to(device),
                    edge_attr=graph_targets.edge_dist.to(device),
                    y=graph_targets.edge_labels,
                    node_y=graph_targets.node_labels,
                    atomic_numbers=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long).cpu(),  
                )
    datalist.append(data)

required_irreps = graph_targets.req_output_irreps
print("required irreps: ", required_irreps)

train_size = len(datalist) - num_val
train_datalist, val_datalist = torch.utils.data.random_split(datalist, [train_size, num_val])
train_dataset = sampleDataset(train_datalist)
val_dataset = sampleDataset(val_datalist)

# Do not use a batch size higher than 1!!
train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=sample_collate_fn, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=sample_collate_fn, shuffle=False, num_workers=0)
data_load_end = time.perf_counter()

print("Time to load dataset: ", data_load_end - data_load_start)

# --> Set up model:
model_setup_start = time.perf_counter()
model = eSEN_Backbone(
    required_irreps,
    sphere_channels=l_embedding_dim,
    hidden_channels=hidden_dim,
    lmax=required_irreps.lmax,
    mmax=required_irreps.lmax,
    use_pbc=is_pbc,
    cutoff=cutoff,
    edge_channels=l_embedding_dim,
    num_layers=num_mp_layers,
    act_type='gate',
    mlp_type = 'spectral',
    num_distance_basis=num_distance_basis
)
model = model.to(device)

restart_file = "outputs/model.pt.pt"
checkpoint = torch.load(restart_file)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
model_setup_end = time.perf_counter()
print("Time to setup model: ", model_setup_end - model_setup_start)

# Rotation utilities:
# -------------------

alpha=180.0 #230.0
beta=0.0 #70.0
gamma= 0.0 #180.0
# alpha=0.0
# beta=0.0
# gamma=0.0

# Cartesian Rotation for the mol:
alpha_rad = torch.deg2rad(torch.tensor(alpha))
beta_rad = torch.deg2rad(torch.tensor(beta))
gamma_rad = torch.deg2rad(torch.tensor(gamma))
Rx = torch.tensor([[1, 0, 0],
                   [0, torch.cos(alpha_rad), -torch.sin(alpha_rad)],
                   [0, torch.sin(alpha_rad), torch.cos(alpha_rad)]])
Ry = torch.tensor([[torch.cos(beta_rad), 0, torch.sin(beta_rad)],
                   [0, 1, 0],
                   [-torch.sin(beta_rad), 0, torch.cos(beta_rad)]])
Rz = torch.tensor([[torch.cos(gamma_rad), -torch.sin(gamma_rad), 0],
                   [torch.sin(gamma_rad), torch.cos(gamma_rad), 0],
                   [0, 0, 1]])
R_cart = torch.matmul(Rz, torch.matmul(Ry, Rx))

# R_cart = rand_matrix() # o3 rand matrix
assert torch.allclose(R_cart.transpose(0, 1), torch.inverse(R_cart)), "R_cart is not orthogonal"
assert torch.allclose(torch.det(R_cart), torch.tensor(1.0)), "R_cart has determinant != 1"

# Spherical Rotation for Irreps:
R_sphere = required_irreps.D_from_matrix(R_cart).to(device)
# assert torch.allclose(R_sphere.conj().transpose(0, 1), torch.inverse(R_sphere)), "R_sphere is not orthogonal"
# assert torch.allclose(torch.abs(R_sphere).pow(2).sum(dim=1), torch.tensor(1.0)), "R_sphere has determinant != 1"

print("R_sphere[3:6, 3:6]: ", R_sphere[0:9, 0:9])
print("R_cart: ", R_cart)


# Commutator test:
# ----------------

# Get a water mol (the one that this network was over-fit on):
batch = train_loader.dataset[0]
batch = batch.to(device)

# -- Get the rotated molecule
original_positions = batch.pos
rotated_positions = torch.zeros_like(batch.pos).to(device)
for i in range(len(batch.pos)):
    rotated_positions[i] = R_cart.to(device) @ batch.pos[i]
batch.pos = rotated_positions

# --> Forward Commutator: Rotate the molecule (use rotated_positions):
print("rotated positions of mol: ", batch.pos)
model.eval()
with torch.no_grad():
    node_output, edge_output = model(batch) 
node_from_forward = node_output[0].detach().cpu()

# --> Backward Commutator: Rotate the output 
batch.pos = original_positions
print("original positions of mol: ", batch.pos)
model.eval()
with torch.no_grad():
    node_output, edge_output = model(batch) 
node_from_backward = (R_sphere.T @ node_output[0]).detach().cpu()

# R_cart_180 = torch.tensor([[0.0, -1.0, 0.0],
#                         [ 1.0, 0.0, 0.0],
#                         [ 0.0, 0.0, 1.0]]).inverse()
# R_sphere_180 = required_irreps.D_from_matrix(R_cart_180).to("cuda:0")
# node_from_backward = (R_sphere_180 @ (R_sphere @ node_from_backward)).detach().cpu()

print("Forward...: ", node_from_forward[:30])
print("Backward...: ", node_from_backward[:30])

plt.imshow(np.abs(node_from_forward.detach().reshape(14, 14)), vmin=0, vmax=1)
plt.colorbar()
plt.savefig('rotated_molecule.png', dpi=300, bbox_inches='tight')
plt.close()


plt.imshow(np.abs(node_from_backward.detach().reshape(14, 14)), vmin=0, vmax=1)
plt.colorbar()
plt.savefig('rotated_output_tensor.png', dpi=300, bbox_inches='tight')
plt.close()

err_matrix = np.abs(node_from_forward.detach().reshape(14, 14) - node_from_backward.detach().reshape(14, 14))
# gt_matrix = np.abs(node_from_forward.detach().reshape(14, 14))
# percentage_err_matrix = (err_matrix / gt_matrix) * 100
plt.imshow(err_matrix) 
plt.colorbar()
plt.savefig('err_percentage.png', dpi=300, bbox_inches='tight')
plt.close()




# # Commutator test:
# # ----------------

# # Get a water mol (the one that this network was over-fit on):
# batch = train_loader.dataset[0]

# # 1. Rotate the molecule
# positions = batch.pos
# print("original positions of mol: ", positions)
# # print("permuted positions: ", positions.permute(1, 0))
# # batch.pos = torch.matmul(positions, R_cart)
# for i in range(len(batch.pos)):
#     batch.pos[i] = batch.pos[i] @ R_cart.T
# # batch.pos = (R_cart @ postions.permute(1, 0)).permute(1, 0)

# # 2. Rotate the irreps:
# # first_edge_rotated_GT = torch.matmul(R_sphere, batch.x[0]) # maybe this is being modified?
# first_edge_rotated_GT = batch.x[0] @ R_sphere.T
# plt.imshow(np.abs(first_edge_rotated_GT.detach().reshape(14, 14)), vmin=0, vmax=1)
# plt.colorbar()
# plt.savefig('first_edge_rotated_GT.png', dpi=300, bbox_inches='tight')
# plt.close()

# # 3. Get the network output:
# batch = batch.to(device)
# data_dict = {
#                 "pos": batch.pos,
#                 "atomic_numbers": batch.atomic_numbers,
#                 "edge_index": torch.tensor(batch.edge_index, dtype=torch.long, device=device).squeeze(0).reshape(2, -1),
#                 "edge_dist": batch.edge_dist,
#                 "nedges": 6,
#                 "natoms": 3,
#             } # assumes that edge_dist is computed within the network!

# print("rotated positions of mol: ", data_dict["pos"])

# model.eval()
# with torch.no_grad():
#     output = model(data_dict) 

# R_cart_180 = torch.tensor([[0.0, -1.0, 0.0],
#                         [ 1.0, 0.0, 0.0],
#                         [ 0.0, 0.0, 1.0]]).inverse()
# R_sphere_180 = required_irreps.D_from_matrix(R_cart_180).to("cuda:0")
# # node_from_backward = (R_sphere_180 @ (R_sphere @ node_from_backward)).detach().cpu()
# first_edge_rotated_model = (R_sphere_180 @ output['node_embedding'][0]).cpu()
# plt.imshow(np.abs(first_edge_rotated_model.detach().reshape(14, 14)), vmin=0, vmax=1)
# plt.colorbar()
# plt.savefig('first_edge_rotated_model.png', dpi=300, bbox_inches='tight')
# plt.close()

# print("first_edge_rotated_GT: ", first_edge_rotated_GT)
# print("first_edge_rotated_model: ", first_edge_rotated_model)


# # Difference:
# # plt.imshow(np.abs(first_edge_rotated_model.detach().reshape(14, 14) - first_edge_rotated_GT.detach().reshape(14, 14)), vmin=0, vmax=0.5)
# err_matrix = np.abs(first_edge_rotated_model.detach().reshape(14, 14) - first_edge_rotated_GT.detach().reshape(14, 14))
# gt_matrix = np.abs(first_edge_rotated_GT.detach().reshape(14, 14))
# percentage_err_matrix = (err_matrix / gt_matrix) * 100
# plt.imshow(percentage_err_matrix, vmin=0, vmax=100) 
# plt.colorbar()
# plt.savefig('equivariance_err_percentage.png', dpi=300, bbox_inches='tight')
# plt.close()