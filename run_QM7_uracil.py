import time
import_start = time.perf_counter()
import os, sys, random
import numpy as np
import torch

from fock_utils import utils_orca_out, fock_targets
from train_utils import loss, utils_compute, splittrainer
from dataset_utils import get_loader
from dataset_utils.ASEDataset import ASEAtomsData
from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

# Models
from equiformer.network import SO2Net
from equiformer.SO3 import CoefficientMappingModule
from esen.esen_new import eSEN_Backbone, Fock_Irreps_Head, Linear_Force_Head     # NO EDGES: .esen_noedges
from e3nn.o3 import Irreps

import_end = time.perf_counter()
print("Time to do imports: ", import_end - import_start)

# Fix random seeds for distributed model initialization
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------------------------------
# Settings (just dumping everything here for now)
# -----------------------------------------------
# ---------------------------
# --> QM7
dbpath = 'fock_datasets/QM7/schnorb_hamiltonian_uracil.db'
database = ASEAtomsData(dbpath)
dataset_name = 'QM7'
output_folder = 'outputs_QM7_uracil_nosym'
# ---------------------------
# ---------------------------
# --> NablaDFT (tiny)
# original_database = HamiltonianDatabase("./fock_datasets/nabla2_DFT/train_2k.db")
# dataset_name = 'nablaDFT'
# output_folder = 'outputs_nablaDFT'
# ---------------------------

# --> Model settings:
l_embedding_dim = 128                   # sphere channels
num_distance_basis = l_embedding_dim    # number of gaussian basis functions used to expand the edge distance
hidden_dim = l_embedding_dim
num_mp_layers = 2
model_name = 'esen'
restart_backbone = True
restart_head = True
restart_optimizer = True

# --> Training settings:
train_or_eval = "eval"
num_val = 500                           # Number of validation structures
num_train = 25000 #25000 
num_test = 4500
num_epochs = 50000
batch_size = 10                         # for training (batch size is always 1 for eval)
rcut_orbitals = 8.0                     # connectivity cutoff (=2xrcut)
rcut_gaussian = 10.0                    # connectivity cutoff (=2xrcut)
gaussian_width = 1.0                    # width of gaussians used to expand edge distance
reflection_symmetry=False                # use only edges i,j where i<j

train_backbone = True
train_head = True

dtype = torch.float64
torch.set_default_dtype(dtype)
lr_init = 1e-3
patience = 20                          # for scheduler
threshold = 1e-8                        # for scheduler

loss_target = 'fock_matrix'
train_loss_fxn = loss.combined_padded_loss
test_loss_fxn = loss.l1_padded_loss
loss_scheduler = loss.MonotonicDecreaseScheduler
backbone_checkpoint = 'backbone.pt'
head_checkpoint = 'head.pt'

# --------------------------------------------
# Initialize compute environment 
# --------------------------------------------

rank = int(os.environ['SLURM_PROCID'])
world_size = int(os.environ['SLURM_NTASKS'])

if rank == 0:
    print(f"Dataset: {dataset_name}, writing results to {output_folder}", flush=True)
    print(f"Dataset - Num molecules used for training: {num_train}", flush=True)
    print(f"Dataset - Num molecules used for validation: {num_val}", flush=True)
    print(f"Dataset - Edge cutoff distance for orbital blocks: {2*rcut_orbitals}", flush=True)
    print(f"Dataset - Edge cutoff distance for gaussian basis: {rcut_gaussian}", flush=True)
    print(f"Model - Num of Message Passing layers: {num_mp_layers}", flush=True)
    print(f"Model - Embedding dimension: {l_embedding_dim}", flush=True)
    print(f"Model - Edge reflections: {reflection_symmetry}")
    print(f"Training - Loss target: {loss_target}", flush=True)
    print(f"Training - Loss function for training: {train_loss_fxn}", flush=True)
    print(f"Training - Initial learning rate: {lr_init}", flush=True)

compute_start = time.perf_counter()
device = utils_compute.setup_env(rank, world_size)
compute_end = time.perf_counter()
print("Time to setup distributed environment: ", compute_end - compute_start)

if rank == 0 and not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --------------------------------------------
# Prepare data
# --------------------------------------------

data_load_start = time.perf_counter()

train_start_mol, train_end_mol, train_local_num_mol = utils_compute.split_indices(rank, world_size, num_train)
val_start_mol, val_end_mol, val_local_num_mol  = utils_compute.split_indices(rank, world_size, num_val)
test_start_mol, test_end_mol, test_local_num_mol = utils_compute.split_indices(rank, world_size, num_test)

val_start_mol += num_train  # the validation molecules start after training ones
val_end_mol += num_train

test_start_mol += num_train+num_val
test_end_mol += num_train+num_val

### DEBUG ###
# train_start_mol = 0
# train_end_mol = 1
# val_start_mol = 0
# val_end_mol = 1
### DEBUG ###

if train_or_eval == 'train':
    train_loader, required_irreps, basis_transformation = get_loader.get_loader(database, train_start_mol, train_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, reflection_symmetry=reflection_symmetry)
    val_loader, _, _ = get_loader.get_loader(database, val_start_mol, val_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, reflection_symmetry=reflection_symmetry)
    print("Size of train loader: ", len(train_loader))
    print("Size of val loader: ", len(val_loader))

else:
    batch_size = 1
    test_loader, required_irreps, basis_transformation = get_loader.get_loader(database, test_start_mol, test_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, reflection_symmetry=reflection_symmetry)
    print("Size of test loader: ", len(test_loader))

data_load_end = time.perf_counter()
print("Time to load dataset: ", data_load_end - data_load_start)

irreps_in = Irreps([(l_embedding_dim, (l, 1)) for l in range(required_irreps.lmax + 1)]) 

# determine output irreps from target type:
if loss_target == 'fock_matrix':
    output_irreps = required_irreps
    node_target = 'node_y'
    edge_target = 'y'
elif loss_target == 'forces':
    output_irreps = '1x1e'
    node_target = 'forces'
    edge_target = None
else:
    output_irreps = '1x0e'
    node_target = 'energy'
    edge_target = None


# --------------------------------------------
# Get model
# --------------------------------------------
if model_name == 'equiformer':
    mappingReduced = CoefficientMappingModule(required_irreps.lmax, required_irreps.lmax)
    edge_channels_list = [l_embedding_dim, l_embedding_dim, l_embedding_dim]

    attn_hidden_channels = 128 
    attn_alpha_channels = 32
    attn_value_channels = 32 
    ffn_hidden_channels = 64 
    num_heads=2

    backbone = SO2Net(num_mp_layers, 
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
    backbone = eSEN_Backbone(
                    required_irreps,
                    sphere_channels=l_embedding_dim,
                    hidden_channels=hidden_dim,
                    lmax=required_irreps.lmax,
                    mmax=required_irreps.lmax,
                    use_pbc=False,
                    cutoff=rcut_gaussian,
                    edge_channels=l_embedding_dim,
                    num_layers=num_mp_layers,
                    act_type='gate',
                    mlp_type = 'spectral',
                    num_distance_basis=num_distance_basis,
                    gaussian_width=gaussian_width
                )

    if loss_target == "fock_matrix":
        head = Fock_Irreps_Head(irreps_in=irreps_in, 
                                irreps_out=output_irreps, 
                                lmax=required_irreps.lmax, 
                                sphere_channels=l_embedding_dim)
    elif loss_target == "forces":
        head = Linear_Force_Head(backbone)

    elif loss_target == "energy":
        print("To be implemented!")


backbone = backbone.to(device)
head = head.to(device)

if train_backbone and train_head:
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr_init)

elif train_head:                            # freeze backbone model
    for param in backbone.parameters(): 
        param.requires_grad = False
    optimizer = torch.optim.Adam(head.parameters(), lr=lr_init)

elif train_backbone:                        # freeze output head
    for param in head.parameters():     
        param.requires_grad = False
    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr_init)

else:
    print("Running evaluation")

print("Number of parameters in backbone: ", sum(p.numel() for p in backbone.parameters()))
print("Number of parameters in output head: ", sum(p.numel() for p in head.parameters()))

if restart_backbone:
    restart_file = output_folder + '/' + backbone_checkpoint
    print("Restarting backbone model from :", restart_file)
    checkpoint = torch.load(restart_file)
    state_dict = checkpoint['model_state_dict']
    
    # Get rid of module prefix if saved with DDP
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  
        new_state_dict[new_key] = value
    backbone.load_state_dict(new_state_dict)

if restart_head:
    restart_file = output_folder + '/' + head_checkpoint
    print("Restarting output head model from :", restart_file)
    checkpoint = torch.load(restart_file)
    state_dict = checkpoint['model_state_dict']

    if restart_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get rid of module prefix if saved with DDP
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  
        new_state_dict[new_key] = value
    head.load_state_dict(new_state_dict)
    

# --------------------------------------------
# Run Training or Evaluation
# --------------------------------------------

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
scheduler = loss_scheduler(optimizer)
trainer = splittrainer.SplitTrainer(backbone=backbone, 
                                    head=head,
                                    head_irreps=output_irreps,
                                    run_name='QM7_uracil_withsym',
                                    save_frequency=20)
if train_or_eval == "train":
    trainer.train(num_epochs, 
                    train_loss_fxn, 
                    optimizer,
                    scheduler, 
                    device,
                    train_loader=train_loader,
                    loss_target_string=loss_target,
                    node_target_name=node_target, 
                    edge_target_name=edge_target,
                    output_folder=output_folder,
                    val_loader=val_loader,
                    train_backbone=train_backbone,
                    train_head=train_head,
                    basis_transform=basis_transformation)
else:
    trainer.evaluate(test_loss_fxn,
                    device,
                    test_loader,
                    loss_target_string=loss_target,
                    node_target_name=node_target,
                    edge_target_name=edge_target, 
                    basis_transform=basis_transformation,
                    output_folder=output_folder,
                    )
        