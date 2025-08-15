import time
import_start = time.perf_counter()
import os, sys, random
import numpy as np
import torch
from e3nn.o3 import Irreps

from fock_utils import utils_orca_out, fock_targets
from train_utils import loss, utils_compute, splittrainer
from dataset_utils import get_loader, get_scale_shift, dataset_analysis
from dataset_utils.ASEDataset import ASEAtomsData
from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

# Models
from esen_full.esen_new import eSEN_Backbone, Fock_Irreps_Head, Linear_Energy_Head

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
dbpath = 'fock_datasets/QM7/schnorb_hamiltonian_water.db'
database = ASEAtomsData(dbpath)
dataset_name = 'QM7'
output_folder = 'outputs_QM7_water_halfedge_nodereduce'
# ---------------------------

# --> Shuffle:
# print("Not shuffling database, using the first molecule only for debugging")
print("Shuffling database...")
indices = list(range(len(database)))
random.shuffle(indices)
database = [database[i] for i in indices]

# --> Model settings:
l_embedding_dim = 128                   # sphere channels
num_distance_basis = 128                # number of gaussian basis functions used to expand the edge distance
hidden_dim = l_embedding_dim
num_mp_layers = 3
restart_backbone = False
restart_head = False
restart_optimizer = False

# --> Training settings:
train_or_eval = "train"
num_val = 500                           # Number of validation structures
num_train = 500
num_test = len(database) - num_train - num_val  # Number of test structures
num_epochs = 5000
batch_size = 1                          # 1 for eval, 10 for train
rcut_orbitals = 8.0                     # connectivity cutoff (=2xrcut)
rcut_gaussian = rcut_orbitals*2         # connectivity cutoff (=2xrcut)
gaussian_width = 1.0                    # width of gaussians used to expand edge distance

# Symmetry reduction settings:
reduce_edge = True                      # use only edge orbital blocks for edge i,j where i<j as labels (edges will be symmetrized in the output head)
reduce_node = True                     # inter-orbital forward/backward interactions are enforced to be equal
reduce_node_intra = True               # intra-orbital interactions are enforced to have 0 odd degrees

train_backbone = True
train_head = True

dtype = torch.float64
torch.set_default_dtype(dtype)
lr_init = 1e-5
patience = 500                          # if ReduceLROnPlateau scheduler
threshold = 1e-5                        # if ReduceLROnPlateau scheduler
scheduler_type = 'cosine'               # 'plateau', 'cosine'
T_max = num_epochs                      # for cosine scheduler - period of cosine annealing
eta_min = 1e-7                         # for cosine scheduler - minimum learning rate

loss_target = 'fock_matrix'
train_loss_fxn = loss.rmse_mse_padded_loss
test_loss_fxn = loss.l1_unpadded_loss
loss_scheduler = loss.MonotonicDecreaseScheduler
backbone_checkpoint = 'backbone.pt'
head_checkpoint = 'head.pt'
head_type = 'gated'                   # 'linear' or 'gated'
include_edges = True

if reduce_edge and batch_size != 1:
    raise ValueError("If using reduce_edge, batch size must be 1! Reverse_edge map is not collated.")

# dataset_analysis.dataset_analysis(database, dataset_name, rcut=rcut_orbitals, dtype=torch.float64, reduce_edge=False)

scale_and_shift = False
scale_shift_file = 'element_scale_shifts_water_' + dataset_name + '.pt'

# Scale and shift the orbital self-interaction scalar components of the dataset
if scale_and_shift:
    print("Getting scale and shift factors...")
    if scale_shift_file not in os.listdir('./fock_datasets'):
        print("[Computing element scale and shift factors for the dataset]")
        get_scale_shift.get_scale_shift(database, dataset_name, rcut_orbitals, dtype=dtype, reduce_edge=reduce_edge)
        print("Done computing scale and shift factors, saving to file:", scale_shift_file)
    else:
        print("[Loading element scale and shift factors from file]")
        scale_shift_data = torch.load('./fock_datasets/' + scale_shift_file)
        scale_shift_data = {
            "element_scalar_means": scale_shift_data["element_scalar_means"],  # dict[int -> list[float]]
            "element_scalar_stds": scale_shift_data["element_scalar_stds"],    # dict[int -> list[float]]
            "scalar_irrep_indices": scale_shift_data["scalar_irrep_indices"]   # list[int]
        }
else:
    print("Not scaling or shifting the dataset")
    scale_shift_data = None

# --------------------------------------------
# Initialize compute environment 
# --------------------------------------------

rank = int(os.environ['SLURM_PROCID'])
world_size = int(os.environ['SLURM_NTASKS'])
print(f"Running on rank {rank} of {world_size} total ranks", flush=True)

if rank == 0:
    print(f"Dataset: {dataset_name}, writing results to {output_folder}", flush=True)
    print(f"Dataset - Num molecules used for training: {num_train}", flush=True)
    print(f"Dataset - Num molecules used for validation: {num_val}", flush=True)
    print(f"Dataset - Edge cutoff distance for orbital blocks: {2*rcut_orbitals}", flush=True)
    print(f"Dataset - Edge cutoff distance for gaussian basis: {rcut_gaussian}", flush=True)
    print(f"Dataset - Scale and shift dataset: {scale_and_shift}", flush=True)
    print(f"Model - Num of Message Passing layers: {num_mp_layers}", flush=True)
    print(f"Model - Embedding dimension: {l_embedding_dim}", flush=True)
    print(f"Model - # Distance basis functions: {num_distance_basis}", flush=True)
    print(f"Model - Edge reduction: {reduce_edge}")
    print(f"Model - Node reduction - interorbital: {reduce_node}")
    print(f"Model - Node reduction - intraorbital: {reduce_node_intra}")
    print(f"Training - Loss target: {loss_target}", flush=True)
    print(f"Training - Loss function: {train_loss_fxn}", flush=True)
    print(f"Training - Initial learning rate: {lr_init}", flush=True)
    print(f"Training - Scheduler type: {scheduler_type}", flush=True)

compute_start = time.perf_counter()
device = utils_compute.setup_env(rank, world_size)
compute_end = time.perf_counter()
print("Time to setup distributed environment: ", compute_end - compute_start, flush=True)

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
# print("USING THE DEBUG MOLECULE", flush=True)
# train_start_mol = 0
# train_end_mol = 1
# val_start_mol = 0
# val_end_mol = 1
# test_start_mol = 0
# test_end_mol = 1
### DEBUG ###

if train_or_eval == 'train':
    train_loader, required_irreps, basis_transformation, orbital_basis = get_loader.get_loader(database, train_start_mol, train_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, half_edges=reduce_edge, scale_shift_data=scale_shift_data)
    val_loader, _, _, _ = get_loader.get_loader(database, val_start_mol, val_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, half_edges=reduce_edge, scale_shift_data=scale_shift_data)
    print("Size of train loader: ", len(train_loader), flush=True)
    print("Size of val loader: ", len(val_loader), flush=True)
else:
    batch_size = 1
    test_loader, required_irreps, basis_transformation, orbital_basis = get_loader.get_loader(database, test_start_mol, test_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, half_edges=reduce_edge, scale_shift_data=scale_shift_data)
    print("Size of test loader: ", len(test_loader), flush=True)

data_load_end = time.perf_counter()
print("Time to load dataset: ", data_load_end - data_load_start, flush=True)

ls_list = train_loader.dataset[0].fock_target_object.ls_list
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
# Get model backbone + head
# --------------------------------------------

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
                gaussian_width=gaussian_width,
                include_edges=include_edges
            )

if loss_target == "fock_matrix":
    head = Fock_Irreps_Head(irreps_in=irreps_in, 
                            irreps_out=output_irreps, 
                            lmax=required_irreps.lmax, 
                            sphere_channels=l_embedding_dim,
                            half_edges=reduce_edge,
                            head_type=head_type,
                            ls_list=ls_list,
                            reduce_node=reduce_node,
                            reduce_node_intra=reduce_node_intra,
                            orbital_basis=orbital_basis)

elif loss_target == "forces":
    head = Linear_Force_Head(backbone)
    # head = Convolution_Force_Head(backbone)
    # head = Gated_Force_Head(backbone, irreps_in)

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

print("Number of parameters in backbone: ", sum(p.numel() for p in backbone.parameters()), flush=True)
print("Number of parameters in output head: ", sum(p.numel() for p in head.parameters()), flush=True)

if restart_backbone:
    restart_file = output_folder + '/' + backbone_checkpoint
    print("Restarting backbone model from :", restart_file, flush=True)
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
    print("Restarting output head model from :", restart_file, flush=True)
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

if scheduler_type == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
elif scheduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, verbose=True)
else:
    raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose 'plateau' or 'cosine'.")
    
# scheduler = loss_scheduler(optimizer)

trainer = splittrainer.SplitTrainer(backbone=backbone, 
                                    head=head,
                                    head_irreps=output_irreps,
                                    run_name='water_final',
                                    save_frequency=10)

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
        
