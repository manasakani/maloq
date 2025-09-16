import time
import_start = time.perf_counter()
import os, sys, random
import numpy as np
import torch

from fock_utils import utils_orca_out, fock_targets, basis_sets
from train_utils import loss, utils_compute, splittrainer
from dataset_utils import get_loader, dataset_analysis, get_scale_shift
from dataset_utils.ASEDataset import ASEAtomsData, ASEDataset
from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
from torch_geometric.loader import DataLoader
import ase.db
from ase.db import connect
from ase import Atoms

# Models
from esen_full.esen_new import eSEN_Backbone, Fock_Irreps_Head, Linear_Force_Head, Linear_Energy_Head     
from e3nn.o3 import Irreps

import_end = time.perf_counter()
print("Time to do imports: ", import_end - import_start)

# Fix random seeds for distributed model initialization
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# torch.autograd.set_detect_anomaly(True)


# --------------------------------------------
# Initialize compute environment 
# --------------------------------------------

rank = int(os.environ['SLURM_PROCID'])
world_size = int(os.environ['SLURM_NTASKS'])

compute_start = time.perf_counter()
device = utils_compute.setup_env(rank, world_size)
compute_end = time.perf_counter()
print("Time to setup distributed environment: ", compute_end - compute_start)

# -----------------------------------------------
# Settings (just dumping everything here for now)
# -----------------------------------------------

# ---------------------------
# --> OMOL 
dataset_folder = '/checkpoint/ocp/manasakani/omol_58k_Sep11/omol_closedshell_58k_train_6.0_alledge_job_'+str(rank)+'.db' 
dtype = torch.float32
output_folder = 'outputs_omol_58k_E128_scaled'
dataset_name = 'omol'
run_name = 'omol_58k_Aug26_E128_scaled'
orbital_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
orbital_basis = dict(sorted(orbital_basis.items(), key=lambda item: len(item[1]), reverse=True)) # put elements with the largest basis first
orbital_basis = {int(k): v for k, v in orbital_basis.items()}

db = ase.db.connect(dataset_folder)
total_rows = db.count()
# ---------------------------

# --> Model settings:
l_embedding_dim = 128                   # sphere channels 
num_distance_basis = l_embedding_dim    # number of gaussian basis functions used to expand the edge distance
hidden_dim = l_embedding_dim
num_mp_layers = 3 
model_name = 'esen'
restart_backbone = False
restart_head = False
restart_optimizer = False

# --> Training settings:
train_or_eval = "train"
num_val = 1                             # Number of validation structures
num_train = total_rows - num_val        # Number of training structures - need equal batches on every gpu (use 840 molecules per gpu if doing a mol-wise split)
num_epochs = 3000
batch_size = 1                          # 1 for not oom (molecule-wise batching for evals)
target_atoms_per_batch = 130            # if not using batch_size (atom-wise batching for train)
target_edges_per_batch = 18000          # Don't use more than 18k (for E128)
rcut_orbitals = 6.0                     # connectivity cutoff (=2xrcut)
rcut_gaussian = rcut_orbitals*2         # connectivity cutoff (=2xrcut)
gaussian_width = 1.0                    # width of gaussians used to expand edge distance

# Additional symmetries:
reduce_edge = False                     # use only edges i,j where i<j (other edges are reflected)
reduce_node = False                     # inter-orbital forward/backward interactions are enforced to be equal
reduce_node_intra = False               # intra-orbital interactions are enforced to have 0 odd degrees

train_backbone = True
train_head = True

torch.set_default_dtype(dtype)
lr_init = 1e-3 
patience = 10                           # for scheduler
threshold = 1e-5                        # for scheduler
scheduler_type = 'plateau'               # 'plateau' or 'cosine'
T_max = num_epochs                      # for cosine scheduler - period of cosine annealing
eta_min = 1e-8                          # for cosine scheduler - minimum learning rate

loss_target = 'fock_matrix'
compute_uncoupled_loss = True          
head_type = 'gated'                     # linear or gated 
train_loss_fxn = loss.rmse_mse_padded_loss   
loss_scheduler = loss.MonotonicDecreaseScheduler
backbone_checkpoint = 'backbone.pt'
head_checkpoint = 'head.pt'

if reduce_edge and batch_size != 1:
    raise ValueError("If using reduce_edge, batch size must be 1! Reverse_edge map is not collated.")

scale_and_shift = True
scale_shift_file = 'element_scale_shifts_' + dataset_name + '.pt'

# Dump all settings to the output file
if rank == 0:
    print(f"Dataset: {dataset_name}, writing results to {output_folder}", flush=True)
    print(f"Dataset - Num molecules used for training: {num_train}", flush=True)
    print(f"Dataset - Num molecules used for validation: {num_val}", flush=True)
    print(f"Dataset - Edge cutoff distance for orbital blocks: {2*rcut_orbitals}", flush=True)
    print(f"Dataset - Edge cutoff distance for gaussian basis: {rcut_gaussian}", flush=True)
    print(f"Dataset - Scaling/shifting data: {scale_and_shift}", flush=True)
    print(f"Model - Num of Message Passing layers: {num_mp_layers}", flush=True)
    print(f"Model - Embedding dimension: {l_embedding_dim}", flush=True)
    print(f"Model - Edge reflections: {reduce_edge}")    
    print(f"Model - Node reduction - interorbital: {reduce_node}")
    print(f"Model - Node reduction - intraorbital: {reduce_node_intra}")
    print(f"Training - Loss target: {loss_target}", flush=True)
    print(f"Training - Loss function: {train_loss_fxn}", flush=True)
    print(f"Training - Initial learning rate: {lr_init}", flush=True)

if rank == 0 and not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --------------------------------------------
# Prepare data
# --------------------------------------------

data_load_start = time.perf_counter()

# Split data between GPUs - each rank has it's own DB now!
# train_start_mol, train_end_mol, train_local_num_mol = utils_compute.split_indices(rank, world_size, num_train)
# val_start_mol, val_end_mol, val_local_num_mol  = utils_compute.split_indices(rank, world_size, num_val)
train_start_mol = 0
train_end_mol = num_train
train_local_num_mol = num_train

val_start_mol = num_train  # the validation molecules start after training ones
val_end_mol = val_start_mol + num_val
val_local_num_mol = num_val

# Query this rank's molecules from the database
train_database = ASEDataset(dataset_folder, orbital_basis, dtype=dtype, world_size=world_size, rank=rank, start_idx=train_start_mol, end_idx=train_end_mol)
val_database = ASEDataset(dataset_folder, orbital_basis, dtype=dtype, world_size=world_size, rank=rank, start_idx=val_start_mol, end_idx=val_end_mol)

# Compute scale and shift factors if required
if scale_and_shift:
    print("Getting scale and shift factors...", flush=True)
    print(f"Scale and shift file: {scale_shift_file}", flush=True)
    if scale_shift_file not in os.listdir('./fock_datasets/'):
        print("[Computing element scale and shift factors for the dataset]", flush=True)
        get_scale_shift.get_scale_shift(train_database, dataset_name, rcut_orbitals, dtype=dtype, reduce_edge=reduce_edge, filename=scale_shift_file)
        scale_shift_data = torch.load('./fock_datasets/' + scale_shift_file)
        print("Done computing scale and shift factors", flush=True)
    else:
        print("[Loading element scale and shift factors from file]", flush=True)
        scale_shift_data = torch.load('./fock_datasets/' + scale_shift_file)
        scale_shift_data = {
            "element_scalar_means": scale_shift_data["element_scalar_means"],  # dict[int -> list[float]]
            "element_scalar_stds": scale_shift_data["element_scalar_stds"],    # dict[int -> list[float]]
            "scalar_irrep_indices": scale_shift_data["scalar_irrep_indices"]   # list[int]
        }
else:
    print("Not scaling or shifting the dataset", flush=True)
    scale_shift_data = None

# Create the fock target analysis objects for each molecule in the dataset, scale and shift the node labels if required
print("Creating data loaders, making fock analysis objects if needed ...", flush=True)

# Create the dataloaders for each GPU's data (if eval, we use the val set)
if train_or_eval == "train":
    train_data = get_scale_shift.scale_shift_database(train_database, 0, train_local_num_mol, rcut_orbitals, orbital_basis, reduce_edge, scale_shift_data, scale_nodes=scale_and_shift, train_or_eval=train_or_eval)
    # train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0)
    train_loader = get_scale_shift.create_edge_balanced_dataloader(
        train_data, 
        target_edges_per_batch=target_edges_per_batch, 
        tolerance=0.15,   
        shuffle=True,
        num_workers=0
    )
    # trim train_loader to have min_train_loader_size batches - train_loader is a SimpleBatchIterator, so we can directly slice its batches
    # communicate between gpus to find min_train_loader_size:

    if world_size > 1:
        torch.distributed.barrier()
        local_size = torch.tensor(len(train_loader), dtype=torch.long).cuda()
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(all_sizes, local_size)
        
        min_train_loader_size = min([size.item() for size in all_sizes])
        
        if len(train_loader) > min_train_loader_size:
            random.shuffle(train_loader.batches)
            train_loader.batches = train_loader.batches[:min_train_loader_size]
        print(f"NOTE: Trimming train loader size on rank {rank} (for batch consistency across ranks) from {local_size.item()} to {min_train_loader_size}", flush=True)

    print(f"Size of train loader: {len(train_loader)}", flush=True)

    val_data = get_scale_shift.scale_shift_database(val_database, 0, val_local_num_mol, rcut_orbitals, orbital_basis, reduce_edge, scale_shift_data, scale_nodes=scale_and_shift, train_or_eval=train_or_eval)
    # val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)
    val_loader = get_scale_shift.create_edge_balanced_dataloader(
        val_data,
        target_edges_per_batch=target_edges_per_batch, 
        tolerance=0.1,
        shuffle=False,
        num_workers=0
    )
else: # molecule-wise batching for evals, only make the val dataloader since that's what evaluated
    val_data = get_scale_shift.scale_shift_database(val_database, 0, val_local_num_mol, rcut_orbitals, orbital_basis, reduce_edge, scale_shift_data, scale_nodes=scale_and_shift, train_or_eval=train_or_eval)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)

print("Size of val loader: ", len(val_loader), flush=True)
basis_transformation = val_data[0].fock_target_object.basis_transformation

# # analyze the node labels for this dataset
# dataset_analysis.dataset_analysis(train_database, dataset_name, rcut=5.0, dtype=dtype, reduce_edge=False, scale_shift_data=None, rank=rank)
# print("Dataset analysis done, exiting", flush=True)

data_load_end = time.perf_counter()
print("Time to load dataset: ", data_load_end - data_load_start, flush=True)

# --> Irrep information for the targets
required_irreps = val_data[0].fock_target_object.req_output_irreps
ls_list = val_data[0].fock_target_object.ls_list
basis_transformation = val_data[0].fock_target_object.basis_transformation
orbital_basis = {k: torch.tensor(v) for k, v in orbital_basis.items()}

print("orbital_basis: ", orbital_basis)
print("required_irreps: ", required_irreps)
print("simplified_out_irreps: ", Irreps(required_irreps).sort()[0].simplify())

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
                            sphere_channels=l_embedding_dim,
                            half_edges=reduce_edge,
                            ls_list=ls_list,
                            reduce_node=reduce_node,
                            reduce_node_intra=reduce_node_intra,
                            orbital_basis=orbital_basis)
elif loss_target == "forces":
    head = Linear_Force_Head(backbone)

elif loss_target == "energy":
    print("To be implemented!")


backbone = backbone.to(device)
head = head.to(device)

if train_backbone and train_head:
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=lr_init, weight_decay=1e-4)

elif train_head:                            # freeze backbone model
    for param in backbone.parameters(): 
        param.requires_grad = False
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr_init, weight_decay=1e-4)

elif train_backbone:                        # freeze output head
    for param in head.parameters():     
        param.requires_grad = False
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr_init, weight_decay=1e-4)

else:
    print("Running evaluation")

print("Number of parameters in backbone: ", sum(p.numel() for p in backbone.parameters()), flush=True)
print("Number of parameters in output head: ", sum(p.numel() for p in head.parameters()), flush=True)

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

if scheduler_type == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
elif scheduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, verbose=True)
else:
    raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose 'plateau' or 'cosine'.")
 
print("Going to training or evaluation", flush=True)

# scheduler = loss_scheduler(optimizer, lag_epochs=100)
trainer = splittrainer.SplitTrainer(backbone=backbone, 
                                    head=head,
                                    head_irreps=output_irreps,
                                    run_name=run_name,
                                    save_frequency=1)

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
                    basis_transform=basis_transformation,
                    train_backbone=train_backbone,
                    train_head=train_head,
                    compute_uncoupled_loss=compute_uncoupled_loss)
else:
    trainer.evaluate(train_loss_fxn,
                    device,
                    val_loader,
                    loss_target_string=loss_target,
                    node_target_name=node_target,
                    edge_target_name=edge_target, 
                    basis_transform=basis_transformation,
                    output_folder=output_folder,
                    )
