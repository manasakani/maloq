import torch
import os
from train_utils import loss, training_workflow
from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

# ---------------------------------------------------------
# nablaDFT Dataset Configuration
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'nablaDFT',
    "dbpath": "/capstor/store/cscs/pasc/c33/manasa/nablaDFT_datasets/train_2k.db",
    "output_folder": 'outputs_nablaDFT',
    "run_name": 'nabla_2k',
    
    # Execution Mode
    "train_or_eval": "train",         # Set to "train" to start learning
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 12081,
    "num_val": 64,
    "num_test": 0,
    "batch_size": 1,                 # 1 for eval, usually 10 for train
    
    # Symmetry Reductions 
    "reduce_edge": False,            # Unused!
    "reduce_node": True,             # Enforce inter-orbital interactions equal
    "reduce_node_intra": True,       # Enforce 0 odd degrees for intra-orbital
    
    # Training Hyperparameters
    "num_epochs": 1000,
    "dtype": torch.float32,          # nablaDFT often uses float64 - 32 here for cupy kernel
    "lr_init": 1e-4,
    "optimizer_type": "adam",        # standard Adam
    "weight_decay": 0.0,
    "scheduler_type": 'cosine',      # 'plateau' or 'cosine'
    "eta_min": 1e-8,                 # For cosine scheduler
    "patience": 500,                 # For plateau scheduler (if swapped)
    "threshold": 1e-8,
    "step_every_epoch": False,       # Cosine usually steps per iteration
    
    # Loss & Checkpointing
    "loss_target": 'fock_matrix',
    "train_loss_fxn": loss.rmse_mse_padded_loss,
    "test_loss_fxn": loss.l1_unpadded_loss,
    "save_frequency": 5,
    "restart_backbone": True,       # Restart options
    "restart_head": True,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": True,         # Scale & shift the node labels

    # Model Architecture
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 10.0,
    "rcut_gaussian": 20.0,           # 2 * rcut_orbitals
    "gaussian_width": 1.0,
    "include_edges": True,           # Based on fock_matrix target
    "head_type": 'gated',
}

if __name__ == "__main__":
    if not os.path.exists(config['output_folder']):
        os.makedirs(config['output_folder'])

    # Initialize the nablaDFT-specific database
    database = HamiltonianDatabase(config['dbpath'])
    
    # Initialize and run the workflow
    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run(database)


# import time
# import_start = time.perf_counter()
# import os, sys, random
# import numpy as np
# import torch

# from fock_utils import utils_orca_out, fock_targets
# from train_utils import loss, utils_compute, splittrainer
# from dataset_utils import get_loader, dataset_analysis, get_scale_shift
# from dataset_utils.ASEDataset import ASEAtomsData
# from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

# # Models
# from helm.esen_new import eSEN_Backbone, Fock_Irreps_Head, Linear_Force_Head
# from e3nn.o3 import Irreps

# import_end = time.perf_counter()
# print("Time to do imports: ", import_end - import_start, flush=True)

# # Fix random seeds for distributed model initialization
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # -----------------------------------------------
# # Settings (just dumping everything here for now)
# # -----------------------------------------------
# # ---------------------------
# # --> NablaDFT (tiny conformers)
# database = HamiltonianDatabase("/capstor/store/cscs/pasc/c33/manasa/nablaDFT_datasets/train_5k.db")
# dataset_name = 'nablaDFT'
# run_name = 'nabla_2k'
# output_folder = 'outputs_nablaDFT'
# # ---------------------------

# # --> Model settings:
# l_embedding_dim = 128                   # sphere channels
# num_distance_basis = l_embedding_dim    # number of gaussian basis functions used to expand the edge distance
# hidden_dim = l_embedding_dim
# num_mp_layers = 3
# model_name = 'esen'
# restart_backbone = True
# restart_head = True
# restart_optimizer = False

# # --> Training settings:
# train_or_eval = "eval"
# compute_total_energy = False
# num_val = 64                             # Number of validation structures
# num_train = 100#len(database) - 64
# num_test = 100#len(database)
# num_epochs = 50
# batch_size = 1                          # 1 for eval, 10 for train
# rcut_orbitals = 10.0                     # connectivity cutoff (=2xrcut)
# rcut_gaussian = 2*10.0                  # connectivity cutoff (=2xrcut)
# gaussian_width = 1.0                    # width of gaussians used to expand edge distance

# # Additional symmetries:
# reduce_edge = False                      # use only edges i,j where i<j (other edges are reflected)
# reduce_node = True                      # inter-orbital forward/backward interactions are enforced to be equal
# reduce_node_intra = True                # intra-orbital interactions are enforced to have 0 odd degrees

# train_backbone = True
# train_head = True
# save_frequency = 1

# dtype = torch.float64
# torch.set_default_dtype(dtype)
# lr_init = 1e-4
# patience = 500                          # for scheduler
# threshold = 1e-8                        # for scheduler
# scheduler_type = 'cosine'               # 'plateau' or 'cosine'
# eta_min = 1e-8                          # for cosine scheduler - minimum learning rate

# loss_target = 'fock_matrix'
# head_type = 'gated'                     # linear or gated
# train_loss_fxn = loss.rmse_mse_padded_loss
# test_loss_fxn = loss.l1_unpadded_loss
# loss_scheduler = loss.MonotonicDecreaseScheduler
# backbone_checkpoint = 'backbone.pt'
# head_checkpoint = 'head.pt'
# include_edges = False if loss_target == 'forces' else True
# scale_and_shift = True

# # Scale and shift the orbital self-interaction scalar components of the dataset
# if scale_and_shift:
#     print("Getting scale and shift factors...", flush=True)
#     scale_shift_file = 'element_scale_shifts_' + dataset_name + '.pt'
#     print(f"Scale and shift file: {scale_shift_file}", flush=True)
#     if scale_shift_file not in os.listdir('./fock_utils/'):
#         print("[Computing element scale and shift factors for the dataset]", flush=True)
#         get_scale_shift.get_scale_shift(database, dataset_name, rcut_orbitals, dtype=dtype, reduce_edge=reduce_edge)
#     else:
#         print("[Loading element scale and shift factors from file]", flush=True)
#         scale_shift_data = torch.load('./fock_utils/' + scale_shift_file)
#         scale_shift_data = {
#             "element_scalar_means": scale_shift_data["element_scalar_means"],  # dict[int -> list[float]]
#             "element_scalar_stds": scale_shift_data["element_scalar_stds"],    # dict[int -> list[float]]
#             "scalar_irrep_indices": scale_shift_data["scalar_irrep_indices"]   # list[int]
#         }
# else:
#     print("Not scaling or shifting the dataset")
#     scale_shift_data = None

# # --------------------------------------------
# # Initialize compute environment
# # --------------------------------------------

# rank = int(os.environ['SLURM_PROCID'])
# world_size = int(os.environ['SLURM_NTASKS'])

# if rank == 0:
#     print(f"Dataset: {dataset_name}, writing results to {output_folder}", flush=True)
#     print(f"Dataset - Num molecules used for training: {num_train}", flush=True)
#     print(f"Dataset - Num molecules used for validation: {num_val}", flush=True)
#     print(f"Dataset - Edge cutoff distance for orbital blocks: {2*rcut_orbitals}", flush=True)
#     print(f"Dataset - Edge cutoff distance for gaussian basis: {rcut_gaussian}", flush=True)
#     print(f"Model - Num of Message Passing layers: {num_mp_layers}", flush=True)
#     print(f"Model - Embedding dimension: {l_embedding_dim}", flush=True)
#     print(f"Model - Edge reduction: {reduce_edge}")
#     print(f"Model - Node reduction - interorbital: {reduce_node}")
#     print(f"Model - Node reduction - intraorbital: {reduce_node_intra}")
#     print(f"Training - Loss target: {loss_target}", flush=True)
#     print(f"Training - Loss function: {train_loss_fxn}", flush=True)
#     print(f"Training - Initial learning rate: {lr_init}", flush=True)

# compute_start = time.perf_counter()
# device = utils_compute.setup_env(rank, world_size)
# compute_end = time.perf_counter()
# print("Time to setup distributed environment: ", compute_end - compute_start)

# if rank == 0 and not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # --------------------------------------------
# # Prepare data
# # --------------------------------------------

# data_load_start = time.perf_counter()

# train_start_mol, train_end_mol, train_local_num_mol = utils_compute.split_indices(rank, world_size, num_train)
# val_start_mol, val_end_mol, val_local_num_mol  = utils_compute.split_indices(rank, world_size, num_val)
# test_start_mol, test_end_mol, test_local_num_mol = utils_compute.split_indices(rank, world_size, num_test)

# val_start_mol += num_train  # the validation molecules start after training ones
# val_end_mol += num_train

# ## DEBUG ### - 22 is the first molecule with a Br atom
# # print("Starting molecule for test!!: ", test_start_mol, flush=True)
# # test_start_mol = 22
# # test_end_mol = test_start_mol + test_local_num_mol
# ## DEBUG ###

# if train_or_eval == 'train':
#     train_loader, required_irreps, basis_transformation, orbital_basis, ls_list = get_loader.get_loader(database, train_start_mol, train_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, half_edges=reduce_edge,  scale_shift_data=scale_shift_data)
#     val_loader, _, _, _, _ = get_loader.get_loader(database, val_start_mol, val_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, half_edges=reduce_edge,  scale_shift_data=scale_shift_data)
#     print("Size of train loader: ", len(train_loader))
# else:
#     batch_size = 1
#     test_loader, required_irreps, basis_transformation, orbital_basis, ls_list = get_loader.get_loader(database, test_start_mol, test_end_mol, dataset_name, rcut_orbitals, batch_size, dtype=dtype, half_edges=reduce_edge, scale_shift_data=scale_shift_data)
#     print("Size of test loader: ", len(test_loader))

# data_load_end = time.perf_counter()
# print("Time to load dataset: ", data_load_end - data_load_start)

# irreps_in = Irreps([(l_embedding_dim, (l, 1)) for l in range(required_irreps.lmax + 1)])

# # determine output irreps from target type:
# if loss_target == "fock_matrix":
#     output_irreps = required_irreps
#     node_target = 'node_y'
#     edge_target = 'y'
# elif loss_target == "forces":
#     output_irreps = '1x1e'
#     node_target = 'forces'
#     edge_target = None
# else:
#     output_irreps = '1x0e'
#     node_target = 'energy'
#     edge_target = None

# # --------------------------------------------
# # Get model
# # --------------------------------------------

# backbone = eSEN_Backbone(
#                 required_irreps,
#                 sphere_channels=l_embedding_dim,
#                 hidden_channels=hidden_dim,
#                 lmax=required_irreps.lmax,
#                 mmax=required_irreps.lmax,
#                 use_pbc=False,
#                 cutoff=rcut_gaussian,
#                 edge_channels=l_embedding_dim,
#                 num_layers=num_mp_layers,
#                 act_type='gate',
#                 mlp_type = 'spectral',
#                 num_distance_basis=num_distance_basis,
#                 gaussian_width=gaussian_width,
#                 include_edges=include_edges
#             )

# if loss_target == "fock_matrix":
#     head = Fock_Irreps_Head(irreps_in=irreps_in,
#                             irreps_out=output_irreps,
#                             lmax=required_irreps.lmax,
#                             sphere_channels=l_embedding_dim,
#                             half_edges=reduce_edge,
#                             ls_list=ls_list,
#                             head_type=head_type,
#                             reduce_node=reduce_node,
#                             reduce_node_intra=reduce_node_intra,
#                             orbital_basis=orbital_basis)

# elif loss_target == "forces":
#     head = Linear_Force_Head(backbone)

# elif loss_target == "energy":
#     print("To be implemented!")


# backbone = backbone.to(device)
# head = head.to(device)

# if train_backbone and train_head:
#     optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr_init)

# elif train_head:
#     for param in backbone.parameters(): # freeze backbone model
#         param.requires_grad = False
#     optimizer = torch.optim.Adam(head.parameters(), lr=lr_init)

# elif train_backbone:
#     for param in head.parameters():     # freeze output head
#         param.requires_grad = False
#     optimizer = torch.optim.Adam(backbone.parameters(), lr=lr_init)

# else:
#     print("Check train recipe (backbone/head)")

# print("Number of parameters in backbone: ", sum(p.numel() for p in backbone.parameters()))
# print("Number of parameters in output head: ", sum(p.numel() for p in head.parameters()))

# if restart_backbone:
#     restart_file = output_folder + '/' + backbone_checkpoint
#     print("Restarting backbone model from :", restart_file)
#     checkpoint = torch.load(restart_file)
#     state_dict = checkpoint['model_state_dict']

#     # Get rid of module prefix if saved with DDP
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key.replace('module.', '')
#         new_state_dict[new_key] = value
#     backbone.load_state_dict(new_state_dict)

# if restart_head:
#     restart_file = output_folder + '/' + head_checkpoint
#     print("Restarting output head model from :", restart_file)
#     checkpoint = torch.load(restart_file)
#     state_dict = checkpoint['model_state_dict']

#     if restart_optimizer:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     # Get rid of module prefix if saved with DDP
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key.replace('module.', '')
#         new_state_dict[new_key] = value
#     head.load_state_dict(new_state_dict)


# # --------------------------------------------
# # Run Training or Evaluation
# # --------------------------------------------

# # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
# scheduler = loss_scheduler(optimizer)
# trainer = splittrainer.SplitTrainer(backbone=backbone,
#                                     head=head,
#                                     head_irreps=output_irreps,
#                                     run_name=run_name,
#                                     save_frequency=save_frequency)

# if train_or_eval == "train":

#     if scheduler_type == 'plateau':
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold)
#     elif scheduler_type == 'cosine':
#         T_max = num_epochs * len(train_loader)                     # for cosine scheduler - period of cosine annealing
#         print("T_max for scheduler: ", T_max)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
#     else:
#         raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose 'plateau' or 'cosine'.")

#     trainer.train(num_epochs,
#                     train_loss_fxn,
#                     optimizer,
#                     scheduler,
#                     device,
#                     train_loader=train_loader,
#                     loss_target_string=loss_target,
#                     node_target_name=node_target,
#                     edge_target_name=edge_target,
#                     output_folder=output_folder,
#                     val_loader=val_loader,
#                     train_backbone=train_backbone,
#                     train_head=train_head,
#                     basis_transform=basis_transformation,
#                     step_every_epoch=False)
# else:

#     trainer.evaluate(test_loss_fxn,
#                     device,
#                     test_loader,
#                     loss_target_string=loss_target,
#                     node_target_name=node_target,
#                     edge_target_name=edge_target,
#                     dataset_name=dataset_name,
#                     basis_transform=basis_transformation,
#                     compute_total_energy=compute_total_energy,
#                     output_folder=output_folder,
#                     orbital_basis=orbital_basis
#                     )
