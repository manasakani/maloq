# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

import torch
import os
from maloq.train_utils import loss, training_workflow

# ---------------------------------------------------------
# OMol Dataset Configuration - closed shell version
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'omol_csh_58k',
    "dbpath": "/capstor/store/cscs/pasc/c33/manasa/omol_datasets/omol_csh_58k/train.h5",
    "output_folder": 'outputs_omol_csh_58k',
    "run_name": 'omol',
    "open_shell": False,
    
    # Execution Mode
    "train_or_eval": "train",               # "train", "eval", "infer" (inference mode)
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 1000,
    "num_val": 32,
    "num_test": 32,
    "batch_size": 1,                        # 1 for eval
    "distribute_graphs": False,             # Distribute graphs and perform communication in the forward pass
    "partition_type": 'linear-edgewise',    # 'linear', 'low_nn', 'metis', 'worstcase'
    "dist_backend": 'nccl',                 # 'gloo' or 'nccl' 

    # Symmetry Reductions 
    "reduce_edge": False,           # Unused!
    "reduce_node": True,            # Enforce inter-orbital interactions equal
    "reduce_node_intra": True,      # Enforce 0 odd degrees for intra-orbital
    
    # Training Hyperparameters
    "num_epochs": 2000,
    "dtype": torch.float32,          # fp32 here for cupy matrix<->label
    "lr_init": 1e-4,
    "optimizer_type": "muon",        # adam, adamw, muon (w/ adam)
    "weight_decay": 0.0,
    "scheduler_type": 'plateau',     # 'plateau' or 'cosine'
    "eta_min": 1e-8,                 # For cosine scheduler
    "patience": 500,                 # For plateau scheduler
    "threshold": 1e-8,
    "step_every_epoch": False,       # In case you want to step per epoch
    
    # Loss & Checkpointing
    "loss_target": 'fock_matrix',
    "train_loss_fxn": loss.rmse_mse_padded_loss,
    "test_loss_fxn": loss.l1_unpadded_loss,
    "save_frequency": 5,
    "restart_backbone": True,       
    "restart_head": True,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": True,            # Scale & shift the node labels

    # Model Architecture
    "wigner_backend": "triton",         # "triton" for fused Triton kernel (requires GPU + triton)
    "basis_transform_backend": "torch", # keep torch, triton implementation is unoptimized
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 6.0,               # 2*rcut_orbitals is the connectivity cutoff for the graph
    "rcut_gaussian": 12.0,              # Should be 2 * rcut_orbitals
    "gaussian_width": 1.0,

    # Evals
    "compute_total_energy": False,    # Whether to compute total energy from fock matrix
}

def main():
    os.makedirs(config['output_folder'], exist_ok=True)
    
    # Initialize and run the workflow
    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()