# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

import torch
import os
from maloq.train_utils import loss, training_workflow

# ---------------------------------------------------------
# OMol Dataset Configuration - closed shell version
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'omol',
    "dbpath": "/capstor/store/cscs/pasc/c33/manasa/omol_raw_datasets/omol_elytes_unsolvated_raw/",
    "output_folder": 'outputs_omol_elytes',
    "run_name": 'omol',
    "open_shell": False,
    
    # Execution Mode
    "train_or_eval": "train",        # Set to "train" to start learning
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 15000, #28900,
    "num_val": 32,
    "num_test": 1,
    "batch_size": 10,                 # 1 for eval, usually 10 for train
    "distribute_graphs": True,       # Distribute graphs and perform communication in the forward pass (ongoing)
    "partition_type": 'linear-edgewise',       # 'linear', 'low_nn', 'metis', 'worstcase'
    "dist_backend": 'nccl',              # 'gloo' for CPU, 'nccl' for GPU (if distributed training is implemented)

    # Symmetry Reductions 
    "reduce_edge": False,            # Unused!
    "reduce_node": True,            # Enforce inter-orbital interactions equal
    "reduce_node_intra": True,      # Enforce 0 odd degrees for intra-orbital
    
    # Training Hyperparameters
    "num_epochs": 200,
    "dtype": torch.float32,          # 32 here for cupy kernel
    "lr_init": 1e-3,
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
    "save_frequency": 20,
    "restart_backbone": False,       # Restart options
    "restart_head": False,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": True,         # Scale & shift the node labels

    # Model Architecture
    "wigner_backend": "triton",  # "triton" for fused Triton kernel (requires GPU + triton)
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 6.0,
    "rcut_gaussian": 12.0,           # 2 * rcut_orbitals
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