import torch
import os
from train_utils import loss, training_workflow
from fock_utils import basis_sets
from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase

# ---------------------------------------------------------
# OMol Dataset Configuration - closed shell version
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'omol',
    "dbpath": "./omol_elytes_unsolvated_raw/",
    "output_folder": 'outputs_omol_elytes',
    "run_name": 'omol',
    "open_shell": False,
    
    # Execution Mode
    "train_or_eval": "train",        # Set to "train" to start learning
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 32, #28900,
    "num_val": 32,
    "num_test": 1,
    "batch_size": 1,                 # 1 for eval, usually 10 for train
    "distribute_graphs": True,       # Distribute graphs and perform communication in the forward pass (ongoing)

    # Symmetry Reductions 
    "reduce_edge": False,            # Unused!
    "reduce_node": False,             # Enforce inter-orbital interactions equal
    "reduce_node_intra": False,       # Enforce 0 odd degrees for intra-orbital
    
    # Training Hyperparameters
    "num_epochs": 100,
    "dtype": torch.float32,          # nablaDFT often uses float64 - 32 here for cupy kernel
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
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 6.0,
    "rcut_gaussian": 12.0,           # 2 * rcut_orbitals
    "gaussian_width": 1.0,
    "include_edges": True,           # Based on fock_matrix target
    "head_type": 'gated',

    # Evals
    "compute_total_energy": False,    # Whether to compute total energy from fock matrix
}

if __name__ == "__main__":
    os.makedirs(config['output_folder'], exist_ok=True)
    
    # Initialize and run the workflow
    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run()