import torch
import os
from train_utils import loss, training_workflow
from fock_utils import basis_sets
from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
from dataset_utils.ASEDataset import ASEDataset

# ---------------------------------------------------------
# nablaDFT Dataset Configuration
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'omol',
    "dbpath": "./created_omol_database/omol_electrolytes_unsolvated_test_raw_job_0.db",
    "output_folder": 'outputs_omol_elytes',
    "run_name": 'omol',
    
    # Execution Mode
    "train_or_eval": "train",         # Set to "train" to start learning
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 7,
    "num_val": 3,
    "num_test": 0,
    "batch_size": 1,                 # 1 for eval, usually 10 for train
    
    # Symmetry Reductions 
    "reduce_edge": False,            # Unused!
    "reduce_node": True,             # Enforce inter-orbital interactions equal
    "reduce_node_intra": True,       # Enforce 0 odd degrees for intra-orbital
    "open_shell": False,
    
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
    "rcut_orbitals": 6.0,
    "rcut_gaussian": 12.0,           # 2 * rcut_orbitals
    "gaussian_width": 1.0,
    "include_edges": True,           # Based on fock_matrix target
    "head_type": 'gated',
}

if __name__ == "__main__":
    if not os.path.exists(config['output_folder']):
        os.makedirs(config['output_folder'])

    # Initialize the omol specific database
    orbital_basis = basis_sets.def2_tzvpd
    database = ASEDataset(config['dbpath'], orbital_basis, dtype=config['dtype'], open_shell=config['open_shell'])
    
    # Initialize and run the workflow
    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run(database)