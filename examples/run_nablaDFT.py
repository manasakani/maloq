# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

import torch
from maloq.train_utils import loss, training_workflow

# ---------------------------------------------------------
# nablaDFT Dataset Training & Evaluation
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'nablaDFT',
    "dbpath": "/capstor/store/cscs/pasc/c33/manasa/nablaDFT_datasets/train_2k.db",
    "output_folder": 'outputs_nablaDFT_test',
    "run_name": 'nabla_2k',          
    "open_shell": False,
    
    # Execution Mode
    "train_or_eval": "train",         
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 12081,
    "num_val": 1,
    "num_test": 1,
    "batch_size": 10,                
    "distribute_graphs": False,      
    "partition_type": 'metis',      
    "dist_backend": 'nccl',          

    # Symmetry Reductions 
    "reduce_edge": False,            
    "reduce_node": True,            
    "reduce_node_intra": True,      
    
    # Training Hyperparameters
    "num_epochs": 700,
    "dtype": torch.float32,         
    "lr_init": 1e-4,
    "optimizer_type": "adam",        
    "weight_decay": 0.0,
    "scheduler_type": 'cosine',      
    "eta_min": 1e-8,                 
    "patience": 500,                 
    "threshold": 1e-8,
    "step_every_epoch": False,      
    
    # Loss & Checkpointing
    "loss_target": 'fock_matrix',
    "train_loss_fxn": loss.rmse_mse_padded_loss,
    "test_loss_fxn": loss.l1_unpadded_loss,
    "save_frequency": 10,
    "restart_backbone": False,      
    "restart_head": False,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": True,        

    # Evals
    "compute_total_energy": False,

    # Model Architecture
    "wigner_backend": "triton", 
    "basis_transform_backend": "torch",
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 8.0,
    "rcut_gaussian": 16.0,          
    "gaussian_width": 1.0,
}

def main():
    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()