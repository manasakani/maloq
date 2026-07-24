# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

import torch
from maloq.train_utils import loss, training_workflow

# ---------------------------------------------------------
# Custom CP2K Dataset Training & Evaluation
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'cp2k_material',
    "dbpath": ['/capstor/scratch/cscs/mdossena/MoS2_Al2O3_SELF_PASS_32_32_21_MH1', 
                '/capstor/scratch/cscs/mdossena/MoS2_HfO2_SELF_PASS_32_32_21_MH1'],
    "output_folder": 'outputs_maloq',
    "run_name": 'cp2k_material',
    "open_shell": False,
    
    # Execution Mode
    "train_or_eval": "train",        
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 50,
    "num_val": 1, 
    "num_test": 1,
    "shuffle": True,
    "batch_size": 1,                 
    "distribute_graphs": True,        
    "partition_type": 'linear-edgewise',       
    
    # Symmetry Reductions 
    "reduce_edge": False,            
    "reduce_node": True,             
    "reduce_node_intra": True,      
    
    # Training Hyperparameters
    "num_epochs": 400,
    "dtype": torch.float32,         
    "lr_init": 1e-6,
    "optimizer_type": "adamw",     
    "weight_decay": 0.0,
    "scheduler_type": 'cosine',    
    "eta_min": 1e-8,               
    "patience": 50,               
    "threshold": 1e-8,
    "step_every_epoch": False,      
    "compute_uncoupled_loss": False, 
    
    # Loss & Checkpointing
    "loss_target": 'fock_matrix',
    "train_loss_fxn": loss.rmse_mse_padded_loss,
    "test_loss_fxn": loss.l1_unpadded_loss,
    "save_frequency": 10,
    "restart_backbone": True,      
    "restart_head": True,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": True,        

    # Evals
    "compute_total_energy": False,

    # Model Architecture
    "wigner_backend": "triton",  
    "l_embedding_dim": 128,
    "hidden_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "message_type": "source-target-message",
    "rcut_orbitals": 5.0, 
    "rcut_gaussian": 12.0,          
    "gaussian_width": 1.0,
}

def main():
    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()