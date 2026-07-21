import torch
from train_utils import loss, training_workflow

# ---------------------------------------------------------
# Custom CP2K Dataset Training & Evaluation
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'cp2k_material',
    "dbpath": ['/capstor/scratch/cscs/mdossena/MoS2_Al2O3_SELF_PASS_32_32_21_MH1', 
                '/capstor/scratch/cscs/mdossena/MoS2_HfO2_SELF_PASS_32_32_21_MH1'],
    "output_folder": 'outputs/maloq',
    "run_name": 'cp2k_material',
    "open_shell": False,
    
    # Execution Mode
    "train_or_eval": "train",         # Set to "train" to start learning
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 50,
    "num_val": 1, 
    "num_test": 1,
    "shuffle": True,
    "batch_size": 1,                 # 1 for eval, usually 10 for train
    "distribute_graphs": True,         # Distribute graphs and perform communication in the forward pass (ongoing)
    "partition_type": 'linear-edgewise',       # 'linear-atomwise', 'low_nn', 'metis', 'worstcase'
    
    # Symmetry Reductions 
    "reduce_edge": False,            # Unused!
    "reduce_node": True,             # Enforce inter-orbital interactions equal
    "reduce_node_intra": True,       # Enforce 0 odd degrees for intra-orbital
    
    # Training Hyperparameters
    "num_epochs": 400,
    "dtype": torch.float32,          # 32 here for cupy kernel
    "lr_init": 1e-6,
    "optimizer_type": "adamw",        # standard Adam
    "weight_decay": 0.0,
    "scheduler_type": 'cosine',      # 'plateau' or 'cosine'
    "eta_min": 1e-8,                 # For cosine scheduler
    "patience": 50,                 # For plateau scheduler (if swapped)
    "threshold": 1e-8,
    "step_every_epoch": False,       # Cosine usually steps per iteration
    "compute_uncoupled_loss": False,  # Compute loss on uncoupled predictions (if True, also requires "scale_and_shift"=True)
    
    # Loss & Checkpointing
    "loss_target": 'fock_matrix',
    "train_loss_fxn": loss.rmse_mse_padded_loss,
    "test_loss_fxn": loss.l1_unpadded_loss,
    "save_frequency": 10,
    "restart_backbone": True,       # Restart options
    "restart_head": True,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": True,         # Scale & shift the node labels

    # Evals
    "compute_total_energy": False,

    # Model Architecture
    "wigner_backend": "triton",  # "triton" for fused Triton kernel (requires GPU + triton)
    "l_embedding_dim": 128,
    "hidden_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "message_type": "source-target-message",
    "rcut_orbitals": 5.0, 
    "rcut_gaussian": 12.0,           # 2 * rcut_orbitals
    "gaussian_width": 1.0,
}

if __name__ == "__main__":

    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run()
