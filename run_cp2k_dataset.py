import torch
from train_utils import loss, training_workflow

# ---------------------------------------------------------
# Custom CP2K Dataset Training & Evaluation
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'cp2k_material',
    "dbpath": "/capstor/store/cscs/pasc/c33/manasa/2D_oxide_datasets/",
    "output_folder": 'outputs_cp2k_material',
    "run_name": 'cp2k_material',
    "open_shell": False,
    
    # Execution Mode
    "train_or_eval": "train",         # Set to "train" to start learning
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 1, 
    "num_val": 1, 
    "num_test": 0,
    "batch_size": 10,                 # 1 for eval, usually 10 for train
    "distribute_graphs": True,       # Distribute graphs and perform communication in the forward pass (ongoing)
    
    # Symmetry Reductions 
    "reduce_edge": False,            # Unused!
    "reduce_node": True,             # Enforce inter-orbital interactions equal
    "reduce_node_intra": True,       # Enforce 0 odd degrees for intra-orbital
    
    # Training Hyperparameters
    "num_epochs": 20000,
    "dtype": torch.float32,          # 32 here for cupy kernel
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
    "save_frequency": 10,
    "restart_backbone": False,       # Restart options
    "restart_head": False,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": False,         # Scale & shift the node labels

    # Evals
    "compute_total_energy": False,

    # Model Architecture
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 4.5,
    "rcut_gaussian": 10.0,           # 2 * rcut_orbitals
    "gaussian_width": 1.0,
    "include_edges": True,           # Based on fock_matrix target
    "head_type": 'gated',
}

if __name__ == "__main__":

    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run()