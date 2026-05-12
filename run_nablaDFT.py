import torch
from train_utils import loss, training_workflow

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
    "train_or_eval": "train",         # Set to "train" to start learning
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 4, #12081, #12081,
    "num_val": 4,#64,
    "num_test": 1,
    "batch_size": 1,                 # if distributing graphs, this is the number of molecules per distributed graph
    "distribute_graphs": True,      # Distribute graphs and perform communication in the forward pass (ongoing)
    "partition_type": 'metis',       # 'linear', 'low_nn', 'metis', 'worstcase'
    "dist_backend": 'nccl',          # 'gloo' for CPU, 'nccl' for GPU (if distributed training is implemented)

    # Symmetry Reductions 
    "reduce_edge": False,            # Unused!
    "reduce_node": True,             # Enforce inter-orbital interactions equal
    "reduce_node_intra": True,       # Enforce 0 odd degrees for intra-orbital
    
    # Training Hyperparameters
    "num_epochs": 700,
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
    "save_frequency": 3,
    "restart_backbone": False,       # Restart options
    "restart_head": False,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": True,         # Scale & shift the node labels

    # Evals
    "compute_total_energy": False,

    # Model Architecture
    "wigner_backend": "triton",  # "triton" for fused Triton kernel (requires GPU + triton)
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

    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run()