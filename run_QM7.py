import torch
from maloq.train_utils import loss, training_workflow

# ---------------------------------------------------------
# QM7 Water Dataset Training & Evaluation
# ---------------------------------------------------------
config = {

    # Dataset Paths & Naming
    "dataset_name": 'QM7',
    "dbpath": '/capstor/store/cscs/userlab/lp16/gnn_datasets/datasets/schnorb_hamiltonian_water.db',
    "output_folder": 'outputs_QM7_water',
    "run_name": 'water',
    "open_shell": False,
    
    # Execution Mode
    "train_or_eval": "train",         # Change to "eval" for testing
    "train_backbone": True,          # Set to False to freeze the backbone
    "train_head": True,              # Set to False to freeze the head
    
    # Data Splitting
    "num_train": 500,
    "num_val": 500,
    "num_test": 4500,
    "shuffle": True,
    "batch_size": 5,          # 1 for eval, 10 for train (set to 1 for water script)
    "distribute_graphs": False,       # Distribute graphs and perform communication in the forward pass (ongoing implementation, not finished)
    
    # Symmetry Reductions
    "reduce_edge": True,
    "reduce_node": True,
    "reduce_node_intra": True,
    
    # Training Hyperparameters
    "num_epochs": 200000,
    "dtype": torch.float32,
    "lr_init": 1e-4,
    "optimizer_type": "adamw",
    "weight_decay": 1e-4,
    "scheduler_type": 'plateau', # 'plateau' or 'cosine'
    "patience": 100,
    "threshold": 1e-5,
    "eta_min": 1e-8,
    "step_every_epoch": True,   # Scheduler steps every epoch for Water
    
    # Loss & Checkpointing
    "loss_target": 'fock_matrix',
    "train_loss_fxn": loss.rmse_mse_padded_loss,
    "test_loss_fxn": loss.l1_unpadded_loss,
    "save_frequency": 5,
    "restart_backbone": False,
    "restart_head": False,
    "restart_optimizer": False,
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": False,   # Set to True to enable scaling

    # Evals
    "compute_total_energy": False,

    # Model Architecture
    "wigner_backend": "torch",  # "triton" for fused Triton kernel (requires GPU + triton)
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 6.0,
    "rcut_gaussian": 12.0,    # rcut_orbitals * 2
    "gaussian_width": 1.0,
}

def main():
    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()