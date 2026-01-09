import torch
from train_utils import loss, training_workflow
from dataset_utils.ASEDataset import ASEAtomsData

# ---------------------------------------------------------
# QM7 Water Dataset Training & Evaluation
# ---------------------------------------------------------
config = {

    # Dataset Paths & Naming
    "dataset_name": 'QM7',
    "dbpath": '/capstor/store/cscs/userlab/lp16/gnn_datasets/datasets/schnorb_hamiltonian_water.db',
    "output_folder": 'outputs_QM7_water_trained',
    "run_name": 'water_final',
    
    # Execution Mode
    "train_or_eval": "eval",         # Change to "eval" for testing
    "train_backbone": True,          # Set to False to freeze the backbone
    "train_head": True,              # Set to False to freeze the head
    
    # Data Splitting
    "num_train": 500,
    "num_val": 500,
    "num_test": 100,
    "batch_size": 10,          # 1 for eval, 10 for train (set to 1 for water script)
    
    # Symmetry Reductions
    "reduce_edge": False,
    "reduce_node": False,
    "reduce_node_intra": False,
    
    # Training Hyperparameters
    "num_epochs": 20000,
    "dtype": torch.float32, # torch.float64!
    "lr_init": 1e-5,
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
    "save_frequency": 10,
    "restart_backbone": True,
    "restart_head": True,
    "restart_optimizer": True,
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": False,   # Set to True to enable scaling

    # Evals
    "compute_total_energy": False,

    # Model Architecture
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 8.0,
    "rcut_gaussian": 16.0,    # rcut_orbitals * 2
    "gaussian_width": 1.0,
    "include_edges": True,
    "head_type": 'gated',     # 'linear' or 'gated'
}

if __name__ == "__main__":
    # Initialize the specific database type
    database = ASEAtomsData(config['dbpath'])
    
    # Initialize and run the workflow
    workflow = training_workflow.TrainingWorkflow(config)
    workflow.run(database)