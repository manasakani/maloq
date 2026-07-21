import argparse
import os

import torch
import torch.distributed as dist
from train_utils import loss, training_workflow

# ---------------------------------------------------------
# nablaDFT Dataset Training & Evaluation
# ---------------------------------------------------------
config = {
    # Dataset Paths & Naming
    "dataset_name": 'nablaDFT',
    "dbpath": "/capstor/store/cscs/pasc/c33/manasa/nablaDFT_datasets/train_2k.db",
    "output_folder": 'outputs/nablaDFT_test',
    "run_name": 'nabla_2k',           # used in config filename saved to output folder
    "open_shell": False,

    # Experiment Tracking
    "use_wandb": False,
    "wandb_project": "maloq",
    "wandb_entity": None,
    "wandb_mode": "online",
    "validation_matrix_metrics": False,
    "validation_matrix_metrics_frequency": 1,
    
    # Execution Mode
    "train_or_eval": "train",         # Set to "train" to start learning
    "train_backbone": True,
    "train_head": True,
    
    # Data Splitting
    "num_train": 12081,
    "num_val": 64,
    "num_test": 1,
    "batch_size": 10,                # if distributing graphs, this is the number of molecules per distributed graph
    "distribute_graphs": False,      # Distribute graphs and perform communication in the forward pass (ongoing)
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
    "optimizer_type": "adam",
    "weight_decay": 0.0,
    # SOAP uses lr_init unless soap_lr is set explicitly. The dimension limit
    # prevents allocating very large quadratic preconditioner matrices.
    "soap_lr": None,
    "soap_betas": (0.95, 0.95),
    "soap_precondition_frequency": 10,
    "soap_max_precondition_dim": 256,
    "soap_precondition_1d": False,
    # Muon is applied to backbone matrices; the output head and vector/scalar
    # parameters use AdamW at muon_adamw_lr (or lr_init when unset).
    "muon_lr": 2e-2,
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_adamw_lr": None,
    "muon_adamw_betas": (0.9, 0.95),
    "scheduler_type": 'cosine',      # 'plateau' or 'cosine'
    "eta_min": 1e-8,                 # For cosine scheduler
    "patience": 500,                 # For plateau scheduler (if swapped)
    "threshold": 1e-8,
    "step_every_epoch": False,       # Cosine usually steps per iteration
    
    # Loss & Checkpointing
    "loss_target": 'energies',
    "train_loss_fxn": loss.rmse_mse_padded_loss,
    "test_loss_fxn": loss.l1_unpadded_loss,
    "save_frequency": 10,
    "restart_backbone": False,       # Restart options
    "restart_head": False,
    "restart_optimizer": False,      
    "backbone_checkpoint": 'backbone.pt',
    "head_checkpoint": 'head.pt',
    "scale_and_shift": True,         # Scale & shift the node labels or energies

    # Evals
    "compute_total_energy": False,

    # Model Architecture
    "wigner_backend": "triton",  # "triton" for fused Triton kernel (requires GPU + triton)
    "l_embedding_dim": 128,
    "num_distance_basis": 128,
    "num_mp_layers": 3,
    "rcut_orbitals": 8.0,
    "rcut_gaussian": 16.0,           # 2 * rcut_orbitals
    "gaussian_width": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate MALOQ on nablaDFT")
    parser.add_argument("--dbpath", default=None)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--wigner-backend", choices=("torch", "triton"), default=None)
    parser.add_argument(
        "--optimizer-type",
        choices=("adam", "adamw", "soap", "muon"),
        default=None,
    )
    parser.add_argument("--soap-lr", type=float, default=None)
    parser.add_argument("--soap-precondition-frequency", type=int, default=None)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=None)
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--muon-adamw-lr", type=float, default=None)
    parser.add_argument("--muon-momentum", type=float, default=None)
    parser.add_argument("--muon-ns-steps", type=int, default=None)
    parser.add_argument(
        "--muon-nesterov", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"), default=None
    )
    parser.add_argument(
        "--validation-matrix-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--validation-matrix-metrics-frequency", type=int, default=None
    )
    parser.add_argument(
        "--scale-and-shift", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def build_config(args):
    run_config = dict(config)
    if args.dbpath is not None:
        run_config["dbpath"] = args.dbpath
    if args.output_folder is not None:
        run_config["output_folder"] = args.output_folder
    if args.wigner_backend is not None:
        run_config["wigner_backend"] = args.wigner_backend
    if args.optimizer_type is not None:
        run_config["optimizer_type"] = args.optimizer_type
    if args.soap_lr is not None:
        run_config["soap_lr"] = args.soap_lr
    if args.soap_precondition_frequency is not None:
        run_config["soap_precondition_frequency"] = args.soap_precondition_frequency
    if args.soap_max_precondition_dim is not None:
        run_config["soap_max_precondition_dim"] = args.soap_max_precondition_dim
    if args.muon_lr is not None:
        run_config["muon_lr"] = args.muon_lr
    if args.muon_adamw_lr is not None:
        run_config["muon_adamw_lr"] = args.muon_adamw_lr
    if args.muon_momentum is not None:
        run_config["muon_momentum"] = args.muon_momentum
    if args.muon_ns_steps is not None:
        run_config["muon_ns_steps"] = args.muon_ns_steps
    if args.muon_nesterov is not None:
        run_config["muon_nesterov"] = args.muon_nesterov
    if args.wandb is not None:
        run_config["use_wandb"] = args.wandb
    if args.wandb_project is not None:
        run_config["wandb_project"] = args.wandb_project
    if args.wandb_mode is not None:
        run_config["wandb_mode"] = args.wandb_mode
    if args.validation_matrix_metrics is not None:
        run_config["validation_matrix_metrics"] = args.validation_matrix_metrics
    if args.validation_matrix_metrics_frequency is not None:
        run_config["validation_matrix_metrics_frequency"] = (
            args.validation_matrix_metrics_frequency
        )
    if args.scale_and_shift is not None:
        run_config["scale_and_shift"] = args.scale_and_shift
    run_config["train_or_eval"] = args.mode
    run_config["restart_backbone"] = args.mode == "eval"
    run_config["restart_head"] = args.mode == "eval"

    if args.smoke:
        run_config.update(
            num_train=2,
            num_val=1,
            num_test=1,
            batch_size=1,
            num_epochs=1,
            save_frequency=1,
            wigner_backend=args.wigner_backend or "torch",
            l_embedding_dim=32,
            num_distance_basis=16,
            num_mp_layers=1,
        )
        if args.output_folder is None:
            run_config["output_folder"] = "outputs/nablaDFT_energy_smoke"
    return run_config


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    workflow = None
    try:
        workflow = training_workflow.TrainingWorkflow(build_config(parse_args()))
        workflow.run()
    finally:
        if workflow is not None:
            workflow.finish_tracking()
        if dist.is_initialized():
            dist.destroy_process_group()
