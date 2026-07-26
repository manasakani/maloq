# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

import os
import time
import random
import math
import numpy as np
import torch
import json
from e3nn.o3 import Irreps
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from pathlib import Path

from . import loss, optimizers, utils_compute, splittrainer
from ..dataset_utils import get_loader, get_scale_shift
from ..dataset_utils.ASEDataset import distribute_data, ASEDataset, ASEAtomsData
from ..dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
from ..helm.esen_osh import eSEN_Backbone, Fock_Irreps_Head, HELM_Force_Head, HELM_Energy_Head
from ..helm.muon_fock_head import MuonFockIrrepsHead
from ..helm.qhflow3_clean import QHFlow3MaloqBackbone
from ..helm.static_te_head import StaticTensorExpansionHead


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


class TrainingWorkflow:

    DEFAULTS = {
        "run_name": "run",
        "output_folder": "outputs/run",
        "seed": 42,
        "backbone_type": "esen",
        "head_type": "maloq",
        "open_shell": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "wigner_backend": "torch",
        "gate_act_type": "tanh",
        "mlp_type": "spectral",
        "message_passing_schedule": "interleaved",
        "initial_edge_state_mode": "edge_degree",
        "num_edge_layers": None,
        "output_l_embedding_dim": None,
        "nte_output_projection_mode": "so3_linear",
        "use_edge_envelope": False,
        "use_edge_scalar_modulation": False,
        "residual_update_scale_mode": "none",
        "residual_update_scale_init": 1.0,
        "residual_update_scale_log_range": 0.0,
        "unscaled_node_layers": (),
        "repeat_system_embedding_each_node_block": False,
        "node_stack_mode": "nte",
        "edge_stack_mode": "recurrent",
        "qhflow3_layer_gaussian_width": 2.0,
        "qhflow3_layer_grid_ffn_chunk_size": 512,
        "qhflow3_exact_pair_rng_aligned": False,
        "edge_atom_norm_type": None,
        "edge_post_residual_norm_type": None,
        "direct_edgewise_layers": (),
        "edge_atomwise_output_mode": "residual_scaled",
        "edge_norm1_position": "post_edgewise",
        "esen_grid_resolution": None,
        "nte_input_conditioning": "none",
        "qhflow3_max_radius": 12.0,
        "qhflow3_radius_embed_dim": 32,
        "qhflow3_grid_resolution": 48,
        "qhflow3_grid_ffn_chunk_size": 512,
        "qhflow3_use_overlap": True,
        "qhflow3_muonize_output_projection": False,
        "static_te_init_mode": "zero",
        "static_te_init_std": 1.0,
        "static_te_gate_degrees": (),
        "static_te_gate_activation": "none",
        "static_te_gate_init": 1.0,
        "distribute_graphs": False,
        "partition_type": None,
        "tiling_dims": None,
        "lr_init": 1e-4,
        "optimizer_type": "adam",
        "soap_lr": None,
        "soap_betas": (0.95, 0.95),
        "soap_shampoo_beta": -1.0,
        "soap_eps": 1e-8,
        "soap_precondition_frequency": 10,
        "soap_max_precondition_dim": 256,
        "soap_precondition_1d": False,
        "soap_normalize_grads": False,
        "muon_lr": 2e-2,
        "muon_momentum": 0.95,
        "muon_nesterov": True,
        "muon_ns_steps": 5,
        "muon_adamw_lr": None,
        "muon_adamw_betas": (0.9, 0.95),
        "muon_adamw_eps": 1e-10,
        "muon_output_projection_policy": "shape_muon",
        "gradient_clip_val": None,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 1000,
        "scheduler_power": 1.0,
        "min_lr_ratio": 0.0,
        "compute_total_energy": False,
        "dist_backend": "nccl" if torch.cuda.is_available() else "gloo",
        "use_wandb": False,
        "wandb_project": "maloq",
        "wandb_entity": None,
        "wandb_mode": "online",
        "wandb_run_name": None,
        "wandb_group": None,
        "wandb_job_type": None,
        "wandb_tags": (),
        "experiment_version": 1,
        "wandb_log_every_n_steps": 10,
        "validation_matrix_metrics": False,
        "validation_matrix_metrics_frequency": 1,
        "scale_shift_mode": "standardize",
    }

    def __init__(self, config):
        self.config = self.DEFAULTS | config
        self.config['output_folder'] = self.resolve_output_folder(
            self.config['output_folder'], self.config['run_name']
        )
        self.wandb_run = None
        self.setup_environment()

        # check_input_config will raise errors if there are incompatible settings
        self.check_input_config()
        self.wandb_run = self.setup_tracking()

    @staticmethod
    def resolve_output_folder(output_folder, run_name):
        """Resolve every model-run output below the project ``outputs`` tree."""
        output_path = Path(
            os.path.expandvars(os.path.expanduser(str(output_folder)))
        )

        if output_path.is_absolute():
            resolved = output_path.resolve()
        else:
            parts = output_path.parts
            if parts and parts[0] == "outputs":
                relative_path = output_path
            elif len(parts) == 1 and parts[0].startswith("outputs_"):
                relative_path = Path("outputs") / parts[0].removeprefix("outputs_")
            else:
                relative_path = Path("outputs") / output_path

            if relative_path == Path("outputs"):
                relative_path /= str(run_name)
            resolved = (PROJECT_ROOT / relative_path).resolve()

        output_root = OUTPUT_ROOT.resolve()
        if resolved != output_root and output_root not in resolved.parents:
            raise ValueError(
                "Model outputs must be stored below "
                f"{output_root}; received {output_folder!r}."
            )
        return str(resolved)

    def setup_tracking(self):
        """Initializes optional experiment tracking on the primary rank."""
        if self.rank != 0 or not self.config.get('use_wandb', False):
            return None

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking was requested, but wandb is not installed."
            ) from exc

        wandb_config = {
            key: (
                value
                if isinstance(value, (str, int, float, bool)) or value is None
                else value.__name__
                if hasattr(value, '__name__')
                else str(value)
            )
            for key, value in self.config.items()
        }
        run = wandb.init(
            project=self.config['wandb_project'],
            entity=self.config.get('wandb_entity'),
            name=self.config.get('wandb_run_name') or self.config['run_name'],
            group=self.config.get('wandb_group'),
            job_type=self.config.get('wandb_job_type'),
            tags=list(self.config.get('wandb_tags') or ()),
            dir=self.config['output_folder'],
            config=wandb_config,
            mode=self.config.get('wandb_mode', 'online'),
        )
        print(
            f"W&B tracking enabled ({self.config.get('wandb_mode', 'online')}): "
            f"{run.name}",
            flush=True,
        )
        return run

    def finish_tracking(self):
        """Finishes the active experiment tracking run, if any."""
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    def setup_environment(self):
        """Initializes seeds, dtypes, and distributed compute environment."""
        seed = int(self.config['seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.set_default_dtype(self.config['dtype'])

        # torchrun, Open MPI, and SLURM distributed setup.
        self.rank, self.world_size, self.local_rank = (
            utils_compute.distributed_context()
        )

        compute_start = time.perf_counter()
        self.device = utils_compute.setup_env(
            self.rank,
            self.world_size,
            backend=self.config['dist_backend'],
            local_rank=self.local_rank,
        )
        compute_end = time.perf_counter()
        
        if self.rank == 0:
            print(f"Time to setup distributed environment: {compute_end - compute_start:.4f}s")
            if not os.path.exists(self.config['output_folder']):
                os.makedirs(self.config['output_folder'])
        dist.barrier()

        if self.config.get('distribute_graphs', False):
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            mpi_world_size = MPI.COMM_WORLD.Get_size()
            if (mpi_rank, mpi_world_size) != (self.rank, self.world_size):
                raise RuntimeError(
                    'Distributed-graph training requires matching MPI and '
                    'torch.distributed ranks. Launch it with mpirun rather '
                    'than torchrun. '
                    f'MPI={mpi_rank}/{mpi_world_size}, '
                    f'torch={self.rank}/{self.world_size}.'
                )

    def check_input_config(self):
        """Validates the configuration for incompatible settings, and writes config to output folder."""

        optimizer_type = self.config.get('optimizer_type', 'adam').lower()
        valid_optimizers = {'adam', 'adamw', 'soap', 'muon'}
        if optimizer_type not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer '{optimizer_type}'. Choose one of "
                f"{sorted(valid_optimizers)}."
            )
        self.config['optimizer_type'] = optimizer_type

        if self.config['gate_act_type'] not in {'tanh', 'sigmoid'}:
            raise ValueError("gate_act_type must be 'tanh' or 'sigmoid'.")
        if self.config['mlp_type'] not in {'spectral', 'grid'}:
            raise ValueError("mlp_type must be 'spectral' or 'grid'.")
        if self.config['message_passing_schedule'] not in {
            'interleaved', 'node_then_edge'
        }:
            raise ValueError(
                "message_passing_schedule must be 'interleaved' or "
                "'node_then_edge'."
            )
        if self.config['initial_edge_state_mode'] not in {
            'edge_degree', 'zero'
        }:
            raise ValueError(
                "initial_edge_state_mode must be 'edge_degree' or 'zero'."
            )
        if self.config['residual_update_scale_mode'] not in {
            'none', 'bounded_degree'
        }:
            raise ValueError(
                "residual_update_scale_mode must be 'none' or "
                "'bounded_degree'."
            )
        self.config['unscaled_node_layers'] = tuple(
            int(index) for index in self.config['unscaled_node_layers']
        )
        if len(set(self.config['unscaled_node_layers'])) != len(
            self.config['unscaled_node_layers']
        ):
            raise ValueError("unscaled_node_layers must not contain duplicates.")
        if any(
            index < 1 or index > int(self.config['num_mp_layers'])
            for index in self.config['unscaled_node_layers']
        ):
            raise ValueError(
                "unscaled_node_layers must contain 1-based indices within "
                "num_mp_layers."
            )
        if (
            self.config['repeat_system_embedding_each_node_block']
            and self.config['nte_input_conditioning'] != 'qhflow3_exact'
        ):
            raise ValueError(
                "repeat_system_embedding_each_node_block requires "
                "nte_input_conditioning='qhflow3_exact'."
            )
        if self.config['node_stack_mode'] not in {'nte', 'qhflow3_exact'}:
            raise ValueError(
                "node_stack_mode must be 'nte' or 'qhflow3_exact'."
            )
        if self.config['nte_output_projection_mode'] not in {
            'so3_linear', 'qhflow3_irrep_linear'
        }:
            raise ValueError(
                "nte_output_projection_mode must be 'so3_linear' or "
                "'qhflow3_irrep_linear'."
            )
        if (
            self.config['nte_output_projection_mode']
            != 'so3_linear'
            and self.config['backbone_type'] != 'esen'
        ):
            raise ValueError(
                "nte_output_projection_mode='qhflow3_irrep_linear' "
                "requires backbone_type='esen'."
            )
        if self.config['edge_stack_mode'] not in {
            'recurrent', 'nte_parallel', 'qhflow3_parallel',
            'qhflow3_exact_parallel'
        }:
            raise ValueError(
                "edge_stack_mode must be 'recurrent', 'nte_parallel', "
                "'qhflow3_parallel', or 'qhflow3_exact_parallel'."
            )
        exact_qhflow3_layers = (
            self.config['node_stack_mode'] == 'qhflow3_exact'
            or self.config['edge_stack_mode'] == 'qhflow3_exact_parallel'
        )
        if (
            self.config['qhflow3_exact_pair_rng_aligned']
            and self.config['edge_stack_mode'] != 'qhflow3_exact_parallel'
        ):
            raise ValueError(
                "qhflow3_exact_pair_rng_aligned requires "
                "edge_stack_mode='qhflow3_exact_parallel'."
            )
        if exact_qhflow3_layers and self.config['backbone_type'] != 'esen':
            raise ValueError(
                "Exact QHFlow3 layer transplants require backbone_type='esen'."
            )
        if exact_qhflow3_layers and self.config['mlp_type'] != 'grid':
            raise ValueError(
                "Exact QHFlow3 layer transplants require mlp_type='grid'."
            )
        if exact_qhflow3_layers and self.config['distribute_graphs']:
            raise ValueError(
                "Exact QHFlow3 layer transplants do not support distributed "
                "graph training."
            )
        if float(self.config['qhflow3_layer_gaussian_width']) <= 0.0:
            raise ValueError("qhflow3_layer_gaussian_width must be positive.")
        qhflow3_layer_chunk = self.config['qhflow3_layer_grid_ffn_chunk_size']
        if qhflow3_layer_chunk is not None and int(qhflow3_layer_chunk) <= 0:
            raise ValueError(
                "qhflow3_layer_grid_ffn_chunk_size must be positive."
            )
        if (
            self.config['edge_stack_mode'] in {
                'nte_parallel', 'qhflow3_parallel',
                'qhflow3_exact_parallel'
            }
            and self.config['message_passing_schedule'] != 'node_then_edge'
        ):
            raise ValueError(
                "Parallel edge stacks require "
                "message_passing_schedule='node_then_edge'."
            )
        if (
            self.config['node_stack_mode'] == 'qhflow3_exact'
            and self.config['message_passing_schedule'] != 'node_then_edge'
        ):
            raise ValueError(
                "The exact QHFlow3 node stack requires node_then_edge."
            )
        valid_edge_norm_types = {
            None, 'layer_norm', 'layer_norm_sh', 'rms_norm_sh'
        }
        for option_name in (
            'edge_atom_norm_type',
            'edge_post_residual_norm_type',
        ):
            if self.config[option_name] not in valid_edge_norm_types:
                raise ValueError(
                    f"{option_name} must be None, 'layer_norm', "
                    "'layer_norm_sh', or 'rms_norm_sh'."
                )
        if len(set(self.config['direct_edgewise_layers'])) != len(
            self.config['direct_edgewise_layers']
        ):
            raise ValueError("direct_edgewise_layers must not contain duplicates.")
        num_edge_layers = (
            int(self.config['num_mp_layers'])
            if self.config['num_edge_layers'] is None
            else int(self.config['num_edge_layers'])
        )
        if any(
            index < 1 or index > num_edge_layers
            for index in self.config['direct_edgewise_layers']
        ):
            raise ValueError(
                "direct_edgewise_layers must contain 1-based indices within "
                "num_edge_layers."
            )
        if self.config['initial_edge_state_mode'] == 'zero':
            if self.config['backbone_type'] != 'esen':
                raise ValueError(
                    "initial_edge_state_mode='zero' requires "
                    "backbone_type='esen'."
                )
            if 'matrix' not in self.config['loss_target']:
                raise ValueError(
                    "initial_edge_state_mode='zero' requires a matrix "
                    "loss target with edge embeddings."
                )
            if self.config['message_passing_schedule'] != 'node_then_edge':
                raise ValueError(
                    "initial_edge_state_mode='zero' requires "
                    "message_passing_schedule='node_then_edge'."
                )
            if self.config['edge_stack_mode'] != 'recurrent':
                raise ValueError(
                    "initial_edge_state_mode='zero' requires "
                    "edge_stack_mode='recurrent'."
                )
            if (
                self.config.get('message_type', 'source-target')
                != 'source-target'
            ):
                raise ValueError(
                    "initial_edge_state_mode='zero' requires "
                    "message_type='source-target'."
                )
            if 1 in self.config['direct_edgewise_layers']:
                raise ValueError(
                    "initial_edge_state_mode='zero' is redundant with "
                    "direct_edgewise_layers containing EdgeBlock 1."
                )
        if self.config['edge_atomwise_output_mode'] not in {
            'residual_scaled', 'direct'
        }:
            raise ValueError(
                "edge_atomwise_output_mode must be "
                "'residual_scaled' or 'direct'."
            )
        if self.config['edge_norm1_position'] not in {
            'post_edgewise', 'pre_node'
        }:
            raise ValueError(
                "edge_norm1_position must be "
                "'post_edgewise' or 'pre_node'."
            )
        if self.config['muon_output_projection_policy'] not in {
            'shape_muon', 'adamw'
        }:
            raise ValueError(
                "muon_output_projection_policy must be 'shape_muon' or 'adamw'."
            )
        if (
            self.config['qhflow3_muonize_output_projection']
            and self.config['backbone_type'] != 'qhflow3_clean'
        ):
            raise ValueError(
                "qhflow3_muonize_output_projection requires "
                "backbone_type='qhflow3_clean'."
            )
        if (
            self.config['qhflow3_muonize_output_projection']
            and self.config['optimizer_type'] != 'muon'
        ):
            raise ValueError(
                "qhflow3_muonize_output_projection requires optimizer_type='muon'."
            )
        if (
            self.config['esen_grid_resolution'] is not None
            and int(self.config['esen_grid_resolution']) <= 0
        ):
            raise ValueError("esen_grid_resolution must be positive or None.")
        if self.config['nte_input_conditioning'] not in {
            'none', 'overlap', 'qhflow3_exact'
        }:
            raise ValueError(
                "nte_input_conditioning must be 'none', 'overlap', or "
                "'qhflow3_exact'."
            )
        if (
            self.config['nte_input_conditioning'] != 'none'
            and self.config['backbone_type'] != 'esen'
        ):
            raise ValueError(
                "nte_input_conditioning is available only for the eSEN backbone."
            )
        if (
            self.config['nte_input_conditioning'] != 'none'
            and self.config['distribute_graphs']
        ):
            raise ValueError(
                "NTE matrix input conditioning requires distribute_graphs=False."
            )
        if self.config['backbone_type'] not in {'esen', 'qhflow3_clean'}:
            raise ValueError("backbone_type must be 'esen' or 'qhflow3_clean'.")
        if self.config['head_type'] not in {
            'maloq',
            'maloq_muon',
            'maloq_semantic_global_muon',
            'maloq_semantic_global_gate_muon',
            'static_te',
        }:
            raise ValueError(
                "head_type must be 'maloq', 'maloq_muon', "
                "'maloq_semantic_global_muon', "
                "'maloq_semantic_global_gate_muon', or 'static_te'."
            )
        if (
            self.config['head_type']
            in {
                'maloq_muon',
                'maloq_semantic_global_muon',
                'maloq_semantic_global_gate_muon',
            }
            and self.config['reduce_edge']
        ):
            raise ValueError(
                "Muon-compatible MALOQ heads currently require reduce_edge=False."
            )
        if self.config['head_type'] == 'static_te':
            if self.config['open_shell']:
                raise ValueError("static_te currently supports closed-shell data only.")
            if self.config['reduce_edge']:
                raise ValueError("static_te currently requires reduce_edge=False.")
            if self.config['static_te_init_mode'] not in {'zero', 'normal'}:
                raise ValueError("static_te_init_mode must be 'zero' or 'normal'.")
            if float(self.config['static_te_init_std']) <= 0.0:
                raise ValueError("static_te_init_std must be positive.")
            gate_degrees = tuple(
                int(degree) for degree in self.config['static_te_gate_degrees']
            )
            if len(set(gate_degrees)) != len(gate_degrees):
                raise ValueError("static_te_gate_degrees must not contain duplicates.")
            if any(degree < 0 for degree in gate_degrees):
                raise ValueError("static_te_gate_degrees must be non-negative.")
            gate_activation = self.config['static_te_gate_activation']
            if gate_activation not in {'none', 'residual_tanh', 'sigmoid'}:
                raise ValueError(
                    "static_te_gate_activation must be 'none', "
                    "'residual_tanh', or 'sigmoid'."
                )
            if gate_degrees and gate_activation == 'none':
                raise ValueError(
                    "static_te_gate_degrees requires an active gate."
                )
            if not gate_degrees and gate_activation != 'none':
                raise ValueError(
                    "An active static_te gate requires static_te_gate_degrees."
                )
            gate_init = float(self.config['static_te_gate_init'])
            if gate_activation == 'residual_tanh' and not 0.0 < gate_init < 2.0:
                raise ValueError(
                    "residual_tanh static_te_gate_init must be between 0 and 2."
                )
            if gate_activation == 'sigmoid' and not 0.0 < gate_init < 1.0:
                raise ValueError(
                    "sigmoid static_te_gate_init must be between 0 and 1."
                )
            self.config['static_te_gate_degrees'] = gate_degrees
        if (
            self.config['backbone_type'] == 'qhflow3_clean'
            and self.config['output_l_embedding_dim'] is None
        ):
            raise ValueError(
                "qhflow3_clean requires output_l_embedding_dim for its bottle width."
            )
        qhflow3_grid_resolution = self.config['qhflow3_grid_resolution']
        if (
            qhflow3_grid_resolution is not None
            and int(qhflow3_grid_resolution) <= 0
        ):
            raise ValueError("qhflow3_grid_resolution must be positive.")
        qhflow3_grid_chunk = self.config['qhflow3_grid_ffn_chunk_size']
        if qhflow3_grid_chunk is not None and int(qhflow3_grid_chunk) <= 0:
            raise ValueError("qhflow3_grid_ffn_chunk_size must be positive.")
        gradient_clip_val = self.config['gradient_clip_val']
        if gradient_clip_val is not None and float(gradient_clip_val) <= 0.0:
            raise ValueError("gradient_clip_val must be positive when specified.")
        gradient_accumulation_steps = int(
            self.config['gradient_accumulation_steps']
        )
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        self.config['gradient_accumulation_steps'] = gradient_accumulation_steps
        if self.config.get('delta_learning', False):
            if self.config['loss_target'] not in {
                'fock_matrix', 'density_matrix'
            }:
                raise ValueError(
                    "delta_learning requires a Hamiltonian or density matrix target."
                )
            if self.config['dataset_name'] != 'QM7':
                raise ValueError(
                    "delta_learning requires the QM7-style ASE data loader."
                )
            if self.config['open_shell']:
                raise ValueError("delta_learning currently supports closed shell only.")
            if self.config['distribute_graphs']:
                raise ValueError(
                    "delta_learning is not implemented for distributed graphs."
                )

        if 'matrix' in self.config['loss_target']:
            self.config['include_edges'] = True
            print("Initializing model with edge embeddings, since loss target involves a matrix.")
        else:
            print("Initializing model without edge embeddings, since loss target does not involve a matrix.")
            self.config['include_edges'] = False

        # wigner_backend exists and is equal to triton
        if self.config.get('wigner_backend', 'torch') == 'triton':
            if self.device.type != 'cuda':
                raise ValueError("Triton Wigner backend requires a CUDA-capable GPU.")
            if self.config['dtype'] == torch.float64:
                raise ValueError("Triton Wigner backend does not support float64 dtype.")

        # Write config settings to the output file if not eval:
        if self.rank == 0 and self.config['train_or_eval'] == 'train':
            config_path = os.path.join(self.config['output_folder'], f"config_{self.config['run_name']}.json")
            serializable_config = {
                k: (v.__name__ if hasattr(v, '__name__') else str(v)) 
                for k, v in self.config.items()
            }
            with open(config_path, 'w') as f:
                json.dump(serializable_config, f, indent=4)

            print(f"Config dumped to {config_path}")

        # if using restart, check that the checkpoint files exist and are not corrupted
        if self.config['restart_backbone']:
            backbone_path = os.path.join(self.config['output_folder'], self.config['backbone_checkpoint'])
            if not os.path.exists(backbone_path):
                raise FileNotFoundError(f"Backbone checkpoint not found at {backbone_path}")
            try:
                torch.load(backbone_path, map_location=self.device)
            except Exception as e:
                raise ValueError(f"Error loading backbone checkpoint from {backbone_path}: {e}")
                
        if self.config['restart_head']:
            head_path = os.path.join(self.config['output_folder'], self.config['head_checkpoint'])
            if not os.path.exists(head_path):
                raise FileNotFoundError(f"Head checkpoint not found at {head_path}")
            try:
                torch.load(head_path, map_location=self.device)
            except Exception as e:
                raise ValueError(f"Error loading head checkpoint from {head_path}: {e}")

        if 'shuffle' not in self.config:
            self.config['shuffle'] = False

        # if partition_type is not specified, set it 'linear-edgewise' if distribute_graphs is True, else None:
        if self.config['distribute_graphs'] and self.config['partition_type'] is None:
            self.config['partition_type'] = 'linear-edgewise'
            print("No partition type specified for distributed graph training; defaulting to 'linear-edgewise'.")   

        # if both reduce_edge and distribute graphs are true, print that there is a known bug!:
        if self.config['reduce_edge'] and self.config['distribute_graphs']:
            raise ValueError("reduce_edge and distribute_graphs cannot both be True, as communication has not been implemented yet in the output head.")
        
        # distribute_graphs cannot be used with non-matrix valued learning targets:
        if self.config['distribute_graphs'] and 'matrix' not in self.config['loss_target']:
            raise ValueError("Distributed graph training is currently only implemented for matrix-valued learning targets (e.g. fock_matrix).")

        if (
            self.config['distribute_graphs']
            and self.config['backbone_type'] == 'qhflow3_clean'
        ):
            raise ValueError(
                "QHFlow3 currently supports multi-GPU data-parallel training, "
                "but not distributed-graph training."
            )

        if self.config['validation_matrix_metrics']:
            if self.config['loss_target'] not in ['fock_matrix', 'density_matrix']:
                raise ValueError(
                    "Validation matrix metrics require a Fock or density matrix loss target."
                )
            if self.config['distribute_graphs']:
                raise ValueError(
                    "Validation matrix metrics are not yet supported with distributed graphs."
                )
        if self.config['validation_matrix_metrics_frequency'] < 1:
            raise ValueError("validation_matrix_metrics_frequency must be at least 1.")
        if self.config['wandb_log_every_n_steps'] < 1:
            raise ValueError("wandb_log_every_n_steps must be at least 1.")
        if not all(
            isinstance(tag, str) and tag.strip()
            for tag in self.config.get('wandb_tags', ())
        ):
            raise ValueError("wandb_tags must contain only non-empty strings.")

        # if hidden_dim is not provided, set it to l_embedding_dim:
        if 'hidden_dim' not in self.config:
            self.config['hidden_dim'] = self.config['l_embedding_dim']
            print(f"hidden_dim not specified; defaulting to l_embedding_dim={self.config['l_embedding_dim']}")

        # if c['message_type'] is not provided, set it to 'source-target':
        if 'message_type' not in self.config:
            self.config['message_type'] = 'source-target'
            print(f"message_type not specified; defaulting to 'source-target'")

    def _handle_scale_shift(self, database=None):
        """Manages the computation or loading of scale/shift factors."""
        if not self.config.get('scale_and_shift'):
            return None

        dataset_name = self.config['dataset_name']
        scale_shift_mode = self.config.get(
            'scale_shift_mode',
            'standardize',
        )
        if scale_shift_mode not in {'standardize', 'shift_only'}:
            raise ValueError(
                "scale_shift_mode must be 'standardize' or 'shift_only'."
            )

        if self.config['loss_target'] in ['fock_matrix', 'density_matrix']:
            configured_path = self.config.get('scale_shift_path')
            if configured_path:
                target_path = Path(configured_path).expanduser()
                if not target_path.is_absolute():
                    target_path = PROJECT_ROOT / target_path
                target_path = target_path.resolve()
                if not target_path.is_file():
                    raise FileNotFoundError(
                        "Configured scale-shift artifact does not exist: "
                        f"{target_path}"
                    )
                print(
                    "[Loading configured scale/shift factors from "
                    f"{target_path}]",
                    flush=True,
                )
                data = torch.load(
                    target_path,
                    map_location='cpu',
                    weights_only=False,
                )
                provenance = data.get('provenance', {})
                expected = {
                    'dataset_name': dataset_name,
                    'loss_target': self.config['loss_target'],
                    'rcut_orbitals': self.config['rcut_orbitals'],
                }
                mismatches = {
                    key: (provenance.get(key), value)
                    for key, value in expected.items()
                    if provenance.get(key) != value
                }
                if mismatches:
                    details = ", ".join(
                        f"{key}={actual!r} (expected {expected_value!r})"
                        for key, (actual, expected_value) in mismatches.items()
                    )
                    raise ValueError(
                        "Scale-shift provenance mismatch: " + details
                    )
                return {
                    "element_scalar_means": data["element_scalar_means"],
                    "element_scalar_stds": data["element_scalar_stds"],
                    "scalar_irrep_indices": data["scalar_irrep_indices"],
                    "normalization_mode": scale_shift_mode,
                }

            if self.config['open_shell']:
                filename = f"element_scale_shifts_osh_{dataset_name}.pt"
            else:
                filename = f"element_scale_shifts_{dataset_name}.pt"

            current_file_path = os.path.dirname(os.path.abspath(__file__))
            target_path = os.path.join(current_file_path, "../fock_utils/", filename)
            file_exists = os.path.exists(target_path)

            if file_exists:
                print(f"[Loading precomputed scale/shift factors for {dataset_name} from {target_path}]")
                data = torch.load(target_path)
                
            # Recompute scale/shift factors for this dataset
            else:
                print(f"[Computing scale/shift factors for {dataset_name}]")
                if database is None:
                    print("Error: Database object is required to compute scale/shift factors but is None.")
                    exit()
                data = get_scale_shift.get_scale_shift(
                    database, dataset_name, self.config['rcut_orbitals'], 
                    dtype=self.config['dtype'], reduce_edge=self.config['reduce_edge'], 
                    filename=filename
                )
                print('Finished computing scale/shift factors! Will use these to scale the node data.')
            
            if self.config['open_shell']:
                return {
                    "element_scalar_means_alpha": data["element_scalar_means_alpha"],  
                    "element_scalar_means_beta": data["element_scalar_means_beta"],   
                    "element_scalar_stds_alpha": data["element_scalar_stds_alpha"],   
                    "element_scalar_stds_beta": data["element_scalar_stds_beta"],    
                    "scalar_irrep_indices": data["scalar_irrep_indices"],
                    "normalization_mode": scale_shift_mode,
                }
            else:
                return {
                    "element_scalar_means": data["element_scalar_means"],
                    "element_scalar_stds": data["element_scalar_stds"],
                    "scalar_irrep_indices": data["scalar_irrep_indices"],
                    "normalization_mode": scale_shift_mode,
                }

        elif self.config['loss_target'] in ['energies']:

            filename = 'stats_nablaDFT/lin_ref_coeffs_nablaDFT.npz'
            current_dir = os.path.dirname(os.path.abspath(__file__))
            energy_ref_file = os.path.join(current_dir, "../dataset_utils/", filename)

            if os.path.exists(energy_ref_file):
                print(f"Loading energy reference coefficients from {energy_ref_file}")
                lin_ref_data = np.load(energy_ref_file)
                element_references_np = lin_ref_data['coeff']  # Shape: (max_atomic_number,)
                
                # Convert to torch tensor and move to device
                element_references = torch.tensor(element_references_np, dtype=self.config['dtype'], device=self.device)
                print(f"Loaded energy references for {len(element_references)} elements")
                
                # Print non-zero references for verification
                nonzero_mask = torch.abs(element_references) > 1e-10
                nonzero_elements = torch.where(nonzero_mask)[0]
                print("Non-zero energy references:")
                for z in nonzero_elements:
                    print(f"  Element Z={z.item()}: {element_references[z].item():.6f} Hartree")
                
                self.config['element_references'] = element_references
                
            else:
                raise FileNotFoundError(f"Energy reference file {energy_ref_file} not found!")

        else:
            raise ValueError(f"Unknown loss target for scale/shift handling: {self.config['loss_target']}")
                        

    def prepare_loaders(self, database_input=None):
        """
        Calculates splits and returns the appropriate DataLoaders.
        database_input can be a single file path, a folder path, or a pre-loaded ASEDataset.
        """
        c = self.config

        db_source = database_input if database_input is not None else c['dbpath']

        if isinstance(db_source, str):
            db_sources = [db_source]
        elif isinstance(db_source, list):
            db_sources = db_source
        else:
            db_sources = []
        is_folder = any(os.path.isdir(src) for src in db_sources)
        # is_folder = isinstance(db_source, str) and os.path.isdir(db_source)

        if is_folder:

            # Custom cp2k datasets - each subfolder contains a structure, hamiltonian, and overlap matrix
            if self.config['dataset_name'] == 'cp2k_material':
                # data_folders = [os.path.join(db_source, f) for f in os.listdir(db_source) if os.path.isdir(os.path.join(db_source, f))]
                data_folders = []
                for src in db_sources:
                    if os.path.isdir(src):
                        subdirs = [
                            os.path.join(src, f) 
                            for f in os.listdir(src) 
                            if os.path.isdir(os.path.join(src, f))
                        ]
                        data_folders.extend(subdirs)
                data_folders.sort()

                if c['shuffle']:
                    random.shuffle(data_folders)
                
                total_needed = c['num_train'] + c['num_val']
                print(f"Found {len(data_folders)} data folders in {db_source}", flush=True)

                train_database = data_folders[:c['num_train']]
                scale_shift_data = self._handle_scale_shift(train_database)

                # if c['dbpath_val'] is provided, take validation folders from there instead of splitting from the training folders:
                if c.get('dbpath_val') is not None:
                    val_db_source = c['dbpath_val']
                    val_data_folders = [os.path.join(val_db_source, f) for f in os.listdir(val_db_source) if os.path.isdir(os.path.join(val_db_source, f))]
                    val_database = val_data_folders[:c['num_val']]
                else:
                    val_database = data_folders[c['num_train']:total_needed] if c['num_val'] > 0 else None

                tr_start, tr_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_train'], c['distribute_graphs'])
                val_start, val_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_val'], c['distribute_graphs'])
                test_start, test_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_test'], c['distribute_graphs'])
            
            # (Omol) folders of db files
            else:
                print(f"Rank {self.rank}: Loading data from folder {db_source}", flush=True)

                # --- 1. Distribute Data across ranks ---
                val_db_source = c.get('dbpath_val')
                if val_db_source is not None:
                    train_data_dict, _ = distribute_data(
                        base_folder=db_source,
                        world_size=self.world_size,
                        rank=self.rank,
                        N_global_train=c['num_train'],
                        N_global_val=0,
                    )
                    _, val_data_dict = distribute_data(
                        base_folder=val_db_source,
                        world_size=self.world_size,
                        rank=self.rank,
                        N_global_train=0,
                        N_global_val=c['num_val'],
                    )
                else:
                    train_data_dict, val_data_dict = distribute_data(
                        base_folder=db_source,
                        world_size=self.world_size,
                        rank=self.rank,
                        N_global_train=c['num_train'],
                        N_global_val=c['num_val'],
                    )

                # --- 2. Load Training Segments ---
                train_datasets = []
                for entry in train_data_dict:
                    ds = ASEDataset(
                        db_path=entry['db_file'],
                        dtype=c['dtype'],
                        open_shell=c['open_shell'],
                        start_idx=entry['start_idx'], 
                        end_idx=entry['end_idx'],
                        matrix_target=c['loss_target'],
                    )
                    train_datasets.append(ds)
                train_database = ConcatDataset(train_datasets) if train_datasets else None

                # --- 3. Load Validation Segments ---
                val_datasets = []
                for entry in val_data_dict:
                    ds = ASEDataset(
                        db_path=entry['db_file'],
                        dtype=c['dtype'],
                        open_shell=c['open_shell'],
                        start_idx=entry['start_idx'],
                        end_idx=entry['end_idx'],
                        matrix_target=c['loss_target'],
                    )
                    val_datasets.append(ds)
                val_database = ConcatDataset(val_datasets) if val_datasets else None

                # When using ConcatDataset, the database is already sliced for the rank.
                # We process the entire local concat object.
                tr_start, tr_end = 0, len(train_database)
                val_start, val_end = 0, len(val_database) if val_database else 0
                
                # Determine Scale/Shift using the local training shard
                scale_shift_data = self._handle_scale_shift(train_database)
            
        else:
            # print(f"Rank {self.rank}: Loading data from single file {db_source}")
            # dist.barrier()

            # We must ensure we have a valid Dataset object here
            # Eg: omol single DB file
            if isinstance(db_source, str):
                print(f"Rank {self.rank}: Initializing ASEDataset from {db_source}")
                db_obj = ASEDataset(
                    db_source,
                    dtype=c['dtype'],
                    open_shell=c['open_shell'],
                    matrix_target=c['loss_target'],
                )
            else:
                db_obj = db_source
            
            # --- SINGLE DB FILE MODE ---
            # 1. Calculate split indices
            tr_start, tr_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_train'], c['distribute_graphs'])
            val_start, val_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_val'], c['distribute_graphs'])
            test_start, test_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_test'], c['distribute_graphs'])

            # 2. Offset validation and test to ensure unique molecules
            val_start += c['num_train']; val_end += c['num_train']
            test_start += (c['num_train'] + c['num_val']); test_end += (c['num_train'] + c['num_val'])

            # 3. Get Scale/Shift
            train_database = db_obj
            val_database = db_obj
            scale_shift_data = self._handle_scale_shift(db_obj)  
            print("Got scaling/shifting factors for this dataset.", flush=True)      

        # 4. Data loading logic
        if c['train_or_eval'] == 'train':
            # Note: argument name in get_loader is 'rcut', not 'rcut_orbitals'
            if train_database is None or len(train_database) == 0:
                train_loader = None
            else:
                train_loader, required_irreps, basis_trans, orb_basis, ls_list = get_loader.get_loader(
                    database=train_database,
                    start_idx=tr_start,
                    end_idx=tr_end,
                    dataset_name=c['dataset_name'],
                    rcut=c['rcut_orbitals'],
                    batch_size=c['batch_size'],
                    dtype=c['dtype'],
                    half_edges=c['reduce_edge'],
                    loss_target_string=c['loss_target'],
                    is_open_shell=c['open_shell'],
                    scale_shift_data=scale_shift_data,
                    distribute_graphs=c['distribute_graphs'],
                    tiling_dims=c['tiling_dims'],
                    partition_type=c['partition_type'],
                    train_or_eval=c['train_or_eval'],
                    delta_learning=c.get('delta_learning', False),
                    load_delta_auxiliary_matrix=(
                        c['backbone_type'] == 'qhflow3_clean'
                        or c['nte_input_conditioning'] == 'qhflow3_exact'
                    ),
                )

                dist.barrier()
                for i in range(self.world_size):
                    if self.rank == i:
                        first_batch = next(iter(train_loader), None)
                        if first_batch is not None:
                            batch = first_batch
                            if not c['open_shell']:
                                num_atoms = batch['node_y'].shape[1] if c['distribute_graphs'] else batch['node_y'].shape[0]
                                num_edges = batch['y'].shape[1] if c['distribute_graphs'] else batch['y'].shape[0]
                            else:
                                num_atoms = batch['node_y_alpha'].shape[1] if c['distribute_graphs'] else batch['node_y_alpha'].shape[0]
                                num_edges = batch['y_alpha'].shape[1] if c['distribute_graphs'] else batch['y_alpha'].shape[0]
                            
                            print(f"Rank {self.rank}: First train batch - Num atoms: {num_atoms}, Num edges: {num_edges}", flush=True)
                    dist.barrier()
            
            if val_database is None or len(val_database) == 0:
                val_loader = None
            else:
                val_loader, *_ = get_loader.get_loader(
                    database=val_database,
                    start_idx=val_start,
                    end_idx=val_end,
                    dataset_name=c['dataset_name'],
                    rcut=c['rcut_orbitals'],
                    batch_size=c['batch_size'],
                    dtype=c['dtype'],
                    half_edges=c['reduce_edge'],
                    loss_target_string=c['loss_target'],
                    is_open_shell=c['open_shell'],
                    scale_shift_data=scale_shift_data,
                    distribute_graphs=c['distribute_graphs'],
                    tiling_dims=c['tiling_dims'],
                    partition_type=c['partition_type'],
                    train_or_eval=c['train_or_eval'],
                    delta_learning=c.get('delta_learning', False),
                    load_delta_auxiliary_matrix=(
                        c['backbone_type'] == 'qhflow3_clean'
                        or c['nte_input_conditioning'] == 'qhflow3_exact'
                    ),
                )
            return train_loader, val_loader, required_irreps, basis_trans, orb_basis, ls_list
            
        elif c['train_or_eval'] == 'eval':
            # print("Using validation set for testing/evaluation.")
            # test_start = val_start 
            # test_end = val_end 

            # Eval mode: force batch_size to 1
            test_loader, required_irreps, basis_trans, orb_basis, ls_list = get_loader.get_loader(
                database=val_database,
                start_idx=test_start,
                end_idx=test_end,
                dataset_name=c['dataset_name'],
                rcut=c['rcut_orbitals'],
                batch_size=1, 
                dtype=c['dtype'],
                half_edges=c['reduce_edge'],
                loss_target_string=c['loss_target'],
                is_open_shell=c['open_shell'],
                scale_shift_data=scale_shift_data,
                distribute_graphs=c['distribute_graphs'],
                tiling_dims=c['tiling_dims'],
                partition_type=c['partition_type'],
                train_or_eval=c['train_or_eval'],
                delta_learning=c.get('delta_learning', False),
                load_delta_auxiliary_matrix=(
                    c['backbone_type'] == 'qhflow3_clean'
                    or c['nte_input_conditioning'] == 'qhflow3_exact'
                ),
            )
        
        # inference mode:
        else:
            print("Inference mode: using the entire dataset for evaluation.")   
            test_loader, required_irreps, basis_trans, orb_basis, ls_list = get_loader.get_loader(
                database=val_database,
                start_idx=val_start,
                end_idx=val_end,
                dataset_name=c['dataset_name'],
                rcut=c['rcut_orbitals'],
                batch_size=1, 
                dtype=c['dtype'],
                half_edges=c['reduce_edge'],
                loss_target_string=c['loss_target'],
                is_open_shell=c['open_shell'],
                scale_shift_data=scale_shift_data,
                distribute_graphs=c['distribute_graphs'],
                tiling_dims=c['tiling_dims'],
                partition_type=c['partition_type'],
                train_or_eval=c['train_or_eval'],
                delta_learning=c.get('delta_learning', False),
                load_delta_auxiliary_matrix=(
                    c['backbone_type'] == 'qhflow3_clean'
                    or c['nte_input_conditioning'] == 'qhflow3_exact'
                ),
            )
        
        return None, test_loader, required_irreps, basis_trans, orb_basis, ls_list
        
    @staticmethod
    def _collect_muon_parameters(backbone, head):
        """Route every trainable matrix through Muon, matching MuonAdamW."""
        return [
            parameter
            for module in (backbone, head)
            for parameter in module.parameters()
            if parameter.ndim >= 2 and parameter.requires_grad
        ]

    @staticmethod
    def _collect_semantic_global_muon_parameters(head):
        """Return explicitly materialized global node/edge head matrices."""
        semantic_parameters = getattr(head, "semantic_matrix_parameters", None)
        if semantic_parameters is None:
            return []
        parameters = [
            parameter
            for parameter in semantic_parameters()
            if parameter.requires_grad
        ]
        parameter_ids = [id(parameter) for parameter in parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("Semantic-global Muon parameters must be unique.")
        if any(parameter.ndim != 2 for parameter in parameters):
            raise ValueError(
                "Semantic-global Muon head parameters must be explicit matrices."
            )
        return parameters

    @staticmethod
    def _collect_semantic_gate_muon_parameters(head):
        """Return the explicitly materialized scalar/gate projection matrices."""
        gate_parameters = getattr(head, "gate_matrix_parameters", None)
        if gate_parameters is None:
            return []
        parameters = [
            parameter
            for parameter in gate_parameters()
            if parameter.requires_grad
        ]
        parameter_ids = [id(parameter) for parameter in parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("Semantic gate Muon parameters must be unique.")
        if any(parameter.ndim != 2 for parameter in parameters):
            raise ValueError(
                "Semantic gate Muon parameters must be explicit matrices."
            )
        return parameters

    @staticmethod
    def _collect_nte_output_projection_parameters(backbone):
        """Return eSEN's degree-batched 128→output projection tensors."""
        parameters = []
        for module_name in ("node_output_projection", "edge_output_projection"):
            module = getattr(backbone, module_name, None)
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is not None and weight.requires_grad:
                parameters.append(weight)
        return parameters

    def build_model(self, required_irreps, orb_basis, ls_list):
        """Initializes backbone, head, optimizer, and scheduler."""
        c = self.config
        
        # 1. Backbone
        delta_learning = c.get('delta_learning', False)
        if c['backbone_type'] == 'qhflow3_clean':
            backbone = QHFlow3MaloqBackbone(
                sh_lmax=required_irreps.lmax,
                hidden_size=c['l_embedding_dim'],
                bottle_hidden_size=c['output_l_embedding_dim'],
                num_gnn_layers=c['num_mp_layers'],
                num_ham_gnn_layers=(
                    c['num_mp_layers']
                    if c['num_edge_layers'] is None
                    else c['num_edge_layers']
                ),
                max_radius=c['qhflow3_max_radius'],
                radius_embed_dim=c['qhflow3_radius_embed_dim'],
                escn_edge_channels=c['hidden_dim'],
                escn_num_distance_basis=c['num_distance_basis'],
                esen_max_radius=c['rcut_gaussian'],
                grid_resolution=c['qhflow3_grid_resolution'],
                grid_ffn_chunk_size=c['qhflow3_grid_ffn_chunk_size'],
                basis=(
                    'def2-svp-nabla'
                    if c['dataset_name'] == 'nablaDFT'
                    else 'def2-svp'
                ),
                delta_learning=delta_learning,
                delta_target=c['loss_target'],
                default_hamiltonian_input=(
                    'init_ham' if delta_learning else 'zero'
                ),
                use_block_S=c['qhflow3_use_overlap'],
                use_block_H=delta_learning,
                muonize_output_projection=(
                    c['qhflow3_muonize_output_projection']
                ),
            ).to(self.device)
        else:
            backbone = eSEN_Backbone(
                required_irreps, sphere_channels=c['l_embedding_dim'],
                hidden_channels=c['hidden_dim'], lmax=required_irreps.lmax,
                mmax=required_irreps.lmax, cutoff=c['rcut_gaussian'],
                grid_resolution=c['esen_grid_resolution'],
                edge_channels=c['l_embedding_dim'], num_layers=c['num_mp_layers'],
                act_type='gate', mlp_type=c['mlp_type'],
                gate_act_type=c['gate_act_type'],
                num_distance_basis=c['num_distance_basis'],
                gaussian_width=c['gaussian_width'], include_edges=c['include_edges'],
                open_shell=c['open_shell'],
                wigner_backend=c.get('wigner_backend', 'torch'),
                distributed_graph_training=c['distribute_graphs'],
                message_type=c['message_type'],
                message_passing_schedule=c['message_passing_schedule'],
                initial_edge_state_mode=c['initial_edge_state_mode'],
                num_edge_layers=c['num_edge_layers'],
                output_sphere_channels=c['output_l_embedding_dim'],
                nte_output_projection_mode=c['nte_output_projection_mode'],
                use_edge_envelope=c['use_edge_envelope'],
                use_edge_scalar_modulation=c['use_edge_scalar_modulation'],
                residual_update_scale_mode=c['residual_update_scale_mode'],
                residual_update_scale_init=c['residual_update_scale_init'],
                residual_update_scale_log_range=c['residual_update_scale_log_range'],
                unscaled_node_layers=c['unscaled_node_layers'],
                repeat_system_embedding_each_node_block=(
                    c['repeat_system_embedding_each_node_block']
                ),
                node_stack_mode=c['node_stack_mode'],
                edge_stack_mode=c['edge_stack_mode'],
                qhflow3_layer_gaussian_width=c['qhflow3_layer_gaussian_width'],
                qhflow3_layer_grid_ffn_chunk_size=c['qhflow3_layer_grid_ffn_chunk_size'],
                qhflow3_exact_pair_rng_aligned=(
                    c['qhflow3_exact_pair_rng_aligned']
                ),
                edge_atom_norm_type=c['edge_atom_norm_type'],
                edge_post_residual_norm_type=(
                    c['edge_post_residual_norm_type']
                ),
                direct_edgewise_layers=c['direct_edgewise_layers'],
                edge_atomwise_output_mode=c['edge_atomwise_output_mode'],
                edge_norm1_position=c['edge_norm1_position'],
                input_conditioning=c['nte_input_conditioning'],
                conditioning_basis=(
                    'def2-svp-nabla'
                    if c['dataset_name'] == 'nablaDFT'
                    else 'def2-svp'
                ),
                conditioning_delta_learning=delta_learning,
                conditioning_delta_target=c['loss_target'],
            ).to(self.device)

        # 2. Head
        head_channels = (
            c['l_embedding_dim']
            if c['output_l_embedding_dim'] is None
            else c['output_l_embedding_dim']
        )
        irreps_in = Irreps(
            [
                (head_channels, (degree, 1))
                for degree in range(required_irreps.lmax + 1)
            ]
        )
        
        if c['loss_target'] == 'fock_matrix' or c['loss_target'] == 'density_matrix':
            if c['head_type'] == 'static_te':
                head = StaticTensorExpansionHead(
                    irreps_out=required_irreps,
                    lmax=required_irreps.lmax,
                    sphere_channels=head_channels,
                    reduce_edge=c['reduce_edge'],
                    open_shell=c['open_shell'],
                    ls_list=ls_list,
                    reduce_node=c['reduce_node'],
                    reduce_node_intra=c['reduce_node_intra'],
                    init_mode=c['static_te_init_mode'],
                    init_std=c['static_te_init_std'],
                    gate_degrees=tuple(c['static_te_gate_degrees']),
                    gate_activation=c['static_te_gate_activation'],
                    gate_init=c['static_te_gate_init'],
                )
            elif c['head_type'] in {
                'maloq_muon',
                'maloq_semantic_global_muon',
                'maloq_semantic_global_gate_muon',
            }:
                head = MuonFockIrrepsHead(
                    irreps_in=irreps_in, irreps_out=required_irreps,
                    lmax=required_irreps.lmax, sphere_channels=head_channels,
                    reduce_edge=c['reduce_edge'], open_shell=c['open_shell'],
                    ls_list=ls_list, reduce_node=c['reduce_node'],
                    reduce_node_intra=c['reduce_node_intra'], orbital_basis=orb_basis,
                    muonize_gate=(
                        c['head_type'] == 'maloq_semantic_global_gate_muon'
                    ),
                )
            else:
                head = Fock_Irreps_Head(
                    irreps_in=irreps_in, irreps_out=required_irreps,
                    lmax=required_irreps.lmax, sphere_channels=head_channels,
                    reduce_edge=c['reduce_edge'], open_shell=c['open_shell'],
                    ls_list=ls_list, reduce_node=c['reduce_node'],
                    reduce_node_intra=c['reduce_node_intra'], orbital_basis=orb_basis
                )
        elif c['loss_target'] == "forces":
            head = HELM_Force_Head(backbone)
            
        elif c['loss_target'] == "energies":
            head = HELM_Energy_Head(backbone)
        
        head = head.to(self.device)

        # Loader construction can consume RNG state differently on each rank.
        # Manual data-parallel gradient averaging therefore needs an explicit
        # rank-0 model broadcast, just as DDP performs during construction.
        if self.world_size > 1:
            with torch.no_grad():
                for module in (backbone, head):
                    tensors = list(module.parameters()) + list(module.buffers())
                    for tensor in tensors:
                        if tensor.numel() == 0:
                            continue
                        synchronized = tensor.detach().contiguous()
                        dist.broadcast(synchronized, src=0)
                        tensor.copy_(synchronized)
            dist.barrier()

        if self.rank == 0:
            is_qhflow3 = c['backbone_type'] == 'qhflow3_clean'
            semantic_global_params = (
                self._collect_semantic_global_muon_parameters(head)
                if c['head_type'] in {
                    'maloq_semantic_global_muon',
                    'maloq_semantic_global_gate_muon',
                }
                else []
            )
            semantic_gate_params = (
                self._collect_semantic_gate_muon_parameters(head)
                if c['head_type'] == 'maloq_semantic_global_gate_muon'
                else []
            )
            model_summary = {
                'model_variant': c.get('model_variant', 'maloq-baseline'),
                'backbone_type': c['backbone_type'],
                'head_type': c['head_type'],
                'muon_routing': (
                    (
                        'shape_matrix_muon_plus_semantic_global_head_muon'
                        '_plus_semantic_gate_muon'
                    )
                    if semantic_gate_params
                    else (
                        'shape_matrix_muon_plus_semantic_global_head_muon'
                        if semantic_global_params
                        else (
                            'ndim_ge_2_muon_except_output_projection_adamw'
                            if c['muon_output_projection_policy'] == 'adamw'
                            else 'all_trainable_ndim_ge_2'
                        )
                    )
                ),
                'semantic_global_head_parameters': sum(
                    parameter.numel()
                    for parameter in semantic_global_params
                ),
                'semantic_gate_parameters': sum(
                    parameter.numel()
                    for parameter in semantic_gate_params
                ),
                'seed': int(c['seed']),
                'backbone_parameters': sum(p.numel() for p in backbone.parameters()),
                'head_parameters': sum(p.numel() for p in head.parameters()),
                'total_parameters': sum(p.numel() for p in backbone.parameters())
                + sum(p.numel() for p in head.parameters()),
                'trainable_parameters': sum(
                    p.numel()
                    for p in list(backbone.parameters()) + list(head.parameters())
                    if p.requires_grad
                ),
                'trunk_channels': int(c['l_embedding_dim']),
                'output_channels': int(head_channels),
                'node_layers': int(c['num_mp_layers']),
                'edge_layers': int(
                    c['num_mp_layers']
                    if c['num_edge_layers'] is None
                    else c['num_edge_layers']
                ),
                'message_passing_schedule': (
                    'qhflow3_node_then_pair'
                    if is_qhflow3
                    else c['message_passing_schedule']
                ),
                'initial_edge_state_mode': (
                    None if is_qhflow3 else c['initial_edge_state_mode']
                ),
                'mlp_type': 'grid' if is_qhflow3 else c['mlp_type'],
                'gate_act_type': 'sigmoid' if is_qhflow3 else c['gate_act_type'],
                'residual_update_scale_mode': (
                    'none' if is_qhflow3 else c['residual_update_scale_mode']
                ),
                'unscaled_node_layers': (
                    None if is_qhflow3 else list(c['unscaled_node_layers'])
                ),
                'repeat_system_embedding_each_node_block': (
                    None
                    if is_qhflow3
                    else bool(c['repeat_system_embedding_each_node_block'])
                ),
                'node_stack_mode': (
                    'qhflow3_exact'
                    if is_qhflow3
                    else c['node_stack_mode']
                ),
                'edge_stack_mode': (
                    'qhflow3_parallel_sum'
                    if is_qhflow3
                    else c['edge_stack_mode']
                ),
                'qhflow3_layer_gaussian_width': (
                    2.0 if is_qhflow3 else c['qhflow3_layer_gaussian_width']
                ),
                'qhflow3_layer_grid_ffn_chunk_size': (
                    c['qhflow3_grid_ffn_chunk_size']
                    if is_qhflow3
                    else c['qhflow3_layer_grid_ffn_chunk_size']
                ),
                'edge_atom_norm_type': (
                    None if is_qhflow3 else c['edge_atom_norm_type']
                ),
                'edge_post_residual_norm_type': (
                    None
                    if is_qhflow3
                    else c['edge_post_residual_norm_type']
                ),
                'direct_edgewise_layers': (
                    None if is_qhflow3 else list(c['direct_edgewise_layers'])
                ),
                'edge_atomwise_output_mode': (
                    None if is_qhflow3 else c['edge_atomwise_output_mode']
                ),
                'edge_norm1_position': (
                    None if is_qhflow3 else c['edge_norm1_position']
                ),
                'nte_output_projection_mode': (
                    None if is_qhflow3 else c['nte_output_projection_mode']
                ),
                'nte_output_projection_rng_contract': (
                    None
                    if is_qhflow3
                    else (
                        'legacy_so3_linear_aligned'
                        if c['nte_output_projection_mode']
                        == 'qhflow3_irrep_linear'
                        else 'native_so3_linear'
                    )
                ),
                'muon_output_projection_policy': (
                    c['muon_output_projection_policy']
                    if c['optimizer_type'] == 'muon'
                    else None
                ),
                'esen_grid_resolution': (
                    None if is_qhflow3 else c['esen_grid_resolution']
                ),
                'nte_input_conditioning': (
                    None if is_qhflow3 else c['nte_input_conditioning']
                ),
                'delta_learning': bool(c.get('delta_learning', False)),
                'prediction_contract': (
                    (
                        'final_density=initial_density+predicted_delta'
                        if c['loss_target'] == 'density_matrix'
                        else 'final_hamiltonian=initial_hamiltonian+predicted_delta'
                    )
                    if c.get('delta_learning', False)
                    else 'absolute_target'
                ),
            }
            if is_qhflow3:
                model_summary.update(
                    qhflow3_primary_matrix_input=(
                        (
                            'initial_density_matrix'
                            if c['loss_target'] == 'density_matrix'
                            else 'initial_hamiltonian'
                        )
                        if c.get('delta_learning', False) else 'zero'
                    ),
                    qhflow3_auxiliary_matrix_input=(
                        (
                            'initial_hamiltonian'
                            if c['loss_target'] == 'density_matrix'
                            else 'initial_density_matrix'
                        )
                        if c.get('delta_learning', False) else None
                    ),
                    qhflow3_basis=backbone.basis,
                    qhflow3_overlap_input=(
                        'native_loader_atom_diagonal_blocks'
                        if c['qhflow3_use_overlap']
                        else 'disabled'
                    ),
                    qhflow3_grid_resolution=(
                        None
                        if c['qhflow3_grid_resolution'] is None
                        else int(c['qhflow3_grid_resolution'])
                    ),
                    qhflow3_grid_ffn_chunk_size=c['qhflow3_grid_ffn_chunk_size'],
                    qhflow3_output_projection_optimizer=(
                        'muon' if c['qhflow3_muonize_output_projection'] else 'adamw'
                    ),
                )
            summary_path = Path(c['output_folder']) / 'model_summary.json'
            summary_path.write_text(json.dumps(model_summary, indent=2) + '\n')
            print(
                f"Model parameters: {model_summary['total_parameters']:,} "
                f"({model_summary['model_variant']})",
                flush=True,
            )

        # 3. Optimizer
        backbone_params = []
        head_params = []
        if c['train_backbone']:
            backbone_params = list(backbone.parameters())
        else: 
            for p in backbone.parameters(): p.requires_grad = False
            
        if c['train_head']:
            head_params = list(head.parameters())
        else:
            for p in head.parameters(): p.requires_grad = False

        params = backbone_params + head_params
        optimizer_type = c['optimizer_type']
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params, lr=c['lr_init'])
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=c['lr_init'],
                weight_decay=c.get('weight_decay', 0.0),
            )
        elif optimizer_type == 'soap':
            soap_lr = c['lr_init'] if c.get('soap_lr') is None else c['soap_lr']
            optimizer = optimizers.SOAP(
                params,
                lr=soap_lr,
                betas=tuple(c['soap_betas']),
                shampoo_beta=c['soap_shampoo_beta'],
                eps=c['soap_eps'],
                weight_decay=c.get('weight_decay', 0.0),
                precondition_frequency=c['soap_precondition_frequency'],
                max_precond_dim=c['soap_max_precondition_dim'],
                precondition_1d=c['soap_precondition_1d'],
                normalize_grads=c['soap_normalize_grads'],
            )
        elif optimizer_type == 'muon':
            # Ordinary matrices follow ml-dft's shape rule. The explicit
            # semantic policy separates the node/edge global contraction
            # matrices into named Muon groups without changing their update.
            muon_params = self._collect_muon_parameters(backbone, head)
            output_projection_adamw_params = (
                self._collect_nte_output_projection_parameters(backbone)
                if (
                    c['backbone_type'] == 'esen'
                    and c['muon_output_projection_policy'] == 'adamw'
                )
                else []
            )
            output_projection_adamw_ids = {
                id(parameter) for parameter in output_projection_adamw_params
            }
            muon_params = [
                parameter
                for parameter in muon_params
                if id(parameter) not in output_projection_adamw_ids
            ]
            semantic_global_params = (
                self._collect_semantic_global_muon_parameters(head)
                if c['head_type'] in {
                    'maloq_semantic_global_muon',
                    'maloq_semantic_global_gate_muon',
                }
                else []
            )
            semantic_gate_params = (
                self._collect_semantic_gate_muon_parameters(head)
                if c['head_type'] == 'maloq_semantic_global_gate_muon'
                else []
            )
            semantic_global_ids = {
                id(parameter) for parameter in semantic_global_params
            }
            semantic_gate_ids = {
                id(parameter) for parameter in semantic_gate_params
            }
            shape_matrix_params = [
                parameter
                for parameter in muon_params
                if id(parameter) not in semantic_global_ids
                and id(parameter) not in semantic_gate_ids
            ]
            muon_param_ids = {
                id(parameter)
                for parameter in (
                    *shape_matrix_params,
                    *semantic_global_params,
                    *semantic_gate_params,
                )
            }
            auxiliary_params = [p for p in params if id(p) not in muon_param_ids]
            if not muon_params:
                raise ValueError(
                    "Muon requires at least one trainable backbone matrix parameter."
                )

            parameter_groups = []
            if shape_matrix_params:
                parameter_groups.append({
                    'params': shape_matrix_params,
                    'use_muon': True,
                    'lr': c['muon_lr'],
                    'name': 'shape_matrix_muon',
                })
            if semantic_global_params:
                parameter_groups.append({
                    'params': semantic_global_params,
                    'use_muon': True,
                    'lr': c['muon_lr'],
                    'name': 'semantic_global_head_muon',
                })
            if semantic_gate_params:
                parameter_groups.append({
                    'params': semantic_gate_params,
                    'use_muon': True,
                    'lr': c['muon_lr'],
                    'name': 'semantic_gate_muon',
                })
            if auxiliary_params:
                auxiliary_lr = (
                    c['lr_init']
                    if c.get('muon_adamw_lr') is None
                    else c['muon_adamw_lr']
                )
                parameter_groups.append(
                    {
                        'params': auxiliary_params,
                        'use_muon': False,
                        'lr': auxiliary_lr,
                        'betas': tuple(c['muon_adamw_betas']),
                        'eps': c['muon_adamw_eps'],
                        'name': 'auxiliary_adamw',
                    }
                )
            optimizer = optimizers.Muon(
                parameter_groups,
                lr=c['muon_lr'],
                momentum=c['muon_momentum'],
                nesterov=c['muon_nesterov'],
                ns_steps=c['muon_ns_steps'],
                weight_decay=c.get('weight_decay', 0.0),
                betas=tuple(c['muon_adamw_betas']),
                eps=c['muon_adamw_eps'],
            )
            if self.rank == 0:
                print(
                    "Muon optimizer: "
                    f"{sum(p.numel() for p in shape_matrix_params):,} "
                    "shape-routed matrix parameters, "
                    f"{sum(p.numel() for p in semantic_global_params):,} "
                    "semantic-global head parameters, "
                    f"{sum(p.numel() for p in semantic_gate_params):,} "
                    "semantic gate parameters, "
                    f"{sum(p.numel() for p in output_projection_adamw_params):,} "
                    "output-projection AdamW parameters, "
                    f"{sum(p.numel() for p in auxiliary_params):,} AdamW parameters."
                )

        # 4. Restarts
        self._load_checkpoint(backbone, c['backbone_checkpoint'], "backbone")
        self._load_checkpoint(head, c['head_checkpoint'], "head", optimizer if c['restart_optimizer'] else None)

        return backbone, head, optimizer
    
    def _get_scheduler(self, optimizer, train_loader):
        """Initializes scheduler based on training loader length."""
        c = self.config
        optimizer_steps_per_epoch = math.ceil(
            len(train_loader) / c['gradient_accumulation_steps']
        )
        scheduler_steps_per_epoch = (
            1 if c.get('step_every_epoch', False) else optimizer_steps_per_epoch
        )
        if c['scheduler_type'] == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=c['patience'], threshold=c['threshold']
            )
        elif c['scheduler_type'] == 'cosine':
            t_max = c['num_epochs'] * scheduler_steps_per_epoch
            if self.rank == 0: 
                print(f"T_max for scheduler: {t_max}")
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=c['eta_min'])
        elif c['scheduler_type'] == 'warmup_polynomial':
            max_steps = max(1, c['num_epochs'] * scheduler_steps_per_epoch)
            warmup_steps = int(c['warmup_steps'])
            power = float(c['scheduler_power'])
            min_lr_ratio = float(c['min_lr_ratio'])
            if warmup_steps < 0:
                raise ValueError("warmup_steps cannot be negative.")
            if power <= 0.0:
                raise ValueError("scheduler_power must be positive.")
            if not 0.0 <= min_lr_ratio <= 1.0:
                raise ValueError("min_lr_ratio must be between 0 and 1.")

            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return max(min_lr_ratio, float(step + 1) / warmup_steps)
                decay_steps = max(1, max_steps - warmup_steps)
                progress = min(
                    1.0,
                    max(0.0, float(step - warmup_steps) / decay_steps),
                )
                return min_lr_ratio + (1.0 - min_lr_ratio) * (
                    (1.0 - progress) ** power
                )

            if self.rank == 0:
                print(
                    "Warmup-polynomial scheduler: "
                    f"warmup_steps={warmup_steps}, max_steps={max_steps}, "
                    f"power={power}, min_lr_ratio={min_lr_ratio}",
                    flush=True,
                )
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
        else:
            raise ValueError(f"Unknown scheduler: {c['scheduler_type']}")

    def _load_checkpoint(self, model, filename, name, optimizer=None):
        path = os.path.join(self.config['output_folder'], filename)
        if (name == "backbone" and self.config['restart_backbone']) or (name == "head" and self.config['restart_head']):
            if os.path.exists(path):
                if self.rank == 0: print(f"Restarting {name} from {path}")
                ckpt = torch.load(path, map_location=self.device)
                sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
                model.load_state_dict(sd)
                if optimizer and 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def close(self):
        """Best-effort distributed cleanup for explicit shutdown paths."""
        self.finish_tracking()
        utils_compute.cleanup_process_group(sync_barrier=True)

    def run(self):
        try:
            # HELM's data and training pipeline currently supports these datasets:
            if self.config['dataset_name'] == 'QM7':
                target_property = (
                    'density_matrix'
                    if self.config['loss_target'] == 'density_matrix'
                    else 'hamiltonian'
                )
                load_properties = [
                    'energy',
                    'forces',
                    target_property,
                    'overlap',
                ]
                if self.config.get('delta_learning', False):
                    initial_target_property = (
                        'initial_density_matrix'
                        if target_property == 'density_matrix'
                        else 'initial_hamiltonian'
                    )
                    load_properties.append(initial_target_property)
                    if self.config['backbone_type'] == 'qhflow3_clean':
                        auxiliary_property = (
                            'initial_hamiltonian'
                            if target_property == 'density_matrix'
                            else 'initial_density_matrix'
                        )
                        load_properties.append(auxiliary_property)
                # ASEAtomsData's property setter reads DB metadata through the
                # open connection, so apply the selection after construction.
                database = ASEAtomsData(self.config['dbpath'])
                database.load_properties = load_properties
                if self.config['shuffle']:
                    print("Shuffling QM7 dataset for training...")
                    indices = list(range(len(database)))
                    random.shuffle(indices)
                    database = [database[i] for i in indices]

            elif self.config['dataset_name'] == 'nablaDFT':
                database = HamiltonianDatabase(self.config['dbpath'])
            elif self.config['dataset_name'] == 'omol':
                database = None
            elif self.config['dataset_name'] == 'cp2k_material':
                database = None
            else:
                raise ValueError(f"Unknown dataset name: {self.config['dataset_name']}")

            """Main execution loop."""
            loader, val_loader, irreps, basis_trans, orb_basis, ls_list = self.prepare_loaders(database)
            backbone, head, optimizer = self.build_model(irreps, orb_basis, ls_list)
            scheduler = self._get_scheduler(optimizer, loader) if loader else None

            trainer = splittrainer.SplitTrainer(
                backbone=backbone, head=head, head_irreps=irreps, # Note: update if forces
                run_name=self.config.get('run_name', 'run'),
                save_frequency=self.config.get('save_frequency', 10),
                wandb_run=self.wandb_run,
            )

            target_map = {
                'fock_matrix': ('node_y', 'y'),
                'density_matrix': ('node_y', 'y'),
                'forces': ('forces', None),
                'energies': ('energies', None)
            }
            node_target, edge_target = target_map[self.config['loss_target']]

            if self.config['train_or_eval'] == "train":
                trainer.train(
                    self.config['num_epochs'], self.config['train_loss_fxn'],
                    optimizer, scheduler, self.device, train_loader=loader,
                    val_loader=val_loader, loss_target_string=self.config['loss_target'],
                    node_target_name=node_target, edge_target_name=edge_target,
                    output_folder=self.config['output_folder'],
                    train_backbone=self.config['train_backbone'],
                    train_head=self.config['train_head'],
                    basis_transform=basis_trans,
                    compute_uncoupled_loss=self.config.get('compute_uncoupled_loss', False),
                    step_every_epoch=self.config.get('step_every_epoch', True),
                    element_references=self.config.get('element_references', None),
                    validation_matrix_metrics=self.config.get('validation_matrix_metrics', False),
                    validation_matrix_metrics_frequency=self.config.get('validation_matrix_metrics_frequency', 1),
                    gradient_clip_val=self.config.get('gradient_clip_val'),
                    gradient_accumulation_steps=self.config.get(
                        'gradient_accumulation_steps', 1
                    ),
                    wandb_enabled=self.config.get('use_wandb', False),
                    wandb_log_every_n_steps=self.config.get(
                        'wandb_log_every_n_steps', 10
                    ),
                )
            elif self.config['train_or_eval'] == "eval":
                trainer.evaluate(
                    self.config['test_loss_fxn'], self.device, val_loader,
                    loss_target_string=self.config['loss_target'],
                    node_target_name=node_target, edge_target_name=edge_target, compute_total_energy=self.config['compute_total_energy'],
                    basis_transform=basis_trans, output_folder=self.config['output_folder'],
                    dataset_name=self.config['dataset_name'], orbital_basis=orb_basis,
                    element_references=self.config.get('element_references', None),
                    distributed_graphs=self.config['distribute_graphs']
                )
            else:
                trainer.infer(
                    self.config['test_loss_fxn'], self.device, val_loader,
                    loss_target_string=self.config['loss_target'],
                    compute_total_energy=self.config['compute_total_energy'],
                    basis_transform=basis_trans, output_folder=self.config['output_folder'],
                    dataset_name=self.config['dataset_name'], orbital_basis=orb_basis,
                    element_references=self.config.get('element_references', None),
                    distributed_graphs=self.config['distribute_graphs']
                )
        finally:
            self.close()
