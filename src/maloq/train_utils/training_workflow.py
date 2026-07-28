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

from . import optimizers, utils_compute, splittrainer
from ..dataset_utils import get_loader, get_scale_shift
from ..dataset_utils.ASEDataset import distribute_data, ASEDataset, ASEAtomsData
from ..dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
from ..dataset_utils.omol_csh_58k_dataset_utils import (
    OMolCSH58kDatabase,
    load_key_manifest,
)
from ..helm.esen_osh import (
    eSEN_Backbone,
    Fock_Irreps_Head,
    HELM_Force_Head,
    HELM_Energy_Head,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


class TrainingWorkflow:
    """Canonical workflow for the original MALOQ eSEN/Fock-head model."""

    SUPPORTED_BACKBONE_TYPES = frozenset({"esen"})
    SUPPORTED_HEAD_TYPES = frozenset({"maloq"})
    DEFAULTS = {
        "run_name": "run",
        "output_folder": "outputs/run",
        "seed": 42,
        "backbone_type": "esen",
        "head_type": "maloq",
        "atom_scalar_embedding_mode": "element_charge_spin",
        "open_shell": False,
        "dataset_format": "auto",
        "omol_csh_metadata_policy": "preserve",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "wigner_backend": "torch",
        "mlp_type": "spectral",
        "esen_grid_resolution": None,
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
        "gradient_clip_val": None,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 1000,
        "scheduler_power": 1.0,
        "min_lr_ratio": 0.0,
        "compute_total_energy": False,
        "compute_eigenvalues": True,
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
        "compute_uncoupled_loss": False,
    }

    def __init__(self, config):
        self.config = self.DEFAULTS | config
        self.config["output_folder"] = self.resolve_output_folder(
            self.config["output_folder"], self.config["run_name"]
        )
        self.wandb_run = None
        self.setup_environment()

        # check_input_config will raise errors if there are incompatible settings
        self.check_input_config()
        self.wandb_run = self.setup_tracking()

    @staticmethod
    def resolve_output_folder(output_folder, run_name):
        """Resolve every model-run output below the project ``outputs`` tree."""
        output_path = Path(os.path.expandvars(os.path.expanduser(str(output_folder))))

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
        if self.rank != 0 or not self.config.get("use_wandb", False):
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
                if hasattr(value, "__name__")
                else str(value)
            )
            for key, value in self.config.items()
        }
        run = wandb.init(
            project=self.config["wandb_project"],
            entity=self.config.get("wandb_entity"),
            name=self.config.get("wandb_run_name") or self.config["run_name"],
            group=self.config.get("wandb_group"),
            job_type=self.config.get("wandb_job_type"),
            tags=list(self.config.get("wandb_tags") or ()),
            dir=self.config["output_folder"],
            config=wandb_config,
            mode=self.config.get("wandb_mode", "online"),
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
        seed = int(self.config["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.set_default_dtype(self.config["dtype"])

        # torchrun, Open MPI, and SLURM distributed setup.
        self.rank, self.world_size, self.local_rank = (
            utils_compute.distributed_context()
        )

        compute_start = time.perf_counter()
        self.device = utils_compute.setup_env(
            self.rank,
            self.world_size,
            backend=self.config["dist_backend"],
            local_rank=self.local_rank,
        )
        compute_end = time.perf_counter()

        if self.rank == 0:
            print(
                f"Time to setup distributed environment: {compute_end - compute_start:.4f}s"
            )
            if not os.path.exists(self.config["output_folder"]):
                os.makedirs(self.config["output_folder"])
        dist.barrier()

        if self.config.get("distribute_graphs", False):
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            mpi_world_size = MPI.COMM_WORLD.Get_size()
            if (mpi_rank, mpi_world_size) != (self.rank, self.world_size):
                raise RuntimeError(
                    "Distributed-graph training requires matching MPI and "
                    "torch.distributed ranks. Launch it with mpirun rather "
                    "than torchrun. "
                    f"MPI={mpi_rank}/{mpi_world_size}, "
                    f"torch={self.rank}/{self.world_size}."
                )

    def _validate_backbone_feature_config(self):
        """Reject feature-owned configuration in the canonical workflow."""
        selected_feature = self.config.get("experimental_feature")
        if selected_feature is not None:
            raise ValueError(
                "Canonical MALOQ cannot run experimental feature "
                f"{selected_feature!r}; use the explicit experimental "
                "workflow."
            )

    def _supports_atom_scalar_embedding(self):
        """Whether the selected backbone consumes eSEN atom scalars."""
        return True

    def _uses_matrix_input_conditioning(self):
        """Whether the backbone consumes an auxiliary matrix input."""
        return False

    def check_input_config(self):
        """Validates the configuration for incompatible settings, and writes config to output folder."""

        optimizer_type = self.config.get("optimizer_type", "adam").lower()
        valid_optimizers = {"adam", "adamw", "soap", "muon"}
        if optimizer_type not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer '{optimizer_type}'. Choose one of "
                f"{sorted(valid_optimizers)}."
            )
        self.config["optimizer_type"] = optimizer_type

        if self.config["dataset_format"] not in {"auto", "ase", "omol_csh_h5"}:
            raise ValueError("dataset_format must be 'auto', 'ase', or 'omol_csh_h5'.")
        if self.config["omol_csh_metadata_policy"] not in {
            "preserve",
            "paper_contract",
        }:
            raise ValueError(
                "omol_csh_metadata_policy must be 'preserve' or 'paper_contract'."
            )
        if self.config["atom_scalar_embedding_mode"] not in {
            "element_charge_spin",
            "element_only",
        }:
            raise ValueError(
                "atom_scalar_embedding_mode must be 'element_charge_spin' "
                "or 'element_only'."
            )
        if (
            self.config["atom_scalar_embedding_mode"] != "element_charge_spin"
            and not self._supports_atom_scalar_embedding()
        ):
            raise ValueError(
                "atom_scalar_embedding_mode is available only for the eSEN backbone."
            )
        if self.config["dataset_format"] == "omol_csh_h5":
            if self.config["dataset_name"] != "omol":
                raise ValueError(
                    "dataset_format='omol_csh_h5' requires dataset_name='omol'."
                )
            if self.config["open_shell"]:
                raise ValueError("The published OMol_CSH H5 target is closed shell.")
            if self.config["loss_target"] != "fock_matrix":
                raise ValueError(
                    "The published OMol_CSH H5 files contain only Fock targets."
                )
            if self.config["distribute_graphs"]:
                raise ValueError(
                    "OMol_CSH H5 streaming supports data-parallel training, "
                    "not distributed-graph training."
                )
            if self.config["backbone_type"] != "esen":
                raise ValueError(
                    "OMol_CSH H5 currently supports the original eSEN/HELM path only."
                )
            if self._uses_matrix_input_conditioning():
                raise ValueError(
                    "OMol_CSH H5 has no overlap or initial-matrix conditioning."
                )
            if self.config["compute_eigenvalues"]:
                raise ValueError(
                    "OMol_CSH H5 has no overlap matrix; generalized "
                    "eigenvalue metrics must be disabled."
                )
            if self.config["compute_total_energy"]:
                raise ValueError(
                    "OMol_CSH H5 has no density/overlap data for total energy."
                )
            if self.config["atom_scalar_embedding_mode"] != "element_only":
                raise ValueError(
                    "Public OMol_CSH H5 metadata is audit-only and includes "
                    "spin values outside the charge/spin embedding range. "
                    "Use atom_scalar_embedding_mode='element_only'."
                )
            for split_name in ("num_train", "num_val", "num_test"):
                split_size = int(self.config[split_name])
                if split_size and (
                    split_size < self.world_size or split_size % self.world_size != 0
                ):
                    raise ValueError(
                        f"OMol_CSH {split_name}={split_size} must be zero or "
                        f"a positive multiple of world_size={self.world_size}; "
                        "the streaming data-parallel loader requires every "
                        "rank to receive the same non-empty split."
                    )
            if (
                self.config["train_or_eval"] == "train"
                and int(self.config["num_train"]) == 0
            ):
                raise ValueError("OMol_CSH training requires num_train > 0.")
            if (
                self.config["train_or_eval"] == "eval"
                and int(self.config["num_test"]) == 0
            ):
                raise ValueError("OMol_CSH evaluation requires num_test > 0.")

        if self.config["mlp_type"] not in {"spectral", "grid"}:
            raise ValueError("mlp_type must be 'spectral' or 'grid'.")
        self._validate_backbone_feature_config()
        if self.config["backbone_type"] not in self.SUPPORTED_BACKBONE_TYPES:
            supported = ", ".join(sorted(self.SUPPORTED_BACKBONE_TYPES))
            raise ValueError(f"backbone_type must be one of: {supported}.")
        if self.config["head_type"] not in self.SUPPORTED_HEAD_TYPES:
            supported = ", ".join(sorted(self.SUPPORTED_HEAD_TYPES))
            raise ValueError(f"head_type must be one of: {supported}.")
        if (
            self.config["esen_grid_resolution"] is not None
            and int(self.config["esen_grid_resolution"]) <= 0
        ):
            raise ValueError("esen_grid_resolution must be positive or None.")
        gradient_clip_val = self.config["gradient_clip_val"]
        if gradient_clip_val is not None and float(gradient_clip_val) <= 0.0:
            raise ValueError("gradient_clip_val must be positive when specified.")
        gradient_accumulation_steps = int(self.config["gradient_accumulation_steps"])
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        self.config["gradient_accumulation_steps"] = gradient_accumulation_steps
        if self.config.get("delta_learning", False):
            if self.config["loss_target"] not in {"fock_matrix", "density_matrix"}:
                raise ValueError(
                    "delta_learning requires a Hamiltonian or density matrix target."
                )
            if self.config["dataset_name"] != "QM7":
                raise ValueError(
                    "delta_learning requires the QM7-style ASE data loader."
                )
            if self.config["open_shell"]:
                raise ValueError("delta_learning currently supports closed shell only.")
            if self.config["distribute_graphs"]:
                raise ValueError(
                    "delta_learning is not implemented for distributed graphs."
                )

        if "matrix" in self.config["loss_target"]:
            self.config["include_edges"] = True
            print(
                "Initializing model with edge embeddings, since loss target involves a matrix."
            )
        else:
            print(
                "Initializing model without edge embeddings, since loss target does not involve a matrix."
            )
            self.config["include_edges"] = False

        # wigner_backend exists and is equal to triton
        if self.config.get("wigner_backend", "torch") == "triton":
            if self.device.type != "cuda":
                raise ValueError("Triton Wigner backend requires a CUDA-capable GPU.")
            if self.config["dtype"] == torch.float64:
                raise ValueError(
                    "Triton Wigner backend does not support float64 dtype."
                )

        # Write config settings to the output file if not eval:
        if self.rank == 0 and self.config["train_or_eval"] == "train":
            config_path = os.path.join(
                self.config["output_folder"], f"config_{self.config['run_name']}.json"
            )
            serializable_config = {
                k: (v.__name__ if hasattr(v, "__name__") else str(v))
                for k, v in self.config.items()
            }
            with open(config_path, "w") as f:
                json.dump(serializable_config, f, indent=4)

            print(f"Config dumped to {config_path}")

        # if using restart, check that the checkpoint files exist and are not corrupted
        if self.config["restart_backbone"]:
            backbone_path = os.path.join(
                self.config["output_folder"], self.config["backbone_checkpoint"]
            )
            if not os.path.exists(backbone_path):
                raise FileNotFoundError(
                    f"Backbone checkpoint not found at {backbone_path}"
                )
            try:
                torch.load(backbone_path, map_location=self.device)
            except Exception as e:
                raise ValueError(
                    f"Error loading backbone checkpoint from {backbone_path}: {e}"
                )

        if self.config["restart_head"]:
            head_path = os.path.join(
                self.config["output_folder"], self.config["head_checkpoint"]
            )
            if not os.path.exists(head_path):
                raise FileNotFoundError(f"Head checkpoint not found at {head_path}")
            try:
                torch.load(head_path, map_location=self.device)
            except Exception as e:
                raise ValueError(f"Error loading head checkpoint from {head_path}: {e}")

        if "shuffle" not in self.config:
            self.config["shuffle"] = False

        # if partition_type is not specified, set it 'linear-edgewise' if distribute_graphs is True, else None:
        if self.config["distribute_graphs"] and self.config["partition_type"] is None:
            self.config["partition_type"] = "linear-edgewise"
            print(
                "No partition type specified for distributed graph training; defaulting to 'linear-edgewise'."
            )

        # if both reduce_edge and distribute graphs are true, print that there is a known bug!:
        if self.config["reduce_edge"] and self.config["distribute_graphs"]:
            raise ValueError(
                "reduce_edge and distribute_graphs cannot both be True, as communication has not been implemented yet in the output head."
            )

        # distribute_graphs cannot be used with non-matrix valued learning targets:
        if (
            self.config["distribute_graphs"]
            and "matrix" not in self.config["loss_target"]
        ):
            raise ValueError(
                "Distributed graph training is currently only implemented for matrix-valued learning targets (e.g. fock_matrix)."
            )

        if self.config["validation_matrix_metrics"]:
            if self.config["loss_target"] not in ["fock_matrix", "density_matrix"]:
                raise ValueError(
                    "Validation matrix metrics require a Fock or density matrix loss target."
                )
            if self.config["distribute_graphs"]:
                raise ValueError(
                    "Validation matrix metrics are not yet supported with distributed graphs."
                )
        if self.config["validation_matrix_metrics_frequency"] < 1:
            raise ValueError("validation_matrix_metrics_frequency must be at least 1.")
        if self.config["wandb_log_every_n_steps"] < 1:
            raise ValueError("wandb_log_every_n_steps must be at least 1.")
        if not all(
            isinstance(tag, str) and tag.strip()
            for tag in self.config.get("wandb_tags", ())
        ):
            raise ValueError("wandb_tags must contain only non-empty strings.")

        # if hidden_dim is not provided, set it to l_embedding_dim:
        if self.config.get("hidden_dim") is None:
            self.config["hidden_dim"] = self.config["l_embedding_dim"]
            print(
                f"hidden_dim not specified; defaulting to l_embedding_dim={self.config['l_embedding_dim']}"
            )

        # if c['message_type'] is not provided, set it to 'source-target':
        if "message_type" not in self.config:
            self.config["message_type"] = "source-target"
            print("message_type not specified; defaulting to 'source-target'")

    def _handle_scale_shift(self, database=None):
        """Manages the computation or loading of scale/shift factors."""
        if not self.config.get("scale_and_shift"):
            return None

        dataset_name = self.config["dataset_name"]
        scale_shift_mode = self.config.get(
            "scale_shift_mode",
            "standardize",
        )
        if scale_shift_mode not in {"standardize", "shift_only"}:
            raise ValueError("scale_shift_mode must be 'standardize' or 'shift_only'.")

        if self.config["loss_target"] in ["fock_matrix", "density_matrix"]:
            configured_path = self.config.get("scale_shift_path")
            if configured_path:
                target_path = Path(configured_path).expanduser()
                if not target_path.is_absolute():
                    target_path = PROJECT_ROOT / target_path
                target_path = target_path.resolve()
                if not target_path.is_file():
                    raise FileNotFoundError(
                        f"Configured scale-shift artifact does not exist: {target_path}"
                    )
                print(
                    f"[Loading configured scale/shift factors from {target_path}]",
                    flush=True,
                )
                data = torch.load(
                    target_path,
                    map_location="cpu",
                    weights_only=False,
                )
                provenance = data.get("provenance", {})
                expected = {
                    "dataset_name": dataset_name,
                    "loss_target": self.config["loss_target"],
                    "rcut_orbitals": self.config["rcut_orbitals"],
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
                    raise ValueError("Scale-shift provenance mismatch: " + details)
                return {
                    "element_scalar_means": data["element_scalar_means"],
                    "element_scalar_stds": data["element_scalar_stds"],
                    "scalar_irrep_indices": data["scalar_irrep_indices"],
                    "normalization_mode": scale_shift_mode,
                }

            if self.config["open_shell"]:
                filename = f"element_scale_shifts_osh_{dataset_name}.pt"
            else:
                filename = f"element_scale_shifts_{dataset_name}.pt"

            current_file_path = os.path.dirname(os.path.abspath(__file__))
            target_path = os.path.join(current_file_path, "../fock_utils/", filename)
            file_exists = os.path.exists(target_path)

            if file_exists:
                print(
                    f"[Loading precomputed scale/shift factors for {dataset_name} from {target_path}]"
                )
                data = torch.load(target_path)

            # Recompute scale/shift factors for this dataset
            else:
                if self.config["dataset_format"] == "omol_csh_h5":
                    raise FileNotFoundError(
                        "OMol_CSH H5 training requires a precomputed "
                        "scale-shift artifact. Automatic H5 recomputation is "
                        "disabled because the legacy OMol statistics path is "
                        "not streaming- or DDP-safe."
                    )
                print(f"[Computing scale/shift factors for {dataset_name}]")
                if database is None:
                    print(
                        "Error: Database object is required to compute scale/shift factors but is None."
                    )
                    exit()
                data = get_scale_shift.get_scale_shift(
                    database,
                    dataset_name,
                    self.config["rcut_orbitals"],
                    dtype=self.config["dtype"],
                    reduce_edge=self.config["reduce_edge"],
                    filename=filename,
                )
                print(
                    "Finished computing scale/shift factors! Will use these to scale the node data."
                )

            if self.config["open_shell"]:
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

        elif self.config["loss_target"] in ["energies"]:
            filename = "stats_nablaDFT/lin_ref_coeffs_nablaDFT.npz"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            energy_ref_file = os.path.join(current_dir, "../dataset_utils/", filename)

            if os.path.exists(energy_ref_file):
                print(f"Loading energy reference coefficients from {energy_ref_file}")
                lin_ref_data = np.load(energy_ref_file)
                element_references_np = lin_ref_data[
                    "coeff"
                ]  # Shape: (max_atomic_number,)

                # Convert to torch tensor and move to device
                element_references = torch.tensor(
                    element_references_np,
                    dtype=self.config["dtype"],
                    device=self.device,
                )
                print(
                    f"Loaded energy references for {len(element_references)} elements"
                )

                # Print non-zero references for verification
                nonzero_mask = torch.abs(element_references) > 1e-10
                nonzero_elements = torch.where(nonzero_mask)[0]
                print("Non-zero energy references:")
                for z in nonzero_elements:
                    print(
                        f"  Element Z={z.item()}: {element_references[z].item():.6f} Hartree"
                    )

                self.config["element_references"] = element_references

            else:
                raise FileNotFoundError(
                    f"Energy reference file {energy_ref_file} not found!"
                )

        else:
            raise ValueError(
                f"Unknown loss target for scale/shift handling: {self.config['loss_target']}"
            )

    def prepare_loaders(self, database_input=None):
        """
        Calculates splits and returns the appropriate DataLoaders.
        database_input can be a single file path, a folder path, or a pre-loaded ASEDataset.
        """
        c = self.config

        db_source = database_input if database_input is not None else c["dbpath"]

        if isinstance(db_source, str):
            db_sources = [db_source]
        elif isinstance(db_source, list):
            db_sources = db_source
        else:
            db_sources = []
        is_folder = any(os.path.isdir(src) for src in db_sources)
        # is_folder = isinstance(db_source, str) and os.path.isdir(db_source)

        if c["dataset_format"] == "omol_csh_h5":
            if not isinstance(db_source, str):
                raise TypeError(
                    "OMol_CSH H5 loading requires dbpath to be a file path."
                )
            all_keys = load_key_manifest(db_source)
            total_requested = (
                int(c["num_train"]) + int(c["num_val"]) + int(c["num_test"])
            )
            if total_requested > len(all_keys):
                raise ValueError(
                    f"Requested {total_requested} OMol_CSH samples from "
                    f"a manifest containing {len(all_keys)}."
                )

            tr_start_g, tr_end_g, _ = utils_compute.split_indices(
                self.rank, self.world_size, c["num_train"], False
            )
            val_start_g, val_end_g, _ = utils_compute.split_indices(
                self.rank, self.world_size, c["num_val"], False
            )
            test_start_g, test_end_g, _ = utils_compute.split_indices(
                self.rank, self.world_size, c["num_test"], False
            )
            val_start_g += c["num_train"]
            val_end_g += c["num_train"]
            test_start_g += c["num_train"] + c["num_val"]
            test_end_g += c["num_train"] + c["num_val"]

            rank_indices = (
                list(range(tr_start_g, tr_end_g))
                + list(range(val_start_g, val_end_g))
                + list(range(test_start_g, test_end_g))
            )
            db_obj = OMolCSH58kDatabase(
                db_source,
                indices=rank_indices,
                metadata_policy=c["omol_csh_metadata_policy"],
            )
            num_local_train = tr_end_g - tr_start_g
            num_local_val = val_end_g - val_start_g
            num_local_test = test_end_g - test_start_g
            tr_start, tr_end = 0, num_local_train
            val_start, val_end = tr_end, tr_end + num_local_val
            test_start, test_end = val_end, val_end + num_local_test

            train_database = db_obj if num_local_train else None
            val_database = db_obj if (num_local_val or num_local_test) else None
            scale_shift_data = self._handle_scale_shift(db_obj)
            print(
                f"Rank {self.rank}: OMol_CSH H5 local split has "
                f"train={num_local_train}, val={num_local_val}, "
                f"test={num_local_test}.",
                flush=True,
            )

        elif is_folder:
            # Custom cp2k datasets - each subfolder contains a structure, hamiltonian, and overlap matrix
            if self.config["dataset_name"] == "cp2k_material":
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

                if c["shuffle"]:
                    random.shuffle(data_folders)

                total_needed = c["num_train"] + c["num_val"]
                print(
                    f"Found {len(data_folders)} data folders in {db_source}", flush=True
                )

                train_database = data_folders[: c["num_train"]]
                scale_shift_data = self._handle_scale_shift(train_database)

                # if c['dbpath_val'] is provided, take validation folders from there instead of splitting from the training folders:
                if c.get("dbpath_val") is not None:
                    val_db_source = c["dbpath_val"]
                    val_data_folders = [
                        os.path.join(val_db_source, f)
                        for f in os.listdir(val_db_source)
                        if os.path.isdir(os.path.join(val_db_source, f))
                    ]
                    val_database = val_data_folders[: c["num_val"]]
                else:
                    val_database = (
                        data_folders[c["num_train"] : total_needed]
                        if c["num_val"] > 0
                        else None
                    )

                tr_start, tr_end, _ = utils_compute.split_indices(
                    self.rank, self.world_size, c["num_train"], c["distribute_graphs"]
                )
                val_start, val_end, _ = utils_compute.split_indices(
                    self.rank, self.world_size, c["num_val"], c["distribute_graphs"]
                )
                test_start, test_end, _ = utils_compute.split_indices(
                    self.rank, self.world_size, c["num_test"], c["distribute_graphs"]
                )

            # (Omol) folders of db files
            else:
                print(
                    f"Rank {self.rank}: Loading data from folder {db_source}",
                    flush=True,
                )

                # --- 1. Distribute Data across ranks ---
                val_db_source = c.get("dbpath_val")
                if val_db_source is not None:
                    train_data_dict, _ = distribute_data(
                        base_folder=db_source,
                        world_size=self.world_size,
                        rank=self.rank,
                        N_global_train=c["num_train"],
                        N_global_val=0,
                    )
                    _, val_data_dict = distribute_data(
                        base_folder=val_db_source,
                        world_size=self.world_size,
                        rank=self.rank,
                        N_global_train=0,
                        N_global_val=c["num_val"],
                    )
                else:
                    train_data_dict, val_data_dict = distribute_data(
                        base_folder=db_source,
                        world_size=self.world_size,
                        rank=self.rank,
                        N_global_train=c["num_train"],
                        N_global_val=c["num_val"],
                    )

                # --- 2. Load Training Segments ---
                train_datasets = []
                for entry in train_data_dict:
                    ds = ASEDataset(
                        db_path=entry["db_file"],
                        dtype=c["dtype"],
                        open_shell=c["open_shell"],
                        start_idx=entry["start_idx"],
                        end_idx=entry["end_idx"],
                        matrix_target=c["loss_target"],
                    )
                    train_datasets.append(ds)
                train_database = (
                    ConcatDataset(train_datasets) if train_datasets else None
                )

                # --- 3. Load Validation Segments ---
                val_datasets = []
                for entry in val_data_dict:
                    ds = ASEDataset(
                        db_path=entry["db_file"],
                        dtype=c["dtype"],
                        open_shell=c["open_shell"],
                        start_idx=entry["start_idx"],
                        end_idx=entry["end_idx"],
                        matrix_target=c["loss_target"],
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
                    dtype=c["dtype"],
                    open_shell=c["open_shell"],
                    matrix_target=c["loss_target"],
                )
            else:
                db_obj = db_source

            # --- SINGLE DB FILE MODE ---
            # 1. Calculate split indices
            tr_start, tr_end, _ = utils_compute.split_indices(
                self.rank, self.world_size, c["num_train"], c["distribute_graphs"]
            )
            val_start, val_end, _ = utils_compute.split_indices(
                self.rank, self.world_size, c["num_val"], c["distribute_graphs"]
            )
            test_start, test_end, _ = utils_compute.split_indices(
                self.rank, self.world_size, c["num_test"], c["distribute_graphs"]
            )

            # 2. Offset validation and test to ensure unique molecules
            val_start += c["num_train"]
            val_end += c["num_train"]
            test_start += c["num_train"] + c["num_val"]
            test_end += c["num_train"] + c["num_val"]

            # 3. Get Scale/Shift
            train_database = db_obj
            val_database = db_obj
            scale_shift_data = self._handle_scale_shift(db_obj)
            print("Got scaling/shifting factors for this dataset.", flush=True)

        # 4. Data loading logic
        if c["train_or_eval"] == "train":
            # Note: argument name in get_loader is 'rcut', not 'rcut_orbitals'
            if train_database is None or len(train_database) == 0:
                train_loader = None
            else:
                train_loader, required_irreps, basis_trans, orb_basis, ls_list = (
                    get_loader.get_loader(
                        database=train_database,
                        start_idx=tr_start,
                        end_idx=tr_end,
                        dataset_name=c["dataset_name"],
                        rcut=c["rcut_orbitals"],
                        batch_size=c["batch_size"],
                        dtype=c["dtype"],
                        half_edges=c["reduce_edge"],
                        loss_target_string=c["loss_target"],
                        is_open_shell=c["open_shell"],
                        scale_shift_data=scale_shift_data,
                        distribute_graphs=c["distribute_graphs"],
                        tiling_dims=c["tiling_dims"],
                        partition_type=c["partition_type"],
                        train_or_eval=c["train_or_eval"],
                        delta_learning=c.get("delta_learning", False),
                        shuffle=c["shuffle"],
                        load_delta_auxiliary_matrix=self._needs_delta_auxiliary_matrix(),
                    )
                )

                dist.barrier()
                for i in range(self.world_size):
                    if self.rank == i:
                        first_batch = next(iter(train_loader), None)
                        if first_batch is not None:
                            batch = first_batch
                            if not c["open_shell"]:
                                num_atoms = (
                                    batch["node_y"].shape[1]
                                    if c["distribute_graphs"]
                                    else batch["node_y"].shape[0]
                                )
                                num_edges = (
                                    batch["y"].shape[1]
                                    if c["distribute_graphs"]
                                    else batch["y"].shape[0]
                                )
                            else:
                                num_atoms = (
                                    batch["node_y_alpha"].shape[1]
                                    if c["distribute_graphs"]
                                    else batch["node_y_alpha"].shape[0]
                                )
                                num_edges = (
                                    batch["y_alpha"].shape[1]
                                    if c["distribute_graphs"]
                                    else batch["y_alpha"].shape[0]
                                )

                            print(
                                f"Rank {self.rank}: First train batch - Num atoms: {num_atoms}, Num edges: {num_edges}",
                                flush=True,
                            )
                    dist.barrier()

            if val_database is None or len(val_database) == 0:
                val_loader = None
            else:
                val_loader, *_ = get_loader.get_loader(
                    database=val_database,
                    start_idx=val_start,
                    end_idx=val_end,
                    dataset_name=c["dataset_name"],
                    rcut=c["rcut_orbitals"],
                    batch_size=c["batch_size"],
                    dtype=c["dtype"],
                    half_edges=c["reduce_edge"],
                    loss_target_string=c["loss_target"],
                    is_open_shell=c["open_shell"],
                    scale_shift_data=scale_shift_data,
                    distribute_graphs=c["distribute_graphs"],
                    tiling_dims=c["tiling_dims"],
                    partition_type=c["partition_type"],
                    train_or_eval=c["train_or_eval"],
                    compute_outside_cutoff_reference_stats=c.get(
                        "validation_matrix_metrics", False
                    ),
                    delta_learning=c.get("delta_learning", False),
                    load_delta_auxiliary_matrix=self._needs_delta_auxiliary_matrix(),
                )
            return (
                train_loader,
                val_loader,
                required_irreps,
                basis_trans,
                orb_basis,
                ls_list,
            )

        elif c["train_or_eval"] == "eval":
            # print("Using validation set for testing/evaluation.")
            # test_start = val_start
            # test_end = val_end

            # Eval mode: force batch_size to 1
            test_loader, required_irreps, basis_trans, orb_basis, ls_list = (
                get_loader.get_loader(
                    database=val_database,
                    start_idx=test_start,
                    end_idx=test_end,
                    dataset_name=c["dataset_name"],
                    rcut=c["rcut_orbitals"],
                    batch_size=1,
                    dtype=c["dtype"],
                    half_edges=c["reduce_edge"],
                    loss_target_string=c["loss_target"],
                    is_open_shell=c["open_shell"],
                    scale_shift_data=scale_shift_data,
                    distribute_graphs=c["distribute_graphs"],
                    tiling_dims=c["tiling_dims"],
                    partition_type=c["partition_type"],
                    train_or_eval=c["train_or_eval"],
                    compute_outside_cutoff_reference_stats=c.get(
                        "validation_matrix_metrics", False
                    ),
                    delta_learning=c.get("delta_learning", False),
                    load_delta_auxiliary_matrix=self._needs_delta_auxiliary_matrix(),
                )
            )

        # inference mode:
        else:
            print("Inference mode: using the entire dataset for evaluation.")
            test_loader, required_irreps, basis_trans, orb_basis, ls_list = (
                get_loader.get_loader(
                    database=val_database,
                    start_idx=val_start,
                    end_idx=val_end,
                    dataset_name=c["dataset_name"],
                    rcut=c["rcut_orbitals"],
                    batch_size=1,
                    dtype=c["dtype"],
                    half_edges=c["reduce_edge"],
                    loss_target_string=c["loss_target"],
                    is_open_shell=c["open_shell"],
                    scale_shift_data=scale_shift_data,
                    distribute_graphs=c["distribute_graphs"],
                    tiling_dims=c["tiling_dims"],
                    partition_type=c["partition_type"],
                    train_or_eval=c["train_or_eval"],
                    delta_learning=c.get("delta_learning", False),
                    load_delta_auxiliary_matrix=self._needs_delta_auxiliary_matrix(),
                )
            )

        return None, test_loader, required_irreps, basis_trans, orb_basis, ls_list

    def _needs_delta_auxiliary_matrix(self):
        """Whether delta-learning needs the non-target initial matrix."""
        return False

    def _backbone_summary(self, backbone):
        """Return provenance for the original MALOQ backbone."""
        del backbone
        return {
            "message_passing_schedule": "interleaved",
            "initial_edge_state_mode": "edge_degree",
            "initial_edge_degree_envelope": False,
            "post_atomwise_edge_residual_layers": [],
            "mlp_type": self.config["mlp_type"],
            "esen_grid_resolution": self.config["esen_grid_resolution"],
        }

    def _architecture_name(self, backbone):
        return getattr(backbone, "architecture", "MALOQ-eSEN")

    def _edge_layer_count(self, backbone):
        del backbone
        return int(self.config["num_mp_layers"])

    @staticmethod
    def _collect_muon_parameters(backbone, head):
        """Route every trainable matrix through Muon, matching MuonAdamW."""
        return [
            parameter
            for module in (backbone, head)
            for parameter in module.parameters()
            if parameter.ndim >= 2 and parameter.requires_grad
        ]

    def _collect_output_projection_adamw_parameters(self, backbone):
        """Return explicit node/edge output projections when present."""
        parameters = []
        for module_name in (
            "node_output_projection",
            "edge_output_projection",
        ):
            module = getattr(backbone, module_name, None)
            weight = getattr(module, "weight", None)
            if weight is not None and weight.requires_grad:
                parameters.append(weight)
        return parameters

    def _head_channels(self, backbone):
        del backbone
        return int(self.config["l_embedding_dim"])

    def _build_esen_backbone(self, required_irreps):
        """Build the canonical MALOQ backbone through a neutral seam."""
        c = self.config
        return eSEN_Backbone(
            required_irreps,
            sphere_channels=c["l_embedding_dim"],
            hidden_channels=c["hidden_dim"],
            lmax=required_irreps.lmax,
            mmax=required_irreps.lmax,
            cutoff=c["rcut_gaussian"],
            grid_resolution=c["esen_grid_resolution"],
            edge_channels=c["l_embedding_dim"],
            num_layers=c["num_mp_layers"],
            act_type="gate",
            mlp_type=c["mlp_type"],
            num_distance_basis=c["num_distance_basis"],
            gaussian_width=c["gaussian_width"],
            include_edges=c["include_edges"],
            open_shell=c["open_shell"],
            atom_scalar_embedding_mode=c["atom_scalar_embedding_mode"],
            wigner_backend=c.get("wigner_backend", "torch"),
            distributed_graph_training=c["distribute_graphs"],
            message_type=c["message_type"],
        ).to(self.device)

    def _build_backbone(self, required_irreps):
        return self._build_esen_backbone(required_irreps)

    def _build_matrix_head(
        self,
        *,
        irreps_in,
        required_irreps,
        head_channels,
        orb_basis,
        ls_list,
    ):
        c = self.config
        return Fock_Irreps_Head(
            irreps_in=irreps_in,
            irreps_out=required_irreps,
            lmax=required_irreps.lmax,
            sphere_channels=head_channels,
            reduce_edge=c["reduce_edge"],
            open_shell=c["open_shell"],
            ls_list=ls_list,
            reduce_node=c["reduce_node"],
            reduce_node_intra=c["reduce_node_intra"],
            orbital_basis=orb_basis,
        )

    def build_model(self, required_irreps, orb_basis, ls_list):
        """Initializes backbone, head, optimizer, and scheduler."""
        c = self.config

        # 1. Backbone
        backbone = self._build_backbone(required_irreps)

        # 2. Head
        head_channels = self._head_channels(backbone)
        irreps_in = Irreps(
            [(head_channels, (degree, 1)) for degree in range(required_irreps.lmax + 1)]
        )

        if c["loss_target"] in {"fock_matrix", "density_matrix"}:
            head = self._build_matrix_head(
                irreps_in=irreps_in,
                required_irreps=required_irreps,
                head_channels=head_channels,
                orb_basis=orb_basis,
                ls_list=ls_list,
            )
        elif c["loss_target"] == "forces":
            head = HELM_Force_Head(backbone)

        elif c["loss_target"] == "energies":
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
            projection_policy = c.get(
                "muon_output_projection_policy",
                "shape_muon",
            )
            model_summary = {
                "architecture": self._architecture_name(backbone),
                "model_variant": c.get("model_variant", "maloq-baseline"),
                "backbone_type": c["backbone_type"],
                "head_type": c["head_type"],
                "optimizer_type": c["optimizer_type"],
                "muon_routing": (
                    "ndim_ge_2_muon_except_output_projection_adamw"
                    if (c["optimizer_type"] == "muon" and projection_policy == "adamw")
                    else (
                        "all_trainable_ndim_ge_2"
                        if c["optimizer_type"] == "muon"
                        else None
                    )
                ),
                "seed": int(c["seed"]),
                "backbone_parameters": sum(
                    parameter.numel() for parameter in backbone.parameters()
                ),
                "head_parameters": sum(
                    parameter.numel() for parameter in head.parameters()
                ),
                "total_parameters": sum(
                    parameter.numel() for parameter in backbone.parameters()
                )
                + sum(parameter.numel() for parameter in head.parameters()),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in (
                        *list(backbone.parameters()),
                        *list(head.parameters()),
                    )
                    if parameter.requires_grad
                ),
                "trunk_channels": int(c["l_embedding_dim"]),
                "output_channels": int(head_channels),
                "node_layers": int(c["num_mp_layers"]),
                "edge_layers": self._edge_layer_count(backbone),
                "muon_output_projection_policy": (
                    projection_policy if c["optimizer_type"] == "muon" else None
                ),
                "scale_and_shift": bool(c.get("scale_and_shift", False)),
                "scale_shift_mode": (
                    c.get("scale_shift_mode")
                    if c.get("scale_and_shift", False)
                    else None
                ),
                "delta_learning": bool(c.get("delta_learning", False)),
                "prediction_contract": (
                    (
                        "final_density=initial_density+predicted_delta"
                        if c["loss_target"] == "density_matrix"
                        else "final_hamiltonian=initial_hamiltonian+predicted_delta"
                    )
                    if c.get("delta_learning", False)
                    else "absolute_target"
                ),
            }
            model_summary.update(self._backbone_summary(backbone))
            summary_path = Path(c["output_folder"]) / "model_summary.json"
            summary_path.write_text(json.dumps(model_summary, indent=2) + "\n")
            print(
                f"Model parameters: {model_summary['total_parameters']:,} "
                f"({model_summary['model_variant']})",
                flush=True,
            )

        # 3. Optimizer
        backbone_params = []
        head_params = []
        if c["train_backbone"]:
            backbone_params = list(backbone.parameters())
        else:
            for parameter in backbone.parameters():
                parameter.requires_grad = False

        if c["train_head"]:
            head_params = list(head.parameters())
        else:
            for parameter in head.parameters():
                parameter.requires_grad = False

        params = backbone_params + head_params
        optimizer_type = c["optimizer_type"]
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(params, lr=c["lr_init"])
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=c["lr_init"],
                weight_decay=c.get("weight_decay", 0.0),
            )
        elif optimizer_type == "soap":
            soap_lr = c["lr_init"] if c.get("soap_lr") is None else c["soap_lr"]
            optimizer = optimizers.SOAP(
                params,
                lr=soap_lr,
                betas=tuple(c["soap_betas"]),
                shampoo_beta=c["soap_shampoo_beta"],
                eps=c["soap_eps"],
                weight_decay=c.get("weight_decay", 0.0),
                precondition_frequency=c["soap_precondition_frequency"],
                max_precond_dim=c["soap_max_precondition_dim"],
                precondition_1d=c["soap_precondition_1d"],
                normalize_grads=c["soap_normalize_grads"],
            )
        elif optimizer_type == "muon":
            muon_params = self._collect_muon_parameters(backbone, head)
            projection_policy = c.get(
                "muon_output_projection_policy",
                "shape_muon",
            )
            output_projection_adamw_params = (
                self._collect_output_projection_adamw_parameters(backbone)
                if projection_policy == "adamw"
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
            muon_param_ids = {id(parameter) for parameter in muon_params}
            auxiliary_params = [
                parameter for parameter in params if id(parameter) not in muon_param_ids
            ]
            if not muon_params:
                raise ValueError(
                    "Muon requires at least one trainable matrix parameter."
                )

            parameter_groups = [
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": c["muon_lr"],
                    "name": "matrix_muon",
                }
            ]
            if auxiliary_params:
                auxiliary_lr = (
                    c["lr_init"]
                    if c.get("muon_adamw_lr") is None
                    else c["muon_adamw_lr"]
                )
                parameter_groups.append(
                    {
                        "params": auxiliary_params,
                        "use_muon": False,
                        "lr": auxiliary_lr,
                        "betas": tuple(c["muon_adamw_betas"]),
                        "eps": c["muon_adamw_eps"],
                        "name": "auxiliary_adamw",
                    }
                )
            optimizer = optimizers.Muon(
                parameter_groups,
                lr=c["muon_lr"],
                momentum=c["muon_momentum"],
                nesterov=c["muon_nesterov"],
                ns_steps=c["muon_ns_steps"],
                weight_decay=c.get("weight_decay", 0.0),
                betas=tuple(c["muon_adamw_betas"]),
                eps=c["muon_adamw_eps"],
            )
            if self.rank == 0:
                print(
                    "Muon optimizer: "
                    f"{sum(p.numel() for p in muon_params):,} "
                    "matrix parameters, "
                    f"{sum(p.numel() for p in output_projection_adamw_params):,} "
                    "output-projection AdamW parameters, "
                    f"{sum(p.numel() for p in auxiliary_params):,} "
                    "auxiliary AdamW parameters."
                )

        # 4. Restarts
        self._load_checkpoint(backbone, c["backbone_checkpoint"], "backbone")
        self._load_checkpoint(
            head,
            c["head_checkpoint"],
            "head",
            optimizer if c["restart_optimizer"] else None,
        )

        return backbone, head, optimizer

    def _get_scheduler(self, optimizer, train_loader):
        """Initializes scheduler based on training loader length."""
        c = self.config
        optimizer_steps_per_epoch = math.ceil(
            len(train_loader) / c["gradient_accumulation_steps"]
        )
        scheduler_steps_per_epoch = (
            1 if c.get("step_every_epoch", False) else optimizer_steps_per_epoch
        )
        if c["scheduler_type"] == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=c["patience"],
                threshold=c["threshold"],
            )
        elif c["scheduler_type"] == "cosine":
            t_max = c["num_epochs"] * scheduler_steps_per_epoch
            if self.rank == 0:
                print(f"T_max for scheduler: {t_max}")
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=c["eta_min"]
            )
        elif c["scheduler_type"] == "warmup_polynomial":
            max_steps = max(1, c["num_epochs"] * scheduler_steps_per_epoch)
            warmup_steps = int(c["warmup_steps"])
            power = float(c["scheduler_power"])
            min_lr_ratio = float(c["min_lr_ratio"])
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
                return min_lr_ratio + (1.0 - min_lr_ratio) * ((1.0 - progress) ** power)

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
        path = os.path.join(self.config["output_folder"], filename)
        if (name == "backbone" and self.config["restart_backbone"]) or (
            name == "head" and self.config["restart_head"]
        ):
            if os.path.exists(path):
                if self.rank == 0:
                    print(f"Restarting {name} from {path}")
                ckpt = torch.load(path, map_location=self.device)
                sd = {
                    k.replace("module.", ""): v
                    for k, v in ckpt["model_state_dict"].items()
                }
                model.load_state_dict(sd)
                if optimizer and "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def close(self):
        """Best-effort distributed cleanup for explicit shutdown paths."""
        self.finish_tracking()
        utils_compute.cleanup_process_group(sync_barrier=True)

    def _build_trainer(self, *, backbone, head, head_irreps):
        """Construct the trainer used by :meth:`run`.

        Research workflows may override this feature-neutral factory while
        retaining the canonical data, optimizer, checkpoint, and run paths.
        """
        return splittrainer.SplitTrainer(
            backbone=backbone,
            head=head,
            head_irreps=head_irreps,
            run_name=self.config.get("run_name", "run"),
            save_frequency=self.config.get("save_frequency", 10),
            wandb_run=self.wandb_run,
        )

    def run(self):
        try:
            # HELM's data and training pipeline currently supports these datasets:
            if self.config["dataset_name"] == "QM7":
                target_property = (
                    "density_matrix"
                    if self.config["loss_target"] == "density_matrix"
                    else "hamiltonian"
                )
                load_properties = [
                    "energy",
                    "forces",
                    target_property,
                    "overlap",
                ]
                if self.config.get("delta_learning", False):
                    initial_target_property = (
                        "initial_density_matrix"
                        if target_property == "density_matrix"
                        else "initial_hamiltonian"
                    )
                    load_properties.append(initial_target_property)
                    if self._needs_delta_auxiliary_matrix():
                        auxiliary_property = (
                            "initial_hamiltonian"
                            if target_property == "density_matrix"
                            else "initial_density_matrix"
                        )
                        load_properties.append(auxiliary_property)
                # ASEAtomsData's property setter reads DB metadata through the
                # open connection, so apply the selection after construction.
                database = ASEAtomsData(self.config["dbpath"])
                database.load_properties = load_properties
                if self.config["shuffle"]:
                    print("Shuffling QM7 dataset for training...")
                    indices = list(range(len(database)))
                    random.shuffle(indices)
                    database = [database[i] for i in indices]

            elif self.config["dataset_name"] == "nablaDFT":
                database = HamiltonianDatabase(self.config["dbpath"])
            elif self.config["dataset_name"] == "omol":
                database = None
            elif self.config["dataset_name"] == "cp2k_material":
                database = None
            else:
                raise ValueError(f"Unknown dataset name: {self.config['dataset_name']}")

            """Main execution loop."""
            loader, val_loader, irreps, basis_trans, orb_basis, ls_list = (
                self.prepare_loaders(database)
            )
            backbone, head, optimizer = self.build_model(irreps, orb_basis, ls_list)
            scheduler = self._get_scheduler(optimizer, loader) if loader else None

            trainer = self._build_trainer(
                backbone=backbone,
                head=head,
                head_irreps=irreps,  # Note: update if forces
            )

            target_map = {
                "fock_matrix": ("node_y", "y"),
                "density_matrix": ("node_y", "y"),
                "forces": ("forces", None),
                "energies": ("energies", None),
            }
            node_target, edge_target = target_map[self.config["loss_target"]]

            if self.config["train_or_eval"] == "train":
                trainer.train(
                    self.config["num_epochs"],
                    self.config["train_loss_fxn"],
                    optimizer,
                    scheduler,
                    self.device,
                    train_loader=loader,
                    val_loader=val_loader,
                    loss_target_string=self.config["loss_target"],
                    node_target_name=node_target,
                    edge_target_name=edge_target,
                    output_folder=self.config["output_folder"],
                    train_backbone=self.config["train_backbone"],
                    train_head=self.config["train_head"],
                    basis_transform=basis_trans,
                    compute_uncoupled_loss=self.config.get(
                        "compute_uncoupled_loss", False
                    ),
                    step_every_epoch=self.config.get("step_every_epoch", True),
                    element_references=self.config.get("element_references", None),
                    validation_matrix_metrics=self.config.get(
                        "validation_matrix_metrics", False
                    ),
                    validation_matrix_metrics_frequency=self.config.get(
                        "validation_matrix_metrics_frequency", 1
                    ),
                    gradient_clip_val=self.config.get("gradient_clip_val"),
                    gradient_accumulation_steps=self.config.get(
                        "gradient_accumulation_steps", 1
                    ),
                    wandb_enabled=self.config.get("use_wandb", False),
                    wandb_log_every_n_steps=self.config.get(
                        "wandb_log_every_n_steps", 10
                    ),
                )
            elif self.config["train_or_eval"] == "eval":
                trainer.evaluate(
                    self.config["test_loss_fxn"],
                    self.device,
                    val_loader,
                    loss_target_string=self.config["loss_target"],
                    compute_eigenvalues=self.config["compute_eigenvalues"],
                    node_target_name=node_target,
                    edge_target_name=edge_target,
                    compute_total_energy=self.config["compute_total_energy"],
                    basis_transform=basis_trans,
                    output_folder=self.config["output_folder"],
                    dataset_name=self.config["dataset_name"],
                    orbital_basis=orb_basis,
                    element_references=self.config.get("element_references", None),
                    distributed_graphs=self.config["distribute_graphs"],
                )
            else:
                trainer.infer(
                    self.config["test_loss_fxn"],
                    self.device,
                    val_loader,
                    loss_target_string=self.config["loss_target"],
                    compute_total_energy=self.config["compute_total_energy"],
                    basis_transform=basis_trans,
                    output_folder=self.config["output_folder"],
                    dataset_name=self.config["dataset_name"],
                    orbital_basis=orb_basis,
                    element_references=self.config.get("element_references", None),
                    distributed_graphs=self.config["distribute_graphs"],
                )
        finally:
            self.close()
