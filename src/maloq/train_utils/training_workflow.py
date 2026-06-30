import os
import time
import random
import numpy as np
import torch
import json
from e3nn.o3 import Irreps
from torch.utils.data import ConcatDataset
import torch.distributed as dist

from . import loss, utils_compute, splittrainer
from ..dataset_utils import get_loader, get_scale_shift
from ..dataset_utils.ASEDataset import distribute_data, ASEDataset, ASEAtomsData
from ..dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
from ..helm.esen_osh import eSEN_Backbone, Fock_Irreps_Head, HELM_Force_Head, HELM_Energy_Head

class TrainingWorkflow:

    DEFAULTS = {
        "run_name": "run",
        "output_folder": "outputs",
        "open_shell": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "wigner_backend": "torch",
        "distribute_graphs": False,
        "partition_type": None,
        "tiling_dims": None,
        "lr_init": 1e-4,
        "optimizer_type": "adam",
        "compute_total_energy": False,
        "dist_backend": "nccl" if torch.cuda.is_available() else "gloo",
    }

    def __init__(self, config):
        self.config = self.DEFAULTS | config
        self.setup_environment()

        # check_input_config will raise errors if there are incompatible settings
        self.check_input_config()
        
    def setup_environment(self):
        """Initializes seeds, dtypes, and distributed compute environment."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.set_default_dtype(self.config['dtype'])

        # SLURM / Distributed setup
        self.rank = int(os.environ.get('SLURM_PROCID', 0))
        self.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        
        compute_start = time.perf_counter()
        self.device = utils_compute.setup_env(self.rank, self.world_size, backend=self.config['dist_backend'])
        compute_end = time.perf_counter()
        
        if self.rank == 0:
            print(f"Time to setup distributed environment: {compute_end - compute_start:.4f}s")
            if not os.path.exists(self.config['output_folder']):
                os.makedirs(self.config['output_folder'])
        dist.barrier()

    def check_input_config(self):
        """Validates the configuration for incompatible settings, and writes config to output folder."""

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

    def _handle_scale_shift(self, database=None):
        """Manages the computation or loading of scale/shift factors."""
        if not self.config.get('scale_and_shift'):
            return None

        dataset_name = self.config['dataset_name']

        if self.config['loss_target'] in ['fock_matrix', 'density_matrix']:

            if self.config['open_shell']:
                filename = f"element_scale_shifts_osh_{dataset_name}.pt"
            else:
                filename = f"element_scale_shifts_{dataset_name}.pt"

            target_path = os.path.join("./fock_utils/", filename)
            file_exists = os.path.exists(target_path)

            if file_exists:
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
                    "scalar_irrep_indices": data["scalar_irrep_indices"]  
                }
            else:
                return {
                    "element_scalar_means": data["element_scalar_means"],
                    "element_scalar_stds": data["element_scalar_stds"],
                    "scalar_irrep_indices": data["scalar_irrep_indices"]
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
                train_data_dict, val_data_dict = distribute_data(
                    base_folder=db_source,
                    world_size=self.world_size,
                    rank=self.rank,
                    N_global_train=c['num_train'],
                    N_global_val=c['num_val']
                )

                # --- 2. Load Training Segments ---
                train_datasets = []
                for entry in train_data_dict:
                    ds = ASEDataset(
                        db_path=entry['db_file'],
                        dtype=c['dtype'],
                        open_shell=c['open_shell'],
                        start_idx=entry['start_idx'], 
                        end_idx=entry['end_idx']     
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
                        end_idx=entry['end_idx']
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
                    db_source, dtype=c['dtype'], open_shell=c['open_shell']
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
                train_or_eval=c['train_or_eval']
            )

            dist.barrier()
            for i in range(self.world_size):
                if self.rank == i:
                    for batch in train_loader:
                        if not c['open_shell']:
                            num_atoms = batch['node_y'].shape[1] if c['distribute_graphs'] else batch['node_y'].shape[0]
                            num_edges = batch['y'].shape[1] if c['distribute_graphs'] else batch['y'].shape[0]
                        else:
                            num_atoms = batch['node_y_alpha'].shape[1] if c['distribute_graphs'] else batch['node_y_alpha'].shape[0]
                            num_edges = batch['y_alpha'].shape[1] if c['distribute_graphs'] else batch['y_alpha'].shape[0]
                        
                        print(f"Rank {self.rank}: Train batch - Num atoms: {num_atoms}, Num edges: {num_edges}", flush=True)
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
                    train_or_eval=c['train_or_eval']
                )
            return train_loader, val_loader, required_irreps, basis_trans, orb_basis, ls_list
            
        else:
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
                train_or_eval=c['train_or_eval']
            )
        
        return test_loader, None, required_irreps, basis_trans, orb_basis, ls_list

    def build_model(self, required_irreps, orb_basis, ls_list):
        """Initializes backbone, head, optimizer, and scheduler."""
        c = self.config
        
        # 1. Backbone
        backbone = eSEN_Backbone(
            required_irreps, sphere_channels=c['l_embedding_dim'],
            hidden_channels=c['l_embedding_dim'], lmax=required_irreps.lmax,
            mmax=required_irreps.lmax, cutoff=c['rcut_gaussian'],
            edge_channels=c['l_embedding_dim'], num_layers=c['num_mp_layers'],
            act_type='gate', mlp_type='spectral', 
            num_distance_basis=c['num_distance_basis'],
            gaussian_width=c['gaussian_width'], include_edges=c['include_edges'],
            open_shell=c['open_shell'],
            wigner_backend=c.get('wigner_backend', 'torch'),
            distributed_graph_training=c['distribute_graphs']
        ).to(self.device)

        # 2. Head
        irreps_in = Irreps([(c['l_embedding_dim'], (l, 1)) for l in range(required_irreps.lmax + 1)])
        
        if c['loss_target'] == 'fock_matrix' or c['loss_target'] == 'density_matrix':
            head = Fock_Irreps_Head(
                irreps_in=irreps_in, irreps_out=required_irreps,
                lmax=required_irreps.lmax, sphere_channels=c['l_embedding_dim'],
                reduce_edge=c['reduce_edge'], open_shell=c['open_shell'],
                ls_list=ls_list, reduce_node=c['reduce_node'],
                reduce_node_intra=c['reduce_node_intra'], orbital_basis=orb_basis
            )
        elif c['loss_target'] == "forces":
            head = HELM_Force_Head(backbone)
            
        elif c['loss_target'] == "energies":
            head = HELM_Energy_Head(backbone)
        
        head = head.to(self.device)

        # 3. Optimizer
        params = []
        if c['train_backbone']: 
            params += list(backbone.parameters())
        else: 
            for p in backbone.parameters(): p.requires_grad = False
            
        if c['train_head']: 
            params += list(head.parameters())
        else:
            for p in head.parameters(): p.requires_grad = False

        if c.get('optimizer_type', 'adam').lower() == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=c['lr_init'], weight_decay=c.get('weight_decay', 0.0))
        else:
            optimizer = torch.optim.Adam(params, lr=c['lr_init'])

        # 4. Restarts
        self._load_checkpoint(backbone, c['backbone_checkpoint'], "backbone")
        self._load_checkpoint(head, c['head_checkpoint'], "head", optimizer if c['restart_optimizer'] else None)

        return backbone, head, optimizer
    
    def _get_scheduler(self, optimizer, train_loader):
        """Initializes scheduler based on training loader length."""
        c = self.config
        if c['scheduler_type'] == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=c['patience'], threshold=c['threshold']
            )
        elif c['scheduler_type'] == 'cosine':
            t_max = c['num_epochs'] * len(train_loader)
            if self.rank == 0: 
                print(f"T_max for scheduler: {t_max}")
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=c['eta_min'])
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

    def run(self):

        # HELM's data and training pipeline currently supports these datasets:
        if self.config['dataset_name'] == 'QM7':
            database = ASEAtomsData(self.config['dbpath'])
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
            save_frequency=self.config.get('save_frequency', 10)
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
            )
        else:
            trainer.evaluate(
                self.config['test_loss_fxn'], self.device, loader,
                loss_target_string=self.config['loss_target'],
                node_target_name=node_target, edge_target_name=edge_target, compute_total_energy=self.config['compute_total_energy'],
                basis_transform=basis_trans, output_folder=self.config['output_folder'],
                dataset_name=self.config['dataset_name'], orbital_basis=orb_basis,
                element_references=self.config.get('element_references', None),
                distributed_graphs=self.config['distribute_graphs']
            )