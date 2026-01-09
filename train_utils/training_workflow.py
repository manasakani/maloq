import os
import time
import random
import numpy as np
import torch
from e3nn.o3 import Irreps

from train_utils import loss, utils_compute, splittrainer
from dataset_utils import get_loader, get_scale_shift
from helm.esen_new import eSEN_Backbone, Fock_Irreps_Head, Linear_Force_Head

class TrainingWorkflow:
    def __init__(self, config):
        self.config = config
        self.setup_environment()
        
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
        self.device = utils_compute.setup_env(self.rank, self.world_size)
        compute_end = time.perf_counter()
        
        if self.rank == 0:
            print(f"Time to setup distributed environment: {compute_end - compute_start:.4f}s")
            if not os.path.exists(self.config['output_folder']):
                os.makedirs(self.config['output_folder'])

    def _handle_scale_shift(self, database):
        """Manages the computation or loading of scale/shift factors."""
        if not self.config.get('scale_and_shift'):
            return None

        dataset_name = self.config['dataset_name']
        filename = f"element_scale_shifts_{dataset_name}.pt"
        target_path = os.path.join("./fock_utils/", filename)
        file_exists = os.path.exists(target_path)

        if file_exists:
            data = torch.load(target_path)
            
        # Recompute scale/shift factors for this dataset
        else:
            print(f"[Computing scale/shift factors for {dataset_name}]")
            data = get_scale_shift.get_scale_shift(
                database, dataset_name, self.config['rcut_orbitals'], 
                dtype=self.config['dtype'], reduce_edge=self.config['reduce_edge'], 
                filename=filename
            )
            
        return {
            "element_scalar_means": data["element_scalar_means"],
            "element_scalar_stds": data["element_scalar_stds"],
            "scalar_irrep_indices": data["scalar_irrep_indices"]
        }

    def prepare_loaders(self, database):
        """Calculates splits and returns the appropriate DataLoaders."""
        c = self.config
        
        # 1. Calculate split indices
        tr_start, tr_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_train'])
        val_start, val_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_val'])
        test_start, test_end, _ = utils_compute.split_indices(self.rank, self.world_size, c['num_test'])

        # 2. Offset validation and test to ensure unique molecules
        val_start += c['num_train']; val_end += c['num_train']
        test_start += (c['num_train'] + c['num_val']); test_end += (c['num_train'] + c['num_val'])

        # 3. Get Scale/Shift
        scale_shift_data = self._handle_scale_shift(database)
        
        # 4. Data loading logic
        if c['train_or_eval'] == 'train':
            # Note: argument name in get_loader is 'rcut', not 'rcut_orbitals'
            train_loader, required_irreps, basis_trans, orb_basis, ls_list = get_loader.get_loader(
                database=database,
                start_idx=tr_start,
                end_idx=tr_end,
                dataset_name=c['dataset_name'],
                rcut=c['rcut_orbitals'],
                batch_size=c['batch_size'],
                dtype=c['dtype'],
                half_edges=c['reduce_edge'],
                scale_shift_data=scale_shift_data
            )
            
            val_loader, *_ = get_loader.get_loader(
                database=database,
                start_idx=val_start,
                end_idx=val_end,
                dataset_name=c['dataset_name'],
                rcut=c['rcut_orbitals'],
                batch_size=c['batch_size'],
                dtype=c['dtype'],
                half_edges=c['reduce_edge'],
                scale_shift_data=scale_shift_data
            )
            return train_loader, val_loader, required_irreps, basis_trans, orb_basis, ls_list
            
        else:
            # Eval mode: force batch_size to 1
            test_loader, required_irreps, basis_trans, orb_basis, ls_list = get_loader.get_loader(
                database=database,
                start_idx=test_start,
                end_idx=test_end,
                dataset_name=c['dataset_name'],
                rcut=c['rcut_orbitals'],
                batch_size=1, 
                dtype=c['dtype'],
                half_edges=c['reduce_edge'],
                scale_shift_data=scale_shift_data
            )
        
        return test_loader, None, required_irreps, basis_trans, orb_basis, ls_list

    def build_model(self, required_irreps, orb_basis, ls_list):
        """Initializes backbone, head, optimizer, and scheduler."""
        c = self.config
        
        # 1. Backbone
        backbone = eSEN_Backbone(
            required_irreps, sphere_channels=c['l_embedding_dim'],
            hidden_channels=c['l_embedding_dim'], lmax=required_irreps.lmax,
            mmax=required_irreps.lmax, use_pbc=False, cutoff=c['rcut_gaussian'],
            edge_channels=c['l_embedding_dim'], num_layers=c['num_mp_layers'],
            act_type='gate', mlp_type='spectral', 
            num_distance_basis=c['num_distance_basis'],
            gaussian_width=c['gaussian_width'], include_edges=c['include_edges']
        ).to(self.device)

        # 2. Head
        irreps_in = Irreps([(c['l_embedding_dim'], (l, 1)) for l in range(required_irreps.lmax + 1)])
        
        if c['loss_target'] == 'fock_matrix':
            head = Fock_Irreps_Head(
                irreps_in=irreps_in, irreps_out=required_irreps,
                lmax=required_irreps.lmax, sphere_channels=c['l_embedding_dim'],
                half_edges=c['reduce_edge'], head_type=c['head_type'],
                ls_list=ls_list, reduce_node=c['reduce_node'],
                reduce_node_intra=c['reduce_node_intra'], orbital_basis=orb_basis
            )
        elif c['loss_target'] == "forces":
            head = Linear_Force_Head(backbone)
        
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

    def run(self, database):

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
            'forces': ('forces', None),
            'energy': ('energy', None)
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
                basis_transform=basis_trans, step_every_epoch=self.config.get('step_every_epoch', True)
            )
        else:
            trainer.evaluate(
                self.config['test_loss_fxn'], self.device, loader,
                loss_target_string=self.config['loss_target'],
                node_target_name=node_target, edge_target_name=edge_target,
                basis_transform=basis_trans, output_folder=self.config['output_folder'],
                dataset_name=self.config['dataset_name'], orbital_basis=orb_basis
            )