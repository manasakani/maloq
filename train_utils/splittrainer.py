import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from e3nn.o3 import Irreps
import wandb
import matplotlib.pyplot as plt
import os
from dataset_utils import get_scale_shift
from fock_utils.get_energy_from_fock import build_density, get_integrals, get_permute_phase, permute_mat
from fock_utils.utils_orca_out import periodic_table_number, sort_by_m, read_orca_out, periodic_table
from fock_utils import basis_sets
import json
from pyscf import gto

# note: removing amp to get better precision for now
def disable_amp(func):
    def wrapper(*args, **kwargs):
        print("Disabling torch amp")
        with torch.cuda.amp.autocast(enabled=False):
            return func(*args, **kwargs)
    return wrapper

def get_timestamp_uid() -> str:
    return datetime.datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())[:4]

class SplitTrainer():

    def __init__(self, backbone, head, head_irreps, save_frequency=10, run_id=None, run_name='noname'):

        self.backbone = backbone      # takes atom graph, outputs internal embeddings
        self.head = head              # takes internal embeddings, outputs fixed irrep size
        self.head_irreps = head_irreps
        self.save_frequency = save_frequency

        if not run_id:
            run_id = str(get_timestamp_uid)
        
        # config: any dictionary, add the training parameters
        config = {}

        wandb.init(config=config,    
                   id=run_id,
                   name=run_name,
                   project='fockmatrices',
                   entity='manasakani')
        
    # -- Train model --
    # @disable_amp
    def train(self, 
            num_epochs, 
            loss_fxn, 
            optimizer, 
            scheduler, 
            device, 
            train_loader, 
            loss_target_string, 
            node_target_name, 
            val_loader=None, 
            edge_target_name=None, 
            output_folder='outputs',
            num_warmup_epochs=0,
            train_backbone=True,
            train_head=True,
            basis_transform=None,
            compute_uncoupled_loss=True,
            element_references=None,
            min_lr=1e-10):

        print(f"Loss Targets: {node_target_name}, {edge_target_name}", flush=True)
        # torch.autograd.set_detect_anomaly(True)

        # Torch compile:
        # self.backbone = torch.compile(self.backbone, fullgraph=True)
        # self.head = torch.compile(self.head, fullgraph=True)

        if not val_loader:
            print("Note: using training dataset for scheduler updates")
            val_loader = train_loader

        dist.barrier()
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank() 
            world_size = dist.get_world_size()
            if train_backbone: 
                self.backbone = nn.parallel.DistributedDataParallel(self.backbone, device_ids=[0], output_device=0, find_unused_parameters=False, gradient_as_bucket_view=True)
            if train_head:
                self.head = nn.parallel.DistributedDataParallel(self.head, device_ids=[0], output_device=0, find_unused_parameters=False, gradient_as_bucket_view=True)
        else:
            rank = 0
        
        # Ensure that the ranks have the same number of batches! - need to be careful of this due to the custom data distribution
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)
        self.check_batch_consistency(num_train_batches, num_val_batches, device)

        include_edges = False
        if edge_target_name:
            include_edges = True
        
        initial_lr = optimizer.param_groups[0]['lr']

        track_loss_node = []
        track_loss_node_val = []
        if include_edges:
            track_loss_edge = []
            track_loss_edge_val = []
        
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            
            if train_backbone:
                self.backbone.train() 
                
            if train_head: 
                self.head.train()

            train_loss_node = 0.0
            train_loss_edge = 0.0
            torch.cuda.reset_peak_memory_stats()
            for batch_idx, batch in enumerate(train_loader):

                optimizer.zero_grad()

                # -- Forward -- 
                forward_start = time.perf_counter()
                batch = batch.to(device)

                backbone_start = time.perf_counter()
                backbone_out = self.backbone(batch) 
                backbone_end = time.perf_counter()

                if loss_target_string == 'fock_matrix':
                    node_output, edge_output_fwd, edge_output_bwd, edge_perm, edge_refl = self.head(backbone_out, batch)
                    head_end = time.perf_counter()
                    
                    this_node_target = getattr(batch, node_target_name)
                    this_edge_target = getattr(batch, edge_target_name)

                    loss_start = time.perf_counter()
                    loss_node, loss_edge, loss = self.compute_fock_loss(
                        node_output, edge_output_fwd, edge_output_bwd,
                        this_node_target, this_edge_target,
                        edge_perm, edge_refl, loss_fxn, self.head_irreps,
                        basis_transform, compute_uncoupled_loss
                    )
                    train_loss_node += loss_node.item()
                    train_loss_edge += loss_edge.item()
                    loss_end = time.perf_counter()
                    if rank == 0:
                        print(f"--> Rank {rank} batch {batch_idx} loss: ", loss.item(), flush=True)
                        current_mem = torch.cuda.memory_allocated() / (1024 * 1024)   
                        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) 
                        print(f"Current: {current_mem:.2f} MB, Peak: {peak_mem:.2f} MB")
                        # print(torch.cuda.memory.memory_summary(), flush=True)
                        print(f"Backbone time: {backbone_end - backbone_start:.4f}s, Head time: {head_end - backbone_end:.4f}s, Loss time: {loss_end - loss_start:.4f}s", flush=True)

                elif loss_target_string == 'forces':
                    node_output = self.head(backbone_out, batch)
                    this_node_target = getattr(batch, node_target_name)

                    this_node_target = this_node_target[:, [1, 2, 0]] # match edge permutations
                    loss = loss_fxn(node_output['forces'], this_node_target, self.head_irreps) 

                    train_loss_node += loss.item()
                    
                elif loss_target_string == 'energies':
                    node_output = self.head(backbone_out, batch)
                    scaled_energies = get_scale_shift.apply_energy_refs(batch, batch.energies, element_references, operation="subtract") # Apply energy reference scaling
                    loss = loss_fxn(node_output['energies'], scaled_energies, self.head_irreps)
                    # print("predicted and reference energies for last train batch: ", node_output['energies'].tolist(), scaled_energies.tolist())
                    train_loss_node += loss.item()

                else:
                    raise ValueError(f"Unknown loss target string: {loss_target_string}")

                forward_end = time.perf_counter()

                # -- Backwards -- 
                backward_start = time.perf_counter()
                loss.backward()
                optimizer.step()
                backward_end = time.perf_counter()

                # Garbage collection 
                if loss_target_string == 'fock_matrix':
                    del node_output, edge_output_fwd, edge_output_bwd, this_node_target, this_edge_target
                    del loss_node, loss_edge, loss
                else:
                    del node_output, loss
                del batch, backbone_out
                
                # if rank == 0:
                #     print("Time per forward pass: ", forward_end - forward_start, flush=True)
                #     print("Time for backward pass: ", backward_end - backward_start, flush=True)

            # -- Output dump -- 
            if loss_target_string == 'fock_matrix':
                track_loss_node.append(train_loss_node/num_train_batches) 
                track_loss_edge.append(train_loss_edge/num_train_batches)
            else:
                track_loss_node.append(train_loss_node/num_train_batches) 

            if rank == 0:
                if loss_target_string == 'fock_matrix':
                    print(f"Epoch {epoch+1}, Train Loss: [node] {track_loss_node[-1]} [edge] {track_loss_edge[-1]}", flush=True)    
                else:
                    print(f"Epoch {epoch+1}, Train Loss: [node] {track_loss_node[-1]}", flush=True)    

            dist.barrier()

            # Validation step
            self.backbone.eval()
            self.head.eval()
            val_loss = 0.0
            val_loss_node = 0.0
            val_loss_edge = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    # -- Forward --
                    batch = batch.to(device)
                    # with autocast():
                    backbone_out = self.backbone(batch) 
                    
                    # -- Loss --
                    if loss_target_string == 'fock_matrix':
                        node_output, edge_output_fwd, edge_output_bwd, edge_perm, edge_refl = self.head(backbone_out, batch)
                        
                        this_node_target = getattr(batch, node_target_name)
                        this_edge_target = getattr(batch, edge_target_name)

                        loss_node, loss_edge, loss = self.compute_fock_loss(
                            node_output, edge_output_fwd, edge_output_bwd,
                            this_node_target, this_edge_target,
                            edge_perm, edge_refl, loss_fxn, self.head_irreps,
                            basis_transform, compute_uncoupled_loss
                        )

                        val_loss_node += loss_node.item()
                        val_loss_edge += loss_edge.item()

                    elif loss_target_string == 'forces':
                        node_output = self.head(backbone_out, batch)
                        this_node_target = getattr(batch, node_target_name)

                        if self.head_irreps == '1x1e':             # permute force vectors to match edge permutations
                            this_node_target = this_node_target[:, [1, 2, 0]]
                            loss = loss_fxn(node_output['forces'], this_node_target, self.head_irreps) 
                        val_loss_node += loss.item()
                    elif loss_target_string == 'energies':
                        node_output = self.head(backbone_out, batch)
                        scaled_energies = get_scale_shift.apply_energy_refs(batch, batch.energies, element_references)
                        loss = loss_fxn(node_output['energies'], scaled_energies, self.head_irreps)
                        # print("predicted and reference energies for last val batch: ", node_output['energies'].tolist(), ref_energies.tolist())

                        val_loss_node += loss.item()

                    else:
                        raise ValueError(f"Unknown loss target string: {loss_target_string}")

                    val_loss += loss.item()

                    # Garbage collection for validation stuff
                    if loss_target_string == 'fock_matrix':
                        del node_output, edge_output_fwd, edge_output_bwd, this_node_target, this_edge_target
                        del loss_node, loss_edge, loss
                    else:
                        del node_output, loss
                    del batch, backbone_out
                    # torch.cuda.empty_cache()
                    # torch.cuda.synchronize()

            # -- Output dump -- 
            if loss_target_string == 'fock_matrix':
                track_loss_node_val.append(val_loss_node/num_val_batches) 
                track_loss_edge_val.append(val_loss_edge/num_val_batches)
            else:
                track_loss_node_val.append(val_loss_node/num_val_batches) 

            if rank == 0:
                if loss_target_string == 'fock_matrix':
                    print(f"Epoch {epoch+1}, Val Loss: [node] {track_loss_node_val[-1]} [edge] {track_loss_edge_val[-1]}", flush=True)    
                else:
                    print(f"Epoch {epoch+1}, Val Loss: [node] {track_loss_node_val[-1]}", flush=True)   
                print(torch.cuda.memory.memory_summary(), flush=True)
            
            if dist.is_initialized():
                val_loss_tensor = torch.tensor(val_loss, device=device)
                val_loss_node_tensor = torch.tensor(val_loss_node, device=device)
                val_loss_edge_tensor = torch.tensor(val_loss_edge, device=device)

                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_loss_node_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_loss_edge_tensor, op=dist.ReduceOp.SUM)

                world_size = dist.get_world_size()
                val_loss = val_loss_tensor.item() / world_size
                val_loss_node = val_loss_node_tensor.item() / world_size
                val_loss_edge = val_loss_edge_tensor.item() / world_size

            # -- Scheduler -- 
            if hasattr(scheduler, 'patience'):  # ReduceLROnPlateau
                scheduler.step(val_loss)
            else:                               # CosineAnnealingLR 
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if rank == 0:
                print("Current learning rate: ", current_lr)
            
            epoch_end = time.perf_counter()
            if rank == 0:
                print("Time per epoch: ", epoch_end - epoch_start)

            # log to wandb:
            # if (epoch + 1) % self.save_frequency == 0:
            update_dict = {"node_loss": float(track_loss_node[-1]), 
                        "node_val_loss": float(track_loss_node_val[-1]),
                        "learning_rate": float(current_lr)}
            if loss_target_string == 'fock_matrix':
                update_dict.update({"edge_loss": float(track_loss_edge[-1]), 
                                    "edge_val_loss": float(track_loss_edge_val[-1])})
            wandb.log(update_dict)
            
            # save state
            if rank == 0:
                if (epoch + 1) % self.save_frequency == 0:
                    if loss_target_string == 'fock_matrix':
                        self.save_training_state(epoch, self.backbone, optimizer, track_loss_node, track_loss_node_val, 'backbone', output_folder, track_loss_edge, track_loss_edge_val)
                        self.save_training_state(epoch, self.head, optimizer, track_loss_node, track_loss_node_val, 'head', output_folder, track_loss_edge, track_loss_edge_val)     
                    else:
                        self.save_training_state(epoch, self.backbone, optimizer, track_loss_node, track_loss_node_val, 'backbone', output_folder)
                        self.save_training_state(epoch, self.head, optimizer, track_loss_node, track_loss_node_val, 'head', output_folder)
    
            # End condition is based on the learning rate:
            min_lr_reached = torch.tensor(float(current_lr == min_lr), device='cuda')
            if min_lr_reached:
                print("Reached minimum learning rate, finished training.")
                if loss_target_string == 'fock_matrix':
                    self.save_training_state(epoch, self.backbone, optimizer, track_loss_node, track_loss_node_val, 'backbone', output_folder, track_loss_edge, track_loss_edge_val)
                    self.save_training_state(epoch, self.head, optimizer, track_loss_node, track_loss_node_val, 'head', output_folder, track_loss_edge, track_loss_edge_val)     
                else:
                    self.save_training_state(epoch, self.backbone, optimizer, track_loss_node, track_loss_node_val, 'backbone', output_folder)
                    self.save_training_state(epoch, self.head, optimizer, track_loss_node, track_loss_node_val, 'head', output_folder)
                return

    # -- Evaluate model --
    def evaluate(self, 
                loss_fxn, 
                device, 
                eval_loader, 
                loss_target_string, 
                node_target_name, 
                edge_target_name=None, 
                basis_transform=None,
                element_references=None,
                output_folder='outputs',
                dataset_name='omol',
                orbital_basis=None,
                compute_total_energy=True):
        
        print(f"Loss Targets: {node_target_name}, {edge_target_name}" )
        print("Running eval.")
        self.backbone.eval() 
        self.head.eval() 

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank() 
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        num_eval_batches = len(eval_loader)
        print(f"Running {num_eval_batches} batches through eval.")

        include_edges = False
        if edge_target_name:
            include_edges = True

        track_loss = []
        track_loss_node = []
        if loss_target_string == 'fock_matrix':
            track_loss_edge = []
        
        # -- Evaluate everything in the train_loader -- 
        # with torch.no_grad():  # NOTE: there is a bug with torch.no_grad() and e3nn_linear (used in the output head!) for e3nn v. 0.5.6. Using 0.5.5 instead.

        # dictionaries to store the orbital blocks, they get rewritten by each batch
        node_outputs = {}
        node_labels = {}
        eigenvalue_maes = []
        total_energy_errors = []
        num_atoms_in_molecule_list = []
        if loss_target_string == 'fock_matrix':
            edge_outputs = {}
            edge_labels = {}
        
        for index, batch in enumerate(eval_loader):
            print(f"Processing molecule {index}...", flush=True)   
            print(f"Number of atoms in molecule {index}: {batch.num_nodes}", flush=True)

            with torch.no_grad():
                batch = batch.to(device)
                backbone_out = self.backbone(batch) 

            # print("Writing embeddings to file...", flush=True)
            # self.write_embeddings_to_file(backbone_out, batch, index, output_folder, rank)
            # self.visualize_embeddings(backbone_out["node_embeddings"], output_folder, keyword='node')
            # self.visualize_embeddings(backbone_out["edge_embeddings"], output_folder, keyword='edge')

            # pass all the batches through:
            if loss_target_string == 'fock_matrix':

                with torch.no_grad():
                    node_output, edge_output, edge_output_bwd, edge_perm, edge_refl  = self.head(backbone_out, batch)

                this_node_target = getattr(batch, node_target_name)
                this_edge_target = getattr(batch, edge_target_name)

                # Undo scale/shift layers:
                print("Undoing scale/shift...", flush=True)
                node_output = batch.fock_target_object[0].undo_scale_shift(node_output)

                # for nabladft, we left the node scaling in
                if dataset_name == 'omol' or dataset_name == 'nablaDFT':
                    this_node_target = batch.fock_target_object[0].undo_scale_shift(this_node_target) # note: remove node scaling from evals

                # # write the corresponding node and edge outputs and targets to file:
                # if rank == 0:
                #     np.save(os.path.join(output_folder, 'node_outputs_' + str(index) + '.npy'),
                #             node_output.cpu().detach().numpy())
                #     np.save(os.path.join(output_folder, 'edge_outputs_' + str(index) + '.npy'),
                #             edge_output.cpu().detach().numpy())
                #     np.save(os.path.join(output_folder, 'node_targets_' + str(index) + '.npy'),
                #             this_node_target.cpu().detach().numpy())
                #     np.save(os.path.join(output_folder, 'edge_targets_' + str(index) + '.npy'),
                #             this_edge_target.cpu().detach().numpy())
                #     # write the required irreps to file:    
                #     with open(os.path.join(output_folder, 'irreps.txt'), 'w') as f:
                #         f.write(f"Head irreps: {self.head_irreps}\n")

                # Transform back to uncoupled basis:
                print("Transforming to uncoupled basis...", flush=True)
                uncoupled_node_outputs = basis_transform.get_H(node_output)
                uncoupled_edge_outputs = basis_transform.get_H(edge_output)
                uncoupled_node_labels = basis_transform.get_H(this_node_target)
                uncoupled_edge_labels = basis_transform.get_H(this_edge_target)

                # Unpad them into the hamiltonian orbital blocks
                print("Unpadding orbital blocks...", flush=True)
                atomic_numbers = batch.atomic_numbers.cpu().detach().numpy()
                node_orbital_blocks_output = batch.fock_target_object[0].unpad_node_blocks(uncoupled_node_outputs, atomic_numbers=atomic_numbers)
                edge_orbital_blocks_output = batch.fock_target_object[0].unpad_edge_blocks(uncoupled_edge_outputs, atomic_numbers=atomic_numbers)
                node_orbital_blocks_label = batch.fock_target_object[0].unpad_node_blocks(uncoupled_node_labels, atomic_numbers=atomic_numbers)
                edge_orbital_blocks_label = batch.fock_target_object[0].unpad_edge_blocks(uncoupled_edge_labels, atomic_numbers=atomic_numbers)
                
                # reassemble the matrix (use the model output for the predicted matrix, and reconstruct the label matrix from the orbital blocks if needed)
                print("Reconstructing matrices...", flush=True)
                output_fock_matrix = batch.fock_target_object[0].reconstruct_matrix(node_orbital_blocks_output, edge_orbital_blocks_output, symmetrize_matrix_if_needed=True)
                if hasattr(batch.fock_target_object[0], 'fock_matrix') and batch.fock_target_object[0].fock_matrix is not None:
                    label_fock_matrix = batch.fock_target_object[0].fock_matrix
                else:
                    # reconstructed from targets:
                    label_fock_matrix = batch.fock_target_object[0].reconstruct_matrix(node_orbital_blocks_label, edge_orbital_blocks_label, symmetrize_matrix_if_needed=True)
                    
                # detach the matrices:
                output_fock_matrix = output_fock_matrix.cpu().detach().numpy()
                label_fock_matrix = label_fock_matrix.cpu().detach().numpy()

                # -------- Debugging code --------
                # test - get label fock from orca output file:
                # label_fock_matrix, elements, coordinates, _ = read_orca_out('/home/manasakani/ocp-modeling-dev/manasakani/fock_datasets/single_water_molecule/rot1/orca.out') 
                # full_basis = {periodic_table[element]: basis_sets.def2_tzvpd[element] for element in basis_sets.def2_tzvpd.keys()}
                # full_basis = dict(sorted(full_basis.items(), key=lambda item: len(item[1]), reverse=True)) # put elements with the largest basis first
                # basis = {element: full_basis[element] for element in elements} 
                # label_fock_matrix = sort_by_m(label_fock_matrix, basis, np.array(elements))  # Re-arrange matrix blocks to yzx notation (m=0 is in the middle)
                # -------- Debugging code --------

                # matrix_out = output_fock_matrix.copy()
                # matrix_out[np.abs(matrix_out) < 1e-5] = 0.0
                # plt.imshow(np.log(np.abs(matrix_out)), vmin=-5.0, vmax=5.0)
                # matrix_symmetry_error = np.abs(matrix_out - np.transpose(matrix_out)).sum() / matrix_out.size
                # print("Matrix symmetry error: ", matrix_symmetry_error)
                # plt.colorbar()
                # plt.savefig("predicted_fock.png", dpi=300, bbox_inches='tight')
                # plt.close()

                # matrix_out = label_fock_matrix.copy()
                # matrix_out[np.abs(matrix_out) < 1e-5] = 0.0
                # plt.imshow(np.log(np.abs(matrix_out)), vmin=-5.0, vmax=5.0)
                # plt.colorbar()
                # plt.savefig("label_fock.png", dpi=300, bbox_inches='tight')
                # plt.close()

                # Compute the eigenvalues and eigenvalue error
                print("Solving generalized eigenvalue problem...", flush=True)
                if hasattr(batch, 'overlap_matrix') and batch.overlap_matrix is not None:
                    overlap_matrix = batch.overlap_matrix.detach().cpu().numpy() 
                    label_eigenvalues = sp.linalg.eigvalsh(label_fock_matrix, overlap_matrix)
                    pred_eigenvalues = sp.linalg.eigvalsh(output_fock_matrix, overlap_matrix)
                else:
                    print("Building overlap matrix and computing eigenvalues...", flush=True)
                    label_eigenvalues, pred_eigenvalues = self.get_overlap_and_eigs(batch, output_fock_matrix, label_fock_matrix, orbital_basis, dataset_name)

                # take the first half (occupied):
                num_occupied = len(label_eigenvalues) // 2
                label_eigenvalues = label_eigenvalues[:num_occupied]
                pred_eigenvalues = pred_eigenvalues[:num_occupied]
                
                eigenvalue_MAE = np.abs(label_eigenvalues - pred_eigenvalues).sum() / len(label_eigenvalues)
                eigenvalue_maes.append(eigenvalue_MAE)
                print("MAE error in eigenvalues: ", eigenvalue_MAE, flush=True)
                self.plot_eigenvalues(label_eigenvalues, pred_eigenvalues, s=5, alpha=0.2)
                # self.plot_eigenvalue_diff(label_fock_matrix, output_fock_matrix, s=5, alpha=0.3, label='Eigenvalue Difference', color='darkgreen')

                num_atoms_in_molecule_list.append(batch.num_atoms_in_molecule.cpu().detach().numpy().tolist()[0])

                # Compute error in total energy from predicted and label Fock matrices:
                if compute_total_energy:
                    print("Computing total energy...", flush=True)
                    total_energy_label = self.get_total_energy(batch, label_fock_matrix, orbital_basis, dataset_name)
                    print("Total energy from label Fock matrix: ", total_energy_label, flush=True)
                    # total_energy_label_recon = self.get_total_energy(batch, label_fock_matrix_recon, orbital_basis, dataset_name)
                    total_energy_pred = self.get_total_energy(batch, output_fock_matrix, orbital_basis, dataset_name)
                    print("Total energy from predicted Fock matrix: ", total_energy_pred, flush=True)
                    total_energy_errors.append(np.abs(total_energy_pred - total_energy_label))
                    # print("Total energy from reconstructed label Fock matrix: ", total_energy_label_recon, flush=True)
                    print("Total energy error from predicted Fock matrix: ", total_energy_errors[-1], flush=True)
                    print("Energy from database: ", batch.energies.cpu().detach().numpy(), flush=True)
                else:
                    total_energy_errors.append(0.0)

                node_outputs.update(node_orbital_blocks_output)
                edge_outputs.update(edge_orbital_blocks_output)
                node_labels.update(node_orbital_blocks_label)
                edge_labels.update(edge_orbital_blocks_label)

            elif loss_target_string == 'energies':
                with torch.no_grad():
                    node_output = self.head(backbone_out, batch)
                ref_energies = batch.energies

                # undo energy referencing:
                unscaled_energies = get_scale_shift.apply_energy_refs(batch, node_output['energies'], element_references, operation="add")

                # divide by number of atoms to get per-atom MAE:
                unscaled_energies /= batch.num_atoms_in_molecule
                ref_energies /= batch.num_atoms_in_molecule

                loss = torch.abs(unscaled_energies - ref_energies).mean()  # use MAE for eval                
                print("predicted and reference energies for last val batch: ", unscaled_energies.tolist(), ref_energies.tolist())
                
            else:
                with torch.no_grad():
                    node_output = self.head(backbone_out, batch)

                this_node_target = getattr(batch, node_target_name)

                if self.head_irreps == '1x1e':             
                    this_node_target = this_node_target[:, [1, 2, 0]] # match edge permutations
                    loss = loss_fxn(node_output['forces'], this_node_target) 
                else:
                    print("To be implemented!") 

            # -- Track -- 
            if loss_target_string == 'fock_matrix':

                print("Tracking loss for batch ", index, flush=True)
                edge_multiplier = 2 if batch.fock_target_object[0].half_edges else 1
                total_node_element_loss = 0
                total_edge_element_loss = 0   
                num_node_block_elements = 0
                num_edge_block_elements = 0

                for node_out, node_label in zip(node_outputs.values(), node_labels.values()):
                    total_node_element_loss += torch.abs(node_out - node_label).sum()
                    num_node_block_elements += node_out.numel()

                for edge_out, edge_label in zip(edge_outputs.values(), edge_labels.values()):
                    total_edge_element_loss += edge_multiplier*(torch.abs(edge_out - edge_label).sum())
                    num_edge_block_elements += edge_multiplier*edge_out.numel()

                # convert matrices to torch tensors for loss computation
                output_fock_matrix = torch.tensor(output_fock_matrix)
                label_fock_matrix = torch.tensor(label_fock_matrix)
                
                total_matrix_mae_loss = torch.abs(output_fock_matrix - label_fock_matrix).sum() / output_fock_matrix.numel()
                track_loss_node.append(total_node_element_loss / num_node_block_elements)
                track_loss_edge.append(total_edge_element_loss / num_edge_block_elements)
                track_loss.append(total_matrix_mae_loss)

            else:
                track_loss.append(loss.item())

            # do output dump in append mode:
            if loss_target_string == 'fock_matrix':
                with open(output_folder + "/" + 'model_fock_' + '_eval_' + str(rank) + '.txt', 'a') as f:
                    f.write(f"{track_loss_edge[-1]:.10f}\t{track_loss_node[-1]:.10f}\t{track_loss[-1]:.10f}\t{eigenvalue_maes[-1]:.10f}\t{total_energy_errors[-1]:.10f}\n")
            elif loss_target_string == 'energies':
                with open(output_folder + "/" + 'model_energies_' + '_eval_' + str(rank) + '.txt', 'a') as f:
                    # f.write(f"{track_loss[-1]:.10f}\n")
                    f.write(f"{unscaled_energies.item():.10f}\t{ref_energies.item():.10f}\t{track_loss[-1]:.10f}\n")
            else:
                raise ValueError(f"Unknown loss target string: {loss_target_string}")

            # remove from gpu memory
            print("Removing batch from GPU memory", flush=True)
            del batch, backbone_out, node_output
            if include_edges:
                del edge_output, this_edge_target, output_fock_matrix, label_fock_matrix, node_orbital_blocks_output, edge_orbital_blocks_output, node_orbital_blocks_label, edge_orbital_blocks_label
            
            node_outputs.clear()
            node_labels.clear()
            if loss_target_string == 'fock_matrix':
                edge_outputs.clear()
                edge_labels.clear()
                
            for param1, param2 in zip(self.backbone.parameters(), self.head.parameters()):
                param1.grad = None 
                param2.grad = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


        print(f"Writing eval outputs to file in {output_folder}...", flush=True)

        # -- Output dump -- 
        if loss_target_string == 'fock_matrix':
            with open(output_folder + "/" + 'model' + '_eval_' + str(rank) + '.txt', 'w') as f:
                f.write(f"Edge_MAE\tNode_MAE\tTotal_MAE\tEigenvalue_MAE\tTotal_Energy_Error\tNum_Atoms\n")
                for edge, node, total, eig, energy, num_atoms in zip(track_loss_edge, track_loss_node, track_loss, eigenvalue_maes, total_energy_errors, num_atoms_in_molecule_list):
                    f.write(f"{edge:.10f}\t{node:.10f}\t{total:.10f}\t{eig:.10f}\t{energy:.10f}\t{num_atoms}\n")
        else:
            with open(output_folder + "/" + 'model' + '_eval_' + str(rank) + '.txt', 'w') as f:
                f.write(f"Energy error in Eh/atom\n")
                for node in track_loss:
                    f.write(f"{node:.10f}\n")
    
    # -- Helper functions for training and evaluation --
    def check_batch_consistency(self, num_train_batches, num_val_batches, device):

        if dist.is_available() and dist.is_initialized():
            train_batches_tensor = torch.tensor([num_train_batches], device=device)
            val_batches_tensor = torch.tensor([num_val_batches], device=device)
            train_batches_list = [torch.zeros_like(train_batches_tensor) for _ in range(dist.get_world_size())]
            val_batches_list = [torch.zeros_like(val_batches_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(train_batches_list, train_batches_tensor)
            dist.all_gather(val_batches_list, val_batches_tensor)

            dist.barrier()

            if not all(train_batches_list[0] == tb for tb in train_batches_list):
                print("Mismatch in number of training batches across ranks!", flush=True)
                print(train_batches_list, flush=True)
                raise ValueError("Mismatch in number of training batches across ranks!", flush=True)
            if not all(val_batches_list[0] == vb for vb in val_batches_list):
                print("Mismatch in number of validation batches across ranks!", flush=True)
                raise ValueError("Mismatch in number of validation batches across ranks!", flush=True)

    def compute_fock_loss(self, node_output, edge_output_fwd, edge_output_bwd, this_node_target, this_edge_target, edge_perm, edge_refl, loss_fxn, head_irreps, basis_transform, compute_uncoupled_loss):
        """Computes the Fock loss for the given outputs and targets."""
        
        # In this case, we only have labels for half the edges (edge_mask), so we need to construct the other half using parity rules + permutation
        # the required transformation is pre-computed in the output head as edge_perm and edge_refl
        if edge_perm is not None:

            edge_target_fwd = this_edge_target
            edge_target_bwd = edge_target_fwd[:, edge_perm] * edge_refl # construct the backward edge targets

            output = torch.cat([node_output, edge_output_fwd, edge_output_bwd], dim=0)
            labels = torch.cat([this_node_target, edge_target_fwd, edge_target_bwd], dim=0)

            # Transform from direct sum of irreps to matrix elements
            if compute_uncoupled_loss:
                output = basis_transform.get_H(output)
                labels = basis_transform.get_H(labels)

            loss_node = loss_fxn(node_output, this_node_target, self.head_irreps)
            edge_output = torch.cat([edge_output_fwd, edge_output_bwd], dim=0)
            edge_labels = torch.cat([edge_target_fwd, edge_target_bwd], dim=0)
            loss_edge = loss_fxn(edge_output, edge_labels, self.head_irreps) 

            loss = loss_fxn(output, labels, self.head_irreps)

        # otherwise, edge_output_fwd has all the edges and we can use it directly
        else:
            output = torch.cat([node_output, edge_output_fwd], dim=0)
            labels = torch.cat([this_node_target, this_edge_target], dim=0)

            # Transform from direct sum of irreps to matrix elements
            if compute_uncoupled_loss:
                output = basis_transform.get_H(output)
                labels = basis_transform.get_H(labels)
                
            loss_node = loss_fxn(node_output, this_node_target, self.head_irreps)
            loss_edge = loss_fxn(edge_output_fwd, this_edge_target, self.head_irreps) 
            loss = loss_fxn(output, labels, self.head_irreps)
            # loss = loss_node + loss_edge

        return loss_node, loss_edge, loss
    
    def write_embeddings_to_file(self, backbone_out, batch, index, output_folder, rank):
        """
        Write the embeddings to file for later analysis
        """

        if rank == 0:
            if not os.path.exists(os.path.join(output_folder, 'embeddings')):
                os.makedirs(os.path.join(output_folder, 'embeddings'))
        dist.barrier()

        # Save as binary .npy files (most efficient and preserves full precision)
        np.save(os.path.join(output_folder, 'embeddings/node_embeddings_' + str(index) + '.npy'),
                backbone_out['node_embeddings'].cpu().detach().numpy())
        if backbone_out['edge_embeddings'] is not None:
            np.save(os.path.join(output_folder, 'embeddings/edge_embeddings_' + str(index) + '.npy'),
                    backbone_out['edge_embeddings'].cpu().detach().numpy())
        np.save(os.path.join(output_folder, 'embeddings/edge_distances_' + str(index) + '.npy'),
                batch.edge_attr.cpu().detach().numpy())

        # save positions and elements ("pos": structure.atoms.get_positions(),"atomic_numbers": structure.atomic_numbers,):
        np.save(os.path.join(output_folder, 'embeddings/positions_' + str(index) + '.npy'),
                batch.pos.cpu().detach().numpy())
        np.save(os.path.join(output_folder, 'embeddings/atomic_numbers_' + str(index) + '.npy'),
                batch.atomic_numbers.cpu().detach().numpy())

        # Also save shapes and metadata as text
        with open(os.path.join(output_folder, 'embeddings/embeddings_metadata.txt'), 'w') as f:
            f.write(f"Node embeddings shape: {backbone_out['node_embeddings'].shape}\n")
            if backbone_out['edge_embeddings'] is not None:
                f.write(f"Edge embeddings shape: {backbone_out['edge_embeddings'].shape}\n")
            f.write(f"Edge distances shape: {batch.edge_attr.shape}\n")
        
        del backbone_out


    def adjust_learning_rate(self, optimizer, epoch, warmup_epochs, initial_lr, final_lr):
        """Adjusts the learning rate linearly during the warmup phase."""
        if epoch < warmup_epochs:
            lr = initial_lr + (final_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warmup epoch {epoch+1}: setting learning rate to {lr}")
    
    def get_orbital(self, tensor, irreps_list, l):
        """
        Extract and return all irreps of type l from the tensor
        """

        collect = []
        pointer = 0
        for irrep in irreps_list:
            irrep_l = int(str(irrep).split('x')[-1][0])
            if irrep_l == l:
                collect.append(tensor[pointer : pointer + 2*l+1])

            pointer += 2*irrep_l+1
        
        return collect
                
    def visualize_embeddings(self, embs, output_folder, keyword, plot_log=True):

        # make a folder called "embeddings" in the output folder if it doesn't exist
        if not os.path.exists(os.path.join(output_folder, 'embeddings')):
            os.makedirs(os.path.join(output_folder, 'embeddings'))
        output_folder = os.path.join(output_folder, 'embeddings')

        # for i, emb in enumerate(embs):
        for i in range(embs.shape[0]):
            emb = embs[i, :, :]

            if not plot_log:
                plt.imshow(emb.cpu().detach().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
            else:
                # plt.imshow(np.log(np.abs(emb.cpu().detach().numpy())), cmap='RdBu_r', vmin=-5.0, vmax=1.0)
                plt.imshow(np.log(np.abs(emb.cpu().detach().numpy())), cmap='viridis', vmin=-5.0, vmax=1.0)
                plt.colorbar()
            plt.savefig(output_folder+"/" + keyword + "_emb_"+str(i)+".png", dpi=300, bbox_inches='tight')
            plt.close()

    def plot_eigenvalues(self, label_eigs, pred_eigs, s=1, alpha=0.3):
        """
        Here for convinience, just plots the eigenvalues of the matrix
        """
        plt.figure(figsize=(4, 3))
        plt.scatter(range(len(label_eigs)), label_eigs, s=1, alpha=alpha, label='label energy eigs', color='blue', edgecolors='none')
        plt.scatter(range(len(pred_eigs)), pred_eigs, s=0.5, alpha=alpha, label='predicted energy eigs', color='red', edgecolors='none')
        plt.xlabel('Eigenvalue #')
        plt.ylabel('Eigenvalue ($E_h$)')
        # plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig("eigenvalues_fock.png", dpi=500, bbox_inches='tight')
        plt.close()

    def plot_eigenvalue_diff(self, matrix1, matrix2, s=1, alpha=0.3, label='', color='blue'):
        """
        Here for convinience, just plots the eigenvalues of the matrix
        """
        eigenvalues_1 = np.linalg.eigvalsh(matrix1)
        eigenvalues_2 = np.linalg.eigvalsh(matrix2)
        eigenvalues = np.abs(eigenvalues_1 - eigenvalues_2)
        plt.scatter(range(len(eigenvalues)), eigenvalues, s=s, alpha=alpha, label=label, color=color, edgecolors='none')
    
    def get_total_energy(self, batch, fock_matrix, orbital_basis, dataset_name):
        """
        Compute the total energy error from the Fock matrix 
        """

        atomic_numbers = batch.atomic_numbers.cpu().numpy()
        positions = batch.pos.cpu().numpy()
        bohr_to_angstrom = 0.529177249

        # nablaDFT positions are in bohr (psi4), QM7 and omol are in angstrom (orca)
        if dataset_name == 'nablaDFT':
            positions *= bohr_to_angstrom 

        atom_list = []
        for z, pos in zip(atomic_numbers, positions):
            element_symbol = periodic_table_number[z] 
            atom_list.append([element_symbol, pos])

        if dataset_name == 'omol':
            basis = 'def2-tzvpd'
            functional = 'wb97m-v'

            # the folder names for omol have the format "X_1_1'", where the end is _charge_spin
            try:
                folder_name = batch.folder_name[0]
                charge = int(folder_name.split('_')[-2])
                spin = int(folder_name.split('_')[-1])
            except:
                print("Warning: folder name not in expected format, assuming neutral molecule.")
                charge = 0
                spin = 1

            # Create molecule
            mol = gto.M(
                atom=atom_list,
                basis=basis,  
                unit='Angstrom',
                ecp='def2-tzvpd',
                charge=charge,
            )

            # orca to pyscf ordering:
            with open('./train_utils/element_perm_omol.json', 'r') as fh:
                json_data = json.loads(fh.read())
            elt_reorder = json_data['element_permuations']
            elt_phase = json_data['element_phases']

            # Reverse the sort so that we can use the orca to pyscf permutation:
            fock_matrix = sort_by_m(fock_matrix, orbital_basis, atomic_numbers, direction="e3nn_to_orca") 

            # reorder to PySCF ordering (this is done atomic element wise)
            F = fock_matrix
            perm, phase = get_permute_phase(mol, elt_reorder, elt_phase)
            F = permute_mat(fock_matrix, perm, phase) 

        elif dataset_name == 'QM7':
            basis = 'def2-svp'
            functional = 'pbe'

            F = sort_by_m(fock_matrix, orbital_basis, atomic_numbers, direction="e3nn_to_pyscf") 

            # Create molecule
            mol = gto.M(
                atom=atom_list,
                basis=basis,  
                unit='Angstrom'
            )    
        elif dataset_name == 'nablaDFT':
            basis = 'def2-svp'
            functional = 'wb97x-d'

            F = sort_by_m(fock_matrix, orbital_basis, atomic_numbers, direction="e3nn_to_pyscf") 

            # Create molecule
            mol = gto.M(
                atom=atom_list,
                basis=basis,  
                unit='Angstrom'
            )    

        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # Get intermediate quantities
        P = build_density(mol, F)
        H, E_xc, V_xc = get_integrals(mol, P, functional, dataset_name)

        # Compute energy
        E_nn = mol.energy_nuc()
        total_energy = 0.5 * np.einsum('ij,ji', P,  H + F - V_xc) + E_xc + E_nn

        return total_energy
    
    def get_overlap_and_eigs(self, batch, output_hamiltonian, label_hamiltonian, orbital_basis, dataset_name):
        assert dataset_name == 'omol', "Overlap computation currently only implemented for omol dataset"

        atomic_numbers = batch.atomic_numbers.cpu().numpy()
        positions = batch.pos.cpu().numpy()

        atom_list = []
        for z, pos in zip(atomic_numbers, positions):
            element_symbol = periodic_table_number[z] 
            atom_list.append([element_symbol, pos])

        basis = 'def2-tzvpd'
        functional = 'wb97m-v'

        # the folder name has the format "X_1_1'", where the end is _charge_spin
        try:
            folder_name = batch.folder_name[0]
            charge = int(folder_name.split('_')[-2])
            spin = int(folder_name.split('_')[-1])
        except:
            print("Warning: folder name not in expected format, assuming neutral molecule.")
            charge = 0
            spin = 1

        # Create molecule
        mol = gto.M(
            atom=atom_list,
            basis=basis,  
            unit='Angstrom',
            ecp='def2-tzvpd',
            charge=charge,
        )

        # orca to pyscf ordering:
        with open('./train_utils/element_perm_omol.json', 'r') as fh:
            json_data = json.loads(fh.read())
        elt_reorder = json_data['element_permuations']
        elt_phase = json_data['element_phases']
        perm, phase = get_permute_phase(mol, elt_reorder, elt_phase)

        # get the overlap matrix, permute focks to pyscf order, and diagonalize them
        overlap = mol.intor('int1e_ovlp')
        
        output_F = permute_mat(output_hamiltonian, perm, phase) 
        label_F = permute_mat(label_hamiltonian, perm, phase)

        e_pred, _ = sp.linalg.eigh(output_F, overlap)
        e_label, _ = sp.linalg.eigh(label_F, overlap)

        return e_pred, e_label

    def save_training_state(self, step, model, optimizer, track_loss_node, track_validation_node, save_file, output_folder, track_loss_edge=None, track_validation_edge=None):
        """
        Save the training state of the model and optimizer
        """

        # Save the model if it was trainable
        if any(param.requires_grad for param in model.parameters()):
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, output_folder + "/" + save_file + '.pt')
            torch.save(model.state_dict(), output_folder + "/" + save_file + '_state_dic.pt')

        if track_loss_edge:
            with open(output_folder + "/" + save_file + '_training_loss.txt', 'w') as f:
                for edge, node in zip(track_loss_edge, track_loss_node):
                    f.write(f"{edge:.10f}\t{node:.10f}\n")

            with open(output_folder + "/" + save_file + '_validation_loss.txt', 'w') as f:
                for edge, node in zip(track_validation_edge, track_validation_node):
                    f.write(f"{edge:.10f}\t{node:.10f}\n")
        else:
            with open(output_folder + "/" + save_file + '_training_loss.txt', 'w') as f:
                for node in track_loss_node:
                    f.write(f"{node:.10f}\n")

            with open(output_folder + "/" + save_file + '_validation_loss.txt', 'w') as f:
                for node in track_validation_node:
                    f.write(f"{node:.10f}\n")

        plt.figure(figsize=(4, 3))
        plt.plot(track_loss_node, '-', c='blue', label='node')

        if track_loss_edge:
            plt.plot(track_loss_edge, '-', c='red', label='edge')

        plt.plot(track_validation_node, '--', c='blue', label='validation node')
        if track_loss_edge:
            plt.plot(track_validation_edge,  '--', c='red', label='validation edge')
            
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend(frameon=False)
        plt.savefig(output_folder + "/" + save_file + '_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
