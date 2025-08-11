import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt
import numpy as np
from e3nn.o3 import Irreps
import wandb
import matplotlib.pyplot as plt
import os

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
    @disable_amp
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
            min_lr=1e-15):

        print(f"Loss Targets: {node_target_name}, {edge_target_name}", flush=True)
        # torch.autograd.set_detect_anomaly(True)

        if not val_loader:
            print("Note: using training dataset for scheduler updates")
            val_loader = train_loader

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank() 
            world_size = dist.get_world_size()
            if train_backbone: 
                self.backbone = nn.parallel.DistributedDataParallel(self.backbone, device_ids=[0], output_device=0, find_unused_parameters=False)
            if train_head:
                self.head = nn.parallel.DistributedDataParallel(self.head, device_ids=[0], output_device=0, find_unused_parameters=False)
        else:
            rank = 0
        
        # scaler = GradScaler()  # for mixed precision training

        # Ensure that the ranks have the same number of batches! - need to be careful of this due to the custom data distribution
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)
        self.check_batch_consistency(num_train_batches, num_val_batches, device)

        include_edges = False
        if edge_target_name:
            include_edges = True
        
        warmup_epochs = num_warmup_epochs
        initial_lr = optimizer.param_groups[0]['lr']

        track_loss_node = []
        track_loss_node_val = []
        if include_edges:
            track_loss_edge = []
            track_loss_edge_val = []
        
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()

            self.adjust_learning_rate(optimizer, epoch, warmup_epochs, initial_lr, initial_lr*10)
            
            if train_backbone:
                self.backbone.train() 
                
            if train_head: 
                self.head.train()

            train_loss_node = 0.0
            train_loss_edge = 0.0
            for batch in train_loader:

                optimizer.zero_grad()

                # -- Forward -- 
                forward_start = time.perf_counter()
                batch = batch.to(device)
                # torch.cuda.reset_peak_memory_stats()
                # with autocast():
                backbone_out = self.backbone(batch) 

                if loss_target_string == 'fock_matrix':
                    node_output, edge_output = self.head(backbone_out, batch)
                    
                    this_node_target = getattr(batch, node_target_name)
                    this_edge_target = getattr(batch, edge_target_name)

                    # compute loss in the uncoupled basis:
                    if compute_uncoupled_loss:
                        node_output = basis_transform.get_H(node_output)
                        edge_output = basis_transform.get_H(edge_output)
                        this_node_target = basis_transform.get_H(this_node_target)
                        this_edge_target = basis_transform.get_H(this_edge_target)
                
                    output = torch.cat([node_output, edge_output], dim=0)
                    labels = torch.cat([this_node_target, this_edge_target], dim=0)
                    loss_node = loss_fxn(node_output, this_node_target, self.head_irreps)
                    loss_edge = loss_fxn(edge_output, this_edge_target, self.head_irreps) 
                    loss = loss_fxn(output, labels, self.head_irreps)

                    train_loss_node += loss_node
                    train_loss_edge += loss_edge

                elif loss_target_string == 'forces':
                    node_output = self.head(backbone_out, batch)
                    this_node_target = getattr(batch, node_target_name)

                    this_node_target = this_node_target[:, [1, 2, 0]] # match edge permutations
                    loss = loss_fxn(node_output['forces'], this_node_target, self.head_irreps) 

                    train_loss_node += loss
                    
                elif loss_target_string == 'energies':
                    node_output = self.head(backbone_out, batch)
                    this_node_target = getattr(batch, node_target_name)
                    loss = loss_fxn(node_output['energies'], this_node_target, self.head_irreps)

                    train_loss_node += loss

                else:
                    raise ValueError(f"Unknown loss target string: {loss_target_string}")

                forward_end = time.perf_counter()

                # if rank == 0:
                #     peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) 
                #     print(f"Peak memory allocation: {peak_mem:.2f} MB")

                # -- Backwards -- 
                # scaler.scale(loss).backward()
                loss.backward()
                # scaler.step(optimizer)
                optimizer.step()
                # scaler.update()
                backward_end = time.perf_counter()

                # if rank == 0:
                #     print("Time per forward pass: ", forward_end - forward_start)
                #     print("Time for both forward and backward pass: ", backward_end - forward_start)
                
            # -- Output dump -- 
            if loss_target_string == 'fock_matrix':
                track_loss_node.append(train_loss_node.cpu().detach().numpy()/num_train_batches) 
                track_loss_edge.append(train_loss_edge.cpu().detach().numpy()/num_train_batches)
            else:
                track_loss_node.append(train_loss_node.cpu().detach().numpy()/num_train_batches) 

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
                        node_output, edge_output = self.head(backbone_out, batch)
                        
                        this_node_target = getattr(batch, node_target_name)
                        this_edge_target = getattr(batch, edge_target_name)

                        # Fock matrix loss
                        if compute_uncoupled_loss:
                            node_output = basis_transform.get_H(node_output)
                            edge_output = basis_transform.get_H(edge_output)
                            this_node_target = basis_transform.get_H(this_node_target)
                            this_edge_target = basis_transform.get_H(this_edge_target)

                        output = torch.cat([node_output, edge_output], dim=0)
                        labels = torch.cat([this_node_target, this_edge_target], dim=0)
                        loss_node = loss_fxn(node_output, this_node_target, self.head_irreps)
                        loss_edge = loss_fxn(edge_output, this_edge_target, self.head_irreps) 
                        loss = loss_fxn(output, labels, self.head_irreps)

                        val_loss_node += loss_node
                        val_loss_edge += loss_edge

                    elif loss_target_string == 'forces':
                        node_output = self.head(backbone_out, batch)
                        this_node_target = getattr(batch, node_target_name)

                        if self.head_irreps == '1x1e':             # permute force vectors to match edge permutations
                            this_node_target = this_node_target[:, [1, 2, 0]]
                            loss = loss_fxn(node_output['forces'], this_node_target, self.head_irreps) 
                        else:
                            print("To be implemented!")  

                        val_loss_node += loss
                    elif loss_target_string == 'energies':
                        node_output = self.head(backbone_out, batch)
                        this_node_target = getattr(batch, node_target_name)

                        loss = loss_fxn(node_output['energies'], this_node_target, self.head_irreps)
                        val_loss_node += loss

                    else:
                        raise ValueError(f"Unknown loss target string: {loss_target_string}")

                    val_loss += loss.item()
            
            # -- Output dump -- 
            if loss_target_string == 'fock_matrix':
                track_loss_node_val.append(val_loss_node.cpu().detach().numpy()/num_val_batches) 
                track_loss_edge_val.append(val_loss_edge.cpu().detach().numpy()/num_val_batches)
            else:
                track_loss_node_val.append(val_loss_node.cpu().detach().numpy()/num_val_batches) 

            if rank == 0:
                if loss_target_string == 'fock_matrix':
                    print(f"Epoch {epoch+1}, Val Loss: [node] {track_loss_node_val[-1]} [edge] {track_loss_edge_val[-1]}", flush=True)    
                else:
                    print(f"Epoch {epoch+1}, Val Loss: [node] {track_loss_node_val[-1]}", flush=True)   
            
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
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            if rank == 0:
                print("Current learning rate: ", current_lr)
            
            epoch_end = time.perf_counter()
            if rank == 0:
                print("Time per epoch: ", epoch_end - epoch_start)

            # log to wandb:
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
                raise ValueError("Mismatch in number of training batches across ranks!", flush=True)
            if not all(val_batches_list[0] == vb for vb in val_batches_list):
                print("Mismatch in number of validation batches across ranks!", flush=True)
                raise ValueError("Mismatch in number of validation batches across ranks!", flush=True)

    def adjust_learning_rate(self, optimizer, epoch, warmup_epochs, initial_lr, final_lr):
        """Adjusts the learning rate linearly during the warmup phase."""
        if epoch < warmup_epochs:
            lr = initial_lr + (final_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warmup epoch {epoch+1}: setting learning rate to {lr}")
    
    # -- Evaluate model --
    def evaluate(self, 
                loss_fxn, 
                device, 
                eval_loader, 
                loss_target_string, 
                node_target_name, 
                edge_target_name=None, 
                basis_transform=None,
                output_folder='outputs'):
        
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
        if loss_target_string == 'fock_matrix':
            edge_outputs = {}
            edge_labels = {}
        
        for index, batch in enumerate(eval_loader):
            print(f"Processing molecule {index}...", flush=True)      

            batch = batch.to(device)
            backbone_out = self.backbone(batch) 

            print("Writing embeddings to file...", flush=True)
            if rank == 0:
                # Save as binary .npy files (most efficient and preserves full precision)
                np.save(os.path.join(output_folder, 'node_embeddings_' + str(index) + '.npy'),
                        backbone_out['node_embeddings'].cpu().detach().numpy())
                np.save(os.path.join(output_folder, 'edge_embeddings_' + str(index) + '.npy'),
                        backbone_out['edge_embeddings'].cpu().detach().numpy())
                np.save(os.path.join(output_folder, 'edge_distances_' + str(index) + '.npy'),
                        batch.edge_attr.cpu().detach().numpy())

                # save positions and elements ("pos": structure.atoms.get_positions(),"atomic_numbers": structure.atomic_numbers,):
                np.save(os.path.join(output_folder, 'positions_' + str(index) + '.npy'),
                        batch.pos.cpu().detach().numpy())
                np.save(os.path.join(output_folder, 'atomic_numbers_' + str(index) + '.npy'),
                        batch.atomic_numbers.cpu().detach().numpy())

                # Also save shapes and metadata as text
                with open(os.path.join(output_folder, 'embeddings_metadata.txt'), 'w') as f:
                    f.write(f"Node embeddings shape: {backbone_out['node_embeddings'].shape}\n")
                    f.write(f"Edge embeddings shape: {backbone_out['edge_embeddings'].shape}\n")
                    f.write(f"Edge distances shape: {batch.edge_attr.shape}\n")

            # self.visualize_embeddings(backbone_out["node_embeddings"], output_folder, keyword='node')
            # self.visualize_embeddings(backbone_out["edge_embeddings"], output_folder, keyword='edge')

            # pass all the batches through:
            if loss_target_string == 'fock_matrix':

                node_output, edge_output = self.head(backbone_out, batch)
                this_node_target = getattr(batch, node_target_name)
                this_edge_target = getattr(batch, edge_target_name)

                # Undo scale/shift layers:
                node_output = batch.fock_target_object[0].undo_scale_shift(node_output)
                this_node_target = batch.fock_target_object[0].undo_scale_shift(this_node_target)

                # write the corresponding node and edge outputs and targets to file:
                if rank == 0:
                    np.save(os.path.join(output_folder, 'node_outputs_' + str(index) + '.npy'),
                            node_output.cpu().detach().numpy())
                    np.save(os.path.join(output_folder, 'edge_outputs_' + str(index) + '.npy'),
                            edge_output.cpu().detach().numpy())
                    np.save(os.path.join(output_folder, 'node_targets_' + str(index) + '.npy'),
                            this_node_target.cpu().detach().numpy())
                    np.save(os.path.join(output_folder, 'edge_targets_' + str(index) + '.npy'),
                            this_edge_target.cpu().detach().numpy())
                    # write the required irreps to file:    
                    with open(os.path.join(output_folder, 'irreps.txt'), 'w') as f:
                        f.write(f"Head irreps: {self.head_irreps}\n")

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
                
                # reassemble the matrix 
                print("Reconstructing matrices...", flush=True)
                output_fock_matrix = batch.fock_target_object[0].reconstruct_matrix(node_orbital_blocks_output, edge_orbital_blocks_output, symmetrize_matrix_if_needed=True)
                label_fock_matrix = batch.fock_target_object[0].reconstruct_matrix(node_orbital_blocks_label, edge_orbital_blocks_label, symmetrize_matrix_if_needed=True)

                matrix_out = output_fock_matrix.cpu().detach().numpy()
                # matrix_out[np.abs(matrix_out) < 1e-6] = 0.0
                plt.imshow(np.log(np.abs(matrix_out)), vmin=-10.0, vmax=5.0)
                matrix_symmetry_error = np.abs(matrix_out - np.transpose(matrix_out)).sum() / matrix_out.size
                print("Matrix symmetry error: ", matrix_symmetry_error)
                plt.colorbar()
                plt.savefig("predicted_fock_tranpose.png", dpi=300, bbox_inches='tight')
                plt.close()

                matrix_out = label_fock_matrix.cpu().detach().numpy()
                plt.imshow(np.log(np.abs(matrix_out)), vmin=-10.0, vmax=5.0)
                plt.colorbar()
                plt.savefig("label_fock.png", dpi=300, bbox_inches='tight')
                plt.close()

                matrix_out = np.abs(np.abs(label_fock_matrix.detach().cpu().numpy()) - np.abs(output_fock_matrix.detach().cpu().numpy()))
                plt.imshow(matrix_out, vmin=0.0, vmax=0.002)
                plt.colorbar()
                plt.savefig("diff_fock.png", dpi=300, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(4, 3))
                self.plot_eigenvalues(label_fock_matrix.detach().cpu().numpy(), s=5, alpha=0.2, label='Labeled Fock', color='red')
                self.plot_eigenvalues(output_fock_matrix.detach().cpu().numpy(), s=2, alpha=0.5, label='Predicted Fock', color='blue')
                # self.plot_eigenvalue_diff(label_fock_matrix.detach().cpu().numpy(), output_fock_matrix.detach().cpu().numpy(), s=5, alpha=0.3, label='Eigenvalue Difference', color='darkgreen')

                # Compute the eigenvalues and eigenvalue error
                print("Computing eigenvalues...", flush=True)
                label_eigenvalues = np.linalg.eigvalsh(label_fock_matrix.detach().cpu().numpy())
                pred_eigenvalues = np.linalg.eigvalsh(output_fock_matrix.detach().cpu().numpy())
                eigenvalue_MAE = np.abs(label_eigenvalues - pred_eigenvalues).sum() / len(label_eigenvalues)
                eigenvalue_maes.append(eigenvalue_MAE)
                print("MAE error in eigenvalues: ", eigenvalue_MAE, flush=True)

                plt.xlabel('Eigenvalue #')
                plt.ylabel('Eigenvalue ($E_h$)')
                plt.yscale('log')
                plt.legend()
                plt.grid(True)
                plt.savefig("eigenvalues_fock.png", dpi=500, bbox_inches='tight')
                plt.close()

                node_outputs.update(node_orbital_blocks_output)
                edge_outputs.update(edge_orbital_blocks_output)
                node_labels.update(node_orbital_blocks_label)
                edge_labels.update(edge_orbital_blocks_label)

            else:
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
                edge_multiplier = 2 if batch.fock_target_object[0].reflection_symmetry else 1
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
                
                total_matrix_mae_loss = torch.abs(output_fock_matrix - label_fock_matrix).sum() / output_fock_matrix.numel()
                track_loss_node.append(total_node_element_loss / num_node_block_elements)
                track_loss_edge.append(total_edge_element_loss / num_edge_block_elements)
                track_loss.append(total_matrix_mae_loss)

            else:
                track_loss.append(loss.cpu().detach().numpy()) 

            # do output dump in append mode:
            if loss_target_string == 'fock_matrix':
                with open(output_folder + "/" + 'model_fock_' + '_eval_' + str(rank) + '.txt', 'a') as f:
                    f.write(f"{track_loss_edge[-1]:.10f}\t{track_loss_node[-1]:.10f}\t{track_loss[-1]:.10f}\t{eigenvalue_maes[-1]:.10f}\n")
            
            # remove from gpu
            print("Removing batch from GPU memory", flush=True)
            del batch, backbone_out, node_output, this_node_target
            if include_edges:
                del edge_output, this_edge_target, output_fock_matrix, label_fock_matrix, node_orbital_blocks_output, edge_orbital_blocks_output, node_orbital_blocks_label, edge_orbital_blocks_label
            for param1, param2 in zip(self.backbone.parameters(), self.head.parameters()):
                param1.grad = None 
                param2.grad = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"Writing eval outputs to file in {output_folder}...", flush=True)

        # -- Output dump -- 
        if loss_target_string == 'fock_matrix':
            with open(output_folder + "/" + 'model' + '_eval_' + str(rank) + '.txt', 'w') as f:
                    for edge, node, total, eig in zip(track_loss_edge, track_loss_node, track_loss, eigenvalue_maes):
                        f.write(f"{edge:.10f}\t{node:.10f}\t{total:.10f}\t{eig:.10f}\n")
        else:
            with open(output_folder + "/" + 'model' + '_eval_' + str(rank) + '.txt', 'w') as f:
                    for node in track_loss:
                        f.write(f"{node:.10f}\n")
    
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

        for i, emb in enumerate(embs):

            if not plot_log:
                plt.imshow(emb.cpu().detach().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
            else:
                # plt.imshow(np.log(np.abs(emb.cpu().detach().numpy())), cmap='RdBu_r', vmin=-5.0, vmax=1.0)
                plt.imshow(np.log(np.abs(emb.cpu().detach().numpy())), cmap='viridis', vmin=-5.0, vmax=1.0)
                plt.colorbar()
            plt.savefig(output_folder+"/" + keyword + "_emb_"+str(i)+".png", dpi=300, bbox_inches='tight')
            plt.close()

    def plot_eigenvalues(self, matrix, s=1, alpha=0.3, label='', color='blue'):
        """
        Here for convinience, just plots the eigenvalues of the matrix
        """
        eigenvalues = np.linalg.eigvalsh(matrix)
        plt.scatter(range(len(eigenvalues)), eigenvalues, s=s, alpha=alpha, label=label, color=color, edgecolors='none')
    
    def plot_eigenvalue_diff(self, matrix1, matrix2, s=1, alpha=0.3, label='', color='blue'):
        """
        Here for convinience, just plots the eigenvalues of the matrix
        """
        eigenvalues_1 = np.linalg.eigvalsh(matrix1)
        eigenvalues_2 = np.linalg.eigvalsh(matrix2)
        eigenvalues = np.abs(eigenvalues_1 - eigenvalues_2)
        plt.scatter(range(len(eigenvalues)), eigenvalues, s=s, alpha=alpha, label=label, color=color, edgecolors='none')

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
