# Description: This file contains the functions to train the model. 

import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
import utils
import time
import torch.optim as optim
import compute_env as env
import gc
from mpi4py import MPI
import numpy as np
from torch import cuda
import os
from torch.cuda.amp import autocast, GradScaler
import scipy
import scipy.sparse as sp

DEBUG = os.environ.get("DEBUG", False)

@env.only_rank_zero
def save_training_state(model, optimizer, track_loss_edge, track_loss_node, track_validation_edge, track_validation_node, save_file):
    """
    Save the training state of the model and optimizer
    """
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, save_file + '.pt')
    torch.save(model.state_dict(), save_file + '_state_dic.pt')

    with open(save_file + '_training_loss.txt', 'w') as f:
        for edge, node in zip(track_loss_edge, track_loss_node):
            f.write(f"{edge:.8f}\t{node:.8f}\n")

    with open(save_file + '_validation_loss.txt', 'w') as f:
        for edge, node in zip(track_validation_edge, track_validation_node):
            f.write(f"{edge:.8f}\t{node:.8f}\n")

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    plt.plot(track_loss_edge, label='edge')
    plt.plot(track_validation_node, label='validation node')
    plt.plot(track_validation_edge, label='validation edge')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_file + '_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

############################################################
# Functions to compute the loss with different filtering
############################################################

def get_loss_full_batch(node_output, edge_output, batch, criterion):
    
    """
    Process a batch of data (forward pass + loss) for labels and targets in the uncoupled basis
    """

    # Compute the loss
    loss_node = criterion(node_output, batch.node_y)            # node_y is the node label
    loss_edge = criterion(edge_output, batch.y)                 # y is the edge label
    output = torch.cat([node_output, edge_output], dim=0)
    labels = torch.cat([batch.node_y, batch.y], dim=0)
    loss = criterion(output, labels)    

    return loss_node, loss_edge, loss

# def get_loss_full_batch(node_output, edge_output, batch, criterion, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, partition):
    
#     """
#     Process a batch of data (forward pass + loss) for labels and targets in the uncoupled basis
#     """

#     start_node = partition.start_node
#     end_node = partition.end_node

#     # arange_tensor = torch.arange(labelled_node_size).unsqueeze(0)
#     arange_tensor = torch.arange(end_node - start_node).unsqueeze(0)
#     onsite_edges = torch.cat((arange_tensor, arange_tensor), 0) # edge_index for self-loop (nodes)

#     # Process node predictions
#     # flattened_node_labels = construct_kernel.get_H(batch.node_y)
#     flattened_node_labels = batch.node_y
#     flattened_node_pred = construct_kernel.get_H(node_output)

#     node_label = utils.unflatten(flattened_node_labels, batch.x, onsite_edges,
#                                 equivariant_blocks, atom_orbitals, out_slices)
    
#     node_pred = utils.unflatten(flattened_node_pred, batch.x, onsite_edges,
#                                 equivariant_blocks, atom_orbitals, out_slices)

#     node_label_tensor = torch.cat([matrix.flatten() for matrix in node_label.values()])
#     node_pred_tensor = torch.cat([matrix.flatten() for matrix in node_pred.values()])

#     # Process edge predictions
#     # flattened_edge_labels = construct_kernel.get_H(batch.y)
#     flattened_edge_labels = batch.y
#     flattened_edge_pred = construct_kernel.get_H(edge_output)

#     edge_label = utils.unflatten(flattened_edge_labels, partition.global_atomic_numbers,
#                                     batch.edge_index,
#                                     equivariant_blocks, atom_orbitals, out_slices)
    
#     edge_pred = utils.unflatten(flattened_edge_pred, partition.global_atomic_numbers,
#                                 batch.edge_index,
#                                 equivariant_blocks, atom_orbitals, out_slices)

#     edge_label_tensor = torch.cat([matrix.flatten() for matrix in edge_label.values()])
#     edge_pred_tensor = torch.cat([matrix.flatten() for matrix in edge_pred.values()])

#     # Compute the loss
#     loss_node = criterion(node_pred_tensor, node_label_tensor)
#     loss_edge = criterion(edge_pred_tensor, edge_label_tensor)
#     pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
#     label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
#     loss = criterion(pred_tensor, label_tensor)  

#     return loss_node, loss_edge, loss

def get_loss_flattened(node_output, edge_output, batch, criterion):
    """
    Process a batch of data (forward pass + loss) for labels and targets in the uncoupled basis
    """

    if hasattr(batch, 'labelled_node_size'):
        labelled_node_size = batch.labelled_node_size.item()
        labelled_edge_size = batch.labelled_edge_size.item()
    else:
        batch_size = len(batch)
        labelled_node_size = batch[0].num_nodes * batch_size
        labelled_edge_size = batch[0].num_edges * batch_size

    # Compute the loss
    loss_node = criterion(node_output[0:labelled_node_size], batch.node_y[0:labelled_node_size])            # node_y is the node label
    loss_edge = criterion(edge_output[0:labelled_edge_size], batch.y[0:labelled_edge_size])                 # y is the edge label
    output = torch.cat([node_output[0:labelled_node_size], edge_output[0:labelled_edge_size]], dim=0)
    labels = torch.cat([batch.node_y[0:labelled_node_size], batch.y[0:labelled_edge_size]], dim=0)
    loss = criterion(output, labels)     

    return loss_node, loss_edge, loss

def get_loss_unflattened(node_output, edge_output, batch, criterion, construct_kernel, equivariant_blocks, atom_orbitals, out_slices):
    """
    Process a batch of data (forward pass + loss) for labels and targets in the coupled basis
    """
     
    if hasattr(batch, 'labelled_node_size'):
        labelled_node_size = batch.labelled_node_size.item()
        labelled_edge_size = batch.labelled_edge_size.item()
    else:
        batch_size = len(batch)
        labelled_node_size = batch[0].num_nodes * batch_size
        labelled_edge_size = batch[0].num_edges * batch_size

    arange_tensor = torch.arange(labelled_node_size).unsqueeze(0)
    torch_cat_tensor = torch.cat((arange_tensor, arange_tensor), 0) # edge_index for self-loop (nodes)

    # Process node predictions
    flattened_node_labels = construct_kernel.get_H(batch.node_y[0:labelled_node_size])
    flattened_node_pred = construct_kernel.get_H(node_output[:labelled_node_size])

    node_label = utils.unflatten(flattened_node_labels, batch.x[0:labelled_node_size], torch_cat_tensor,
                                equivariant_blocks, atom_orbitals, out_slices)
    
    node_pred = utils.unflatten(flattened_node_pred, batch.x[0:labelled_node_size], torch_cat_tensor,
                                equivariant_blocks, atom_orbitals, out_slices)

    node_label_tensor = torch.cat([matrix.flatten() for matrix in node_label.values()])
    node_pred_tensor = torch.cat([matrix.flatten() for matrix in node_pred.values()])

    # Process edge predictions
    flattened_edge_labels = construct_kernel.get_H(batch.y[0:labelled_edge_size])
    flattened_edge_pred = construct_kernel.get_H(edge_output[0:labelled_edge_size])

    edge_label = utils.unflatten(flattened_edge_labels, batch.x[0:labelled_node_size],
                                    batch.edge_index[:, 0:labelled_edge_size],
                                    equivariant_blocks, atom_orbitals, out_slices)
    
    edge_pred = utils.unflatten(flattened_edge_pred, batch.x[0:labelled_node_size],
                                batch.edge_index[:, 0:labelled_edge_size],
                                equivariant_blocks, atom_orbitals, out_slices)

    edge_label_tensor = torch.cat([matrix.flatten() for matrix in edge_label.values()])
    edge_pred_tensor = torch.cat([matrix.flatten() for matrix in edge_pred.values()])

    # Compute the loss
    loss_node = criterion(node_pred_tensor, node_label_tensor)
    loss_edge = criterion(edge_pred_tensor, edge_label_tensor)
    pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
    label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
    loss = criterion(pred_tensor, label_tensor)  

    return loss_node, loss_edge, loss

############################################################
# Training the model
############################################################

# Define the loss functions
mse_loss = nn.MSELoss(reduction='mean')
l1_loss = nn.L1Loss(reduction='mean')

# Combine them in a custom way
def combined_loss(output, target):
    return mse_loss(output, target) + l1_loss(output, target)

def train_and_validate_model_subgraph(model, optimizer, partition, training_loader, validation_loader, 
                                      num_epochs=5000, loss_tol=0.0001, patience=500, threshold=1e-3, min_lr=1e-5, 
                                      save_file='model.pth', schedule=False, dtype=torch.float32,
                                      unflatten=False, construct_kernel=None, equivariant_blocks=None, atom_orbitals=None, out_slices=None, criterion='mse'):
    
    device = next(model.parameters()).device  
    # criterion = combined_loss
    if criterion == 'mse':
        print("Using MSE loss...")
        criterion = nn.MSELoss()
    elif criterion == 'mae':
        print("Using MAE loss...")
        criterion = nn.L1Loss()
    else:
        print("Using combined loss...")
        criterion = combined_loss

    p_train = partition['train']
    p_val = partition['validate']

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
    
    if dist.is_available() and dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=False)
    world_size = dist.get_world_size()

    scaler = GradScaler()  # required for mixed precision training

    track_loss_node = []
    track_loss_edge = []
    track_validation_node = []
    track_validation_edge = []
    track_training_loss = [] # node + edge
    track_validation_loss = [] # node + edge

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_push("Per epoch")


        # model.train()
        epoch_start_time = time.time()

        for batch in training_loader:

            if DEBUG:
                cuda.synchronize()
                cuda.nvtx.range_push("Per batch")

                cuda.synchronize()
                dist.barrier()
                cuda.nvtx.range_push("Zero grad + batch to device")


            # _________________________________________________________
            # Gradient cleanup
            optimizer.zero_grad() 

            # _________________________________________________________
            # Forward pass 
            # [Communication of embeddings occurs within the SO3_Embedding class]
            batch = batch.to(device)

            if DEBUG:
                cuda.synchronize()
                dist.barrier()
                cuda.nvtx.range_pop()
                cuda.nvtx.range_push("Forward pass")

            cuda.synchronize()
            dist.barrier()
            batch_start_time = time.time()

            # Forward pass with autocast
            with autocast():
                node_output, edge_output = model(batch, p_train)
                cuda.synchronize()
                forward_pass_time = time.time()

                if DEBUG:
                    cuda.synchronize()
                    dist.barrier()
                    cuda.nvtx.range_pop()
                    cuda.nvtx.range_push("Loss computation")


                # _________________________________________________________
                # Loss computation
                # loss_node, loss_edge, local_loss = get_loss_full_batch(node_output, edge_output, batch, criterion, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, p_train)
                loss_node, loss_edge, local_loss = get_loss_full_batch(node_output, edge_output, batch, criterion)

                if DEBUG:
                    cuda.synchronize()
                    cuda.nvtx.range_pop()
                    cuda.nvtx.range_push("Loss reduction")

            global_loss = local_loss.clone()
            global_loss_node = loss_node.clone()
            global_loss_edge = loss_edge.clone()

            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_loss_node, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_loss_edge, op=dist.ReduceOp.SUM)

            global_loss /= world_size
            global_loss_node /= world_size
            global_loss_edge /= world_size


            if DEBUG:
                cuda.synchronize()
                dist.barrier()  
                cuda.nvtx.range_pop()
                cuda.nvtx.range_push("Backwards pass")

            # _________________________________________________________
            # Backward pass
            cuda.synchronize()
            dist.barrier()
            loss_time = time.time()            
            scaler.scale(global_loss).backward()

            if DEBUG: 
                cuda.synchronize()
                dist.barrier()  
                cuda.nvtx.range_pop()   

                            
            # _________________________________________________________
            # Parameter update
            scaler.step(optimizer)
            scaler.update()

            if DEBUG: 
                cuda.synchronize()
                dist.barrier()
                cuda.nvtx.range_pop()

            cuda.synchronize()
            backward_pass_time = time.time()  

            batch_end_time = time.time()
            forward_pass_duration = forward_pass_time - batch_start_time
            loss_duration = loss_time - forward_pass_time
            backward_pass_duration = backward_pass_time - loss_time
            batch_duration = batch_end_time - batch_start_time

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        track_loss_node.append(global_loss_node.cpu().detach().numpy()) 
        track_loss_edge.append(global_loss_edge.cpu().detach().numpy())
        track_training_loss.append(global_loss.cpu().detach().numpy())
            
 
        print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds", flush=True)
        print(f"--> Forward Pass Time: {forward_pass_duration:.4f} seconds", flush=True)
        print(f"--> Loss Computation Time: {loss_duration:.4f} seconds", flush=True)
        print(f"--> Backward Pass Time: {backward_pass_duration:.4f} seconds", flush=True)
        print(f"--> Total Batch process time: {batch_duration:.4f} seconds", flush=True)
        print("--> Peak memory allocated: " + str(torch.cuda.max_memory_allocated(device)/1e9) + " GB", flush=True)
        print("--> Current memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + " GB", flush=True)
        print(f"--> Memory info: {torch.cuda.mem_get_info(device)}", flush=True)
        print("Epoch: " + str(epoch)+ " loss: " + str(global_loss), flush=True)

        # Validate the model
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)

                # Forward pass
                with autocast():
                    node_output, edge_output = model(batch, p_val) 

                    # Loss computation
                    # loss_node, loss_edge, local_loss = get_loss_full_batch(node_output, edge_output, batch, criterion, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, p_val)
                    loss_node, loss_edge, local_loss = get_loss_full_batch(node_output, edge_output, batch, criterion)

                global_val_loss = local_loss.clone()
                global_loss_node = loss_node.clone()
                global_loss_edge = loss_edge.clone()

                dist.all_reduce(global_val_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_loss_node, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_loss_edge, op=dist.ReduceOp.SUM)

                world_size = dist.get_world_size()
                global_val_loss /= world_size
                global_loss_node /= world_size
                global_loss_edge /= world_size

                validation_loss += global_val_loss.cpu().detach().numpy()

        track_validation_node.append(global_loss_node.cpu().detach().numpy())
        track_validation_edge.append(global_loss_edge.cpu().detach().numpy())
        track_validation_loss.append(global_val_loss.cpu().detach().numpy())

        @env.only_rank_zero
        def print_val_info():
            print("Validation loss: ", validation_loss)
            print("Validation node loss: ", global_loss_node.cpu().detach().numpy())
            print("Validation edge loss: ", global_loss_edge.cpu().detach().numpy())
        print_val_info()

        # save the model and the current training status every 100 epochs
        if epoch % 100 == 0:
            save_training_state(model, optimizer, track_loss_edge, track_loss_node, track_validation_edge, track_validation_node, save_file)
        
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Current Learning Rate: {current_lr:.8f}")
        if current_lr <= min_lr:
            print("Learning rate has reached the minimum threshold. Stopping training.")
            break

        if global_loss < loss_tol:
            print("Loss has reached the minimum threshold. Stopping training.")
            break

        if DEBUG: 
            cuda.synchronize()
            cuda.nvtx.range_pop()

    print("Final loss: ", global_loss) 
    save_training_state(model, optimizer, track_loss_edge, track_loss_node, track_validation_edge, track_validation_node, save_file)


############################################################
# Evaluating/Testing the model
############################################################

def evaluate_model(model, partition, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file='./',reconstruct_ham=False, compute_total_loss=True, plot=True, upper_triangular = False):
    model.eval() 
    local_node_labels = []
    local_node_preds = []
    local_edge_labels = []
    local_edge_preds = []

    start_node = partition.start_node
    end_node = partition.end_node

    # currently only testing on a single rank with 1 batch, need to fix for multiple ranks and batches
    # all examples are set up with 1 batch
    assert len(data_loader) == 1

    if dist.is_available() and dist.is_initialized():
        # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the directed ones
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
        rank = dist.get_rank()
        size = dist.get_world_size()
        comm = MPI.COMM_WORLD
    
    with torch.no_grad(): 

        node_pred_dic = {}
        edge_pred_dic = {}

        start_time = time.time()

        for i, test_batch in enumerate(data_loader):
            print(f"Loading batch {i}/{len(data_loader)}...")
            test_batch = test_batch.to(device)

            # Forward pass
            local_test_node, local_test_edge = model(test_batch, partition)
            print("--> Memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")
            torch.cuda.synchronize()  

            local_test_node = local_test_node.cpu()
            local_test_edge = local_test_edge.cpu()

            arange_tensor = torch.arange(start_node, end_node).unsqueeze(0) #change to global index instead of local index 
            onsite_edges = torch.cat((arange_tensor, arange_tensor), 0)

            # Process node predictions
            flattened_node_pred = construct_kernel.get_H(local_test_node)
            node_pred = utils.unflatten(flattened_node_pred, partition.global_atomic_numbers,
                                        onsite_edges, equivariant_blocks, atom_orbitals, out_slices)

            # Process edge predictions
            flattened_edge_pred = construct_kernel.get_H(local_test_edge)
            edge_pred = utils.unflatten(flattened_edge_pred, partition.global_atomic_numbers,
                                        test_batch.edge_index,
                                        equivariant_blocks, atom_orbitals, out_slices)

            node_pred_dic.update(node_pred)
            edge_pred_dic.update(edge_pred)
                    
            # Clear cache after processing each batch
            torch.cuda.empty_cache()
            gc.collect()  # Python garbage collection
            torch.cuda.synchronize()  
            print("--> Memory allocated (after gc): " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")
    
    end_time = time.time()
    unflatten_time = end_time - start_time

    print(f"Unflatten_time: {unflatten_time:.2f} seconds")
    
    if compute_total_loss == True:

        flattened_node_labels = construct_kernel.get_H(test_batch.node_y.cpu())
        node_label = utils.unflatten(flattened_node_labels, partition.global_atomic_numbers,
                                        onsite_edges, equivariant_blocks, atom_orbitals, out_slices)

        flattened_edge_labels = construct_kernel.get_H(test_batch.y.cpu())
        edge_label = utils.unflatten(flattened_edge_labels, partition.global_atomic_numbers,
                                        test_batch.edge_index,
                                        equivariant_blocks, atom_orbitals, out_slices)
        

        H_block_edge_labels = [matrix.flatten() for matrix in edge_label.values()]
        edge_label_tensor = torch.cat(H_block_edge_labels)
        H_block_edge_pred = [matrix.flatten() for matrix in edge_pred.values()]
        edge_pred_tensor = torch.cat(H_block_edge_pred)
        H_block_node_labels = [matrix.flatten() for matrix in node_label.values()]
        node_label_tensor = torch.cat(H_block_node_labels)
        H_block_node_pred = [matrix.flatten() for matrix in node_pred_dic.values()]
        node_pred_tensor = torch.cat(H_block_node_pred)


        # Compute the MAE
        pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
        label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
        MAEloss_total = torch.mean(torch.abs(pred_tensor - label_tensor))

        hartree_to_eV = 27.21138602

        local_node_labels.append(node_label_tensor)
        local_node_preds.append(node_pred_tensor)
        local_edge_labels.append(edge_label_tensor)
        local_edge_preds.append(edge_pred_tensor)


        local_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        with open(save_file + '_MAE_rank_' + str(local_rank) + '_batch_' + str(i) + '_size_' + str(len(data_loader)) + '.txt', 'w') as f:
            f.write(f"Local Mean Absolute Node Error in mHartree: {torch.mean(torch.abs(node_pred_tensor - node_label_tensor)) * 1e3}\n")
            f.write(f"Local Mean Absolute Edge Error in mHartree: {torch.mean(torch.abs(edge_pred_tensor - edge_label_tensor)) * 1e3}\n")
            f.write(f"Local Mean Absolute Error in mHartree: {MAEloss_total * 1e3}\n")

        # Clear cache after processing each batch
        del local_test_node, local_test_edge, test_batch, node_label, node_pred, edge_label, edge_pred 
        del flattened_node_labels, flattened_node_pred, flattened_edge_labels, flattened_edge_pred, H_block_node_labels, H_block_node_pred, H_block_edge_labels, H_block_edge_pred
        del pred_tensor, label_tensor
        torch.cuda.empty_cache()
        gc.collect()  # Python garbage collection
        torch.cuda.synchronize()  
        print("--> Memory allocated (after gc): " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")

        # Concatenate all results
        local_node_labels = torch.cat(local_node_labels)
        local_node_preds = torch.cat(local_node_preds)
        local_edge_labels = torch.cat(local_edge_labels)
        local_edge_preds = torch.cat(local_edge_preds)

        all_node_labels = partition.comm.gather(local_node_labels, root=0)
        all_node_preds = partition.comm.gather(local_node_preds, root=0)
        all_edge_labels = partition.comm.gather(local_edge_labels, root=0)
        all_edge_preds = partition.comm.gather(local_edge_preds, root=0)

        @env.only_rank_zero
        def print_results():

            all_pred_tensor = torch.cat([torch.cat(all_node_preds), torch.cat(all_edge_preds)])
            all_label_tensor = torch.cat([torch.cat(all_node_labels), torch.cat(all_edge_labels)])
            print("Mean Absolute Node Error in mHartree: ", torch.mean(torch.abs(torch.cat(all_node_labels) - torch.cat(all_node_preds))) * 1e3)
            print("Mean Absolute Edge Error in mHartree: ", torch.mean(torch.abs(torch.cat(all_edge_labels) - torch.cat(all_edge_preds))) * 1e3)
            print("Mean Absolute Error in mHartree: ", torch.mean(torch.abs(all_label_tensor - all_pred_tensor)) * 1e3)

        print_results()

    if reconstruct_ham == True:
        print("Reconstructing Hamiltonian matrix...")
        pred_dic = node_pred_dic.copy()
        pred_dic.update(edge_pred_dic)
        reconstruct_hamiltonian(pred_dic, partition.global_atomic_numbers, partition.comm, local_rank, atom_orbitals, save_file=save_file, upper_triangular=upper_triangular)
    
    # Plotting
    @env.only_rank_zero
    def plot_results():

        # downsample = 1
        # all_node_labels = all_node_labels[::downsample]
        # all_node_preds = all_node_preds[::downsample]
        # all_edge_labels = all_edge_labels[::downsample]
        # all_edge_preds = all_edge_preds[::downsample]

        print("Plotting")
        plt.figure(figsize=(4, 3))
        plt.scatter(torch.cat(all_edge_labels).detach().numpy(), torch.cat(all_edge_preds).detach().numpy(), s=1, alpha=0.5, edgecolor='none', color='crimson', label='Edge')
        plt.scatter(torch.cat(all_node_labels).detach().numpy(), torch.cat(all_node_preds).detach().numpy(), s=1, alpha=0.5, edgecolor='none', color='blue', label='Node')
        plt.plot(torch.cat(all_node_labels).detach().numpy(), torch.cat(all_node_labels).detach().numpy(), c='k', linestyle='dashed', linewidth=0.1, alpha=0.3)
        plt.xlabel("Real $H_{ij}$")
        plt.ylabel("Predicted  $H_{ij}$")
        plt.legend()
        # plt.text(0.5, 0.1, 'Node loss = '+str(MAE_node.item())+', Edge loss = '+str(MAE_edge.item()), fontsize=5, transform=plt.gca().transAxes)
        plt.savefig(save_file+'_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()

    
    if plot == True and compute_total_loss == True:
        plot_results()


def reconstruct_hamiltonian(
    local_pred_dic,
    numbers,
    comm,
    rank,
    atom_orbitals,
    save_file="model_in_training.pth",
    upper_triangular=False,
):
    local_keys = local_pred_dic.keys()
    filtered_local_keys = []

    if upper_triangular == True:
        for key in local_keys:
            if key[0] >= key[1]:
                filtered_local_keys.append(tuple(key))

    else:
        for key in local_keys:
            if key[0] <= key[1]:  # remove all duplicate offsite blocks
                filtered_local_keys.append(tuple(key))

    print("filtering done")

    local_positions = []
    local_values = []

    # Start timing
    start_time = time.time()

    # Convert global_atomic_numbers to NumPy array
    global_atomic_numbers = np.array(numbers.tolist())

    # Compute the number of orbitals for each atom type
    num_orbitals_per_atom = np.array(
        [
            np.sum(2 * np.array(atom_orbitals[str(atom)]) + 1)
            for atom in global_atomic_numbers
        ]
    )

    # Compute starting indices (Hamiltonian indices start from 1)
    starting_indices = (
        np.cumsum(num_orbitals_per_atom) + 1
    )  # switch from 0-based to 1-based indexing
    starting_indices = np.insert(starting_indices, 0, 1)[
        :-1
    ]  # add 1 at the beginning and remove last element

    # Extract atom indices from keys
    keys_array = np.array(filtered_local_keys)  # Shape: (N, 2)
    atom_i_indices = keys_array[:, 0]
    atom_j_indices = keys_array[:, 1]

    starting_i = starting_indices[atom_i_indices]
    starting_j = starting_indices[atom_j_indices]

    H_blocks = [local_pred_dic[tuple(k)] for k in filtered_local_keys]

    # Process each block separately due to varying sizes
    local_positions = []
    local_values = []

    for H_block, s_i, s_j in zip(H_blocks, starting_i, starting_j):
        row_idx, col_idx = np.indices(H_block.shape)

        global_i = s_i + row_idx
        global_j = s_j + col_idx

        # Apply triangular condition
        if upper_triangular:
            mask = global_i >= global_j
        else:
            mask = global_i <= global_j

        H_block = H_block.detach().numpy()
        mask &= H_block != 0

        # Collect valid positions and values
        local_positions.append(np.column_stack((global_i[mask], global_j[mask])))
        local_values.append(H_block[mask])

    # Flatten into single arrays
    local_positions = np.concatenate(local_positions, axis=0)
    local_values = np.concatenate(local_values, axis=0)

    # Step 2: Gather results at the root rank
    all_positions = comm.gather(local_positions, root=0)
    all_values = comm.gather(local_values, root=0)

    # Step 3: Root rank processes and writes
    if rank == 0:
        # Combine results from all ranks
        combined_positions = np.concatenate(all_positions, axis=0)
        combined_values = np.concatenate(all_values, axis=0)

        # Sort by positions
        paired = zip(combined_positions, combined_values)
        sorted_pairs = sorted(paired, key=lambda pair: pair[0][0])
        positions_sorted, values_sorted = zip(*sorted_pairs)

        # Write to the output file
        with open(save_file, "w") as file:
            for (i, j), value in zip(positions_sorted, values_sorted):
                file.write(f"       {i}        {j}  {value:.8e}\n")

        print(f"Hamiltonian matrix written to {save_file}")

    # End timing
    end_time = time.time()

    # Print total time taken
    reconstruct_time = end_time - start_time
    # if rank == 0:
    print(f"Reconstruct_time: {reconstruct_time:.2f} seconds")


@env.only_rank_zero
def plot_eigenvalue_comparison(reference_path, test_path, save_file = "model"):

    plt.rcParams.update({'font.size': 14})
    w = np.load(test_path)

    w_ref = np.load(reference_path)

    print(np.linalg.norm(w - w_ref, ord=2) / np.linalg.norm(w_ref, ord=2))
    print(np.linalg.norm(w - w_ref, ord=1) / np.linalg.norm(w_ref, ord=1))

    plt.figure(figsize=(6, 4))
    plt.scatter(np.arange(len(w)), w, s=1.2, alpha=0.2, c="tomato")  # make dot size smaller
    plt.scatter(0, -10, s=10, c="tomato", label=r"$\mathbf{H}_{ij}^{pred}$") 
    plt.scatter(np.arange(len(w_ref)), w_ref, s=1.2, alpha=0.2, c="mediumslateblue")  # make dot size smaller
    plt.scatter(0, -10, s=10, c="mediumslateblue",  label=r"$\mathbf{H}_{ij}^{GT}$") 

    # y Eigenvale
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue ($\mathbf{H})\;[E_h]$")
    plt.ylim(-2.1, 1.1)

    plt.legend(frameon=False, loc='lower right', title=r"[$\alpha$=0.2]")
    plt.savefig(save_file+"_comparison_eigenvalue"+"_zoom.png", dpi=700, bbox_inches='tight')
    plt.close("all")

@env.only_rank_zero
def compute_eigenvalues(base_path_lower, S_path, base_path_upper = None, symmetrize=False, save_file = "model"):

    # Load the data
    S = np.loadtxt(S_path)

    S_row_ind = S[:, 0].astype(np.int32) - 1
    S_col_ind = S[:, 1].astype(np.int32) - 1
    S_data = S[:, 2]

    S_matrix = sp.coo_matrix((S_data, (S_row_ind, S_col_ind)))   

    H_lower_diagonal_name = base_path_lower
    
    H = np.loadtxt(H_lower_diagonal_name)

    H_row_ind = H[:, 0].astype(np.int32) - 1
    H_col_ind = H[:, 1].astype(np.int32) - 1
    H_data = H[:, 2]

    H_matrix = sp.coo_matrix((H_data, (H_row_ind, H_col_ind)))
    H_matrix = H_matrix.toarray()
    tmp = H_matrix.conj().T.copy()
    # set diagonal to zero
    np.fill_diagonal(tmp, 0)
    H_lower_matrix = H_matrix + tmp

    assert np.allclose(H_lower_matrix, H_lower_matrix.conj().T)

    if symmetrize:
        assert base_path_upper is not None
        H_upper_diagonal_name = base_path_upper

        H = np.loadtxt(H_upper_diagonal_name)

        H_row_ind = H[:, 0].astype(np.int32) - 1
        H_col_ind = H[:, 1].astype(np.int32) - 1
        H_data = H[:, 2]

        H_matrix = sp.coo_matrix((H_data, (H_row_ind, H_col_ind)))
        H_matrix = H_matrix.toarray()
        tmp = H_matrix.conj().T.copy()
        # set diagonal to zero
        np.fill_diagonal(tmp, 0)
        H_upper_matrix = H_matrix + tmp

        assert np.allclose(H_upper_matrix, H_upper_matrix.conj().T)

        H_full_matrix = (H_lower_matrix + H_upper_matrix)/2

    else:
        H_full_matrix = H_lower_matrix


    S_matrix = S_matrix.toarray()
    tmp = S_matrix.conj().T.copy()
    # set diagonal to zero
    np.fill_diagonal(tmp, 0)
    S_matrix = S_matrix + tmp

    assert np.allclose(H_full_matrix, H_full_matrix.conj().T)
    assert np.allclose(S_matrix, S_matrix.conj().T)

    start = time.time()
    w, v = scipy.linalg.eigh(H_full_matrix, S_matrix, lower=True)
    end = time.time()

    print("Time: ", end - start)

    # save the eigenvalues and eigenvectors
    np.save(save_file+'eigenvalues.npy', w)
    np.save(save_file+'eigenvectors.npy', v)