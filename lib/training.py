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

def train_and_validate_model_subgraph(model, optimizer, partition, training_loader, validation_loader, 
                                      num_epochs=5000, loss_tol=0.0001, patience=500, threshold=1e-3, min_lr=1e-5, 
                                      save_file='model.pth', schedule=False, dtype=torch.float32,
                                      unflatten=False, construct_kernel=None, equivariant_blocks=None, atom_orbitals=None, out_slices=None):
    
    device = next(model.parameters()).device  
    criterion = nn.MSELoss(reduction='mean')

    p_train = partition['train']
    p_val = partition['validate']

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
    
    if dist.is_available() and dist.is_initialized():
        # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the one-way edges
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    world_size = dist.get_world_size()

    track_loss_node = []
    track_loss_edge = []
    track_validation_node = []
    track_validation_edge = []
    track_training_loss = [] # node + edge
    track_validation_loss = [] # node + edge

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):

        # model.train()
        epoch_start_time = time.time()

        for batch in training_loader:

            batch_start_time = time.time()

            # _________________________________________________________
            # Gradient cleanup
            optimizer.zero_grad() 

            # _________________________________________________________
            # Forward pass 
            # [Communication of embeddings occurs within the SO3_Embedding class]
            batch = batch.to(device)
            node_output, edge_output = model(batch, p_train)
            forward_pass_time = time.time()

            # _________________________________________________________
            # Loss computation
            loss_node, loss_edge, local_loss = get_loss_full_batch(node_output, edge_output, batch, criterion)

            global_loss = local_loss.clone()
            global_loss_node = loss_node.clone()
            global_loss_edge = loss_edge.clone()

            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_loss_node, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_loss_edge, op=dist.ReduceOp.SUM)

            global_loss /= world_size
            global_loss_node /= world_size
            global_loss_edge /= world_size

            # _________________________________________________________
            # Backward pass
            global_loss.backward()    
            backward_pass_time = time.time()                              
                        
            # _________________________________________________________
            # Parameter update
            optimizer.step()

            batch_end_time = time.time()
            forward_pass_duration = forward_pass_time - batch_start_time
            backward_pass_duration = backward_pass_time - forward_pass_time
            batch_duration = batch_end_time - batch_start_time

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        track_loss_node.append(global_loss_node.cpu().detach().numpy()) 
        track_loss_edge.append(global_loss_edge.cpu().detach().numpy())
        track_training_loss.append(global_loss.cpu().detach().numpy())
            
        @env.only_rank_zero
        def print_train_info(): 
            print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds")
            print(f"--> Forward Pass Time: {forward_pass_duration:.4f} seconds")
            print(f"--> Backward Pass Time: {backward_pass_duration:.4f} seconds")
            print(f"--> Total Batch process time: {batch_duration:.4f} seconds")
            print("--> Memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")
            print(f"--> Memory info: {torch.cuda.mem_get_info(device)}")
            print("Epoch: " + str(epoch)+ " loss: " + str(global_loss))
        print_train_info()

        # Validate the model
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)

                # Forward pass
                node_output, edge_output = model(batch, p_val) 

                # Loss computation
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
            
    print("Final loss: ", global_loss) 
    save_training_state(model, optimizer, track_loss_edge, track_loss_node, track_validation_edge, track_validation_node, save_file)


############################################################
# Evaluating/Testing the model
############################################################

def evaluate_model(model, partition, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file='./'):
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
        MAEloss_total = 0.0

        for i, test_batch in enumerate(data_loader):
            print(f"Loading batch {i}/{len(data_loader)}...")
            test_batch = test_batch.to(device)

            # Forward pass
            local_test_node, local_test_edge = model(test_batch, partition)
            print("--> Memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")
            torch.cuda.synchronize()  

            local_test_node = local_test_node.cpu()
            local_test_edge = local_test_edge.cpu()

            arange_tensor = torch.arange(end_node - start_node).unsqueeze(0)
            # arange_tensor = torch.arange(start_node, end_node).unsqueeze(0)
            onsite_edges = torch.cat((arange_tensor, arange_tensor), 0)

            # Process node predictions
            flattened_node_labels = construct_kernel.get_H(test_batch.node_y.cpu())
            flattened_node_pred = construct_kernel.get_H(local_test_node)

            node_label = utils.unflatten(flattened_node_labels, test_batch.x, #partition.global_atomic_numbers,
                                         onsite_edges, equivariant_blocks, atom_orbitals, out_slices)
            
            node_pred = utils.unflatten(flattened_node_pred, test_batch.x, #partition.global_atomic_numbers,
                                        onsite_edges, equivariant_blocks, atom_orbitals, out_slices)
                        
            H_block_node_labels = [matrix.flatten() for matrix in node_label.values()]
            node_label_tensor = torch.cat(H_block_node_labels)
            H_block_node_pred = [matrix.flatten() for matrix in node_pred.values()]
            node_pred_tensor = torch.cat(H_block_node_pred)

            # Process edge predictions
            flattened_edge_labels = construct_kernel.get_H(test_batch.y.cpu())
            flattened_edge_pred = construct_kernel.get_H(local_test_edge)

            edge_label = utils.unflatten(flattened_edge_labels, partition.global_atomic_numbers,
                                         test_batch.edge_index,
                                         equivariant_blocks, atom_orbitals, out_slices)
            
            edge_pred = utils.unflatten(flattened_edge_pred, partition.global_atomic_numbers,
                                        test_batch.edge_index,
                                        equivariant_blocks, atom_orbitals, out_slices)
                    
            H_block_edge_labels = [matrix.flatten() for matrix in edge_label.values()]
            edge_label_tensor = torch.cat(H_block_edge_labels)
            H_block_edge_pred = [matrix.flatten() for matrix in edge_pred.values()]
            edge_pred_tensor = torch.cat(H_block_edge_pred)

            # Compute the MAE
            pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
            label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
            MAEloss_total += torch.mean(torch.abs(pred_tensor - label_tensor))

            hartree_to_eV = 27.21138602
            print("Mean Absolute Node Error in mHartree: ", torch.mean(torch.abs(node_pred_tensor - node_label_tensor)) * 1e3)
            print("Mean Absolute Edge Error in mHartree: ", torch.mean(torch.abs(edge_pred_tensor - edge_label_tensor)) * 1e3)
            print("Mean Absolute Error in mHartree: ", MAEloss_total * 1e3)
            print("Mean Absolute Error in meV: ", MAEloss_total * hartree_to_eV * 1e3)

            # Collect results for plotting
            local_node_labels.append(node_label_tensor)
            local_node_preds.append(node_pred_tensor)
            local_edge_labels.append(edge_label_tensor)
            local_edge_preds.append(edge_pred_tensor)

            local_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            with open(save_file + '_MAE_rank_' + str(local_rank) + '_batch_' + str(i) + '_size_' + str(len(data_loader)) + '.txt', 'w') as f:
                f.write(f"Mean Absolute Node Error in mHartree: {torch.mean(torch.abs(node_pred_tensor - node_label_tensor)) * 1e3}\n")
                f.write(f"Mean Absolute Edge Error in mHartree: {torch.mean(torch.abs(edge_pred_tensor - edge_label_tensor)) * 1e3}\n")
                f.write(f"Mean Absolute Error in mHartree: {MAEloss_total * 1e3}\n")

            # Clear cache after processing each batch
            del local_test_node, local_test_edge, test_batch, node_label, node_pred, edge_label, edge_pred 
            del flattened_node_labels, flattened_node_pred, flattened_edge_labels, flattened_edge_pred, H_block_node_labels, H_block_node_pred, H_block_edge_labels, H_block_edge_pred
            del pred_tensor, label_tensor
            torch.cuda.empty_cache()
            gc.collect()  # Python garbage collection
            torch.cuda.synchronize()  
            print("--> Memory allocated (after gc): " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")

        # average the loss over all batches
        MAEloss_total = MAEloss_total / len(data_loader)

    # Concatenate all results
    local_node_labels = torch.cat(local_node_labels)
    local_node_preds = torch.cat(local_node_preds)
    local_edge_labels = torch.cat(local_edge_labels)
    local_edge_preds = torch.cat(local_edge_preds)

    # **** Allgather results --> TURNED OFF FOR NOW
    all_node_labels = local_node_labels
    all_node_preds = local_node_preds
    all_edge_labels = local_edge_labels
    all_edge_preds = local_edge_preds
    # all_node_labels = env.allgatherv_cpu_numpy_1D(local_node_labels, device, partition.comm)
    # all_node_preds = env.allgatherv_cpu_numpy_1D(local_node_preds, device, partition.comm)
    # all_edge_labels = env.allgatherv_cpu_numpy_1D(local_edge_labels, device, partition.comm)
    # all_edge_preds = env.allgatherv_cpu_numpy_1D(local_edge_preds, device, partition.comm)
    # **** Allgather finished

    # * Note: testing is always done on a single rank with 1 batch so far, need to modify this for multiple ranks and batches
    # collect the loss from all ranks:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(torch.tensor(MAEloss_total, device=device), op=dist.ReduceOp.SUM)
    
    local_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    with open(save_file + '_MAE_rank_' + str(local_rank) + '_batch_' + str(i) + '_size_' + str(len(data_loader)) + '.txt', 'w') as f:
        f.write(f"Mean Absolute Node Error in mHartree: {torch.mean(torch.abs(node_pred_tensor - node_label_tensor)) * 1e3}\n")
        f.write(f"Mean Absolute Edge Error in mHartree: {torch.mean(torch.abs(edge_pred_tensor - edge_label_tensor)) * 1e3}\n")
        f.write(f"Mean Absolute Error in mHartree: {MAEloss_total * 1e3}\n")

    # downsample for plotting:
    downsample = 1
    all_node_labels = all_node_labels[::downsample]
    all_node_preds = all_node_preds[::downsample]
    all_edge_labels = all_edge_labels[::downsample]
    all_edge_preds = all_edge_preds[::downsample]

    # Plotting
    @env.only_rank_zero
    def plot_results():
        plt.figure(figsize=(4, 3))
        plt.scatter(all_edge_labels, all_edge_preds, s=1, alpha=0.5, edgecolor='none', color='crimson', label='Edge')
        plt.scatter(all_node_labels, all_node_preds, s=1, alpha=0.5, edgecolor='none', color='blue', label='Node')
        plt.plot(all_node_labels, all_node_labels, c='k', linestyle='dashed', linewidth=0.1, alpha=0.3)
        plt.xlabel(r"$(H_{ij})_{\alpha \beta}^{GT}$")
        plt.ylabel(r"$(H_{ij})_{\alpha \beta}^{pred}$")
        plt.legend()
        plt.savefig(save_file+'_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()
    plot_results()



# need to rewrite 
def assemble_hamiltonian_matrix(model, test_batch, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file='model_in_training.pth'):
    """
    Evaluate the model on the test set and return the mean absolute error for the node and edge predictions after reconstructing the Hamiltonian matrices from the predictions.

    """

    test_batch = test_batch.to(device)
    test_node, test_edge = model(test_batch)

    test_info = {}

    test_node = test_node.cpu()
    test_edge = test_edge.cpu()

    flattened_node_labels = construct_kernel.get_H(test_batch.node_y[0:test_batch.labelled_node_size].cpu()) #convert into flattened Hamiltonian form
    flattened_node_pred = construct_kernel.get_H(test_node[0:test_batch.labelled_node_size].cpu())

    flattened_edge_labels = construct_kernel.get_H(test_batch.y[0:test_batch.labelled_edge_size].cpu())
    flattened_edge_pred = construct_kernel.get_H(test_edge[0:test_batch.labelled_edge_size].cpu())

    onsite_edge_index = torch.cat((torch.arange(test_batch.labelled_node_size).unsqueeze(0),torch.arange(test_batch.labelled_node_size).unsqueeze(0)),0)
    numbers = test_batch.x[0:test_batch.labelled_node_size]

    label_orbital_dic = utils.assemble_hamiltonian(flattened_node_labels,numbers, onsite_edge_index,equivariant_blocks,atom_orbitals,out_slices)
    label_orbital_dic_offsite = utils.assemble_hamiltonian(flattened_edge_labels,numbers, test_batch.edge_index[:,0:test_batch.labelled_edge_size],equivariant_blocks,atom_orbitals,out_slices)
    label_orbital_dic.update(label_orbital_dic_offsite)

    torch.save(label_orbital_dic, save_file+'label_ham_dic.pt')

    pred_orbital_dic = utils.assemble_hamiltonian(flattened_node_pred,numbers, onsite_edge_index,equivariant_blocks,atom_orbitals,out_slices)
    pred_orbital_dic_offsite = utils.assemble_hamiltonian(flattened_edge_pred,numbers, test_batch.edge_index[:,0:test_batch.labelled_edge_size],equivariant_blocks,atom_orbitals,out_slices)
    pred_orbital_dic.update(pred_orbital_dic_offsite)

    torch.save(pred_orbital_dic, save_file+'pred_ham_dic.pt')


def reconstruct_hamiltonian_dic(model, test_batch, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file='model_in_training.pth'):
    """
    Evaluate the model on the test set and return the mean absolute error for the node and edge predictions after reconstructing the Hamiltonian matrices from the predictions.

    """
    test_batch = test_batch.to(device)
    test_node, test_edge = model(test_batch)

    test_node = test_node.cpu()
    test_edge = test_edge.cpu()

    flattened_node_labels = construct_kernel.get_H(test_batch.node_y[0:test_batch.labelled_node_size].cpu()) #convert into flattened Hamiltonian form
    flattened_node_pred = construct_kernel.get_H(test_node[0:test_batch.labelled_node_size].cpu())

    flattened_edge_labels = construct_kernel.get_H(test_batch.y[0:test_batch.labelled_edge_size].cpu())
    flattened_edge_pred = construct_kernel.get_H(test_edge[0:test_batch.labelled_edge_size].cpu())

    onsite_edge_index = torch.cat((torch.arange(test_batch.labelled_node_size).unsqueeze(0),torch.arange(test_batch.labelled_node_size).unsqueeze(0)),0)
    numbers = test_batch.x[0:test_batch.labelled_node_size]

    label_dic = utils.unflatten(flattened_node_labels,numbers, onsite_edge_index,equivariant_blocks,atom_orbitals,out_slices)
    label_offsite_dic = utils.unflatten(flattened_edge_labels,numbers, test_batch.edge_index[:,0:test_batch.labelled_edge_size],equivariant_blocks,atom_orbitals,out_slices)
    label_dic.update(label_offsite_dic)

    pred_dic = utils.unflatten(flattened_node_pred,numbers, onsite_edge_index,equivariant_blocks,atom_orbitals,out_slices)
    pred_offsite_dic = utils.unflatten(flattened_edge_pred,numbers, test_batch.edge_index[:,0:test_batch.labelled_edge_size],equivariant_blocks,atom_orbitals,out_slices)
    pred_dic.update(pred_offsite_dic)

    return label_dic,pred_dic