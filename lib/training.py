# Description: This file contains the functions to train the model. 

from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import lib.data as data
import lib.utils as utils
import time
import torch.optim as optim


# Training scheme to train the network on small molecules stored in batches of graphs
def train_and_validate_model_SO2(model, optimizer, training_loader, validation_loader, num_epochs=5000, loss_tol=0.0001, save_file='model_in_training.pth', dtype=torch.float64):
    device = next(model.parameters()).device  

    criterion = nn.MSELoss()                                        # Mean square error
    # criterion = nn.L1Loss()                                       # Mean average error

    track_loss_node = []
    track_loss_edge = []
    track_training_loss = [] #node + edge
    track_validation_loss = [] #node + edge

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        
        # every 100 epochs, reduce the learning rate by half
        if epoch % 100 == 0:
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 1e-8:
                    param_group['lr'] = param_group['lr']/1.5
    
        for batch in training_loader:

            start_time = time.time()
            optimizer.zero_grad() 
            zero_grad_time = time.time()

            batch = batch.to(device)
            memory_transfer_time = time.time()

            node_output, edge_output = model(batch) 
            forward_pass_time = time.time()

            node_output = node_output.to(device)
            edge_output = edge_output.to(device)

            print("Node Output: ", node_output)
            print("Edge Output: ", edge_output)
            print("Node output shape: ", node_output.shape)
            print("Edge output shape: ", edge_output.shape)
            print("Node Label: ", batch.node_y)
            print("Edge Label: ", batch.y)
            print("Node label shape: ", batch.node_y.shape)
            print("Edge label shape: ", batch.y.shape)

            loss_node = criterion(node_output, batch.node_y)                # node_y is the node label
            loss_edge = criterion(edge_output, batch.y)                     # y is the edge label
            
            combined_outputs = torch.cat([node_output, edge_output], dim=0)
            combined_labels = torch.cat([batch.node_y, batch.y], dim=0)
            loss = criterion(combined_outputs, combined_labels)

            loss_computation_time = time.time()
            
            loss.backward()                                  
            backward_pass_time = time.time()

            # Update parameters 
            optimizer.step()
            optimizer_update_time = time.time()

            end_time = time.time()

            zero_grad_duration = zero_grad_time - start_time
            memory_transfer_duration = memory_transfer_time - zero_grad_time
            forward_pass_duration = forward_pass_time - memory_transfer_time
            loss_computation_duration = loss_computation_time - forward_pass_time
            backward_pass_duration = backward_pass_time - loss_computation_time
            optimizer_update_duration = optimizer_update_time - backward_pass_time
            epoch_duration = end_time - start_time

        
        track_loss_node.append(loss_node.cpu().detach().numpy())
        track_loss_edge.append(loss_edge.cpu().detach().numpy())
        track_training_loss.append(loss_node.cpu().detach().numpy() + loss_edge.cpu().detach().numpy())
        print("epoch (training error): "+str(epoch)+" "+str(loss))

        # validate the model after each epoch
        validation_loss = 0.0
        for batch in validation_loader:
            batch = batch.to(device)

            node_output, edge_output = model(batch) 
            node_output = node_output.to(device)
            edge_output = edge_output.to(device)

            loss_node = criterion(node_output, batch.node_y)
            loss_edge = criterion(edge_output, batch.y)
            validation_loss += loss_node.cpu().detach().numpy() + loss_edge.cpu().detach().numpy()

        print("epoch (validation error): "+str(validation_loss))

        track_validation_loss.append(validation_loss/len(validation_loader))

        if epoch % 100 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_file+'.pt')

        # Training end condition
        if loss < loss_tol:
            break
    
    print("Final loss: ", loss)
    
    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    plt.plot(track_loss_edge, label='edge')
    plt.plot(track_training_loss, label='training error')
    plt.plot(track_validation_loss, label='validation error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_file+'.pt')


# Training scheme which takes a dataset of subgraphs, and only computes the loss on undirected edges
def train_model_subgraph(model, optimizer, loader, num_epochs=5000, loss_tol=0.0001, save_file='model_in_training.pth', schedule = False, dtype=torch.float32):
    device = next(model.parameters()).device  # Get the device of the model
    
    if dist.is_available() and dist.is_initialized():
        # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the directed ones
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()

    track_loss_node = []
    track_loss_edge = []

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Learning rate scheduler
        if schedule == True:
            if epoch % 1000 == 0:
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > 1e-8:
                        param_group['lr'] = param_group['lr']/1.5
    
        for batch in loader:

            batch_start_time = time.time()

            # Zero the gradients
            optimizer.zero_grad() 
            zero_grad_time = time.time()

            batch = batch.to(device)
            memory_transfer_time = time.time()

            # Forward pass
            node_output, edge_output = model(batch)
            forward_pass_time = time.time()
            # print("Node label shape: ", batch.node_y.shape)
            # print("Edge label shape: ", batch.y.shape)
            # print("Node output shape: ", node_output.shape)
            # print("Edge output shape: ", edge_output.shape)

            # Compute the loss
            loss_node = criterion(node_output[0:batch.labelled_node_size], batch.node_y[0:batch.labelled_node_size])            # node_y is the node label
            loss_edge = criterion(edge_output[0:batch.labelled_edge_size], batch.y[0:batch.labelled_edge_size])                 # y is the edge label
            output = torch.cat([node_output[0:batch.labelled_node_size], edge_output[0:batch.labelled_edge_size]], dim=0)
            labels = torch.cat([batch.node_y[0:batch.labelled_node_size], batch.y[0:batch.labelled_edge_size]], dim=0)
            loss = criterion(output, labels)      

            loss_computation_time = time.time()

            # Backward pass
            loss.backward()    
            backward_pass_time = time.time()                              
                        
            # Update parameters 
            optimizer.step()

            batch_end_time = time.time()

            forward_pass_duration = forward_pass_time - memory_transfer_time
            backward_pass_duration = backward_pass_time - loss_computation_time
            batch_duration = batch_end_time - batch_start_time

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
            
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:  
                print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds")
                print(f"--> Forward Pass Time: {forward_pass_duration:.4f} seconds")
                print(f"--> Backward Pass Time: {backward_pass_duration:.4f} seconds")
                print(f"--> Total Batch process time: {batch_duration:.4f} seconds")
                print("--> Memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")
                print("Epoch: " + str(epoch)+ " loss: " + str(loss))
        else:
            print("Epoch: " + str(epoch)+ " loss: " + str(loss))
            print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds")
            track_loss_node.append(loss_node.cpu().detach().numpy()) 
            track_loss_edge.append(loss_edge.cpu().detach().numpy())

        if epoch % 100 == 0:
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:  # Save only on rank 0
                    torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_file+'.pt')
                    torch.save(model.state_dict(), save_file+'_state_dic.pt')
            else:
                torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_file+'.pt')
                print("Model saved")
                torch.save(model.state_dict(), save_file+'_state_dic.pt')
                

        if loss < loss_tol:
            break
            
    print("Final loss: ", loss) 

    # save loss in plain txt file
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:  
            world_size = dist.get_world_size()
            with open('track_loss_'+str(world_size)+'_batches.txt', 'w') as f:
                for edge, node in zip(track_loss_edge, track_loss_node):
                    f.write(f"{edge:.8f}\t{node:.8f}\n")  

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    plt.plot(track_loss_edge, label='edge')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:  # Save only on rank 0
            torch.save({'model_state_dict': model.module.state_dict(), # Remove module 
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_file+'.pt')
            torch.save(model.state_dict(), save_file+'_state_dic.pt')
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_file+'.pt')
        
        torch.save(model.state_dict(), save_file+'_state_dic.pt')



def train_and_validate_model_subgraph(model, optimizer, loader, validation_loader, num_epochs=5000, loss_tol=0.0001, save_file='model_in_training.pth', schedule = False, dtype=torch.float32):
    device = next(model.parameters()).device  # Get the device of the model
    
    if dist.is_available() and dist.is_initialized():
        # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the directed ones
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()

    track_loss_node = []
    track_loss_edge = []

    track_validation_node = []
    track_validation_edge = []

    min_lr = 1e-5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)


    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()

        for batch in loader:

            batch_start_time = time.time()

            # Zero the gradients
            optimizer.zero_grad() 
            zero_grad_time = time.time()

            batch = batch.to(device)
            memory_transfer_time = time.time()

            # Forward pass
            node_output, edge_output = model(batch)
            forward_pass_time = time.time()
            # print("Node label shape: ", batch.node_y.shape)
            # print("Edge label shape: ", batch.y.shape)
            # print("Node output shape: ", node_output.shape)
            # print("Edge output shape: ", edge_output.shape)

            # Compute the loss
            loss_node = criterion(node_output[0:batch.labelled_node_size], batch.node_y[0:batch.labelled_node_size])            # node_y is the node label
            loss_edge = criterion(edge_output[0:batch.labelled_edge_size], batch.y[0:batch.labelled_edge_size])                 # y is the edge label
            output = torch.cat([node_output[0:batch.labelled_node_size], edge_output[0:batch.labelled_edge_size]], dim=0)
            labels = torch.cat([batch.node_y[0:batch.labelled_node_size], batch.y[0:batch.labelled_edge_size]], dim=0)
            loss = criterion(output, labels)      

            loss_computation_time = time.time()

            # Backward pass
            loss.backward()    
            backward_pass_time = time.time()                              
                        
            # Update parameters 
            optimizer.step()

            batch_end_time = time.time()

            forward_pass_duration = forward_pass_time - memory_transfer_time
            backward_pass_duration = backward_pass_time - loss_computation_time
            batch_duration = batch_end_time - batch_start_time

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
            
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:  
                print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds")
                print(f"--> Forward Pass Time: {forward_pass_duration:.4f} seconds")
                print(f"--> Backward Pass Time: {backward_pass_duration:.4f} seconds")
                print(f"--> Total Batch process time: {batch_duration:.4f} seconds")
                print("--> Memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")
                print("Epoch: " + str(epoch)+ " loss: " + str(loss))
                track_loss_node.append(loss_node.cpu().detach().numpy()) 
                track_loss_edge.append(loss_edge.cpu().detach().numpy())
        else:
            print("Epoch: " + str(epoch)+ " loss: " + str(loss))
            print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds")
            track_loss_node.append(loss_node.cpu().detach().numpy()) 
            track_loss_edge.append(loss_edge.cpu().detach().numpy())

        if epoch % 100 == 0:
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:  # Save only on rank 0
                    torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_file+'.pt')
                    torch.save(model.state_dict(), save_file+'_state_dic.pt')
            else:
                torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_file+'.pt')
                print("Model saved")
                torch.save(model.state_dict(), save_file+'_state_dic.pt')
                

        #validate the model
        validation_loss = 0.0
        model.eval()

        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                node_output, edge_output = model(batch)
                loss_node = criterion(node_output[0:batch.labelled_node_size], batch.node_y[0:batch.labelled_node_size])
                loss_edge = criterion(edge_output[0:batch.labelled_edge_size], batch.y[0:batch.labelled_edge_size])
              # y is the edge label
                output = torch.cat([node_output[0:batch.labelled_node_size], edge_output[0:batch.labelled_edge_size]], dim=0)
                labels = torch.cat([batch.node_y[0:batch.labelled_node_size], batch.y[0:batch.labelled_edge_size]], dim=0)
                validation_loss += criterion(output, labels)  
        
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:  
                print("Validation loss: ", validation_loss)
                print("Validation node loss: ", loss_node)
                print("Validation edge loss: ", loss_edge)
                track_validation_node.append(loss_node.cpu().detach().numpy())
                track_validation_edge.append(loss_edge.cpu().detach().numpy())  

        else: 
            print("epoch (validation error): "+str(validation_loss))
            print("epoch (validation node error): "+str(loss_node))
            track_validation_node.append(loss_node.cpu().detach().numpy())
            track_validation_edge.append(loss_edge.cpu().detach().numpy())  


        if schedule == True: 
            scheduler.step(validation_loss)


        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.8f}")
        if current_lr <= min_lr:
            print("Learning rate has reached the minimum threshold. Stopping training.")
            break

        if loss < loss_tol:
            break
            
    print("Final loss: ", loss) 

    # save loss in plain txt file
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:  
            world_size = dist.get_world_size()
            with open('track_loss_'+str(world_size)+'_batches.txt', 'w') as f:
                for edge, node in zip(track_loss_edge, track_loss_node):
                    f.write(f"{edge:.8f}\t{node:.8f}\n")  

            with open('track_validation_loss_'+str(world_size)+'_batches.txt', 'w') as f:
                for edge, node in zip(track_validation_edge, track_validation_node):
                    f.write(f"{edge:.8f}\t{node:.8f}\n")  

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    plt.plot(track_loss_edge, label='edge')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:  # Save only on rank 0
            torch.save({'model_state_dict': model.module.state_dict(), # Remove module 
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_file+'.pt')
            torch.save(model.state_dict(), save_file+'_state_dic.pt')
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_file+'.pt')
        
        torch.save(model.state_dict(), save_file+'_state_dic.pt')





















# DGL version - Training scheme which takes a batch of subgraphs and computes the loss on all edges
def train_model_DGL_full(model, optimizer, loader, total_num_nodes, num_epochs=5000, loss_tol=0.0001, save_file='model_in_training.pth', dtype=torch.float32):
    # device = next(model.parameters()).device  # Get the device of the model
    
    ### WARNING: EXPECTS TO SEE ONLY 1 GPU PER NODE, FIGURE OUT HOW TO SET DEVICE CORRECTLY LATER
    device = torch.device("cuda:0")

    # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the directed ones
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)#, output_device=device,

    criterion = nn.MSELoss()

    track_loss = []
    track_loss_node = []
    track_loss_edge = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print("Epoch: ", epoch)

        # every 100 epochs, reduce the learning rate by half
        if epoch % 200 == 0 and dist.get_rank() == 0:
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 1e-8:
                    param_group['lr'] = param_group['lr']/2
    
        epoch_loss = 0.0
        MAE_loss = 0.0

        # model.join() is probably a context manager 
        with model.join():
            for batch_id, (input_nodes, output_nodes, subgraphs) in enumerate(loader):
                optimizer.zero_grad()

                print("Batch ID: ", batch_id)
                print("Input Nodes: ", input_nodes)
                print("Output Nodes: ", output_nodes)
                print("Number of Subgraphs: ", len(subgraphs))
                print("**************************************")

                # Upload subgraphs to GPU
                subgraphs = [sg.to(device) for sg in subgraphs]

                # Forward pass
                node_outputs, edge_outputs = model(subgraphs, total_num_nodes)
                print("--> Memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")

                # Concatenate node and edge outputs if they are lists - do we need this?
                if isinstance(node_outputs, list):
                    node_outputs = torch.cat(node_outputs, dim=0)
                if isinstance(edge_outputs, list):
                    edge_outputs = torch.cat(edge_outputs, dim=0)
                
                # Concatenate the node and edge labels from all subgraphs
                node_labels = torch.cat([sg.ndata['_N/node_label']['_N'].to(device) for sg in subgraphs], dim=0)
                edge_labels = torch.cat([sg.edata['_E/label'].to(device) for sg in subgraphs], dim=0) 

                print("rank ", dist.get_rank(), "node_labels: ", node_labels)
                print("rank ", dist.get_rank(), "edge_labels: ", edge_labels)

                # Compute the loss
                loss_node = criterion(node_outputs, node_labels)
                loss_edge = criterion(edge_outputs, edge_labels)
                combined_outputs = torch.cat([node_outputs, edge_outputs], dim=0)
                combined_labels = torch.cat([node_labels, edge_labels], dim=0)
                loss = criterion(combined_outputs, combined_labels)

                loss.backward()

                # also calculate the L1 loss and print it
                criterion_L1 = nn.L1Loss()
                MAE_loss += criterion_L1(combined_outputs, combined_labels)

                # Update parameters
                optimizer.step()

                epoch_loss += loss.item()

                # testing garbage collection (add this back in before running on large structures)
                # del subgraphs, node_outputs, edge_outputs, node_labels, edge_labels, combined_outputs, combined_labels
                # torch.cuda.empty_cache()  # free GPU memory
            
        # Synchronize loss across all processes
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(torch.tensor(epoch_loss, device=device), op=dist.ReduceOp.SUM)
            dist.all_reduce(torch.tensor(MAE_loss, device=device), op=dist.ReduceOp.SUM)
            epoch_loss /= dist.get_world_size()
            MAE_loss /= dist.get_world_size()

        # Print loss only from rank 0
        if dist.get_rank() == 0:
            print(f"--> Epoch: {epoch} | MSE Loss: {epoch_loss} | MAE Loss: {MAE_loss}")
            epoch_duration = time.time() - epoch_start_time
            print(f"--> Epoch {epoch} - Time: {epoch_duration:.4f} seconds")


        # if dist.is_available() and dist.is_initialized():
        #     if dist.get_rank() == 0: 
        #         print(f"Epoch: {epoch} | MSE Loss: {loss.item()}")
        #         # print(f"---> Mean Absolute Error (mH): {MAE_loss.item()*1e3/len(loader)}")
        #         epoch_end_time = time.time()
        #         epoch_duration = epoch_end_time - epoch_start_time
        #         print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds")

        track_loss_node.append(loss_node.cpu().detach().numpy()) 
        track_loss_edge.append(loss_edge.cpu().detach().numpy())
        track_loss.append(loss.cpu().detach().numpy())

        if epoch % 100 == 0:
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:  # Save only on rank 0
                    torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_file+'.pt')
            else:
                torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_file+'.pt')

        if loss < loss_tol:
            break
            
    print("Final loss: ", loss) 

    # save loss in plain txt file
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:  
            world_size = dist.get_world_size()
            with open('track_loss_'+str(world_size)+'_batches.txt', 'w') as f:
                for edge, node in zip(track_loss_edge, track_loss_node):
                    f.write(f"{edge:.8f}\t{node:.8f}\n")  

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node loss')
    plt.plot(track_loss_edge, label='edge loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:  # Save only on rank 0
            torch.save({'model_state_dict': model.module.state_dict(), # Remove module 
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_file+'.pt')
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_file+'.pt')


def evaluate_model(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device):
    model.eval()  # Set the model to evaluation mode
    all_node_labels = []
    all_node_preds = []
    all_edge_labels = []
    all_edge_preds = []

    if dist.is_available() and dist.is_initialized():
        # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the directed ones
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)

    with torch.no_grad():  # Disable gradient computation
        for test_batch in data_loader:
            test_batch = test_batch.to(device)

            # Forward pass
            test_node, test_edge = model(test_batch)
            test_node = test_node.cpu()
            test_edge = test_edge.cpu()

            # Convert size values to Python numbers
            labelled_node_size = test_batch.labelled_node_size.item()
            labelled_edge_size = test_batch.labelled_edge_size.item()

            # Process node predictions
            flattened_node_labels = construct_kernel.get_H(test_batch.node_y[:labelled_node_size].cpu())
            flattened_node_pred = construct_kernel.get_H(test_node[:labelled_node_size].cpu())

            node_label = utils.unflatten(flattened_node_labels, test_batch.x[:labelled_node_size],
                                         torch.cat((torch.arange(labelled_node_size).unsqueeze(0),
                                                    torch.arange(labelled_node_size).unsqueeze(0)), 0),
                                         equivariant_blocks, atom_orbitals, out_slices)
            
            node_pred = utils.unflatten(flattened_node_pred, test_batch.x[:labelled_node_size],
                                        torch.cat((torch.arange(labelled_node_size).unsqueeze(0),
                                                   torch.arange(labelled_node_size).unsqueeze(0)), 0),
                                        equivariant_blocks, atom_orbitals, out_slices)
            
            H_block_node_labels = [matrix.flatten() for matrix in node_label.values()]
            node_label_tensor = torch.cat(H_block_node_labels)
            H_block_node_pred = [matrix.flatten() for matrix in node_pred.values()]
            node_pred_tensor = torch.cat(H_block_node_pred)

            # Process edge predictions
            flattened_edge_labels = construct_kernel.get_H(test_batch.y[0:labelled_edge_size].cpu())
            flattened_edge_pred = construct_kernel.get_H(test_edge[0:labelled_edge_size].cpu())

            edge_label = utils.unflatten(flattened_edge_labels, test_batch.x[0:labelled_node_size],
                                         test_batch.edge_index[:, 0:labelled_edge_size],
                                         equivariant_blocks, atom_orbitals, out_slices)
            
            edge_pred = utils.unflatten(flattened_edge_pred, test_batch.x[0:labelled_node_size],
                                        test_batch.edge_index[:, 0:labelled_edge_size],
                                        equivariant_blocks, atom_orbitals, out_slices)
            
            H_block_edge_labels = [matrix.flatten() for matrix in edge_label.values()]
            edge_label_tensor = torch.cat(H_block_edge_labels)
            H_block_edge_pred = [matrix.flatten() for matrix in edge_pred.values()]
            edge_pred_tensor = torch.cat(H_block_edge_pred)

            # Compute the MAE
            pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
            label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
            MAEloss_total = torch.mean(torch.abs(pred_tensor - label_tensor)) * 1e3

            print("Mean Absolute Node Error in mHartree: ", torch.mean(torch.abs(node_pred_tensor - node_label_tensor)) * 1e3)
            print("Mean Absolute Edge Error in mHartree: ", torch.mean(torch.abs(edge_pred_tensor - edge_label_tensor)) * 1e3)
            print("Mean Absolute Error in mHartree: ", MAEloss_total)


            # Collect results for plotting
            all_node_labels.append(node_label_tensor)
            all_node_preds.append(node_pred_tensor)
            all_edge_labels.append(edge_label_tensor)
            all_edge_preds.append(edge_pred_tensor)

            # Clear cache after processing each batch
            del test_node, test_edge
            torch.cuda.empty_cache()

    # Concatenate all results
    all_node_labels = torch.cat(all_node_labels)
    all_node_preds = torch.cat(all_node_preds)
    all_edge_labels = torch.cat(all_edge_labels)
    all_edge_preds = torch.cat(all_edge_preds)

    # Plotting
    plt.figure(figsize=(4, 3))
    plt.scatter(all_edge_labels.cpu().numpy(), all_edge_preds.cpu().numpy(), s=1, alpha=0.5, edgecolor='none', color='crimson', label='Edge')
    plt.scatter(all_node_labels.cpu().numpy(), all_node_preds.cpu().numpy(), s=1, alpha=0.5, edgecolor='none', color='blue', label='Node')
    plt.plot(all_node_labels.cpu().numpy(), all_node_labels.cpu().numpy(), c='k', linestyle='dashed', linewidth=0.1, alpha=0.3)
    plt.xlabel("Real $H_{ij}$")
    plt.ylabel("Predicted  $H_{ij}$")
    plt.legend()
    plt.savefig('prediction.png', dpi=300, bbox_inches='tight')
    plt.close()



def analyze_model(model, test_batch, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file='model_in_training.pth'):
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

    onsite_edge_index = torch.cat((torch.arange(test_batch.labelled_node_size).unsqueeze(0),torch.arange(test_batch.labelled_node_size).unsqueeze(0)),0)
    numbers = test_batch.x[0:test_batch.labelled_node_size]

    node_label = utils.unflatten(flattened_node_labels,numbers, onsite_edge_index,equivariant_blocks,atom_orbitals,out_slices)
    node_pred = utils.unflatten(flattened_node_pred,numbers, onsite_edge_index,equivariant_blocks,atom_orbitals,out_slices)

    H_block_node_labels = [matrix.flatten() for matrix in node_label.values()]
    node_label_tensor = torch.cat(H_block_node_labels)

    H_block_node_pred = [matrix.flatten() for matrix in node_pred.values()]
    node_pred_tensor = torch.cat(H_block_node_pred)


    test_info['flattened_node_labels'] = flattened_node_labels
    test_info['flattened_node_pred'] = flattened_node_pred
    test_info['node_label'] = node_label_tensor
    test_info['node_pred'] = node_pred_tensor


    flattened_edge_labels = construct_kernel.get_H(test_batch.y[0:test_batch.labelled_edge_size].cpu())
    flattened_edge_pred = construct_kernel.get_H(test_edge[0:test_batch.labelled_edge_size].cpu())

    edge_label = utils.unflatten(flattened_edge_labels,numbers, test_batch.edge_index[:,0:test_batch.labelled_edge_size],equivariant_blocks,atom_orbitals,out_slices)
    edge_pred = utils.unflatten(flattened_edge_pred,numbers, test_batch.edge_index[:,0:test_batch.labelled_edge_size],equivariant_blocks,atom_orbitals,out_slices)

    H_block_edge_labels = [matrix.flatten() for matrix in edge_label.values()]
    edge_label_tensor = torch.cat(H_block_edge_labels)

    H_block_edge_pred = [matrix.flatten() for matrix in edge_pred.values()]
    edge_pred_tensor = torch.cat(H_block_edge_pred)

    test_info['flattened_edge_labels'] = flattened_edge_labels
    test_info['flattened_edge_pred'] = flattened_edge_pred
    test_info['edge_label'] = edge_label_tensor
    test_info['edge_pred'] = edge_pred_tensor

    torch.save(test_info, save_file+'_test_info.pt')


    MAE_node = torch.mean(torch.abs(node_label_tensor - node_pred_tensor))
    MAE_edge = torch.mean(torch.abs(edge_label_tensor - edge_pred_tensor))

    return MAE_node, MAE_edge






def evaluate_model_DGL(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device):
    model.eval()  # Set the model to evaluation mode
    all_node_labels = []
    all_node_preds = []
    all_edge_labels = []
    all_edge_preds = []

    with torch.no_grad():  # Disable gradient computation
        for batch_id, (input_nodes, output_nodes, subgraphs) in enumerate(data_loader):
            # subgraph = subgraph.to(device)

            subgraphs = [sg.to(device) for sg in subgraphs]

            # Forward pass
            node_outputs, edge_outputs = model(subgraphs)

            for i, (node_output, edge_output, subgraph) in enumerate(zip(node_outputs, edge_outputs, subgraphs)):
            
                node_output = node_output.cpu()
                edge_output = edge_output.cpu()

                # Concatenate the node and edge labels from all subgraphs
                node_labels = subgraph.ndata['_N/node_label']['_N'].cpu()
                edge_labels = subgraph.edata['_E/label'].cpu() 
                num_nodes = subgraph.ndata['_N/feat']['_N'].shape[0]
                print("Number of Nodes: ", num_nodes)

                flattened_node_labels = construct_kernel.get_H(node_labels)
                flattened_node_pred = construct_kernel.get_H(node_output)
                
                atomic_numbers = subgraph.ndata['_N/feat']['_N']
                node_self_indices = torch.cat((torch.arange(num_nodes).unsqueeze(0),
                                               torch.arange(num_nodes).unsqueeze(0)),0)
                atomic_numbers = atomic_numbers.cpu().numpy()
                node_self_indices = node_self_indices.cpu().numpy()
                
                print("Atomic Numbers: ", atomic_numbers)
                print("Node Self Indices: ", node_self_indices)
                print("Type of Node Self Indices: ", type(node_self_indices))

                unflattened_node_labels = utils.unflatten(flattened_node_labels, atomic_numbers, node_self_indices,
                                                          equivariant_blocks, atom_orbitals, out_slices)
                
                unflattened_node_pred = utils.unflatten(flattened_node_pred, atomic_numbers, node_self_indices,
                                                        equivariant_blocks, atom_orbitals, out_slices)

                H_block_node_labels = [matrix.flatten() for matrix in unflattened_node_labels.values()]
                node_label_tensor = torch.cat(H_block_node_labels)
                H_block_node_pred = [matrix.flatten() for matrix in unflattened_node_pred.values()]
                node_pred_tensor = torch.cat(H_block_node_pred)

                # Process edge predictions
                flattened_edge_labels = construct_kernel.get_H(edge_labels)
                flattened_edge_pred = construct_kernel.get_H(edge_output)
                edge_index = torch.stack(subgraph.edges(), dim=0)

                unflattened_edge_labels = utils.unflatten(flattened_edge_labels, atomic_numbers, edge_index,
                                                        equivariant_blocks, atom_orbitals, out_slices)
                
                unflattened_edge_pred = utils.unflatten(flattened_edge_pred, atomic_numbers, edge_index,
                                                        equivariant_blocks, atom_orbitals, out_slices)
                
                H_block_edge_labels = [matrix.flatten() for matrix in unflattened_edge_labels.values()]
                edge_label_tensor = torch.cat(H_block_edge_labels)
                H_block_edge_pred = [matrix.flatten() for matrix in unflattened_edge_pred.values()]
                edge_pred_tensor = torch.cat(H_block_edge_pred)

                # Compute the MAE
                pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
                label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
                MAEloss_total = torch.mean(torch.abs(pred_tensor - label_tensor)) * 1e3
                # pred_tensor = node_pred_tensor
                # label_tensor = node_label_tensor
                # MAEloss_total = torch.mean(torch.abs(pred_tensor - label_tensor)) * 1e3

                print("Mean Absolute Error in mHartree: ", MAEloss_total)

                # Collect results for plotting
                all_node_labels.append(node_label_tensor)
                all_node_preds.append(node_pred_tensor)
                all_edge_labels.append(edge_label_tensor)
                all_edge_preds.append(edge_pred_tensor)

                # Clear cache after processing each batch
                del subgraphs, node_output, edge_output
                torch.cuda.empty_cache()

            #     print("Testing only one batch - break")
            #     break
            # break

    # Concatenate all results
    all_node_labels = torch.cat(all_node_labels)
    all_node_preds = torch.cat(all_node_preds)
    all_edge_labels = torch.cat(all_edge_labels)
    all_edge_preds = torch.cat(all_edge_preds)

    # downsample: take every 100th element
    # downsample = 100    
    # all_node_labels = all_node_labels[::downsample]
    # all_node_preds = all_node_preds[::downsample]
    # all_edge_labels = all_edge_labels[::downsample]
    # all_edge_preds = all_edge_preds[::downsample]

    # Plotting
    plt.figure(figsize=(4, 3))
    plt.scatter(all_edge_labels.cpu().numpy(), all_edge_preds.cpu().numpy(), s=3, alpha=0.1, edgecolor='none', color='crimson', label='Edge (downsampled)')
    plt.scatter(all_node_labels.cpu().numpy(), all_node_preds.cpu().numpy(), s=3, alpha=0.1, edgecolor='none', color='blue', label='Node (downsampled)')
    plt.plot(all_node_labels.cpu().numpy(), all_node_labels.cpu().numpy(), c='k', linestyle='dashed', linewidth=0.1, alpha=0.3)
    plt.xlabel("Real $H_{ij}$")
    plt.ylabel("Predicted  $H_{ij}$")
    plt.legend()
    plt.savefig('prediction.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_model_SO2(construct_kernel, model, test_batch, mask_type='none', dtype=torch.float32):
     
    device = next(model.parameters()).device 
    test_batch = test_batch.to(device)    

    test_node, test_edge = model(test_batch)

    # Edges
    test_labels = construct_kernel.get_H(test_batch.y)
    testing_edge = construct_kernel.get_H(test_edge)
    pred_values_edge = np.concatenate([batch_edge.detach().cpu().numpy().flatten() for batch_edge in testing_edge])
    label_values_edge = np.concatenate([batch_edge.detach().cpu().numpy().flatten() for batch_edge in test_labels])

    # Nodes
    test_labels = construct_kernel.get_H(test_batch.node_y)
    testing_node = construct_kernel.get_H(test_node)
    pred_values_node = np.concatenate([batch_edge.detach().cpu().numpy().flatten() for batch_edge in testing_node])
    label_values_node = np.concatenate([batch_edge.detach().cpu().numpy().flatten() for batch_edge in test_labels])

    plt.figure(figsize=(4, 3))
    plt.scatter(label_values_edge, pred_values_edge, s=3, alpha=0.05, edgecolor='none', color='crimson', label='Edge')
    plt.scatter(label_values_node, pred_values_node, s=3, alpha=0.05, edgecolor='none', color='blue', label='Node')
    plt.plot(label_values_node, label_values_node, c='k',linestyle='dashed', linewidth=0.1, alpha=0.3)

    plt.xlabel("Real $H_{ij}$")
    plt.ylabel("Predicted  $H_{ij}$")
    plt.legend()
 
    # compute mean average error bewteen pred_values_edge and label_values_edge
    pred_values = np.concatenate([pred_values_edge, pred_values_node])
    label_values = np.concatenate([label_values_edge, label_values_node])
    MAEloss_total = np.mean(np.abs(pred_values - label_values))*1e3

    print("Mean Absolute Error in mHartree: ", MAEloss_total)

    plt.savefig('prediction.png', dpi=300, bbox_inches='tight')




# # Training scheme which takes a dataset of subgraphs, and only computes the loss on undirected edges
# def train_model_DGL_sampled(model, optimizer, loader, num_epochs=5000, loss_tol=0.0001, save_file='model_in_training.pth', dtype=torch.float32):
#     device = next(model.parameters()).device  # Get the device of the model
    
#     if dist.is_available() and dist.is_initialized():
#         # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the directed ones
#         model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
#     else:
#         model = nn.DataParallel(model)

#     criterion = nn.MSELoss()

#     track_loss_node = []
#     track_loss_edge = []

#     for epoch in range(num_epochs):
#         epoch_start_time = time.time()

#         # every 100 epochs, reduce the learning rate by half
#         if epoch % 500 == 0:
#             for param_group in optimizer.param_groups:
#                 if param_group['lr'] > 1e-8:
#                     param_group['lr'] = param_group['lr']/1.5
    
#         MAE_loss = 0.0
#         for batch_id, subgraph in enumerate(loader):

#             optimizer.zero_grad()

#             subgraph = subgraph.to(device)

#             node_output, edge_output = model(subgraph)

#             # Assuming 'feat' is the node feature and 'label' is the edge label stored in the subgraph
#             loss_node = criterion(node_output, subgraph.ndata['node_label'].to(device))
#             loss_edge = criterion(edge_output, subgraph.edata['label'].to(device))
#             loss = loss_node + loss_edge

#             loss.backward()

#             # also calculate the L1 loss and print it
#             criterion_L1 = nn.L1Loss()
#             total_output = torch.cat([node_output, edge_output])
#             total_label = torch.cat([subgraph.ndata['node_label'], subgraph.edata['label']])
#             MAE_loss += criterion_L1(total_output, total_label)
            

#             # Update parameters
#             optimizer.step()

#         if dist.is_available() and dist.is_initialized():
#             if dist.get_rank() == 0: 
#                 print(f"Epoch: {epoch} | MSE Loss: {loss.item()}")
#                 print(f"---> Mean Absolute Error (mH): {MAE_loss.item()*1e3/len(loader)}")
#                 epoch_end_time = time.time()
#                 epoch_duration = epoch_end_time - epoch_start_time
#                 print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds")

#         track_loss_node.append(loss_node.cpu().detach().numpy()) 
#         track_loss_edge.append(loss_edge.cpu().detach().numpy())

#         if epoch % 100 == 0:
#             if dist.is_available() and dist.is_initialized():
#                 if dist.get_rank() == 0:  # Save only on rank 0
#                     torch.save({'model_state_dict': model.module.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         }, save_file)
#             else:
#                 torch.save({'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     }, save_file)
                

#         if loss < loss_tol:
#             break
            
#     print("Final loss: ", loss) 

#     # save loss in plain txt file
#     if dist.is_available() and dist.is_initialized():
#         if dist.get_rank() == 0:  
#             world_size = dist.get_world_size()
#             with open('track_loss_'+str(world_size)+'_batches.txt', 'w') as f:
#                 for edge, node in zip(track_loss_edge, track_loss_node):
#                     f.write(f"{edge:.8f}\t{node:.8f}\n")  

#     plt.figure(figsize=(4, 3))
#     plt.plot(track_loss_node, label='node')
#     plt.plot(track_loss_edge, label='edge')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.yscale('log')
#     plt.legend()
#     plt.savefig('loss.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     if dist.is_available() and dist.is_initialized():
#         if dist.get_rank() == 0:  # Save only on rank 0
#             torch.save({'model_state_dict': model.module.state_dict(), # Remove module 
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         }, save_file)
#     else:
#         torch.save({'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     }, save_file)


# def train_and_validate_model_HfO2(model, loader, test_batch, node_embedding_type, num_epochs=5000, learning_rate = 1e-4, loss_tol=0.0001, save_file='model_in_training.pth', construct_kernel=None, equivariant_blocks=None, atom_orbitals=None, out_slices=None, dtype=torch.float32):
    
#     device = next(model.parameters()).device  # Get the device of the model

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Define optimizer.
#     criterion = nn.MSELoss()

#     track_loss_node = []
#     track_loss_edge = []

#     validation_node_loss = []
#     validation_edge_loss = []

#     for epoch in range(num_epochs):
#         # total_loss = 0.0  # Initialize total loss for the epoch
    
#         for id, batch in enumerate(loader):

#             start_time = time.time()
#             optimizer.zero_grad() 

#             # Move input data to the same device as the model
#             batch = batch.to(device)

#             node_output, edge_output = model(batch)

#             loss_node = criterion(node_output[0:batch.labelled_node_size], batch.node_y[0:batch.labelled_node_size])
#             loss_edge = criterion(edge_output[0:batch.labelled_edge_size], batch.y[0:batch.labelled_edge_size])

#             loss = (loss_node*batch.labelled_node_size+loss_edge*batch.labelled_edge_size)/(batch.labelled_node_size+batch.labelled_edge_size)                       
#             loss.backward()                                  # Derive gradients.
                        
#             # Update parameters 
#             optimizer.step()
#             end_time = time.time()
#             epoch_duration = end_time - start_time

#             print("Number of Nodes: ", batch.node_y.shape[0], "Number of Edges: ", batch.y.shape[0])
#             # print("Number of Edges: ", batch.y.shape[0])
#             # print(f"Epoch {epoch+1} - Time: {epoch_duration:.4f} seconds")
            
#             print("epoch: "+str(epoch)+" "+str(loss*1000), f" Time: {epoch_duration:.4f} seconds")

#         track_loss_node.append(loss_node.cpu().detach().numpy()) #tracks loss of last batch 
#         track_loss_edge.append(loss_edge.cpu().detach().numpy())

#         torch.save(track_loss_edge, 'track_loss_edge'+save_file+'.pt')
#         torch.save(track_loss_node, 'track_loss_node'+save_file+'.pt')

#         if epoch % 500 == 0:
            
#             #convert model trained to GPU to model on cpu
    
#             torch.save(model.state_dict(), save_file+'.pt')
#             model_cpu = torch.load(save_file+'_cpu.pt') #find the previously saved model on cpu
#             model_cpu.load_state_dict(torch.load(save_file+'.pt', map_location=torch.device('cpu')))

#             torch.save(model_cpu, save_file+'.pt') #overwrite the previous state_dict file with the model on cpu

#             MAE_node, MAE_edge = evaluate_model(model_cpu, test_batch, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device = 'cpu', save_file=save_file)
#             validation_node_loss.append(MAE_node)
#             validation_edge_loss.append(MAE_edge)

#             torch.save(validation_node_loss, 'validation_node_loss_'+save_file+'.pt')
#             torch.save(validation_edge_loss, 'validation_edge_loss_'+save_file+'.pt')

#         if loss < loss_tol:
#             break
            
    
#     print("Final loss: ", loss)

#     plt.figure(figsize=(4, 3))
#     plt.plot(track_loss_node, label='node')
#     plt.plot(track_loss_edge, label='edge')
#     plt.xlabel('Epoch (x100)')
#     plt.ylabel('Loss')
#     plt.yscale('log')
#     plt.legend()
#     plt.savefig('loss.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     torch.save(model, save_file+'.pt')
