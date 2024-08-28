# Description: This file contains the functions to train the model. 

from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import lib.data as data
import lib.utils as utils
import time

def create_mask(data, train_ratio, option):

    train_mask = torch.zeros(len(data.edge_index[0]), dtype=torch.bool)
    test_mask = torch.zeros(len(data.edge_index[0]), dtype=torch.bool)

    if option == 'random':
        test_ratio = 1 - train_ratio
        train_nodes, test_nodes = train_test_split(range(len(data.edge_index[0])), test_size=test_ratio, random_state=42)

        train_mask[train_nodes] = 1
        test_mask[test_nodes] = 1
    
    # use all nodes for training and testing
    if option ==  'none':
        train_mask[:] = 1
        test_mask[:] = 1

    if option == 'onsite':
        for i in range(len(data.edge_index[0])):
            if data.edge_index[0][i] == data.edge_index[1][i]:
                train_mask[i] = 1
            else:
                test_mask[i] = 1

    return train_mask, test_mask

# update loss after each epoch --> use for multiple graphs
def train_model_SO2(model, loader, node_embedding_type, num_epochs=5000, loss_tol=0.0001, mask_type='none', save_file='model_in_training.pth', dtype=torch.float32):
    device = next(model.parameters()).device  # Get the device of the model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Define optimizer.
    criterion = nn.MSELoss()

    track_loss_node = []
    track_loss_edge = []

    for epoch in range(num_epochs):
        # total_loss = 0.0  # Initialize total loss for the epoch
    
        for id, batch in enumerate(loader):

            optimizer.zero_grad() 

            # Move input data to the same device as the model
            batch = batch.to(device)

            node_output, edge_output = model(batch)

            loss_node = criterion(node_output, batch.node_y) # node_y is the node label
            loss_edge = criterion(edge_output, batch.y)      # y is the edge label
            loss = loss_node+loss_edge
            loss.backward()                                  # Derive gradients.

            # Update parameters 
            optimizer.step()

        track_loss_node.append(loss_node.cpu().detach().numpy())
        track_loss_edge.append(loss_edge.cpu().detach().numpy())

        print("epoch: "+str(epoch)+" "+str(loss*1000))

        if epoch % 100 == 0:
            torch.save(model.state_dict(), save_file)

        if loss < loss_tol:
            break
    
    print("Final loss: ", loss)

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    plt.plot(track_loss_edge, label='edge')
    plt.xlabel('Epoch (x100)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    torch.save(model, save_file)


def train_model_HfO2(model, loader, num_epochs=5000, learning_rate = 1e-4, loss_tol=0.0001, mask_type='none', save_file='model_in_training.pth', dtype=torch.float32):
    device = next(model.parameters()).device  # Get the device of the model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Define optimizer.
    criterion = nn.MSELoss()

    track_loss_node = []
    track_loss_edge = []

    for epoch in range(num_epochs):
        # total_loss = 0.0  # Initialize total loss for the epoch
    
        for id, batch in enumerate(loader):

            start_time = time.time()
            optimizer.zero_grad() 

            # Move input data to the same device as the model
            batch = batch.to(device)

            node_output, edge_output = model(batch)

            loss_node = criterion(node_output[0:batch.labelled_node_size], batch.node_y[0:batch.labelled_node_size])
            loss_edge = criterion(edge_output[0:batch.labelled_edge_size], batch.y[0:batch.labelled_edge_size])

            loss = (loss_node*batch.labelled_node_size+loss_edge*batch.labelled_edge_size)/(batch.labelled_node_size+batch.labelled_edge_size)                       
            loss.backward()                                  # Derive gradients.
                        
            # Update parameters 
            optimizer.step()
            end_time = time.time()
            epoch_duration = end_time - start_time

            print("Number of Nodes: ", batch.node_y.shape[0])
            print("Number of Edges: ", batch.y.shape[0])
            print(f"Epoch {epoch+1} - Time: {epoch_duration:.4f} seconds")
            
            print("epoch: "+str(epoch)+" "+str(loss*1000))

        track_loss_node.append(loss_node.cpu().detach().numpy()) #tracks loss of last batch 
        track_loss_edge.append(loss_edge.cpu().detach().numpy())

        torch.save(track_loss_edge, 'track_loss_edge'+save_file+'.pt')
        torch.save(track_loss_node, 'track_loss_node'+save_file+'.pt')

        if epoch % 100 == 0:
            torch.save(model, save_file+'.pt')

        if loss < loss_tol:
            break
            
    
    print("Final loss: ", loss)

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    plt.plot(track_loss_edge, label='edge')
    plt.xlabel('Epoch (x100)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    torch.save(model, save_file)


def train_and_validate_model_HfO2(model, loader, test_batch, node_embedding_type, num_epochs=5000, learning_rate = 1e-4, loss_tol=0.0001, save_file='model_in_training.pth', construct_kernel=None, equivariant_blocks=None, atom_orbitals=None, out_slices=None, dtype=torch.float32):
    
    device = next(model.parameters()).device  # Get the device of the model

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Define optimizer.
    criterion = nn.MSELoss()

    track_loss_node = []
    track_loss_edge = []

    validation_node_loss = []
    validation_edge_loss = []

    for epoch in range(num_epochs):
        # total_loss = 0.0  # Initialize total loss for the epoch
    
        for id, batch in enumerate(loader):

            start_time = time.time()
            optimizer.zero_grad() 

            # Move input data to the same device as the model
            batch = batch.to(device)

            node_output, edge_output = model(batch)

            loss_node = criterion(node_output[0:batch.labelled_node_size], batch.node_y[0:batch.labelled_node_size])
            loss_edge = criterion(edge_output[0:batch.labelled_edge_size], batch.y[0:batch.labelled_edge_size])

            loss = (loss_node*batch.labelled_node_size+loss_edge*batch.labelled_edge_size)/(batch.labelled_node_size+batch.labelled_edge_size)                       
            loss.backward()                                  # Derive gradients.
                        
            # Update parameters 
            optimizer.step()
            end_time = time.time()
            epoch_duration = end_time - start_time

            print("Number of Nodes: ", batch.node_y.shape[0], "Number of Edges: ", batch.y.shape[0])
            # print("Number of Edges: ", batch.y.shape[0])
            # print(f"Epoch {epoch+1} - Time: {epoch_duration:.4f} seconds")
            
            print("epoch: "+str(epoch)+" "+str(loss*1000), f" Time: {epoch_duration:.4f} seconds")

        track_loss_node.append(loss_node.cpu().detach().numpy()) #tracks loss of last batch 
        track_loss_edge.append(loss_edge.cpu().detach().numpy())

        torch.save(track_loss_edge, 'track_loss_edge'+save_file+'.pt')
        torch.save(track_loss_node, 'track_loss_node'+save_file+'.pt')

        if epoch % 500 == 0:
            
            #convert model trained to GPU to model on cpu
    
            torch.save(model.state_dict(), save_file+'.pt')
            model_cpu = torch.load(save_file+'_cpu.pt') #find the previously saved model on cpu
            model_cpu.load_state_dict(torch.load(save_file+'.pt', map_location=torch.device('cpu')))

            torch.save(model_cpu, save_file+'.pt') #overwrite the previous state_dict file with the model on cpu

            MAE_node, MAE_edge = evaluate_model(model_cpu, test_batch, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device = 'cpu', save_file=save_file)
            validation_node_loss.append(MAE_node)
            validation_edge_loss.append(MAE_edge)

            torch.save(validation_node_loss, 'validation_node_loss_'+save_file+'.pt')
            torch.save(validation_edge_loss, 'validation_edge_loss_'+save_file+'.pt')

        if loss < loss_tol:
            break
            
    
    print("Final loss: ", loss)

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    plt.plot(track_loss_edge, label='edge')
    plt.xlabel('Epoch (x100)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    torch.save(model, save_file+'.pt')




def evaluate_model(model, test_batch, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file='model_in_training.pth'):
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















# update loss after each epoch --> use for multiple graphs
def train_model_minibatch(model, loader, node_embedding_type, num_epochs=5000, loss_tol=0.00001, mask_type='none', save_file='model_in_training.pth'):
    device = next(model.parameters()).device  # Get the device of the model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Define optimizer.
    criterion = nn.MSELoss()

    track_loss = []

    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize total loss for the epoch
        optimizer.zero_grad() 

        print("Epoch: ", epoch)

        for id, batch in enumerate(loader):

            # Move input data to the same device as the model
            batch = batch.to(device)

            out = model(atom_attr=batch.x,
                        node_embedding_type=node_embedding_type,
                        edge_idx=batch.edge_index,
                        edge_attr=batch.edge_feature,
                        batch=batch.batch)

            if epoch == 0:
                train_mask, test_mask = create_mask(batch,              # data batch
                                                    0.8,                # training ratio for this batch
                                                    mask_type)             # random or onsite or nones (how to pick training and testing subsets)
            
            # Compute the output only based on the training nodes
            training_output = out[train_mask]                      
            training_label = batch.y[train_mask]        

            # Reshape the output and label to match the training nodes
            training_output = training_output.reshape(training_label.shape[0],training_label.shape[1],training_label.shape[2])

            # Compute loss, get gradients
            loss = criterion(training_output, training_label)
            loss.backward() 

            # Accumulate the loss for the epoch
            total_loss += loss.item()

        # Update parameters after all batches
        optimizer.step()

        track_loss.append(total_loss)

        if epoch % 100 == 0:
            print("Epoch {}, Total Loss: {}".format(epoch, total_loss))

        if total_loss < loss_tol:
            break
    
    print("Final loss: ", total_loss)

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss)
    plt.xlabel('Epoch (x100)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    torch.save(model, save_file)


# update loss after each graph --> use for single graph
def train_model_singlebatch(model, loader, node_embedding_type, num_epochs=5000, loss_tol=0.00001, mask_type='none', save_file='model_in_training.pth'):
    device = next(model.parameters()).device  # Get the device of the model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Define optimizer.
    criterion = nn.MSELoss()

    track_loss = []

    for epoch in range(num_epochs):
        for id, batch in enumerate(loader):

            optimizer.zero_grad() 

            # Move input data to the same device as the model
            batch = batch.to(device)

            out = model(atom_attr=batch.x,
                        node_embedding_type=node_embedding_type,
                        edge_idx=batch.edge_index,
                        edge_attr=batch.edge_feature,
                        batch=batch.batch)

            if epoch == 0:
                train_mask, test_mask = create_mask(batch,              # data batch
                                                    0.8,                # training ratio for this batch
                                                    mask_type)          # random or onsite or nones (how to pick training and testing subsets)
            
            # Compute the output only based on the training nodes
            training_output = out[train_mask]                      
            training_label = batch.y[train_mask]        

            # Reshape the output and label to match the training nodes
            training_output = training_output.reshape(training_label.shape[0],training_label.shape[1],training_label.shape[2])

            # Compute loss, get gradients, update parameters
            loss = criterion(training_output, training_label)
            loss.backward() 
            optimizer.step()
        
            track_loss.append(loss.item())
            if epoch % 100 == 0:
                # print(loss*1000)
                print("Epoch {}, Total Loss: {}".format(epoch, loss*1000))

        if loss*1000 < loss_tol:
            break
    
    print("final loss: ", loss*1000)

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    torch.save(model, save_file)


def printloss(pred,label_batch, color):
    criterion = nn.MSELoss()
    MSEloss = criterion(pred, label_batch) #prints out the average MSE loss across all Hamiltonian entries 

    criterion2 = nn.L1Loss()
    MAEloss = criterion2(pred,label_batch)


    non_zero_indices = np.nonzero(label_batch[0])

    label = []
    prediction = []

    for i in range(len(non_zero_indices)):
        # label.append(label_batch[0][non_zero_indices[i][0],non_zero_indices[i][1]].numpy())
        # prediction.append(pred[0][non_zero_indices[i][0],non_zero_indices[i][1]].detach().numpy())
        label.append(label_batch[0][non_zero_indices[i][0], non_zero_indices[i][1]].cpu().numpy())
        prediction.append(pred[0][non_zero_indices[i][0], non_zero_indices[i][1]].cpu().detach().numpy())


    label = np.array(label)
    prediction = np.array(prediction)
    # print(label)
    # print(prediction)
    # print((label-prediction)/label)

    loss_average = [criterion(pred[i], label_batch[i]) for i in range(len(pred))] #compute MSEloss for each block 

    # pred_average = [np.mean(pred[i].detach().numpy()) for i in range(len(pred))]

    pred_average = [np.mean(pred[i].detach().cpu().numpy()) for i in range(len(pred))]
    y_average = [np.mean(label_batch[i].cpu().numpy()) for i in range(len(label_batch))]

    # plt.figure(figsize=(4,3))
    # plt.plot(y_average, y_average, c='k',linestyle='dashed', linewidth=1.0, alpha=0.3)
    # plt.scatter(y_average, pred_average, s=8, alpha=0.5)
    # plt.xlabel("real avg(orbital block)")
    # plt.ylabel("predicted avg(orbital block)")
    # plt.savefig('prediction.png', dpi=300, bbox_inches='tight')

    pred_values = np.concatenate([pred_batch.detach().cpu().numpy().flatten() for pred_batch in pred])
    label_values = np.concatenate([label_batch.detach().cpu().numpy().flatten() for label_batch in label_batch])

    plt.plot(label_values, label_values, c='k', linestyle='dashed', linewidth=1.0, alpha=0.05)
    plt.scatter(label_values, pred_values, c=color, s=7, alpha=0.3, edgecolor='none')
    plt.xlabel("Real $H_{ij}$")
    plt.ylabel("Predicted  $H_{ij}$")


def test_model(test_structures, model, loader, node_embedding_type, test_batch, mask_type='none'):

    for batch in loader:
        train_mask, test_mask = create_mask(batch,              # data batch
                                            0.8,                # training ratio for this batch
                                            mask_type)             # random or onsite or nones (how to pick training and testing subsets)
            
    # Move input data to the same device as the model
    device = next(model.parameters()).device 
    test_batch = test_batch.to(device)         

    pred = model(atom_attr=test_batch.x, 
                 node_embedding_type=node_embedding_type, 
                 edge_idx=test_batch.edge_index, 
                 edge_attr=test_batch.edge_feature, 
                 batch=test_batch.batch.to(device))

    # single graph, with transductive learning
    if len(loader) == 1:
        test_pred = pred[test_mask]                                                                         # make sure that there is only one batch in dataloader if a mask is used. 
        test_label = test_batch.y[test_mask]
        test_edges = (test_batch.edge_index.t()[test_mask]).t()
    else:
        test_pred = pred
        test_label = test_batch.y
        test_edges = (test_batch.edge_index.t()).t()

    pred = pred.reshape(test_batch.y.shape[0],test_batch.y.shape[1],test_batch.y.shape[2])                  #reshape the prediction for each Hamiltonian block into the same shape
    test_pred = test_pred.reshape(test_label.shape[0],test_label.shape[1],test_label.shape[2])              #reshape the testing subset for each Hamiltonian

    plt.figure(figsize=(4, 3))

    pred, test_batch.y = utils.rotate_data_back(pred, test_batch.y, test_batch.edge_index, test_batch.rotate_dic, test_structures)
    printloss(pred, test_batch.y, 'b')    

    test_pred, test_label = utils.rotate_data_back(test_pred, test_label, test_edges, test_batch.rotate_dic, test_structures)   
    printloss(test_pred, test_label, 'r')

    plt.savefig('prediction.png', dpi=300, bbox_inches='tight')


def test_model_SO2(test_structures, construct_kernel, model, loader, test_batch, mask_type='none', dtype=torch.float32):
     
    device = next(model.parameters()).device 
    test_batch = test_batch.to(device)    

    test_node, test_edge = model(test_batch)  

    
    # Edges
    test_labels = construct_kernel.get_H(test_batch.y.cpu())
    testing_edge = construct_kernel.get_H(test_edge.cpu())
    pred_values_edge = np.concatenate([batch_edge.detach().cpu().numpy().flatten() for batch_edge in testing_edge])
    label_values_edge = np.concatenate([batch_edge.detach().cpu().numpy().flatten() for batch_edge in test_labels])


    # Nodes
    test_labels = construct_kernel.get_H(test_batch.node_y.cpu())
    testing_node = construct_kernel.get_H(test_node.cpu())
    pred_values_node = np.concatenate([batch_edge.detach().cpu().numpy().flatten() for batch_edge in testing_node])
    label_values_node = np.concatenate([batch_edge.detach().cpu().numpy().flatten() for batch_edge in test_labels])


    plt.figure(figsize=(4, 3))
    plt.scatter(label_values_edge, pred_values_edge, s=1, alpha=0.5, color='crimson', label='Edge')
    plt.scatter(label_values_node, pred_values_node, s=1, alpha=0.5, color='blue', label='Node')

    plt.plot(label_values_node, label_values_node, c='k',linestyle='dashed')

    plt.xlabel("Real $H_{ij}$")
    plt.ylabel("Predicted  $H_{ij}$")
    plt.legend()
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)   

    plt.savefig('prediction.png', dpi=300, bbox_inches='tight')
