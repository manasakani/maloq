import torch
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
import time

def profile_code(profile_flag, output_file="trace.json"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if profile_flag:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]) as prof:
                    result = func(*args, **kwargs)
                prof.export_chrome_trace(output_file)
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

# Loss function options
# ----------------------
def mse_padded_loss(output, target):
    mse_loss_type = nn.MSELoss(reduction='mean') 
    return mse_loss_type(output, target)

def l1_padded_loss(output, target):
    l1_loss_type = nn.L1Loss(reduction='mean') 
    return l1_loss_type(output, target)

def combined_padded_loss(output, target):

    mse_loss = nn.MSELoss(reduction='mean') 
    l1_loss = nn.L1Loss(reduction='mean')
    mse = mse_loss(output, target)
    l1 = l1_loss(output, target)

    return mse + l1

def mse_unpadded_loss(output, target):

    mask = (target != 0).float()

    mse_loss = nn.MSELoss(reduction='none') 
    mse = mse_loss(output, target)
    mse_masked = (mse * mask).sum() / mask.sum()

    return mse_masked

def l1_unpadded_loss(output, target):

    mask = (target != 0).float()

    l1_loss = nn.L1Loss(reduction='none')
    l1 = l1_loss(output, target)
    l1_masked = (l1 * mask).sum() / mask.sum()

    return l1_masked

# compute the unpadded loss:
def combined_unpadded_loss(output, target):

    mask = (target != 0).float()

    mse_loss = nn.MSELoss(reduction='none') 
    l1_loss = nn.L1Loss(reduction='none')
    mse = mse_loss(output, target)
    l1 = l1_loss(output, target)

    # mask the nonzero elements to omit padding:
    mse_masked = (mse * mask).sum() / mask.sum()
    l1_masked = (l1 * mask).sum() / mask.sum()

    return mse_masked + l1_masked

# Model training loop
# ----------------------
@profile_code(profile_flag=False)
def train_model(model, optimizer, loss_fxn, loss_target, num_epochs, train_loader, val_loader, scheduler, device, output_folder):

    # if dist.is_available() and dist.is_initialized():
    #     # find_unused_parameters is true because there are multiple target types
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    #     rank = dist.get_rank() 
    # else:
    #     rank = 0

    track_loss_node = []
    track_loss_edge = []
    track_loss_node_val = []
    track_loss_edge_val = []
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        
        model.train()  
        for batch in train_loader:

            optimizer.zero_grad()

            # -- Forward -- 
            batch = batch.to(device)
            out = model(batch) 
            # dist.barrier()

            # -- Loss -- 
            if loss_target == 'fock_matrix':
                output = torch.cat([out["node_rankN"], out["edge_rankN"]], dim=0)
                labels = torch.cat([batch.node_y, batch.y], dim=0)
                loss_node = loss_fxn(out["node_rankN"], batch.node_y)
                loss_edge = loss_fxn(out["edge_rankN"], batch.y) 
                loss = loss_fxn(output, labels)
                # print("loss on just node 0")
                # output = out["node_rankN"][1]
                # labels = batch.node_y[1]
                # loss_node = loss_fxn(out["node_rankN"][1], batch.node_y[1])
                # loss_edge = loss_fxn(out["edge_rankN"], batch.y) 
                # loss = loss_fxn(output, labels)

            elif loss_target == 'forces':
                loss = loss_fxn(out["node_rank1"], batch.forces[:, [1, 2, 0]])  

            elif loss_target == 'energy':
                print("Fix the batch dimension!")
                # print('out["node_rank0"] ',  out["node_rank0"])
                # print('batch.energies ', batch.energies)
                loss = loss_fxn(out["node_rank0"], batch.energies)  

            else: 
                print("unknown loss!") 
            
            # -- Backwards -- 
            loss.backward()
            optimizer.step()
            
        # -- Output dump -- 
        if loss_target == 'fock_matrix':
            track_loss_node.append(loss_node.cpu().detach().numpy() / train_loader.batch_size)
            track_loss_edge.append(loss_edge.cpu().detach().numpy() / train_loader.batch_size)
        else:
            track_loss_node.append(loss.cpu().detach().numpy() / train_loader.batch_size)
        print(f"Epoch {epoch+1}, Train Loss: {track_loss_node[-1]}")        

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # -- Forward --
                out = model(batch)
                
                # -- Loss --
                if loss_target == 'fock_matrix':
                    output = torch.cat([out["node_rankN"], out["edge_rankN"]], dim=0)
                    labels = torch.cat([batch.node_y, batch.y], dim=0)
                    loss_node = loss_fxn(out["node_rankN"], batch.node_y)
                    loss_edge = loss_fxn(out["edge_rankN"], batch.y) 
                    loss = loss_fxn(output, labels)

                    # print("using only node 0")
                    # output = out["node_rankN"][0]
                    # labels = batch.node_y[0]
                    # loss_node = loss_fxn(out["node_rankN"][0], batch.node_y[0])
                    # loss_edge = loss_fxn(out["edge_rankN"], batch.y) 
                    # loss = loss_fxn(output, labels)

                elif loss_target == 'forces':
                    # loss = loss_fxn(out["node_rank1"], batch.forces)  
                    loss = loss_fxn(out["node_rank1"], batch.forces[:, [1, 2, 0]])  

                elif loss_target == 'energy':
                    print("Fix the batch dimension!")
                    loss = loss_fxn(out["node_rank0"], batch.energies)  

                else: 
                    print("unknown target!")
                        
                val_loss += loss.item()

        # -- Output dump -- 
        if loss_target == 'fock_matrix':
            track_loss_node_val.append(loss_node.cpu().detach().numpy() / val_loader.batch_size)
            track_loss_edge_val.append(loss_edge.cpu().detach().numpy() / val_loader.batch_size)
        else:
            track_loss_node_val.append(loss.cpu().detach().numpy() / val_loader.batch_size)
        print(f"Epoch {epoch+1}, Val Loss: {track_loss_node_val[-1]}")
        
        scheduler.step(loss)
        current_lr = optimizer.param_groups[0]['lr']
        print("current Lr: ", current_lr)
        
        epoch_end = time.perf_counter()
        print("Time per epoch: ", epoch_end - epoch_start)
        
        # save state
        # if rank == 0:
        if (epoch + 1) % 20 == 0:
            if loss_target == 'fock_matrix':
                save_training_state(model, optimizer, track_loss_node, track_loss_node_val, 'model.pt', output_folder, track_loss_edge, track_loss_edge_val)
            else:
                save_training_state(model, optimizer, track_loss_node, track_loss_node_val, 'model.pt', output_folder)
    

def save_training_state(model, optimizer, track_loss_node, track_validation_node, save_file, output_folder, track_loss_edge=None, track_validation_edge=None):
    """
    Save the training state of the model and optimizer
    """
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, output_folder + "/" + save_file + '.pt')
    torch.save(model.state_dict(), output_folder + "/" + save_file + '_state_dic.pt')


    if track_loss_edge:
        with open(output_folder + "/" + save_file + '_training_loss.txt', 'w') as f:
            for edge, node in zip(track_loss_edge, track_loss_node):
                f.write(f"{edge:.8f}\t{node:.8f}\n")

        with open(output_folder + "/" + save_file + '_validation_loss.txt', 'w') as f:
            for edge, node in zip(track_validation_edge, track_validation_node):
                f.write(f"{edge:.8f}\t{node:.8f}\n")
    else:
        with open(output_folder + "/" + save_file + '_training_loss.txt', 'w') as f:
            for node in track_loss_node:
                f.write(f"{node:.8f}\n")

        with open(output_folder + "/" + save_file + '_validation_loss.txt', 'w') as f:
            for node in track_validation_node:
                f.write(f"{node:.8f}\n")


    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')

    if track_loss_edge:
        plt.plot(track_loss_edge, label='edge')

    plt.plot(track_validation_node, label='validation node')
    if track_loss_edge:
        plt.plot(track_validation_edge, label='validation edge')
        
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(output_folder + "/" + save_file + '_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
