import torch
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
import time
from e3nn.o3 import Irreps

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
def mse_padded_loss(output, target, req_irreps=None):
    mse_loss_type = nn.MSELoss(reduction='mean') 
    return mse_loss_type(output, target)

def l1_padded_loss(output, target, req_irreps=None):
    l1_loss_type = nn.L1Loss(reduction='mean') 
    return l1_loss_type(output, target)

def rmse_padded_loss(output, target, req_irreps=None):
    mse_loss_type = nn.MSELoss(reduction='mean') 
    mse = mse_loss_type(output, target)
    rmse = torch.sqrt(mse)
    return rmse

def rmse_mse_padded_loss(output, target, req_irreps=None):
    mse_loss_type = nn.MSELoss(reduction='mean') 
    mse = mse_loss_type(output, target)
    rmse = torch.sqrt(mse)
    return rmse + mse

def combined_padded_loss(output, target, req_irreps=None):

    mse_loss = nn.MSELoss(reduction='mean') 
    l1_loss = nn.L1Loss(reduction='mean')

    mse = mse_loss(output, target)
    rmse = torch.sqrt(mse)
    l1 = l1_loss(output, target)

    return rmse + l1

def mse_unpadded_loss(output, target, req_irreps=None):

    mask = (target != 0).float()

    mse_loss = nn.MSELoss(reduction='none') 
    mse = mse_loss(output, target)
    mse_masked = (mse * mask).sum() / mask.sum()

    return mse_masked

def weighted_irrep_mse_loss(output, target, req_irreps, weights=None):
    """
    Compute the weighted MSE loss over irreps of each degree.
    Parameters:
    - output: Tensor of model outputs.
    - target: Tensor of target values.
    - req_irreps: List or object describing the irreps, e.g., ['1x0e', '1x3e', ...].
    - weights: list of weights for each irrep degree. If None, equal weighting is used.
    Using weight equal to the degree of the irrep
    Returns:
    - loss: Weighted MSE loss.
    """
    if weights is None:
        weights = [1.0] * len(req_irreps)
    assert len(weights) == len(req_irreps), "Weights must match the number of irreps."
    
    pointer = 0
    total_loss = 0.0
    total_weight = 0.0
    # for irrep, weight in zip(req_irreps, weights):
    for irrep in req_irreps:

        irrep_l = int(str(irrep).split('x')[-1][0])
        dim = 2 * irrep_l + 1
        # weight = 2 * irrep_l + 1 # equal to full mse
        weight = 1.0               # take the average over irreps, so that higher
                                   # irreps aren't weighted more just because they have more
                                   # ms

        output_irrep = output[:, pointer:pointer + dim]
        target_irrep = target[:, pointer:pointer + dim]

        mse_loss = torch.nn.functional.mse_loss(output_irrep, target_irrep)
        # combined_loss = combined_padded_loss(output_irrep, target_irrep)

        # Accumulate weighted loss
        total_loss += weight * mse_loss
        total_weight += weight

        pointer += dim

    # Normalize by the total weight
    loss = total_loss / total_weight
    return loss

def l1_unpadded_loss(output, target, req_irreps=None):

    mask = (target != 0).float()

    l1_loss = nn.L1Loss(reduction='none')
    l1 = l1_loss(output, target)
    l1_masked = (l1 * mask).sum() / mask.sum()

    return l1_masked

def geometric_mean_loss(output, target, req_irreps=None):

    abs_error = torch.abs(output - target) 
    log_error = torch.log(abs_error) 
    geometric_mean = torch.exp(torch.mean(log_error)) 

    return geometric_mean

# compute the unpadded loss:
def combined_unpadded_loss(output, target, req_irreps=None):

    mask = (target != 0).float()

    mse_loss = nn.MSELoss(reduction='none') 
    l1_loss = nn.L1Loss(reduction='none')
    mse = mse_loss(output, target)
    l1 = l1_loss(output, target)

    # mask the nonzero elements to omit padding:
    mse_masked = (mse * mask).sum() / mask.sum()
    l1_masked = (l1 * mask).sum() / mask.sum()

    return mse_masked + l1_masked

def adjust_learning_rate(optimizer, epoch, warmup_epochs, initial_lr, final_lr):
    """Adjusts the learning rate linearly during the warmup phase."""
    if epoch < warmup_epochs:
        lr = initial_lr + (final_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Warmup epoch {epoch+1}: setting learning rate to {lr}")

# Scheduler:
# ----------------------

class MonotonicDecreaseScheduler:
    def __init__(self, optimizer, factor=0.95, min_lr=1e-9, lag_epochs=10):
        self.optimizer = optimizer
        self.factor = factor
        self.inverse_factor = 1.0/factor
        self.min_lr = min_lr
        self.prev_loss = None
        self.up_counter = 0
        self.down_counter = 0
        self.lag_epochs = lag_epochs    # increases the lr if the loss has been monotonously decreasing for lag_epochs
        print("Using MonotonicDecreaseScheduler for loss.")
    
    def step(self, current_loss):

        if self.prev_loss is not None and current_loss >= self.prev_loss:
            # self.down_counter += 1
            # if self.down_counter == self.lag_epochs:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                # print(f"Learning rate reduced to {new_lr}")
                # self.down_counter = 0

        if self.prev_loss is not None and current_loss <= self.prev_loss:
            self.up_counter += 1
            if self.up_counter == self.lag_epochs:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group['lr'] * self.inverse_factor, self.min_lr)
                    param_group['lr'] = new_lr
                    # print(f"Learning rate increased to {new_lr}")
                    self.up_counter = 0
        self.prev_loss = current_loss

# Model training loop
# ----------------------
@profile_code(profile_flag=False)
def train_model(model, optimizer, loss_fxn, loss_target, num_epochs, train_loader, val_loader, scheduler, device, output_folder):

    # need find_unused_parameters for now because there are multiple target types and might only fit one
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank() 
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    else:
        rank = 0

    print("Loss Target: ", loss_target)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    warmup_epochs = 20
    initial_lr = optimizer.param_groups[0]['lr']

    track_loss_node = []
    track_loss_edge = []
    track_loss_node_val = []
    track_loss_edge_val = []

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()

        # warmup epochs
        adjust_learning_rate(optimizer, epoch, warmup_epochs, initial_lr, initial_lr*10)
        
        model.train()  
        train_loss_node = 0.0
        train_loss_edge = 0.0
        for batch in train_loader:

            optimizer.zero_grad()

            # -- Forward -- 
            batch = batch.to(device)
            out = model(batch) 

            # -- Loss -- 
            if loss_target == 'fock_matrix':
                output = torch.cat([out["node_rankN"], out["edge_rankN"]], dim=0)
                labels = torch.cat([batch.node_y, batch.y], dim=0)
                loss_node = loss_fxn(out["node_rankN"], batch.node_y)
                loss_edge = loss_fxn(out["edge_rankN"], batch.y) 
                loss = loss_fxn(output, labels)

                train_loss_node += loss_node.cpu().detach().numpy()
                train_loss_edge += loss_edge.cpu().detach().numpy()

            elif loss_target == 'forces':
                loss = loss_fxn(out["node_rank1"], batch.forces[:, [1, 2, 0]])  

                train_loss_node += loss.cpu().detach().numpy()

            elif loss_target == 'energy':
                print("Fix the batch dimension!")
                loss = loss_fxn(out["node_rank0"], batch.energies)  

            else: 
                print("unknown loss!") 
            
            # Aggregate loss across all processes
            # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            # loss /= dist.get_world_size()
            
            # -- Backwards -- 
            loss.backward()
            optimizer.step()
            
        # -- Output dump -- 
        if loss_target == 'fock_matrix':
            track_loss_node.append(train_loss_node/num_train_batches) 
            track_loss_edge.append(train_loss_edge/num_train_batches)
        else:
            track_loss_node.append(train_loss_node/num_train_batches) 

        if rank == 0:
            if loss_target == 'fock_matrix':
                print(f"Epoch {epoch+1}, Train Loss: [node] {track_loss_node[-1]} [edge] {track_loss_edge[-1]}", flush=True) 
            else:
                print(f"Epoch {epoch+1}, Train Loss: [node] {track_loss_node[-1]}", flush=True) 

        # for debugging:
        if math.isnan(track_loss_node_val[-1]):
            print("Error! Found a nan in the loss!")
            break

        # Validation step
        model.eval()
        val_loss = 0.0
        val_loss_node = 0.0
        val_loss_edge = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # -- Forward --
                out = model(batch)
                dist.barrier()
                
                # -- Loss --
                if loss_target == 'fock_matrix':
                    output = torch.cat([out["node_rankN"], out["edge_rankN"]], dim=0)
                    labels = torch.cat([batch.node_y, batch.y], dim=0)
                    loss_node = loss_fxn(out["node_rankN"], batch.node_y)
                    loss_edge = loss_fxn(out["edge_rankN"], batch.y) 
                    loss = loss_fxn(output, labels)

                    val_loss_node += loss_node.cpu().detach().numpy()
                    val_loss_edge += loss_edge.cpu().detach().numpy()

                elif loss_target == 'forces':
                    loss = loss_fxn(out["node_rank1"], batch.forces[:, [1, 2, 0]])  

                    val_loss_node += loss.cpu().detach().numpy()

                elif loss_target == 'energy':
                    loss = loss_fxn(out["node_rank0"], batch.energies)  

                    val_loss_node += loss.cpu().detach().numpy()

                else: 
                    print("unknown target!")
                        
                val_loss += loss.item()
        
        # val_loss_tensor = torch.tensor(val_loss, device=device)
        # dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        # val_loss = val_loss_tensor.item() / dist.get_world_size()

        # -- Output dump -- 
        if loss_target == 'fock_matrix':
            track_loss_node_val.append(val_loss_node/num_val_batches) 
            track_loss_edge_val.append(val_loss_edge/num_val_batches)
        else:
            track_loss_node_val.append(val_loss_node/num_val_batches) 

        if rank == 0:
            if loss_target == 'fock_matrix':
                print(f"Epoch {epoch+1}, Val Loss  : [node] {track_loss_node_val[-1]} [edge] {track_loss_edge_val[-1]}", flush=True)
            else:
                print(f"Epoch {epoch+1}, Val Loss  : [node] {track_loss_node_val[-1]}", flush=True)
            
        # -- Scheduler -- 
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print("Current learning rate: ", current_lr)
        
        epoch_end = time.perf_counter()
        if rank == 0:
            print("Time per epoch: ", epoch_end - epoch_start)
        
        # save state
        if rank == 0:
            if (epoch + 1) % 100 == 0:
                if loss_target == 'fock_matrix':
                    save_training_state(model, optimizer, track_loss_node, track_loss_node_val, 'model', output_folder, track_loss_edge, track_loss_edge_val)
                else:
                    save_training_state(model, optimizer, track_loss_node, track_loss_node_val, 'model', output_folder)
    

# Model training loop
# ----------------------
@profile_code(profile_flag=False)
def eval_model(model, optimizer, loss_fxn, loss_target, num_epochs, train_loader, val_loader, scheduler, device, output_folder, basis_transform):

    if dist.is_available() and dist.is_initialized():
        # find_unused_parameters is true because there are multiple target types
        rank = dist.get_rank() 
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    else:
        rank = 0

    print("Eval Target: ", loss_target)
    model.eval()

    total_loss = []
    loss_node = []
    if loss_target == "fock_matrix":
        loss_edge = []
    
    # -- Evaluate everything in the train_loader -- 
    with torch.no_grad():  
        for index, batch in enumerate(train_loader):

            print("index: ", index)
            # -- Forward -- 
            batch = batch.to(device)
            out = model(batch, index, output_folder) 

            # -- Loss -- 
            if loss_target == 'fock_matrix':

                # Transform back to uncoupled basis:
                uncoupled_node_outputs = basis_transform.get_H(out["node_rankN"])
                uncoupled_edge_outputs = basis_transform.get_H(out["edge_rankN"])
                uncoupled_node_labels = basis_transform.get_H(batch.node_y)
                uncoupled_edge_labels = basis_transform.get_H(batch.y)

                output = torch.cat([uncoupled_node_outputs, uncoupled_edge_outputs], dim=0)
                labels = torch.cat([uncoupled_node_labels, uncoupled_edge_labels], dim=0)
                loss_node.append(loss_fxn(uncoupled_node_outputs, uncoupled_node_labels))
                loss_edge.append(loss_fxn(uncoupled_edge_outputs, uncoupled_edge_labels))
                total_loss.append(loss_fxn(output, labels))
                
            elif loss_target == 'forces':
                loss_node.append(loss_fxn(out["node_rank1"], batch.forces[:, [1, 2, 0]]))

            elif loss_target == 'energy':
                print("Fix the batch dimension!")
                loss_node.append(loss_fxn(out["node_rank0"], batch.energies))  

            else: 
                print("unknown loss!") 
            
            # remove from gpu
            del batch, out
            torch.cuda.empty_cache()
            
        # -- Output dump -- 
        if loss_target == 'fock_matrix':
            with open(output_folder + "/" + 'model' + '_eval.txt', 'w') as f:
                    for edge, node, total in zip(loss_edge, loss_node, total_loss):
                        f.write(f"{edge:.8f}\t{node:.8f}\t{total:.8f}\n")
        else:
            with open(output_folder + "/" + 'model' + '_eval.txt', 'w') as f:
                    for node in loss_node:
                        f.write(f"{node:.8f}\n")



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
    plt.plot(track_loss_node, '-', c='blue', label='node')

    if track_loss_edge:
        plt.plot(track_loss_edge, '-', c='red', label='edge')

    plt.plot(track_validation_node, '--', c='blue', label='validation node')
    if track_loss_edge:
        plt.plot(track_validation_edge,  '--', c='red', label='validation edge')
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(output_folder + "/" + save_file + '_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
