import torch
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
import time
from e3nn.o3 import Irreps

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