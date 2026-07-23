# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

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

# Optimizer:
# ----------------------
@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to compute the orthogonalization of G."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + 1e-7)
    
    # Transpose handling for tall matrices
    if G.size(0) > G.size(1):
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """Standalone Muon optimizer for 2D+ tensors."""
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim >= 2, "Muon only supports 2D+ parameters."
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                # Momentum update
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                # Reshape to 2D (e.g., if handling convolutions or 3D tensor products)
                original_shape = buf.shape
                g_2d = buf.view(buf.size(0), -1) if buf.ndim > 2 else buf
                    
                # Orthogonalize via Newton-Schulz
                g_ortho = zeropower_via_newtonschulz5(g_2d, steps=5)
                
                # Scale by aspect ratio to balance training across layers
                g_ortho *= max(1, g_2d.size(0) / g_2d.size(1)) ** 0.5
                g_update = g_ortho.view(original_shape)
                
                # Apply weight decay multiplicatively
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Update weights
                p.add_(g_update, alpha=-lr)
                
        return loss
        
class HybridMuon(torch.optim.Optimizer):
    """
    A unified optimizer that routes 2D+ tensors to Muon and 1D tensors to AdamW.
    """
    def __init__(self, params, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.01):
        params = list(params)
        # Initialize base Optimizer with dummy defaults so type checks pass
        super().__init__(params, defaults={'lr': adamw_lr, 'weight_decay': weight_decay})
        
        # Split parameters based on rank
        muon_params = [p for p in params if p.ndim >= 2]
        adamw_params = [p for p in params if p.ndim < 2]
        
        self.opt_muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
        self.opt_adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay)
        
        # Schedulers (like CosineAnnealing) iterate over this list, directly
        # modifying the underlying dictionaries.
        self.param_groups = self.opt_muon.param_groups + self.opt_adamw.param_groups

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.opt_muon.step()
        self.opt_adamw.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.opt_muon.zero_grad(set_to_none=set_to_none)
        self.opt_adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon_state': self.opt_muon.state_dict(),
            'adamw_state': self.opt_adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.opt_muon.load_state_dict(state_dict['muon_state'])
        self.opt_adamw.load_state_dict(state_dict['adamw_state'])
        
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