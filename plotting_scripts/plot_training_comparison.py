#!/usr/bin/env python3
"""
Script to compare training and validation losses between two model training runs.
Plots side-by-side subplots for easy comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def read_loss_file(filepath):
    """Read loss values from a text file, handling various formats"""
    try:
        losses = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Try to extract number from the line
                    try:
                        # Handle different formats: just numbers, or "epoch: loss" format
                        if ':' in line:
                            loss_str = line.split(':')[-1].strip()
                        else:
                            loss_str = line
                        
                        loss = float(loss_str)
                        losses.append(loss)
                    except ValueError:
                        continue
        return np.array(losses)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return np.array([])

def smooth_losses(losses, alpha=0.1):
    """Apply exponential smoothing to loss values to reduce noise
    
    Args:
        losses: numpy array of loss values
        alpha: smoothing factor (0 < alpha < 1). Lower values = more smoothing
    
    Returns:
        smoothed numpy array
    """
    if len(losses) <= 1:
        return losses
    
    smoothed = np.zeros_like(losses)
    smoothed[0] = losses[0]
    
    for i in range(1, len(losses)):
        smoothed[i] = alpha * losses[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed

def plot_training_comparison(folders=None):
    """Plot training and validation losses for multiple model runs"""
    
    # Use default folder names if not provided
    if folders is None:
        folders = ["outputs_nablaDFT_focktrained_tiny", "outputs_nablaDFT_energytrained_tiny"]
    
    # Read loss files for all folders
    folder_data = {}
    for folder in folders:
        train_loss = read_loss_file(os.path.join(folder, "backbone_training_loss.txt"))
        val_loss = read_loss_file(os.path.join(folder, "backbone_validation_loss.txt"))
        
        # Smooth the loss curves
        smoothing_alpha = 0.1  # Adjust this value: lower = more smoothing
        train_loss_smoothed = smooth_losses(train_loss, smoothing_alpha)
        val_loss_smoothed = smooth_losses(val_loss, smoothing_alpha)
        
        # Create epochs arrays
        epochs_train = np.arange(1, len(train_loss) + 1) if len(train_loss) > 0 else np.array([])
        epochs_val = np.arange(1, len(val_loss) + 1) if len(val_loss) > 0 else np.array([])
        
        folder_data[folder] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_loss_smoothed': train_loss_smoothed,
            'val_loss_smoothed': val_loss_smoothed,
            'epochs_train': epochs_train,
            'epochs_val': epochs_val
        }
    
    # Print statistics
    for i, folder in enumerate(folders):
        data = folder_data[folder]
        print(f"Folder {i+1} ({folder}):")
        print(f"  Training epochs: {len(data['train_loss'])}")
        print(f"  Validation epochs: {len(data['val_loss'])}")
        if len(data['train_loss']) > 0:
            print(f"  Final training loss: {data['train_loss'][-1]:.6f}")
        if len(data['val_loss']) > 0:
            print(f"  Final validation loss: {data['val_loss'][-1]:.6f}")
        print()
    
    # Calculate axis limits for consistency across all plots (if not manually set)
    all_losses = []
    all_epochs = []
    for folder in folders:
        data = folder_data[folder]
        if len(data['train_loss']) > 0:
            all_losses.extend(data['train_loss'])
            all_epochs.extend(data['epochs_train'])
        if len(data['val_loss']) > 0:
            all_losses.extend(data['val_loss'])
            all_epochs.extend(data['epochs_val'])
    
    # Format x-axis with K labels
    def format_thousands(x, pos):
        if x == 0:
            return '0'
        elif x < 1000:
            return f'{int(x)}'
        else:
            return f'{x/1000:.1f}K'
    
    from matplotlib.ticker import FuncFormatter
    
    # Create individual plots for each folder and save them in their respective directories
    colors = ['b', 'r', 'g', 'purple', 'orange', 'brown']  # Colors for different folders
    
    for i, folder in enumerate(folders):
        data = folder_data[folder]
        color = colors[i % len(colors)]
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
        
        if len(data['train_loss_smoothed']) > 0:
            # Plot every 10th training point  
            train_indices = np.arange(0, len(data['train_loss_smoothed']), 10)
            ax.plot(data['epochs_train'][train_indices], data['train_loss_smoothed'][train_indices], 'b-', 
                   linewidth=2, label='Training Loss', alpha=0.5)
        if len(data['val_loss_smoothed']) > 0:
            # Plot every 10th validation point
            val_indices = np.arange(0, len(data['val_loss_smoothed']), 10)
            ax.plot(data['epochs_val'][val_indices], data['val_loss_smoothed'][val_indices], 
                   'ro', markersize=1, label='Validation Loss', alpha=0.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Format x-axis with K labels
        ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
        
        # Set axis limits
        if 'y_limits' in globals() and y_limits is not None:
            ax.set_ylim(y_limits[0], y_limits[1])
        else:
            ax.set_ylim(1e-3, 1e1)  # Default limits
            
        if 'x_limits' in globals() and x_limits is not None:
            ax.set_xlim(x_limits[0], x_limits[1])
        elif all_epochs:
            ax.set_xlim(0, max(all_epochs))
        
        plt.tight_layout()
        
        # Save plot in its folder
        output_filename = os.path.join(folder, 'training_loss_plot.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot {i+1} saved as: {output_filename}")
        plt.close()
    
    print(f"\nAll plots saved successfully!")

if __name__ == "__main__":
    # Configuration - modify these folder names as needed
    # folders = [
    #     "outputs_nablaDFT_focktrained_medium",
    #     "outputs_nablaDFT_energytrained_medium",
    #     "outputs_nablaDFT_focktrained_energyfinetuned_medium"  # Add third folder here
    # ]
    folders = [
        "outputs_omol_58k_energydirect_E128_scaled",
        "outputs_omol_58k_energypretrained_E140"  # Add third folder here
    ]
    
    # Manual axis limits (set to None to use automatic limits)
    y_limits = (1e-1, 1e1)  # (y_min, y_max) for log scale
    x_limits = (0, 700)   # (x_min, x_max) for epochs
    # y_limits = None  # Uncomment to use automatic y-limits
    # x_limits = None  # Uncomment to use automatic x-limits
    
    plot_training_comparison(folders)
