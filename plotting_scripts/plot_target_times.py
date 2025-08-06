#!/usr/bin/env python3
"""
Script to plot histogram of "Time to make targets" distribution from out-makeomol.out file
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def extract_target_times(filename):
    """
    Extract "Time to make targets" values from the output file
    
    Args:
        filename (str): Path to the output file
        
    Returns:
        list: List of time values in seconds
    """
    target_times = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Look for lines containing "Time to make targets:"
            if "Total time for one structure:" in line:
                # Extract the time value using regex
                match = re.search(r"Total time for one structure:\s+([0-9]+\.?[0-9]*)", line)
                if match:
                    time_value = float(match.group(1))
                    target_times.append(time_value)
    
    return target_times

def plot_histogram(times, output_filename="target_times_histogram.png"):
    """
    Plot histogram of target times
    
    Args:
        times (list): List of time values
        output_filename (str): Name of output image file
    """
    plt.figure(figsize=(5, 4))
    
    # Create histogram
    n, bins, patches = plt.hist(times, bins=30, alpha=0.5, color='darkgreen', edgecolor='black')
    
    # Customize plot
    plt.xlabel('Time to make fock targets for one structure (s)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    # plt.title('Distribution of "Time to make targets"', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    mean_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    stats_text = f'''Statistics:
Mean: {mean_time:.2f} s
Median: {median_time:.2f} s
Std Dev: {std_time:.2f} s
Min: {min_time:.2f} s
Max: {max_time:.2f} s
Num Focks: {len(times)}'''
    
    plt.text(0.60, 0.95, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=10, family='monospace')
    
    # Add vertical line for mean
    plt.axvline(mean_time, color='black', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_time:.2f}s')
    
    # # Add vertical line for median
    # plt.axvline(median_time, color='orange', linestyle='--', linewidth=2, 
    #             label=f'Median: {median_time:.2f}s')
    
    # plt.legend()
    plt.tight_layout()
    plt.yscale('log')
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Histogram saved as {output_filename}")
    
    return mean_time, std_time, median_time, min_time, max_time

def main():
    """Main function"""
    input_file = "out-makeomol_8.0.out"
    
    print(f"Extracting target times from {input_file}...")
    target_times = extract_target_times(input_file)
    
    if not target_times:
        print("No 'Time to make targets' entries found in the file.")
        return
    
    print(f"Found {len(target_times)} target time entries")
    
    # Plot histogram and get statistics
    mean_time, std_time, median_time, min_time, max_time = plot_histogram(target_times)
    
    # Print detailed statistics
    print("\n" + "="*50)
    print("DETAILED STATISTICS")
    print("="*50)
    print(f"Number of samples: {len(target_times)}")
    print(f"Mean time: {mean_time:.3f} seconds")
    print(f"Median time: {median_time:.3f} seconds")
    print(f"Standard deviation: {std_time:.3f} seconds")
    print(f"Minimum time: {min_time:.3f} seconds")
    print(f"Maximum time: {max_time:.3f} seconds")
    print(f"Range: {max_time - min_time:.3f} seconds")
    
    # Calculate percentiles
    percentiles = [25, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        value = np.percentile(target_times, p)
        print(f"  {p}th percentile: {value:.3f} seconds")
    
    # Identify outliers (values > mean + 2*std)
    outlier_threshold = mean_time + 2 * std_time
    outliers = [t for t in target_times if t > outlier_threshold]
    print(f"\nOutliers (> mean + 2*std = {outlier_threshold:.3f}s): {len(outliers)}")
    if outliers:
        print(f"Outlier values: {sorted(outliers)}")

if __name__ == "__main__":
    main()
