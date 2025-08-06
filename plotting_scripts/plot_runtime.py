import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np

# Function to extract runtimes from a text file
def extract_runtimes(file_path):
    runtimes = []
    with open(file_path, 'r') as file:
        epoch_count = 0
        for line in file:
            if line.startswith("Epoch"):
                epoch_count += 1
            if line.startswith("Time per epoch:") and epoch_count > 100:
                # Extract the runtime value
                runtime = float(line.split(":")[1].strip())
                runtimes.append(runtime)
    return runtimes

# Get all text files in the current directory
# file_paths = glob.glob("*reduced.out")
file_paths = ['out-uracil_1xreduced.out', 'out-uracil_2xreduced.out', 'out-uracil_3xreduced.out']

# Extract runtimes from each file
all_runtimes = []
for i, file_path in enumerate(file_paths):
    runtimes = extract_runtimes(file_path)
    # max_runtime = np.mean(runtimes)
    # print(f"Max runtime for File {i+1}: {max_runtime:.2f} seconds")
    all_runtimes.append(runtimes)

# Create a boxplot
plt.figure(figsize=(4, 3))
plt.ylim([0, 15])
plt.boxplot(all_runtimes, labels=['edge', 'node-inter', 'node-intra'])
plt.ylabel("Time per Epoch (seconds)")
plt.grid(True)
plt.savefig("runtimes.png", dpi=300, bbox_inches='tight')