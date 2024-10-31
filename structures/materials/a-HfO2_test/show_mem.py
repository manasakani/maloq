import matplotlib.pyplot as plt

# Data input
slice_lengths = [3, 6, 10, 15]
memory_data = {
    "Model": [0.001133568, 0.001133568, 0.001133568, 0.001133568],
    "Peak allocated": [3.094, 5.913, 9.732, 14.409]
}

# Plotting
plt.figure(figsize=(4, 3))
plt.plot(slice_lengths, memory_data["Model"], marker='o', label="Model", color='tab:orange')
plt.ylabel("Model (GiB)", color='tab:orange')
plt.xlabel("Slice Length ($\AA$)")

# right axis:
plt.twinx()
plt.plot(slice_lengths, memory_data["Peak allocated"], marker='o', label="Peak allocated", color='tab:blue')
plt.ylabel("Peak allocated (GiB)", color='tab:blue')

plt.xlabel("Slice Length ($\AA$)")
plt.title("Forward pass")

# legend title "memory (GiB)":
# plt.legend(frameon=False)

# plot a dashed balck line at y = 16:
plt.axhline(y=16, color='tab:blue', linestyle='--', label='GPU Memory Limit')
# plt.grid(True)
plt.tight_layout()

# yticks at 3, 6, 10, 15:
plt.xticks([3, 6, 10, 15])

plt.savefig("memcom_slice.png", dpi=300, bbox_inches='tight')
plt.close()



####################

# Data input
layers = [1, 2, 3, 4, 5] # using 16 embedding dim and 3A slice length
memory_data = {
    "Model": [0.001133568, 0.002243584, 0.0033536, 0.004463616, 0.005573632],
    "Peak allocated": [3.094, 5.692, 8.288, 10.885, 13.481]
}

plt.figure(figsize=(4, 3))
plt.plot(layers, memory_data["Model"], marker='o', label="Model", color='tab:orange')
plt.ylabel("Model (GiB)", color='tab:orange')
plt.xlabel("# MP layers")

# right axis:
plt.twinx()
plt.plot(layers, memory_data["Peak allocated"], marker='o', label="Peak allocated", color='tab:blue')
plt.ylabel("Peak allocated (GiB)", color='tab:blue')

# Labeling and styling
plt.xlabel("# MP layers")
# plt.ylabel("Memory (GiB)")
plt.title("Forward pass")

# plot a dashed balck line at y = 16:
plt.axhline(y=16, color='tab:blue', linestyle='--', label='GPU Memory Limit')
# plt.grid(True)
plt.tight_layout()

# yticks at 3, 6, 10, 15:
plt.xticks([1, 2, 3, 4, 5])

plt.savefig("memcom_mplayers.png", dpi=300, bbox_inches='tight')


####################
