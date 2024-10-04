
import matplotlib.pyplot as plt
num_slices = [1, 2, 3, 4, 5, 6, 7, 8]
num_slice_epoch = [1, 2, 3, 4, 6, 8]

time_per_epoch = [2.5, 1.3, 0.88, 0.72, 0.65, 0.52, 0.43, 0.38]
std_dev = [0.092, 0.040, 0.031, 0.025, 0.053, 0.024, 0.048, 0.023]
num_epoch = [15744, 15628, 15675, 14833, 18492, 20599]

max_time_per_epoch = [time + std for time, std in zip(time_per_epoch, std_dev)]
min_time_per_epoch = [time - std for time, std in zip(time_per_epoch, std_dev)]
speedup_max_time = [max_time_per_epoch[0] / time for time in max_time_per_epoch]
speedup_min_time = [min_time_per_epoch[0] / time for time in min_time_per_epoch]
speedup = [time_per_epoch[0] / time for time in time_per_epoch]

ideal_speedup = num_slices
print(speedup_max_time)

plt.figure(figsize=(3, 3))
plt.fill_between(num_slices, speedup_min_time, speedup_max_time, alpha=0.3)
plt.plot(num_slices, speedup, marker='o', markersize=3, label='Speedup')
plt.plot(num_slices, ideal_speedup, linestyle='--', c='black', label='Ideal speedup')
plt.xticks(num_slices)
plt.xlabel('# Slices ($N_t$)', fontsize=13)
plt.ylabel('Speedup', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.savefig('speedup.png', dpi=500, bbox_inches='tight')
plt.close()

#####

plt.figure(figsize=(3, 3))
plt.fill_between(num_slices, min_time_per_epoch, max_time_per_epoch, alpha=0.3)
plt.plot(num_slices, time_per_epoch, marker='o', markersize=3)
plt.xticks(num_slices)
plt.xlabel('# Slices ($N_t$)', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.ylabel('Time per epoch (s)', fontsize=13)
plt.savefig('time_per_epoch.png', dpi=500, bbox_inches='tight')
plt.close()