import matplotlib.pyplot as plt

categories = [
    'Norm 1', 'Message prep', 'Radial fxn', 'Rotation', 
    'SO(2) conv 1', 'Activation', 'SO(2) conv 2', 
    'Attention', 'Rotation back', 'Projection', 
    'Norm 2', 'Feedforward'
]

times = [
    0.010083437, 0.001844168, 0.017207384, 0.030134439, 
    0.123267412, 0.011451483, 0.031521559, 
    0.009067059, 0.000252962, 0.000756741, 
    0.000923395, 0.001206636
]

# Plot the bar graph
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(categories, times)

# Add labels and title
ax.set_xlabel('Operation')
ax.set_ylabel('Time (seconds)')
ax.set_title('SO(2) Node Update Breakdown')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Display the plot
plt.tight_layout()
plt.savefig('node_update_time_breakdown.png', dpi=300, bbox_inches='tight')



# Reshape time: 0.016777992248535156
# Radial time: 1.5974044799804688e-05
# m=0 time: 0.021765708923339844
# m>0 time: 0.08084535598754883
# Reshape back time: 0.0007302761077880859

# Reshape time: 0.00022172927856445312
# Radial time: 1.4781951904296875e-05
# m=0 time: 0.019711971282958984
# m>0 time: 0.01112985610961914
# Reshape back time: 0.0002570152282714844

# times = [0.016777992248535156, 1.5974044799804688e-05, 0.021765708923339844, 0.08084535598754883, 0.0007302761077880859]
times = [0.00022172927856445312, 1.4781951904296875e-05, 0.019711971282958984, 0.01112985610961914, 0.0002570152282714844]
categories = ['Reshape', 'Radial fxn', 'm=0', 'm>0', 'Reshape back']

# Plot the bar graph
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(categories, times)
ax.set_xlabel('Operation')
ax.set_ylabel('Time (seconds)')
ax.set_title('SO(2) Conv 2 Breakdown')

plt.xticks(rotation=45, ha='right')

# Display the plot
plt.tight_layout()
plt.savefig('so2_conv_breakdown.png', dpi=300, bbox_inches='tight')
