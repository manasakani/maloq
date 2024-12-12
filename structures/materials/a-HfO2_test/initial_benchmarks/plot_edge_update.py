import matplotlib.pyplot as plt

categories = [
    'Norm 1', 'Message prep', 'Radial fxn', 'Rotation', 
    'SO(2) conv 1', 'Activation', 
    'Rotation back', 'Projection', 
    'Norm 2', 'Feedforward'
]

times = [
    0.0006580352783203125, 0.0007331371307373047, 0.0010211467742919922, 0.00038909912109375, 
    0.17316627502441406, 0.0002512931823730469, 0.00021648406982421875, 
    0.00034332275390625, 0.0007166862487792969, 0.0008990764617919922
]

# Plot the bar graph
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(categories, times)

# Add labels and title
ax.set_xlabel('Operation')
ax.set_ylabel('Time (seconds)')
ax.set_title('SO(2) Edge Update Breakdown')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Display the plot
plt.tight_layout()
plt.savefig('edge_update_time_breakdown.png', dpi=300, bbox_inches='tight')



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
times = [0.00042629241943359375, 1.4543533325195312e-05, 0.09167957305908203, 0.08069968223571777, 0.0002658367156982422]
categories = ['Reshape', 'Radial fxn', 'm=0', 'm>0', 'Reshape back']

# Plot the bar graph
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(categories, times)
ax.set_xlabel('Operation')
ax.set_ylabel('Time (seconds)')
ax.set_title('SO(2) Conv 1 Breakdown')

plt.xticks(rotation=45, ha='right')

# Display the plot
plt.tight_layout()
plt.savefig('so2_conv_breakdown_edge.png', dpi=300, bbox_inches='tight')
