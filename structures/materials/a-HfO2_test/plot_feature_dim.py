import matplotlib.pyplot as plt
import numpy as np

# Data for three different slice lengths
slice_lengths = ['3', '6', '10', '15']

# Times for each category for the three slice lengths
init_so3 = [0.022055149, 0.024488926, 0.02281785, 0.024254798889160156]  # Initializing SO3_Embeddings for nodes and edges
create_so3 = [0.421597242, 0.398981094, 0.388842344, 0.366229772567749]  # Creating SO3_Embeddings for nodes and edges
init_forward_pass = [0.603957653, 0.587581635, 0.585030794, 0.5707027912139893]  # Initializing the forward pass (total init time)
process_node = [0.068328857, 0.094401598, 0.114542246, 0.1592397689819336]  # Processing the node embedding
process_edge = [0.026054621, 0.051911116, 0.070274591, 0.10256052017211914]  # Processing the edge embedding
convert_irreps = [0.173114777, 0.198207378, 0.247527361, 0.31453514099121094]  # Converting to irreps
total_forward_pass = [0.871503353, 0.932151794, 1.017413855, 1.1470839977264404]  # Total forward pass

# Define the width of each bar
bar_width = 0.4

# Stacked box plot
fig, ax = plt.subplots(figsize=(10, 6))

# colors from viridis:
colors = plt.cm.viridis(np.linspace(0, 1, 7))

# Convert the times into cumulative stacked values
bar1 = np.array(init_so3)
bar2 = np.array(create_so3)
bar3 = np.array(init_forward_pass) - bar2 - bar1
bar4 = np.array(process_node)
bar5 = np.array(process_edge)
bar6 = np.array(convert_irreps)
bar7 = np.array(total_forward_pass) - bar1 - bar2 - bar3 - bar4 - bar5 - bar6

# Plotting the stacked bars
p1 = plt.bar(slice_lengths, bar1, bar_width, label='Init SO(3) embeddings', color=colors[0])
p2 = plt.bar(slice_lengths, bar2, bar_width, bottom=bar1, label='Create SO(3) embeddings', color=colors[1])
p3 = plt.bar(slice_lengths, bar3, bar_width, bottom=bar1+bar2, label='Init rotations', color=colors[2])
p4 = plt.bar(slice_lengths, bar4, bar_width, bottom=bar1+bar2+bar3, label='Node update block',  color=colors[3])
p5 = plt.bar(slice_lengths, bar5, bar_width, bottom=bar1+bar2+bar3+bar4, label='Edge update block', color=colors[4])
p6 = plt.bar(slice_lengths, bar6, bar_width, bottom=bar1+bar2+bar3+bar4+bar5, label='Convert to Irreps', color=colors[5])
p7 = plt.bar(slice_lengths, bar7, bar_width, bottom=bar1+bar2+bar3+bar4+bar5+bar6, label='Other Forward Pass Time', color=colors[6])



# Adding labels and title
ax.set_xlabel('Slice Length ($\AA$)')
ax.set_ylabel('Time (s)')
ax.set_title('Forward pass breakdown')

# put legend on side of plot:
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Display the plot

fig.set_size_inches(7, 5)
plt.tight_layout()
plt.savefig('timing_analysis.png', dpi=300, bbox_inches='tight')
