import matplotlib.pyplot as plt
import numpy as np

embedding_dims = ['16', '32', '64', '128']

init_so3 = [0.02205944061279297, 0.023173332, 0.022641897, 0.022683144]  # Initializing SO3_Embeddings for nodes and edges
create_so3= [0.3505072593688965, 0.368170500, 0.365329742, 0.356569767]  # Creating SO3_Embeddings for nodes and edges
init_forward_pass = [0.5317161083221436, 0.547608614, 0.544394970, 0.535581589]  # Initializing the forward pass (total init time)
process_node = [0.06425857543945312, 0.074322462, 0.117687225, 0.241494656]  # Processing the node embedding
process_edge= [0.026105403900146484, 0.041088581, 0.072307110, 0.178815842]  # Processing the edge embedding
convert_irreps = [0.16888165473937988, 0.190473080, 0.277225733, 0.426850557]  # Converting to irreps
total_forward_pass = [0.7910044193267822, 0.853543043, 1.011667013, 1.382787228]  # Total forward pass

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
p1 = plt.bar(embedding_dims, bar1, bar_width, label='Init SO(3) embeddings', color=colors[0])
p2 = plt.bar(embedding_dims, bar2, bar_width, bottom=bar1, label='Create SO(3) embeddings', color=colors[1])
p3 = plt.bar(embedding_dims, bar3, bar_width, bottom=bar1+bar2, label='Init rotations', color=colors[2])
p4 = plt.bar(embedding_dims, bar4, bar_width, bottom=bar1+bar2+bar3, label='Node update block',  color=colors[3])
p5 = plt.bar(embedding_dims, bar5, bar_width, bottom=bar1+bar2+bar3+bar4, label='Edge update block', color=colors[4])
p6 = plt.bar(embedding_dims, bar6, bar_width, bottom=bar1+bar2+bar3+bar4+bar5, label='Convert to Irreps', color=colors[5])
p7 = plt.bar(embedding_dims, bar7, bar_width, bottom=bar1+bar2+bar3+bar4+bar5+bar6, label='Other Forward Pass Time', color=colors[6])



# Adding labels and title
ax.set_xlabel('Embedding Dimension')
ax.set_ylabel('Time (s)')
ax.set_title('Forward pass breakdown')

# put legend on side of plot:
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Display the plot

fig.set_size_inches(7, 5)
plt.tight_layout()
plt.savefig('timing_analysis_embeddingDim.png', dpi=300, bbox_inches='tight')
