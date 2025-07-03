import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# data = pd.read_csv('./outputs_QM7_uracil/model_eval.txt', sep='\t', header=None)

# # Assign column names for clarity
# data.columns = ['edges', 'nodes', 'total']
# data *= 1e6 # convert to uEh
# # data *= 1e3 # convert to uEh
# means = data.mean()

# get data:

file_pattern = './outputs_QM7_uracil/model_eval_*.txt'
files = glob.glob(file_pattern)

dataframes = []
for file in files:
    rank_data = pd.read_csv(file, sep='\t', header=None)
    rank_data.columns = ['edges', 'nodes', 'total', 'eigs']
    # rank_data *= 1e3  # convert to mEh
    rank_data *= 1e6  # convert to uEh
    dataframes.append(rank_data)

data = pd.concat(dataframes, ignore_index=True)
means = data.mean()
maxes = data.max()

# print("edge average: ", mean(data['edges']))
# print("node average: ", mean(data['nodes']))
# print("total average: ", mean(data['total']))

# Create a violin plot
plt.figure(figsize=(4, 3))
sns.violinplot(data=data, bw=0.3, cut=0)

# Add mean lines and annotate with mean values
for i, mean in enumerate(means):
    plt.axhline(y=mean, color='k', linestyle='--', linewidth=1)
    plt.text(i-0.3, 1.3*mean, f'{mean:.2f}', color='k', ha='center', va='bottom')

# Set the labels for the x-axis
plt.xticks(ticks=[0, 1, 2, 3], labels=['edges', 'nodes', 'total', 'eigenvalues'])

# Set the title and labels
plt.ylabel('Error ($\mu$E$_h$)')
# plt.ylabel('Error (mE$_h$)')
plt.ylim([-1, 150])
# plt.ylim([-0.1, 1])

# Show the plot
plt.savefig('model_eval.png', dpi=300, bbox_inches='tight')