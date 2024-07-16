import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

# Loading results
grid_searches = pickle.load(
    open('../../Experiments_results/Results/Experiment1.pkl', 'rb'))

# Plotting the results
sns.set(style="whitegrid", context="talk")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
palette = sns.color_palette()
minimum_val = np.amin(np.array(
    [[method.best_estimator_.training_error_, method.best_estimator_.testing_error_] for method in
     grid_searches.values()]))
maximum_val = np.amax(np.array(
    [[method.best_estimator_.training_error_, method.best_estimator_.testing_error_] for method in
     grid_searches.values()]))

# Plot each method's training and testing scores
for ax, (name, method) in zip(axes, grid_searches.items()):
    ax.plot(method.best_estimator_.training_error_, linewidth=2, label='train', color=palette[0], marker='o',
            markersize=5)
    ax.plot(method.best_estimator_.testing_error_, linewidth=2, label='test', color=palette[2], marker='o',
            markersize=5)
    ax.set_title(name, fontsize=18)
    ax.set_ylim(0, maximum_val * 1.1)
    # ax.set_xlim(-1, len(method.training_error_))
    ax.set_xlabel('Number of iterations', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(which='both', alpha=1, visible=True)

plt.tight_layout()
plt.savefig('../../Experiments_results/Plots/Experiment1.pdf', dpi=300,
            bbox_inches='tight', format='pdf')
plt.show()
print('Plotting Experiment1 over')
