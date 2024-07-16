import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns

# Load results for plotting
results = pickle.load(open('../../Experiments_results/Results/Experiment5.pkl', 'rb'))
repetitions = results['repetitions']
n_values = results['n_values']
d_values = results['d_values']
method_labels = ['BKerNN', 'ReLUNN', 'BKRR']
sns.set(style="whitegrid", context="talk")
palette = sns.color_palette()
color_palette = [palette[0], palette[3], palette[2]]

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(18, 9))

# Plot for growing sample sizes (subplot 1)
for method_idx, method in enumerate(method_labels):
    avg_scores = np.mean(results['results'][method]['scores'][0:len(n_values), :], axis=1)  # Average across repetitions
    for rep in range(repetitions):
        axs[0, 0].plot(n_values, results['results'][method]['scores'][0:len(n_values), rep],
                       color=color_palette[method_idx],
                       alpha=0.3)
    axs[0, 0].plot(n_values, avg_scores, label=f'{method} - Average', color=color_palette[method_idx], linewidth=2.5)

axs[0, 0].set_title('Prediction Score vs Sample Size', fontsize=18)
axs[0, 0].set_xlabel('Sample Size', fontsize=16)
axs[0, 0].set_ylabel(r'$R^2$' + ' score on test set', fontsize=16)
axs[0, 0].set_ylim(ymin=-0.05, ymax=1.05)
axs[0, 0].legend(fontsize=10)

# Plot for feature scores with growing sample sizes (subplot 2)
for method_idx, method in enumerate(method_labels):
    if method != 'BKRR':
        avg_feature_scores = np.mean(results['results'][method]['features'][0:len(n_values), :],
                                     axis=1)  # Average across repetitions
        for rep in range(repetitions):
            axs[0, 1].plot(n_values, results['results'][method]['features'][0:len(n_values), rep],
                           color=color_palette[method_idx], alpha=0.3)
        axs[0, 1].plot(n_values, avg_feature_scores, label=f'{method} - Average', color=color_palette[method_idx],
                       linewidth=2.5)

axs[0, 1].set_title('Feature Score vs Sample Size', fontsize=18)
axs[0, 1].set_xlabel('Sample Size', fontsize=16)
axs[0, 1].set_ylim(ymin=-0.05, ymax=1.05)
axs[0, 1].set_ylabel('Feature Score', fontsize=16)
axs[0, 1].legend(fontsize=10)

# Plot for growing dimensions (subplot 3)
for method_idx, method in enumerate(method_labels):
    avg_scores = np.mean(results['results'][method]['scores'][len(n_values):, :], axis=1)  # Average across repetitions
    for rep in range(repetitions):
        axs[1, 0].plot(d_values, results['results'][method]['scores'][len(n_values):, rep],
                       color=color_palette[method_idx],
                       alpha=0.3)
    axs[1, 0].plot(d_values, avg_scores, label=f'{method} - Average', color=color_palette[method_idx], linewidth=2.5)

axs[1, 0].set_title('Prediction Score vs Dimension', fontsize=18)
axs[1, 0].set_xlabel('Dimension', fontsize=16)
axs[1, 0].set_ylabel(r'$R^2$' + ' score on test set', fontsize=16)
axs[1, 0].legend(fontsize=10)
axs[1, 0].set_ylim(ymin=-0.05, ymax=1.05)

# Plot for feature scores with growing dimensions (subplot 4)
for method_idx, method in enumerate(method_labels):
    if method != 'BKRR':
        avg_feature_scores = np.mean(results['results'][method]['features'][len(n_values):, :],
                                     axis=1)  # Average across repetitions
        for rep in range(repetitions):
            axs[1, 1].plot(d_values, results['results'][method]['features'][len(n_values):, rep],
                           color=color_palette[method_idx], alpha=0.3)
        axs[1, 1].plot(d_values, avg_feature_scores, label=f'{method} - Average', color=color_palette[method_idx],
                       linewidth=2.5)

axs[1, 1].set_title('Feature Score vs Dimension', fontsize=18)
axs[1, 1].set_xlabel('Dimension', fontsize=16)
axs[1, 1].set_ylabel('Feature Score', fontsize=16)
axs[1, 1].legend(fontsize=10)
axs[1, 1].set_ylim(ymin=-0.05, ymax=1.05)

plt.tight_layout()
plt.savefig('../../Experiments_results/Plots/Experiment5.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.show()
print("Plotting Experiment 5 over")
