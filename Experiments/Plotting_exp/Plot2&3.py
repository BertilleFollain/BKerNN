import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Loading results
scores = pickle.load(open('../../Experiments_results/Results/Experiment2.pkl', 'rb'))

df = pickle.load(open('../../Experiments_results/Results/Experiment3.pkl', 'rb'))

# Plotting the results
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.set(style="whitegrid", context="talk")
palette = sns.color_palette()

# 1st Subplot
sns.lineplot(x=scores['ms'], y=scores['scores_training_m'], ax=axes[0], color=palette[0], marker='o', label='train set')
sns.lineplot(x=scores['ms'], y=scores['scores_testing_m'], ax=axes[0], color=palette[2], marker='o', label='test set')
axes[0].set_title('Influence of m on score', fontsize=18)
axes[0].set_xlabel('m', fontsize=16)
axes[0].set_ylabel(r'$R^2$' + ' score', fontsize=16)
axes[0].set_ylim(ymin=-0.05, ymax=1.05)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].grid(which='both', alpha=1, visible=True)

# 2nd Subplot
sns.lineplot(x=scores['lambda_vals'], y=scores['scores_training_lambda'], ax=axes[1], color=palette[0], marker='o',
             label='train set')
sns.lineplot(x=scores['lambda_vals'], y=scores['scores_testing_lambda'], ax=axes[1], color=palette[2], marker='o',
             label='test set')
axes[1].set_title("Influence of " + r'$\lambda$ on score', fontsize=18)
axes[1].set_xlabel(r'$\lambda$', fontsize=16)
axes[1].set_ylabel(r'$R^2$' + ' score', fontsize=16)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].set_xscale('log')
axes[1].set_ylim(ymin=-0.05, ymax=1.05)
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].grid(which='both', alpha=1, visible=True)

# 3rd Subplot
axes[2].grid(which='both', alpha=1, visible=True)
axes[2].set_axisbelow(True)
sns.stripplot(
    x='dataset_id', y='score', hue='reg_type', data=df['df'], dodge=0.5, alpha=.3, legend=False,
)
sns.pointplot(x='dataset_id', y='score', hue='reg_type', data=df['df'], errorbar='sd', dodge=0.67,
              marker='_', markersize=20, markeredgewidth=10, linestyles="none", ax=axes[2], capsize=.1,
              err_kws={'color': 'black', 'linewidth': 1, 'zorder': -1, 'alpha': 0.8})
axes[2].set_title('Influence of regularisation type on score', fontsize=18)
axes[2].set_xlabel('Dataset', fontsize=16)
#axes[2].set_ylim(ymin=.75, ymax=1.05)
axes[2].set_ylabel(r'$R^2$' + ' score on test set', fontsize=16)
axes[2].legend(fontsize=10, loc='best')
axes[2].tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.savefig('../../Experiments_results/Plots/Experiment2&3.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.show()
print("Plotting Experiment 2&3 over")
