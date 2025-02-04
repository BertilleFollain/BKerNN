import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

# Load the results
results = pickle.load(open('../../Experiments_results/Results/Experiment6.pkl', 'rb'))['results']
results_extra_yolanda = pickle.load(open('../../Experiments_results/Results/Experiment6_extra_Yolanda.pkl', 'rb'))['results']
results_extra_music = pickle.load(open('../../Experiments_results/Results/Experiment6_extra_geographical_origin_of_music.pkl', 'rb'))['results']
results_extra_tecator = pickle.load(open('../../Experiments_results/Results/Experiment6_extra_tecator.pkl', 'rb'))['results']
results_extra_pah = pickle.load(open('../../Experiments_results/Results/Experiment6_extra_pah.pkl', 'rb'))['results']
results_extra_semeion = pickle.load(open('../../Experiments_results/Results/Experiment6_extra_semeion.pkl', 'rb'))['results']

# Merge results
for dataset in results_extra_yolanda:
    results[dataset] = results_extra_yolanda[dataset]
for dataset in results_extra_music:
    results[dataset] = results_extra_music[dataset]
for dataset in results_extra_tecator:
    results[dataset] = results_extra_tecator[dataset]
for dataset in results_extra_pah:
    results[dataset] = results_extra_pah[dataset]
for dataset in results_extra_semeion:
    results[dataset] = results_extra_semeion[dataset]

# Convert results to a DataFrame
data = []
for dataset, scores in sorted(results.items()):
    data.append([dataset + ', d=' + str(scores[1]), 'BKerNN, Concave_Feature', scores[0][0]])
    data.append([dataset + ', d=' + str(scores[1]), 'BKerNN, Concave_Variable', scores[0][1]])
    data.append([dataset + ', d=' + str(scores[1]), 'BKRR', scores[0][2]])
    data.append([dataset + ', d=' + str(scores[1]), 'ReLUNN', scores[0][3]])

df = pd.DataFrame(data, columns=['Dataset', 'Method', 'Score'])

# Plotting using Seaborn
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid", context="talk")
palette = sns.color_palette()

# Create the plot
sns.pointplot(x='Dataset', y='Score', hue='Method', data=df, dodge=0.4, linestyle='none', markers=["o", "s", "D", "^"])

# Customizing the plot
plt.title('Prediction Scores by Dataset and Method', fontsize=18)
plt.xlabel('Dataset', fontsize=16)
plt.ylabel(r'$R^2$'+ ' score on test set', fontsize=16)
plt.ylim(0, 1)
plt.xticks(rotation=70, ha='right')  # Rotate dataset labels for better readability

plt.legend(bbox_to_anchor=(0, -0.05, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize =12)
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.tight_layout()
plt.savefig('../../Experiments_results/Plots/Experiment6.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.show()
print('Plotting Experiment6 over')
