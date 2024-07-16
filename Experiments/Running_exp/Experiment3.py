import numpy as np
import pandas as pd
from Methods.BKerNN import BKerNN
import pickle
import scipy.stats

# Seed initialization
seed = 1
np.random.seed(seed)

# Parameters
n = 214
n_test = 1024
d = 20
k = 5
std_noise = 0.5
n_repetitions = 20
gamma = 500
max_iter = 25
fixed_m = 20
s = 1


def generate_data(n_bis, d_bis, n_test_bis, k_bis, feature_bis, std_noise_bis, seed_bis):
    np.random.seed(seed_bis)
    if feature_bis:
        p_bis = scipy.stats.ortho_group.rvs(d)
    else:
        p_bis = np.identity(d_bis)
    X_bis = np.random.rand(n_bis, d_bis) * 2 - 1
    X_test_bis = np.random.rand(n_test_bis, d_bis) * 2 - 1
    y_bis = np.sum(np.sin(np.dot(X_bis, p_bis)[:, 0:k_bis]), axis=1) + std_noise_bis * np.random.randn(n_bis)
    y_test_bis = np.sum(np.sin(np.dot(X_test_bis, p_bis)[:, 0:k_bis]), axis=1) + std_noise_bis * np.random.randn(n_test_bis)
    return X_bis, y_bis, X_test_bis, y_test_bis, p_bis


# 3rd Subplot: Influence of `reg_type` on score for different datasets
reg_types = ['Basic', 'Variable', 'Concave_Variable', 'Feature', 'Concave_Feature']
scores_reg_type = {reg_type: [[] for _ in range(n_repetitions)] for reg_type in reg_types}

print("start")

for i_rep in range(n_repetitions):
    data = [0, 0, 0]
    data[0] = generate_data(n, d, n_test, d, False, std_noise*2, seed + i_rep)
    data[1] = generate_data(n, d, n_test, k, False, std_noise, seed + i_rep)
    data[2] = generate_data(n, d, n_test, k, True, std_noise, seed + i_rep)
    for idata, (X, y, X_test, y_test, p) in enumerate(data):
        for reg_type in reg_types:
            print(i_rep, idata, reg_type)
            bkernn = BKerNN(m=fixed_m, reg_type=reg_type, s=s,
                            lambda_val=2 * np.max(np.linalg.norm(X, axis=1)) / n)
            bkernn.fit(X, y, gamma=gamma, max_iter=max_iter)
            scores_reg_type[reg_type][i_rep].append(bkernn.score(X_test, y_test))

# Prepare data for boxplot
boxplot_data = []
data_name = ['no structure', 'few variables', 'few features']
for reg_type in reg_types:
    for idata in range(len(data)):
        scores = [scores_reg_type[reg_type][i_rep][idata] for i_rep in range(n_repetitions)]
        for score in scores:
            boxplot_data.append((reg_type, data_name[idata], score))

# Convert to DataFrame for seaborn
df = pd.DataFrame(boxplot_data, columns=['reg_type', 'dataset_id', 'score'])

# Save results
scores = {'df': df}
pickle.dump(scores, open('../../Experiments_results/Results/Experiment3.pkl', 'wb'))

print('Experiment 3 over')
