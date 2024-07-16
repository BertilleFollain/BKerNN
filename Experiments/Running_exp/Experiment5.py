import pickle
import numpy as np
import scipy.stats
from Methods.BKerNN import BKerNN
from Methods.BKRR import BKRR
from Methods.ReLUNN import ReLUNN

# Parameters
seed = 0
np.random.seed(seed)
n_values = [10, 20, 50, 100, 150, 200, 300, 400, 500]  # Different sample sizes
d_values = [3, 5, 10, 20, 30, 40, 50]  # Different dimensions
fixed_d = 15
fixed_n = 212
ntest = 201
feature = True
std_noise = 0
k = 3
repetitions = 10
max_iter = 20
max_iter_relunn = 1500
batch_size = 16
m = 50
gamma = 500
gamma_relu = 0.05


# Data generating mechanism
def generate_data(n_bis, d_bis, n_test_bis, k_bis, feature, std_noise_bis, seed_bis):
    np.random.seed(seed_bis)
    if feature:
        p_bis = scipy.stats.ortho_group.rvs(d_bis)
    else:
        p_bis = np.identity(d_bis)
    X_bis = np.random.rand(n_bis, d_bis) * 2 - 1
    y_bis = np.abs(np.sum(np.sin(np.dot(X_bis, p_bis)[:, 0:k_bis]), axis=1)) + std_noise_bis * np.random.randn(
        n_bis)
    X_test_bis = np.random.rand(n_test_bis, d_bis) * 2 - 1
    y_test_bis = np.abs(
        np.sum(np.sin(np.dot(X_test_bis, p_bis)[:, 0:k_bis]), axis=1)) + std_noise_bis * np.random.randn(n_test_bis)
    return X_bis, y_bis, X_test_bis, y_test_bis, p_bis[:, 0:k_bis]


# Initialize results storage
results = {
    method: {
        "scores": np.zeros((len(n_values) + len(d_values), repetitions)),
        "features": np.zeros((len(n_values) + len(d_values), repetitions))
    }
    for method in ['BKerNN', 'BKRR', 'ReLUNN']
}

# Perform the experiment
for n_idx, n in enumerate(n_values):
    for rep in range(repetitions):
        print("n, rep:", n_idx, rep)
        # Generate data
        X, y, X_test, y_test, p = generate_data(n, fixed_d, ntest, k, feature, std_noise, seed + rep)

        # BKerNN
        lambda_val_bkernn = 2 * np.max(np.linalg.norm(X, axis=1)) / n
        bkernn = BKerNN(lambda_val=lambda_val_bkernn, m=m, reg_type='Feature')
        bkernn.fit(X, y, max_iter=max_iter, gamma=gamma)
        results['BKerNN']['scores'][n_idx, rep] = bkernn.score(X_test, y_test)
        results['BKerNN']['features'][n_idx, rep] = bkernn.feature_learning_score(p)

        # BKer
        lambda_val_bker = 2 * np.max(np.linalg.norm(X, axis=1)) / n
        bker = BKRR(lambda_val=lambda_val_bker)
        bker.fit(X, y)
        results['BKRR']['scores'][n_idx, rep] = bker.score(X_test, y_test)

        # ReLUNN
        relunn = ReLUNN(m=m)
        relunn.fit(X, y, gamma=gamma_relu, max_iter=max_iter_relunn, batch_size=batch_size)
        results['ReLUNN']['scores'][n_idx, rep] = relunn.score(X_test, y_test)
        results['ReLUNN']['features'][n_idx, rep] = relunn.feature_learning_score(p)

for d_idx, d in enumerate(d_values):
    for rep in range(repetitions):
        print("d, rep:", d_idx, rep)
        # Generate data
        X, y, X_test, y_test, p = generate_data(fixed_n, d, ntest, k, feature, std_noise, seed + rep)

        # BKerNN
        lambda_val_bkernn = 2 * np.max(np.linalg.norm(X, axis=1)) / fixed_n
        bkernn = BKerNN(lambda_val=lambda_val_bkernn, m=m, reg_type='Feature')
        bkernn.fit(X, y, max_iter=max_iter, gamma=gamma)
        results['BKerNN']['scores'][len(n_values) + d_idx, rep] = bkernn.score(X_test, y_test)
        results['BKerNN']['features'][len(n_values) + d_idx, rep] = bkernn.feature_learning_score(p)

        # BKer
        lambda_val_bker = 2 * np.max(np.linalg.norm(X, axis=1)) / fixed_n
        bker = BKRR(lambda_val=lambda_val_bker)
        bker.fit(X, y)
        results['BKRR']['scores'][len(n_values) + d_idx, rep] = bker.score(X_test, y_test)

        # ReLUNN
        relunn = ReLUNN(m=m)
        relunn.fit(X, y, gamma=gamma_relu, max_iter=max_iter_relunn, batch_size=batch_size)
        results['ReLUNN']['scores'][len(n_values) + d_idx, rep] = relunn.score(X_test, y_test)
        results['ReLUNN']['features'][len(n_values) + d_idx, rep] = relunn.feature_learning_score(p)

# Save results
results_to_save = {
    'results': results,
    'n_values': n_values,
    'd_values': d_values,
    'repetitions': repetitions
}
pickle.dump(results_to_save, open('../../Experiments_results/Results/Experiment5.pkl', 'wb'))

print('Experiment 5 over')
