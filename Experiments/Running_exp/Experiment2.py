import numpy as np
from Methods.BKerNN import BKerNN
import pickle

# Seed initialization
seed = 1
np.random.seed(seed)

# Parameters
n = 412
n_test = 1024
d = 20
k = 5
std_noise = 0.1

# Fixed values and data for two first subplots
gamma = 500
max_iter = 50
X = np.random.rand(n, d) * 2 - 1
y = np.sum(np.abs(2 * np.pi * X[:, 0:k]), axis=1) + std_noise * np.random.randn(n)
X_test = np.random.rand(n_test, d) * 2 - 1
y_test = np.sum(np.abs(2 * np.pi * X_test[:, 0:k]), axis=1) + std_noise * np.random.randn(n_test)

# 1st Subplot: Influence of `m` on score with cross-validation on `lambda_val`
ms = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50]
lambda_val = 0.02
scores_training_m = []
scores_testing_m = []

for m in ms:
    bkernn = BKerNN(m=m, reg_type='Basic', lambda_val=lambda_val)
    bkernn.fit(X, y, gamma=gamma, max_iter=max_iter)
    scores_training_m.append(bkernn.score(X, y))
    scores_testing_m.append(bkernn.score(X_test, y_test))

# 2nd Subplot: Influence of `lambda_val` on score for fixed `m`
lambda_vals = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5]
scores_training_lambda = []
scores_testing_lambda = []
fixed_m = 10  # Fixed value for m

for lambda_val in lambda_vals:
    bkernn = BKerNN(m=fixed_m, lambda_val=lambda_val, reg_type='Basic')
    bkernn.fit(X, y, gamma=gamma, max_iter=max_iter)
    scores_training_lambda.append(bkernn.score(X, y))
    scores_testing_lambda.append(bkernn.score(X_test, y_test))

# Save results
scores = {'scores_testing_m': scores_testing_m, 'scores_training_m': scores_training_m,
          'scores_testing_lambda': scores_testing_lambda, 'scores_training_lambda': scores_training_lambda,
          'lambda_vals': lambda_vals, 'ms': ms}
pickle.dump(scores,
            open('../../Experiments_results/Results/Experiment2.pkl', 'wb'))

print('Experiment 2 over')
