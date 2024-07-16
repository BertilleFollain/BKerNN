import pickle
import numpy as np
import scipy.stats
from Methods_with_monitoring.BKerNN import BKerNN
from Methods_with_monitoring.ExpKerNN import ExpKerNN
from Methods_with_monitoring.GaussianKerNN import GaussianKerNN
from sklearn.model_selection import GridSearchCV

# Sample Data
seed = 0
np.random.seed(seed)
n = 214
n_test = 1024
d = 45
k = 5
std_noise = 0.5
p = scipy.stats.ortho_group.rvs(d)
X = np.random.rand(n, d) * 2 - 1
y = np.abs(np.sum(2 * np.pi * np.dot(X, p)[:, 0:k], axis=1)) + std_noise * np.random.randn(n)
X_test = np.random.rand(n_test, d) * 2 - 1
y_test = np.abs(np.sum(2 * np.pi * np.dot(X_test, p)[:, 0:k], axis=1)) + std_noise * np.random.randn(n_test)

# Method parameters
reg_type = 'Basic'
max_iter = 20
max_iter_2 = 200
m = 100
gamma = 500
lambda_vals = np.array([0.05, 0.1, 1, 0.5, 1, 1.5]) * 2 * np.max(np.linalg.norm(X, axis=1)) / n

# Initialize methods
methods = {
    'BKerNN': BKerNN(m=m, reg_type=reg_type),
    'ExpKerNN': ExpKerNN(m=m, reg_type=reg_type),
    'GaussianKerNN': GaussianKerNN(m=m, reg_type=reg_type)
}

grid_searches = {key: GridSearchCV(method, {'lambda_val': lambda_vals}, cv=5, scoring='neg_mean_squared_error') for
                 key, method in methods.items()}

# Fit methods
for grid_search in grid_searches.values():
    grid_search.fit(X, y, max_iter=max_iter, gamma=gamma, backtracking=True, monitoring=True, X_test=X_test,
                    y_test=y_test)
    grid_search.best_estimator_.fit(X, y, max_iter=max_iter_2, gamma=gamma, backtracking=True, monitoring=True,
                                   X_test=X_test,
                                   y_test=y_test)

# Save results
pickle.dump(grid_searches,
            open('../../Experiments_results/Results/Experiment1.pkl', 'wb'))

print('Experiment 1 over')
