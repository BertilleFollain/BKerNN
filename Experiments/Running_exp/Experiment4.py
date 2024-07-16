import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from Methods.ReLUNN import ReLUNN
from Methods.BKerNN import BKerNN
import seaborn as sns

# Seed initialization
seed = 1
np.random.seed(seed)

# Parameters
n = 128
ntest = 1024
fig, axes = plt.subplots(3, 5, figsize=(20, 15))
sns.set(style="whitegrid", context="talk")
palette = sns.color_palette()

# Main loop
for idata in range(3):
    X = np.random.rand(n, 1) * 2 - 1
    X_test = np.linspace(0, ntest - 1, ntest) / (ntest - 1) * 2 - 1
    X_test = X_test.reshape(-1, 1)
    std_noise = 0.2

    if idata == 0:
        y = np.sin(2 * np.pi * X) + std_noise * np.random.randn(n, 1)
        y_test = np.sin(2 * np.pi * X_test)
    elif idata == 1:
        y = np.sign(np.sin(2 * np.pi * X)) + std_noise * np.random.randn(n, 1)
        y_test = np.sign(np.sin(2 * np.pi * X_test))
    elif idata == 2:
        y = 4 * np.abs(X + 1 - 0.25 - np.floor(X + 1 - 0.25) - 0.5) - 1 + std_noise * np.random.randn(n, 1)
        y_test = 4 * np.abs(X_test + 1 - 0.25 - np.floor(X_test + 1 - 0.25) - 0.5) - 1
    y = np.ravel(y)
    y_test = np.ravel(y_test)

    ms = [1, 5, 1, 5, 32]
    method = ['bkernn', 'bkernn', 'relunn', 'relunn', 'relunn']
    max_iter_relunn = 400000
    batch_size = 16
    max_iter_bkernn = 300
    gamma = 0.005
    param_grid = {'lambda_val': [0.005, 0.01, 0.02, 0.05]}

    for im, m in enumerate(ms):
        if method[im] == 'bkernn':
            bkernn = BKerNN(m=m, reg_type='Basic')
            grid_search = GridSearchCV(bkernn, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X, y, max_iter=max_iter_bkernn, gamma=gamma)
            best_bkernn = grid_search.best_estimator_
            y_test_pred = best_bkernn.predict(X_test)
        elif method[im] == 'relunn':
            relunn = ReLUNN(m=m)
            relunn.fit(X, y, gamma=gamma, max_iter=max_iter_relunn, batch_size=batch_size)
            y_test_pred = relunn.predict(X_test)

        ax = axes[idata, im]
        ax.plot(X_test, y_test, color=palette[0], linewidth=3)
        if method[im] == 'bkernn':
            ax.plot(X_test, y_test_pred, color=palette[3], linewidth=3)
        elif method[im] == 'relunn':
            ax.plot(X_test, y_test_pred, color=palette[1], linewidth=3)
        ax.plot(X, y, 'kx', markersize=4)
        ax.legend(['target', 'prediction'], loc='upper right', fontsize=10)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.axis([-1, 1, -1.5, 2])
        if method[im] == 'bkernn':
            ax.set_title(f'BKerNN: m={m}', fontweight='normal', fontsize=18)
        elif method[im] == 'relunn':
            ax.set_title(f'ReLUNN: m={m}', fontweight='normal', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.savefig('../../Experiments_results/Plots/Experiment4.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.show()

print('Experiment 4 over')
