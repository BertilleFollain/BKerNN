import pickle
import numpy as np
from Methods.BKerNN import BKerNN
from Methods.ReLUNN import ReLUNN
from Methods.BKRR import BKRR
from sklearn.preprocessing import StandardScaler

# Run Obtain_data_exp6.py first to obtain the data
datasets = ['abalone', 'Ailerons', 'Bike_Sharing_Demand', 'Brazilian_houses', 'cpu_act',
            'diamonds', 'elevators', 'house_16H', 'house_sales', 'houses', 'medical_charges', 'MiamiHousing2016',
            'nyc-taxi-green-dec-2016', 'pol', 'sulfur', 'superconduct', 'wine_quality']

# Initialize results storage
results = {dataset: [np.zeros(4), 0] for dataset in datasets}

# Parameters
max_iter = 40
max_iter_relunn = 2500  # corresponds to 100 epochs
batch_size = 16
gamma = 50
gamma_relu = 0.01

for dataset in datasets:
    # Loading results
    res = pickle.load(open('../../Experiments_results/Data_for_Experiment6/Experiment6data' + dataset + '.pkl', 'rb'))
    X = res['X'][:400, :]
    y = res['y'][:400]
    X_test = res['X'][400:500, :]
    y_test = res['y'][400:500]
    n = 400
    print(dataset, np.shape(res['X']))
    results[dataset][1] = np.shape(res['X'])[1]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    y_norm = (y - y.mean()) / y.std()
    y_test = (y_test - y.mean()) / y.std()
    y = y_norm

    m = 2 * results[dataset][1]

    # BKerNN, Concave Feature
    bkernn = BKerNN(m=m, reg_type='Concave_Feature', lambda_val=np.max(np.linalg.norm(X, axis=1)) / n)
    bkernn.fit(X, y, max_iter=max_iter, gamma=gamma)
    results[dataset][0][0] = bkernn.score(X_test, y_test)

    # BKerNN, Concave Variable
    bkernn = BKerNN(m=m, reg_type='Concave_Variable', lambda_val=np.max(np.linalg.norm(X, axis=1)) / n)
    bkernn.fit(X, y, max_iter=max_iter, gamma=gamma)
    results[dataset][0][1] = bkernn.score(X_test, y_test)

    # BKRR
    bkrr = BKRR(lambda_val=np.max(np.linalg.norm(X, axis=1)) / n)
    bkrr.fit(X, y)
    results[dataset][0][2] = bkrr.score(X_test, y_test)

    # ReLUNN
    relunn = ReLUNN(m=m)
    relunn.fit(X, y, gamma=gamma_relu, max_iter=max_iter_relunn, batch_size=batch_size)
    results[dataset][0][3] = relunn.score(X_test, y_test)

# Save results
results = {'results': results}
pickle.dump(results, open('../../Experiments_results/Results/Experiment6.pkl', 'wb'))

print('Experiment 6 over')
