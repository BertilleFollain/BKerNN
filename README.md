# BKerNN, Companion code to 
## *Follain, B. and Bach, F. (2024), Enhanced Feature Learning via Regularisation: Integrating Neural Networks and Kernel Methods*

## What is this project for?
This is the companion code to Follain, B. and Bach, F. (2024), Enhanced Feature Learning via Regularisation: Integrating Neural Networks and Kernel Methods https://arxiv.org/abs/2407.17280
It contains the estimator **BKerNN** introduced in the previously cited article, the code to run the experiments from the article
and the results of said experiments. **BKerNN** is a method for non-parametric regression with linear feature learning, 
which consists in regularised empirical risk minimisation on a neural network kernel fusion, with the Brownian kernel. See the article for more details. The method is available through the class BKerNN in '/Methods/BKerNN.py'. It is easy to use thanks to compatibility with Scikit-learn. 
The code is maintained by Bertille Follain (https://bertillefollain.netlify.app/, email address available on website). Do not 
hesitate to reach out if you need help using it.

## Organisation of the code
The regressors used in the experiments are available in the folder 'Methods', while the code corresponding to each 
experiment are available in the folder 'Experiments'.
The results of the experiments (in .pkl format) are in the folder 'Experiments_results/Results', 
while the figures can be found in the folder 'Experiments_results/Plots'. The requirements for use of the code are in 'requirements.txt'.
Note that some packages are necessary for the experiments but not to use **BKerNN**.

## Methods and Methods_with_monitoring
This folder calls for more detailed explanations. In Methods_with_monitoring, you can find a version of **BKerNN** where the training can be following (cost function, score on train and test set...) as well as similarly monitored **GaussianKerNN** and **ExpKerNN** which are similar to **BKERNN** but with different kernels (Gaussian and Exponential respectively). In Methods, you can find a basic one hidden layer neural network with ReLU activation in the class **ReLUNN** and a simple kernel method with the Brownian kernel in **BKER** as well as the non-monitored **BKerNN**. In both folders, 'Scikit-learn_test.py' allows us to check that all the estimators are compatible with the Scikit-learn API (https://scikit-learn.org/stable/). 

## Example
The class **BKerNN** has many parameters, which are detailed in the definition of the class. Here is a (simple) example of 
usage.
```
from Methods.BKerNN import BKerNN
import numpy as np
import scipy.stats

n = 500  # number of samples
n_test = 500  # number of test samples
d = 20  # original dimension
k = 2  # dimension of hidden linear subspace
X = np.sqrt(3) * (2 * np.random.uniform(size=(n, d)) - 1)
X_test = np.sqrt(3) * (2 * np.random.uniform(size=(n_test, d)) - 1)
p = scipy.stats.ortho_group.rvs(d)
p = p[:, 0:k]
y = np.sum(np.dot(X, p) ** 2, axis=1)
y_test = np.sum(np.dot(X_test, p) ** 2, axis=1)
method = BKerNN(lambda_val=0.005, m=100, reg_type="Feature")
method.fit(X, y, max_iter=50, gamma=500, backtracking=True)  # trains the estimator
y_pred = method.predict(X_test)  # predicts on new dataset
score = method.score(X_test, y_test)  # computes R2 score
feature_learning_score = method.feature_learning_score(p)  # computes feature learning score
print('score', score, 'feature learning score', feature_learning_score)
}
```
