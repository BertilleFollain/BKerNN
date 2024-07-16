from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
import numpy as np
from numba import jit, prange


class ReLUNN(RegressorMixin, BaseEstimator):
    """
    ReLUNN Regressor: A feed-forward neural network regressor with ReLU activation and one hidden layer.
    """

    def __init__(self, m=50):
        """
        Initialize the ReLUNN regressor.

        :param m: Number of particles (neurons).
        """
        if int(m) != m:
            raise ValueError("m must be an integer.")
        self.m = m

    def fit(self, X, y, gamma=1e-8, max_iter=100, batch_size=16, random_state=0):
        """
        Fit the ReLUNN model according to the given training data.

        :param X: Training data of shape (n, n_features_in).
        :param y: Target values of shape (n,).
        :param gamma: Learning rate.
        :param max_iter: Number of iterations.
        :param batch_size: Size of mini-batches.
        :param random_state: Seed for random number generator.
        :return: self : Returns an instance of self.
        """
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = self.X_.shape[1]
        self.n_ = self.X_.shape[0]
        if self.n_ == 1:
            raise ValueError("1 sample")

        random_state = check_random_state(random_state)
        self.max_iter_ = max_iter
        self.batch_size_ = np.minimum(batch_size, self.n_)

        # Initialization
        self.W_ = random_state.randn(self.n_features_in_, self.m)
        self.W_ /= np.sqrt(np.sum(self.W_ ** 2, axis=0))
        self.b_ = random_state.rand(self.m) * 2 - 1
        self.eta_ = random_state.randn(self.m) / np.sqrt(self.m / 2)

        # Training
        for iteration in range(self.max_iter_):
            ind = random_state.choice(self.n_, self.batch_size_, replace=False)
            gradW, gradeta, gradb = self.compute_gradient(self.m, self.n_features_in_, self.W_, self.eta_,
                                                          self.b_, self.X_, self.y_, self.batch_size_, ind)
            gradW = np.reshape(gradW, (self.n_features_in_, self.m), order='F')

            self.W_ -= gamma * gradW
            self.eta_ -= gamma * gradeta
            self.b_ -= gamma * gradb

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the ReLUNN model.

        :param X: Test data of shape (n_samples, n_features).
        :return: Predicted values of shape (n_samples,).
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.sum(self.eta_ * np.maximum(np.dot(X, self.W_) + self.b_, 0), axis=1)
        return y_pred

    def _more_tags(self):
        return {'poor_score': True}

    def value(self, W, b, eta):
        """
        Compute the value of the loss function for specified parameters of the model.

        :param W: Weight matrix.
        :param b: Bias vector.
        :param eta: Coefficient vector.
        :return: Value of the loss function.
        """
        y_pred = np.sum(eta * np.maximum(np.dot(self.X_, W) + b, 0), axis=1)
        return np.mean((self.y_ - y_pred) ** 2) / 2

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction. The wort score can be -inf, the constant mean
        predictor score is 0 and the best possible score is 1.

        :param X: Test data of shape (n_samples, n_features).
        :param y: True values for X of shape (n_samples,).
        :param sample_weight: Sample weights (ignored).
        :return: R^2 score.
        """
        y_pred = self.predict(X)
        return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def feature_learning_score(self, p):
        """
        Compute the feature learning score. The best possible score is 1 and the worst score is 0.

        :param p: Ground truth feature matrix.
        :return: Feature learning score.
        """
        k = np.minimum(np.shape(p)[0], np.shape(p)[1])
        U, S, Vh = np.linalg.svd(self.W_, full_matrices=False)
        p_hat = U[:, 0:k]
        pi_p_hat = np.dot(np.dot(p_hat, np.linalg.inv(np.dot(p_hat.T, p_hat))), p_hat.T)
        pi_p = np.dot(np.dot(p, np.linalg.inv(np.dot(p.T, p))), p.T)
        if k <= self.n_features_in_ / 2:
            error = (np.linalg.norm(pi_p - pi_p_hat)) ** 2 / (2 * k)
        elif k == self.n_features_in_:
            error = 0
        else:
            error = (np.linalg.norm(pi_p - pi_p_hat)) ** 2 / (2 * self.n_features_in_ - 2 * k)
        return 1 - error

    def dimension_score(self, k):
        """
        Compute the dimension learning score. The best possible score is 1 and the worst score is 0.

        :param k: Ground truth dimension.
        :return: Dimension learning score.
        """
        U, S, Vh = np.linalg.svd(self.W_, full_matrices=False)
        importance = S / np.sum(S)
        k_hat = np.sum(importance > 1 / self.n_features_in_)
        if k <= self.n_features_in_ / 2:
            error = np.abs(k_hat - k) / (self.n_features_in_ - k)
        else:
            error = np.abs(k_hat - k) / k
        return 1 - error

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def compute_gradient(m, n_features_in, W, eta, b, X, y, batch_size, ind):
        """
        Compute the gradient of the loss function.

        :param m: Number of particles (neurons).
        :param n_features_in: Number of features.
        :param W: Weight matrix.
        :param eta: Coefficient vector.
        :param b: Bias vector.
        :param X: Training data.
        :param y: Target values.
        :param batch_size: Size of mini-batches.
        :param ind: Indices of the current batch.
        :return: Gradients of W, eta, and b.
        """
        gradW = np.zeros(n_features_in * m, dtype=float)
        gradb = np.zeros(m, dtype=float)
        gradeta = np.zeros(m, dtype=float)
        X = X.astype('float64')
        X_batch = X[ind, :]
        y_batch = y[ind]
        W = W.astype('float64')

        for j in prange(m):
            for l in prange(n_features_in):
                gradW[j * n_features_in + l] = -eta[j] * np.sum(
                    (y_batch - np.sum(eta * np.maximum(np.dot(X_batch, W) + b, 0), axis=1)) * (
                            np.dot(X_batch, W[:, j]) + b[j] > 0) * X_batch[:, l]
                ) / batch_size
            gradeta[j] = -np.sum(
                (y_batch - np.sum(eta * np.maximum(np.dot(X_batch, W) + b, 0), axis=1)) * np.maximum(
                    np.dot(X_batch, W[:, j]) + b[j], 0)
            ) / batch_size
            gradb[j] = -eta[j] * np.sum(
                (y_batch - np.sum(eta * np.maximum(np.dot(X_batch, W) + b, 0), axis=1)) * (
                        np.dot(X_batch, W[:, j]) + b[j] > 0)
            ) / batch_size

        return gradW, gradeta, gradb
