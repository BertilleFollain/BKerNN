from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
import numpy as np
import scipy as sp
from numba import jit, prange


class BKerNN(RegressorMixin, BaseEstimator):
    """
    This method implements a regressor based on a fusions between kernel ridge regression with the Brownian kernel and
    a neural network. This estimator is presented in Follain, B. and Bach, F. (2024),
    Feature learning through regularisation in Neural Networks/Kernel fusion.
    """

    def __init__(self, lambda_val=0.00125, m=50, reg_type='Variable', s=1):
        """
        Initialize the BKerNN with regularization parameters.

        :param lambda_val: Regularization parameter.
        :param m: Number of particles.
        :param reg_type: Type of regularization to use.
        :param s: Parameter for specific types of regularization.
        """
        if lambda_val < 0 or int(m) != m or s < 0 or reg_type not in ['Basic', 'Variable', 'Feature',
                                                                      'Concave_Variable', 'Concave_Feature']:
            raise ValueError("Invalid parameter values")
        self.lambda_val = lambda_val
        self.m = m
        self.reg_type = reg_type
        self.s = s

    def fit(self, X, y, gamma=1, max_iter=100, beta_0=1.5, beta_1=0.5, random_state=0, backtracking=True):
        """
        Fit the model to the training data.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Target values of shape (n_samples,).
        :param gamma: Learning rate.
        :param max_iter: Number of iterations.
        :param beta_0: Initial backtracking line search parameter.
        :param beta_1: Backtracking line search scaling parameter.
        :param random_state: Seed for random number generator.
        :param backtracking: Whether to use backtracking line search.
        :return: self : Returns an instance of self.
        """
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y, y_numeric=True)
        self.X_, self.y_ = X, y
        self.n_features_in_ = self.X_.shape[1]
        self.n_ = self.X_.shape[0]
        if self.n_ == 1:
            raise ValueError("1 sample")

        random_state = check_random_state(random_state)
        self.max_iter_ = max_iter
        self.beta_0_, self.beta_1_ = beta_0, beta_1

        # Center the output
        self.y_mean_ = np.mean(self.y_)
        self.y_norm_ = self.y_ - self.y_mean_

        # Initialization
        self.W_ = random_state.randn(self.n_features_in_, self.m) / np.sqrt(self.n_features_in_)

        # Training
        for iteration in range(self.max_iter_):
            # Compute the kernel matrix
            self.K_ = self.K_from_W(self.X_, self.X_, self.W_, self.m)
            K_norm = (np.eye(self.n_) - np.ones((self.n_, self.n_)) / self.n_) @ self.K_ @ (
                    np.eye(self.n_) - np.ones((self.n_, self.n_)) / self.n_)

            # Solve for alpha
            self.alpha_ = np.linalg.solve(K_norm + self.n_ * self.lambda_val * np.identity(self.n_), self.y_norm_)

            # Compute the gradient
            gradW = self.compute_gradient(self.alpha_, self.m, self.n_features_in_, self.n_, self.W_, self.X_,
                                          self.lambda_val)
            gradW = np.reshape(gradW, (self.n_features_in_, self.m), order='F')

            # Compute the learning rate if backtracking is used
            if backtracking:
                gamma = self.beta_0_ * gamma
                gamma = self._backtracking_line_search(gamma, gradW)

            # Update the particles
            self.W_ = self.prox(self.W_ - gamma * gradW, gamma)

        self.is_fitted_ = True
        return self

    def _backtracking_line_search(self, gamma, gradW):
        """
        Perform backtracking line search to determine optimal step size.

        :param gamma: Initial learning rate.
        :param gradW: Gradient of the weights.
        :returns: Updated learning rate.
        """
        count = 0
        while count < 8 and self.G(self.W_ - gamma * self.G_t(self.W_, gradW, gamma)) > self.G(self.W_,
                                                                                               current=True) - gamma * np.trace(
            np.dot(gradW.T, self.G_t(self.W_, gradW, gamma))) + gamma * np.trace(
            np.dot(self.G_t(self.W_, gradW, gamma).T, self.G_t(self.W_, gradW, gamma))) / 2:
            gamma = self.beta_1_ * gamma
            count += 1
        return gamma

    def G(self, W, current=False):
        """
        Compute the objective function value (excluding the penalty), for a specified value of the weights.

        :param W: Weights matrix.
        :param current: Whether to use the current kernel matrix.
        :returns: Objective function value.
        """
        K = self.K_ if current else self.K_from_W(self.X_, self.X_, W, self.m)
        K_norm = (np.eye(self.n_) - np.ones((self.n_, self.n_)) / self.n_) @ K @ (
                np.eye(self.n_) - np.ones((self.n_, self.n_)) / self.n_)
        alpha_ = np.linalg.solve(K_norm + self.n_ * self.lambda_val * np.identity(self.n_), self.y_norm_)
        value = (self.lambda_val / 2) * np.dot(self.y_norm_.T, alpha_)
        return value

    def G_t(self, W, gradW, gamma):
        """
        Compute the gradient transformation.

        :param W: Weights matrix.
        :param gradW: Gradient of the weights.
        :param gamma: Learning rate.
        :returns: Gradient transformation.
        """
        return (W - self.prox(self.W_ - gamma * gradW, gamma)) / gamma

    def penalty(self):
        """
        Compute the regularization penalty based on the specified regularization type.
        :returns: Regularization penalty.
        """
        if self.reg_type == 'Basic':
            return self.lambda_val * np.sum(np.linalg.norm(self.W_, axis=0) / (2 * self.m))
        elif self.reg_type == 'Variable':
            return self.lambda_val * np.sum(np.linalg.norm(self.W_, axis=1)) / (2 * np.sqrt(self.m))
        elif self.reg_type == 'Feature':
            return np.real(
                self.lambda_val * np.trace(sp.linalg.sqrtm(np.dot(self.W_, self.W_.T))) / (2 * np.sqrt(self.m)))
        elif self.reg_type == 'Concave_Variable':
            return self.lambda_val * np.sum(
                np.log(1 + (self.s / np.sqrt(self.m)) * np.linalg.norm(self.W_, axis=1))) / (2 * self.s)
        elif self.reg_type == 'Concave_Feature':
            U, S, Vh = np.linalg.svd(self.W_, full_matrices=False)
            return self.lambda_val * np.sum(np.log(1 + (self.s * S / np.sqrt(self.m)))) / (2 * self.s)
        return self

    def prox(self, W, gamma):
        """
        Proximal operator for regularization.

        :param W: Weights matrix.
        :param gamma: Learning rate.
        :returns: Updated weights matrix.
        """
        if self.reg_type == 'Basic':
            return self._prox_basic(W, gamma)
        elif self.reg_type == 'Variable':
            return self._prox_variable(W, gamma)
        elif self.reg_type == 'Feature':
            return self._prox_feature(W, gamma)
        elif self.reg_type == 'Concave_Variable':
            return self._prox_concave_variable(W, gamma)
        elif self.reg_type == 'Concave_Feature':
            return self._prox_concave_feature(W, gamma)
        return self

    def _prox_basic(self, W, gamma):
        """
        Basic proximal operator.

        :param W: Weights matrix.
        :param gamma: Learning rate.
        :returns: Updated weights matrix.
        """
        norm_W = np.linalg.norm(W, axis=0)
        update_factor = np.maximum(1 - self.lambda_val * gamma / (2 * self.m * norm_W), 0)
        return W * update_factor[np.newaxis, :]

    def _prox_variable(self, W, gamma):
        """
        Variable proximal operator.

        :param W: Weights matrix.
        :param gamma: Learning rate.
        :returns: Updated weights matrix.
        """
        norm_W = np.linalg.norm(W, axis=1)
        update_factor = np.maximum(1 - self.lambda_val * gamma / (2 * np.sqrt(self.m) * norm_W), 0)
        return W * update_factor[:, np.newaxis]

    def _prox_feature(self, W, gamma):
        """
        Feature proximal operator.

        :param W: Weights matrix.
        :param gamma: Learning rate.
        :returns: Updated weights matrix.
        """
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        update_factor = np.maximum(1 - self.lambda_val * gamma / (2 * self.m * S), 0)
        return (U * S) @ np.diag(update_factor) @ Vh

    def _prox_concave_variable(self, W, gamma):
        """
        Concave variable proximal operator.

        :param W: Weights matrix.
        :param gamma: Learning rate.
        :returns: Updated weights matrix.
        """
        poly_a = self.s * np.linalg.norm(W, axis=1) / np.sqrt(self.m)
        poly_b = 1 - poly_a
        poly_c = self.lambda_val * gamma / (2 * np.sqrt(self.m) * np.linalg.norm(W, axis=1)) - 1
        delta = np.abs(poly_b ** 2 - 4 * poly_a * poly_c)  # when delta is negative, this means that the function is
        # striclty increasing, and the argmin is necessarily c_0 which will be the case even with this abs
        c_1 = (-poly_b + np.sqrt(delta)) / (2 * poly_a)
        c_2 = (-poly_b - np.sqrt(delta)) / (2 * poly_a)
        c_0 = [0] * self.n_features_in_
        list_c = np.array([c_0, c_1, c_2])
        f = lambda x: ((np.linalg.norm(W, axis=1)[x[1]] ** 2) * (
                1 - x[0]) ** 2) / 2 + self.lambda_val * gamma * np.log(
            1 + np.abs(x[0]) * self.s * np.linalg.norm(W, axis=1)[x[1]] / np.sqrt(self.m)) / (2 * self.s)
        f_values = np.array([[f([c_0[a], a]), f([c_1[a], a]), f([c_2[a], a])] for a in range(self.n_features_in_)])
        update_factor = np.array([list_c[np.argmin(f_values, axis=1)[d], d] for d in range(self.n_features_in_)])
        return W * update_factor[:, np.newaxis]

    def _prox_concave_feature(self, W, gamma):
        """
        Concave feature proximal operator.

        :param W: Weights matrix.
        :param gamma: Learning rate.
        :returns: Updated weights matrix.
        """
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        poly_a = self.s * S / np.sqrt(self.m)
        poly_b = 1 - poly_a
        poly_c = self.lambda_val * gamma / (2 * np.sqrt(self.m) * S) - 1
        delta = np.abs(poly_b ** 2 - 4 * poly_a * poly_c)  # when delta is negative, this means that the function is
        # striclty increasing, and the argmin is necessarily c_0 which will be the case even with this abs
        c_1 = (-poly_b + np.sqrt(delta)) / (2 * poly_a)
        c_2 = (-poly_b - np.sqrt(delta)) / (2 * poly_a)
        c_0 = [0] * len(S)
        list_c = np.array([c_0, c_1, c_2])
        f = lambda x: ((S[x[1]] ** 2) * (
                1 - x[0]) ** 2) / 2 + self.lambda_val * gamma * np.log(
            1 + np.abs(x[0]) * self.s * S[x[1]] / np.sqrt(self.m)) / (2 * self.s)
        f_values = np.array([[f([c_0[a], a]), f([c_1[a], a]), f([c_2[a], a])] for a in range(len(S))])
        update_factor = np.array([list_c[np.argmin(f_values, axis=1)[d], d] for d in range(len(S))])
        return U @ np.diag(S * update_factor) @ Vh

    def predict(self, X):
        """
        Predict using the ReLUNN model.

        :param X: Test data of shape (n_samples, n_features).
        :return: Predicted values of shape (n_samples,).
        """
        check_is_fitted(self)
        X = check_array(X)
        K = self.K_from_W(self.X_, X, self.W_, self.m)
        y_pred = K @ self.alpha_ + self.y_mean_ - np.mean(self.K_ @ self.alpha_)
        return y_pred

    def _more_tags(self):
        return {'poor_score': True}

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
        if self.reg_type == 'Feature' or self.reg_type == 'Concave_Feature' or self.reg_type == 'Basic':
            U, S, Vh = np.linalg.svd(self.W_, full_matrices=False)
            p_hat = U[:, 0:k]
        elif self.reg_type == 'Variable' or self.reg_type == 'Concave_Variable':
            p_hat = np.zeros((self.n_features_in_, k))
            S = np.linalg.norm(self.W_, axis=1)
            order_variables = np.argsort(S)
            for a in range(k):
                p_hat[order_variables[self.n_features_in_ - a - 1], a] = 1
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
        if self.reg_type == 'Variable':
            importance = np.linalg.norm(self.W_, axis=1)
        elif self.reg_type == 'Feature' or self.reg_type == 'Basic':
            U, S, Vh = np.linalg.svd(self.W_, full_matrices=False)
            importance = S
        elif self.reg_type == 'Concave_Variable':
            importance = np.log(1 + (self.s / np.sqrt(self.m)) * np.linalg.norm(self.W_, axis=1))
        elif self.reg_type == 'Concave_Feature':
            U, S, Vh = np.linalg.svd(self.W_, full_matrices=False)
            importance = np.log(1 + (self.s * S / np.sqrt(self.m)))
        importance = importance / np.sum(importance)
        k_hat = np.sum(importance > 1 / self.n_features_in_)
        if k <= self.n_features_in_ / 2:
            error = np.abs(k_hat - k) / (self.n_features_in_ - k)
        else:
            error = np.abs(k_hat - k) / k
        return 1 - error

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def compute_gradient(alpha, m, n_features_in, n, W, X, lambda_val):
        """
        Compute the gradient of the loss function with regard to the weights, once the kernel problem has been solved

        :param alpha: Coefficients of kernel ridge.
        :param m: Number of particles.
        :param n_features_in: Number of input features.
        :param n: Number of samples.
        :param W: Weights matrix.
        :param X: Training data features.
        :param lambda_val: Regularization parameter.
        :returns: Gradient of the weights.
        """
        gradW = np.zeros(n_features_in * m, dtype='float64')
        Z = np.outer(alpha, alpha)
        sign_dot = np.zeros((n, n), dtype='float64')
        diff = np.zeros((n, n))
        X = X.astype('float64')
        W = W.astype('float64')
        for j in range(m):
            product = np.dot(X, W[:, j][:, np.newaxis])
            for a in prange(n):
                for b in prange(n):
                    sign_dot[a, b] = np.sign(product[a, :] - product[b, :])[0]
            for l in range(n_features_in):
                for c in prange(n):
                    for d in prange(n):
                        diff[c, d] = X[c, l] - X[d, l]
                gradW[j * n_features_in + l] = lambda_val * np.sum(Z * sign_dot * diff) / (4 * m)
        return gradW

    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True)
    def K_from_W(own_X, X, W, m):
        """
        Compute the kernel matrix from weights.

        :param own_X: Training data features.
        :param X: Data features for kernel computation.
        :param W: Weights matrix.
        :param m: Number of particles.
        :returns: Kernel matrix.
        """
        X = X.astype('float64')
        own_X = own_X.astype('float64')
        W = W.astype('float64')
        n, d = own_X.shape
        n_test, _ = X.shape
        K = np.empty((n_test, n))
        for i in prange(n):
            for i_prime in prange(n_test):
                dot_product = np.dot(X[i_prime, :][np.newaxis, :], W) - np.dot(own_X[i, :][np.newaxis, :], W)
                abs_dot_product = np.abs(dot_product) / m
                K[i_prime, i] = -np.sum(abs_dot_product) / 2
        return K
