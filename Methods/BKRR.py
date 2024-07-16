from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class BKRR(RegressorMixin, BaseEstimator):
    """
    BKRR Regressor: A kernel-based regressor with a simple kernel k(x, x_prime) = (|x| + |x_prime| -|x - x_prime|)/2,
    where |x| is the euclidean norm of the vector x.
    """

    def __init__(self, lambda_val=0.00125):
        """
        Initialize the BKRR regressor.

        :param lambda_val: Regularization parameter. Must be non-negative.
        """
        if lambda_val < 0:
            raise ValueError("lambda_val must be non-negative.")
        self.lambda_val = lambda_val

    def fit(self, X, y):
        """
        Fit the BKRR model according to the given training data.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Target values of shape (n_samples,).
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

        # Center the output
        self.y_mean_ = np.mean(self.y_)
        self.y_norm_ = self.y_ - self.y_mean_

        # Compute the kernel matrix
        self.K_ = self.K_matrix(self.X_)
        K_norm = (np.eye(self.n_) - np.ones((self.n_, self.n_)) / self.n_) @ self.K_ @ (
                np.eye(self.n_) - np.ones((self.n_, self.n_)) / self.n_)

        # Solve for alpha
        self.alpha_ = np.linalg.solve(
            K_norm + self.n_ * self.lambda_val * np.identity(self.n_), self.y_norm_)
        self.is_fitted_ = True
        return self

    def K_matrix(self, other_X):
        """
        Compute the kernel matrix between the training data and other data.

        :param other_X: Other data of shape (n_test, self.n_features_in).
        :return: Kernel matrix of shape (n_test, self.n_).
        """
        n_test, _ = other_X.shape
        K = np.empty((n_test, self.n_))
        for i in range(self.n_):
            for i_prime in range(n_test):
                K[i_prime, i] = -np.linalg.norm(self.X_[i, :] - other_X[i_prime, :]) / 2
        return K

    def predict(self, X):
        """
        Predict using the BKRR model.

        :param X: Test data of shape (n_samples, n_features).
        :return: Predicted values of shape (n_samples,).
        """
        check_is_fitted(self)
        X = check_array(X)
        K = self.K_matrix(X)
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
