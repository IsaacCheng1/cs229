import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape()

        x_test = np.reshape(x, (m, -1, n))                        # (m, 1, n)
        x_diff = self.x - x_test                                  # (m, l, n) assuming that self.x.shape == (l, n)
        x_diff_norm = np.linalg.norm(x_diff, ord=2, axis=2)       # (m, l)
        w = np.exp(- np.square(x_diff_norm) / (2 * self.tau**2))  # (m, l)
        w = np.apply_along_axis(np.diag, 1, w)                    # (m, l, l)

        theta = np.linalg.inv(self.x.T @ w @ self.x) @ self.x.T @ w @ self.y # (m, n)

        prediction = np.einsum('ij,ij->i', x, theta)

        return prediction
        # *** END CODE HERE ***
