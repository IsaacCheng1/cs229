import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y = clf.predict(x_eval)
    np.save(pred_path, y)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi = np.sum(y) / m
        mu_0 = np.dot(x.T, y - 1) / np.sum(y - 1)
        mu_1 = np.dot(x.T, y) / np.sum(y)

        y = np.reshape(y, (-1, 1))
        mu_0 = np.reshape(mu_0, (-1, 1))
        mu_1 = np.reshape(mu_1,(-1, 1))
        mu = np.dot(mu_0, (1 - y).T) + np.dot(mu_1, y.T)
        sigma = np.dot(x.T - mu, (x.T - mu).T) / m

        sigma_inv = np.linalg.inv(sigma)
        self.theta = np.dot(sigma_inv, mu_1 - mu_0)
        theta_0 = 0.5 * (mu_0.T @ sigma_inv @ mu_0 - mu_1.T @ sigma_inv @ mu_1) - np.log((1 - phi) / phi)
        self.theta = np.insert(self.theta, 0, theta_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.dot(util.add_intercept(x), self.theta) >= 0
        # *** END CODE HERE
