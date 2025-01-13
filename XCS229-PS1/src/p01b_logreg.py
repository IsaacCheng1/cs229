import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y = clf.predict(x_eval)
    np.save(pred_path, y)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        m, n = x.shape

        # initialize theta as 0 vector
        if self.theta is None:
            self.theta = np.zeros(n)

        while True:
            g = sigmoid(x @ self.theta)                          # (m,)
            gradient = (x.T @ (g - y)) * (1 / m)                 # (n,)
            g = g.reshape((-1, 1))
            hessian = x.T @ (g * (1 - g) * x) * (1 / m)          # n * n

            theta = self.theta - np.linalg.inv(hessian) @ gradient

            if np.linalg.norm(theta - self.theta, ord = 1) < self.eps:
                self.theta = theta
                break

            self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y = x @ self.theta
        y = np.where(y < 0, 0, 1)
        return y
        # *** END CODE HERE ***
