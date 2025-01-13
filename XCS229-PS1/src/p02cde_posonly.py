import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    _, y_train = util.load_dataset(train_path, add_intercept=True)

    x_valid, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    _, y_valid = util.load_dataset(valid_path, add_intercept=True)

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    _, y_test = util.load_dataset(test_path, add_intercept=True)

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    clf_c = LogisticRegression()
    clf_c.fit(x_train, t_train)
    t = clf_c.predict(x_test)
    np.savetxt(pred_path_c, t)

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    clf_c = LogisticRegression()
    clf_c.fit(x_train, y_train)
    y = clf_c.predict(x_test)
    np.savetxt(pred_path_d, y)

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    def sigmoid(x, theta):
        return 1 / (1 + np.exp(-np.dot(x, theta)))
    alpha = np.mean((sigmoid(x_valid, clf_c.theta))[y_valid == 1])
    t_e = sigmoid(x_test, clf_c.theta) / alpha
    np.savetxt(pred_path_e, t_e >= 0.5)
    # *** END CODER HERE
