import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


def split_dataset(X, y, train_frac=0.8):
    """Splits dataset into training and testing sets"""
    index = np.random.choice(len(y), int(len(y) * train_frac))
    X_train, y_train = X[index], y[index]
    X_test, y_test = np.delete(X, index, 0), np.delete(y, index, 0)
    return X_train, y_train, X_test, y_test


def normalize(X, mu=None, sigma=None):
    """Normalizing and centering features and adds bias feature"""
    if mu is None or sigma is None:
        mu, sigma = X.mean(axis=0), X.std(axis=0)
    normalized = (X - mu) / sigma
    return np.concatenate((np.ones((len(X), 1)), normalized), axis=1), mu, sigma


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def compute_cost(theta, X, y, lambda_):
    """regularized cost function of logistic regression"""
    h = sigmoid(X @ theta)
    l2 = lambda_ * sum(theta[1:] ** 2) / (2 * len(y))
    return (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / len(y) + l2


def compute_gradient(theta, X, y, lambda_):
    """regularized gradient of cost function for logistic regression"""
    h = sigmoid(X @ theta)
    l2 = np.append([0], ((lambda_ * theta[1:]) / len(y)))
    return (X.T @ (h - y)) / len(y) + l2


def gradient_descent(theta, X, y, lambda_, max_iters=400):
    """Finds optimal theta"""
    return minimize(
        compute_cost,
        theta,
        args=(X, y, lambda_),
        jac=compute_gradient,
        options={"maxiter": max_iters},
    ).x


def map_features(X_1, X_2, degree=6):
    """All polynomial features from a pair of features"""
    out = np.ones(X_1.shape)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.concatenate((out, (X_1 ** (i - j)) * (X_2 ** j)), axis=1)
    return out


def predict(X, theta):
    """Predictions of trained logistic regression"""
    return np.array([1 if x >= 0.5 else 0 for x in sigmoid(X @ theta)])


def classification_accuracy(real, predictions):
    return sum(real == predictions) / len(real)


def main():
    np.warnings.filterwarnings("ignore")
    print("loading and normalizing dataset")
    X, y = load_breast_cancer(return_X_y=True)

    X_train, y_train, X_test, y_test = split_dataset(X, y)

    X_train_norm, mu, sigma = normalize(X_train)
    X_test_norm, _, _ = normalize(X_test, mu, sigma)

    print("Training our model")
    theta_init = np.zeros((X_train_norm.shape[1],))
    theta_learned = gradient_descent(theta_init, X_train_norm, y_train, 1)
    y_pred = predict(X_test_norm, theta_learned)

    print("training scikit learn model")
    clf = LogisticRegression(penalty="l2", fit_intercept=False, C=1)
    clf.fit(X_train_norm, y_train)
    y_pred_clf = clf.predict(X_test_norm)

    print(
        "Our model's classification accuracy on the breast cancer dataset: "
        f"{classification_accuracy(y_test, y_pred) * 100:.2f}%"
    )
    print(
        "Scikit's model's classification accuracy on the breast cancer dataset: "
        f"{classification_accuracy(y_test, y_pred_clf) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
