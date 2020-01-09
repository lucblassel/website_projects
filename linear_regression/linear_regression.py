import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

sns.set_context('poster')

def split_dataset(X, y, train_frac=0.8):
    index = np.random.choice(len(y), int(len(y) * train_frac))
    X_train, y_train = X[index], y[index]
    X_test, y_test = np.delete(X, index, 0), np.delete(y, index, 0)
    return X_train, y_train, X_test, y_test

def normalize(X, mu=None, sigma=None):
    if mu is None or sigma is None:
        mu, sigma = X.mean(axis=0), X.std(axis=0)
    return ((X - mu) / sigma), mu, sigma


def compute_cost(theta, X, y, lambda_):
    l2 = lambda_ * sum(theta[1:] ** 2) / (2 * len(y))
    return sum((X @ theta - y) ** 2) / (2 * len(y)) + l2


def compute_gradient(theta, X, y, lambda_):
    l2 = np.append([0], ((lambda_ * theta[1:]) / len(y)))
    return (X.T @ (X @ theta - y)) / len(y) + l2


def gradient_descent(X, y, theta, alpha, max_iters, lambda_=0):
    history = []
    for _ in range(max_iters):
        theta = theta - alpha * compute_gradient(theta, X, y, lambda_)
        history.append(compute_cost(theta, X, y, lambda_))
    return theta, history

def predict(X, theta):
    return X @ theta

def RMSE(y_pred, y_true):
    return np.sqrt(
        sum((y_pred - y_true) ** 2) / len(y_true)
        )

if __name__ == "__main__":

    print("loading data")
    X, y = load_boston(return_X_y=True)

    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # feature normalize
    X_train_norm, mu, sigma = normalize(X_train)
    X_test_norm, _, _ = normalize(X_test, mu, sigma)

    # add bias feature
    X_train_norm = np.append(
        np.ones((len(X_train_norm), 1)), X_train_norm, axis=1)
    X_test_norm = np.append(
        np.ones((len(X_test_norm), 1)), X_test_norm, axis=1)

    print("Choosing the correct learning rate")
    # choosing the right learning parameter
    alphas = [1e-4, 4e-4, 1e-3, 4e-3, 1e-2, 4e-2, 1e-1]
    histories = {}
    for alpha in alphas:
        theta_init = np.zeros((len(X_train_norm[0]),))
        theta_learned, cost_history = gradient_descent(
            X_train_norm, y_train, theta_init, alpha, 400)
        histories[alpha] = cost_history

    print("best alpha = 0.1")
    for alpha, history in histories.items():
        plt.plot(history, label=alpha)
    plt.title('Learning curve depending on learning rate alpha')
    plt.xlabel('nb. of iterations')
    plt.ylabel('cost C')
    plt.legend(loc='upper right', title='alpha:')
    plt.show()

    print("Fitting regression")
    # according to the plot alpha = 0.1 is the best
    theta_init = np.zeros((len(X_train_norm[0]),))
    theta_learned, cost_history = gradient_descent(X_train_norm, y_train, theta_init, 0.1, 1000)

    y_pred = predict(X_test_norm, theta_learned)
    
    reg = LinearRegression()
    reg.fit(X_train_norm[:, 1:], y_train)
    preds_sklearn = reg.predict(X_test_norm[:, 1:])

    print(f"Our model RMSE on test set: {RMSE(y_test, y_pred)}")
    print(f"Scikit-model RMSE on test set: {RMSE(y_test, preds_sklearn)}")

    # Regression plot
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot(sorted(y_test), sorted(y_test), linestyle=':', color='red', label="y = x")
    plt.legend()
    plt.xlabel("true value")
    plt.ylabel("predicted value")
    plt.show()







