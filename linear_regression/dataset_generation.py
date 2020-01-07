import numpy as np
import matplotlib.pyplot as plt

def get_diffs(X, y_, coef):
    ymin, ymax = [], []
    for x, y in zip(X, y_):
        ymin.append(min(x * coef, y))
        ymax.append(max(x * coef, y))
    return ymin, ymax

def plot_dataset(X, y, coef):
    plt.scatter(X, y)
    plt.plot(sorted(X), [coef * x for x in sorted(X)], linestyle=':', label=f"y = {coef}*x")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

def plot_diffs(X, y, coef):
    ymin, ymax = get_diffs(X, y, coef)
    plt.vlines(X, ymin, ymax, color='green')
    plot_dataset(X, y, coef)

if __name__ == "__main__":
    X = np.array(
        [0.25464312,  0.81562694, -1.64063088, -0.46683864,  2.09938726,
         0.26128934, -1.56805687, -0.89657469,  0.22647514,  1.2417255,
         -0.75966191, -0.20179728,  0.48668321, -0.53070636, -0.57584333])
    y = np.array(
        [0.52147008,  2.14693935, -3.4493487, -1.55644601,  3.9586241,
         0.72466665, -3.90762308, -2.44370217,  0.15379182,  2.91402644,
         -2.67569134, -0.38408903,  2.02038201, -1.5327161, -1.55876381])

    coef = 2

    