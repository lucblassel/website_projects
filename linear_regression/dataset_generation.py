import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_diffs(X, y_, coef):
    ymin, ymax = [], []
    for x, y in zip(X, y_):
        ymin.append(min(x * coef, y))
        ymax.append(max(x * coef, y))
    return ymin, ymax

def plot_dataset(X, y, coef, ax):
    ax.scatter(X, y, color='blue')
    ax.plot(sorted(X), [coef * x for x in sorted(X)], linestyle=':', label=f"y = {coef}*x", color='red')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_diffs(X, y, coef, ax):
    ymin, ymax = get_diffs(X, y, coef)
    plot_dataset(X, y, coef, ax)
    ax.vlines(X, ymin, ymax, color='green')

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

    sns.set_context("poster")

    # plot dataset
    fig, ax = plt.subplots(1)
    plot_dataset(X, y, coef, ax)
    plt.show()

    # show square distance
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    plot_diffs(X, y, coef, axes[0])
    plot_diffs(X, y, 3 * coef, axes[1])
    plt.show()

    # generate cost gradient showing
    thetas, costs = list(range(-8, 13)), []
    for theta in thetas: 
        cost = sum(((X * theta) - y) ** 2) / (2 * len(y)) 
        costs.append(cost)

    plt.plot(thetas, costs)
    plt.xticks(thetas)
    plt.xlabel('theta')
    plt.ylabel('Cost')
    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()
    plt.hlines(y=min(costs), xmin=-100, xmax=2, linestyle=':', color='red')
    plt.vlines(x=2, ymin=-100, ymax=min(costs), linestyle=':', color='red')
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.annotate(
        'minimal cost for\n theta=2', xy=(2,min(costs)),
        xytext=(3, 7),
        arrowprops=dict(facecolor='black', shrink=0.1, width=1),
    )
    plt.show()