#----------------- Packages
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np

#----------------- Function
def plot_cost_gd_lr(X_train, y_train, clf, method, iterations):
    """
    Iteration vs Cost 

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
            The input data.

    y_train : array-like, shape (n_samples, n_features)
            The input data.

    clf : class object
            Classifier of my logistic regression model

    method : string
            Method of the LR solver

    iterations : integer
            Number of iterations for the solver.


    Returns
    -------
    Figure object.
    """

    # Model
    clf.fit(X_train, y_train)
    cost = clf.total_cost

    # Create figure
    fig = plt.figure(figsize=(20, 10))

    if method == 'BGD':
        title = 'Batch Gradient Descent'
        patches = mpatches.Patch(color='green', label=method)

    elif method == 'SGD':
        title = 'Stochastic Gradient Descent'
        patches = mpatches.Patch(color='maroon', label=method)

    # Create histograms of the true and predicted values
    plt.plot(range(0, iterations, 1), cost, color='green',
             linewidth=1, linestyle='dashed', marker='o', label=method)

    # Figure parameters
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    # Set titles
    plt.xlabel("Iterations", fontsize=18)
    plt.ylabel("Cost (J)", fontsize=18)

    # Append the MSE and Rsq values to title
    plt.title(title, fontsize=20)

    return fig
