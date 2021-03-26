#----------------- Packages
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

#----------------- Function


def ScatterPlot_DT(X_train, y_train, X_test, y_test, reg):
    """ 
    Scatterplot of training and testing sets with regression line.

    Parameters
    ----------
    X_train : array
        X array of training set.

    y_train: array
        y array of training set.

    X_test: array
        X array of testing set.

    y_test: array
        y array of testing set.

    degree : integer
        Degree of the polynomial.

    Returns
    -------
    figure : figure object
        Scatter plot.
    """

    # Set model instances
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # R-squared
    rsq = r2_score(y_test, y_pred)

    # Set figure object
    fig = plt.figure(figsize=(20, 10))

    # Scatter plots
    plt.scatter(X_train.values[:, 0], y_train,
                color='green', label='Training Set', s=150)
    plt.scatter(X_test.values[:, 0], y_test,
                color='blue', label='Testing Set', s=150)
    plt.scatter(X_test.values[:, 0], y_pred,
                color='red', label='Prediction Points', s=150)

    # Set figure parameters
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_1$", fontsize=18)
    plt.title('Scatterplot , $r^2$ = {:.2f} '.format(rsq), fontsize=18)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    # Legend Patches
    green_patch = mpatches.Patch(color='green', label='Training Set')
    blue_patch = mpatches.Patch(color='blue', label='Testing Set')
    red_patch = mpatches.Patch(color='red', label='Predictions')

    patches = [green_patch, blue_patch, red_patch]

    legend = plt.legend(handles=patches, loc='upper right',
                        borderaxespad=0., fontsize=14)

    return fig
