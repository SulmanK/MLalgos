#----------------- Packages
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

#----------------- Function


def ScatterPlot_LWR(X_train, y_train, X_test, y_test, reg, tau, degree):
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
    y_pred = reg.predict(X_train, y_train, X_test)
    theta = reg.theta

    # R-squared
    rsq = r2_score(y_test, y_pred)

    # Set figure object
    fig = plt.figure(figsize=(20, 10))

    # Create x_axis values
    x_tmp = np.arange(0, 1.1, 0.01)
    x_limit = np.arange(0, 1.1, 0.01)
    x_limit = x_limit.reshape((x_limit.shape[0], 1))

    # Transform x_axis values
    poly = PolynomialFeatures(degree)
    x_limit = poly.fit_transform(x_limit)

    # Predictions
    hypothesis = np.dot(x_limit, theta)

    # Scatter plots
    plt.scatter(X_train, y_train, color='green', label='Training Set', s = 150)
    plt.scatter(X_test, y_test, color='blue', label='Testing Set', s = 150)
    plt.scatter(X_test, y_pred, color='red', label='Prediction Points', s = 150)
    plt.plot(x_tmp, hypothesis, color='purple', label='Predictions')

    # Set figure parameters
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_1$", fontsize=18)
    plt.title('Locally Weighted Regression, Degree = {}, $r^2$ = {:.2f} '.format(
        degree, rsq), fontsize=18)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    # Legend Patches
    green_patch = mpatches.Patch(color='green', label='Training Set')
    blue_patch = mpatches.Patch(color='blue', label='Testing Set')
    red_patch = mpatches.Patch(color='red', label='Predictions')
    purple_patch = mpatches.Patch(
        color='purple', label='Locally Weighted Regressor')

    patches = [green_patch, blue_patch, red_patch, purple_patch]

    legend = plt.legend(handles=patches, loc='upper right',
                        borderaxespad=0., fontsize=14)

    return fig
