#----------------- Packages
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
#----------------- Function


def gaussian_weight(x_i, x, tau):
    """ 
    Gaussian weight follows the gaussian distribution.

    Parameters
    ----------
    x_i : array
        Array to train on, usually the training set

    x : float
        Testing sample, to make predictions

    tau : float
        Bandwidth parameter - distance at how much the weight parameters contribute

    Returns
    -------
    w : array
        Array
        The weight matrix.
    """

    # Create an identity matrix of weights using the size of len of the training set
    m = len(x_i)
    w = np.zeros((m, m))

    # Iterate through each sample in the training set
    for i in range(m):
        # Index each training sample
        xi = x_i[i]

        # Set each index of the weight matrix by using all the training samples for each testing sample
        w[i, i] = np.exp(np.dot((xi-x), (xi-x).T)/(-2 * np.power(tau, 2)))

    return w


class myLocallyWeightedRegression:
    """ 
    Locally weighted regression

    Parameters
    ----------
    tau : float
        Bandwidth parameter - distance at how much the weight parameters contribute

    degree : integer
        Degree of the polynomial.

    Returns
    -------
    y_pred : array
        Predicted labels.
    """

    def __init__(self, tau, degree):
        self.tau = tau
        self.degree = degree

    def predict(self, X_train, y_train, X_test):
        """ 
        Prediction function.

        Parameters
        ----------
        X_train : array
            X array of training set.

        y_train: array
            y array of training set.

        X_test: array
            X array of testing set.

        Returns
        -------
        y_pred : array
            Predicted labels.
        """

        # Reshape arrays
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

        # Set the number of degrees for polynomial transformation
        poly = PolynomialFeatures(self.degree)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)

        # Predicted Label list
        y_pred = []

        # Theta list
        average_theta = []
        # Iterate through each sample in the testing set
        for x in X_test:

            # Initialize gaussian weights
            w = gaussian_weight(x_i=X_train, x=x, tau=self.tau)

            # Closed form solution of Theta
            theta = np.dot(
                np.linalg.pinv(
                    np.dot(
                        np.dot(X_train.T, w),
                        X_train)
                ),
                np.dot(
                    np.dot(X_train.T, w), y_train)
            )

            # Average theta
            average_theta.append(theta)

            # Make prediction using optimal theta
            hypothesis = np.dot(x, theta)
            y_pred.append(hypothesis)

        # Set theta
        average_theta = np.array(average_theta)
        average_theta = np.mean(average_theta, axis=0)
        self.theta = average_theta

        # Convert predicted label list into an array and select subindex
        y_pred = np.array(y_pred)
        tmp = []
        for x in y_pred.flatten():
            tmp.append(x)

        y_pred = np.array(tmp)
        return y_pred
