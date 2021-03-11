#----------------- Packages
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import rbf_kernel

import cvxopt
import numpy as np
import pandas as pd
import random
#----------------- Function


def kernel_cvxopt(name, x, y, degree=3, sigma=0.5):
    """ 
    Various kernels used for SVM.

    Parameters
    ----------
    name : string
        Kernel method used in the algorithm.

    x : array
        Input array used to perform the transformation.

    degree : int, optional (default: 3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    sigma : float, optional (default: 0.5)
        Parameter for RBF kernel.
        Ignored by all other kernels.
    """

    # Linear
    if name == 'linear':
        tmp = np.dot(x, y.T)

    # Poly
    elif name == 'poly':
        tmp = (1 + np.dot(x, y.T)) ** degree

    # Radial Basis Function
    elif name == 'rbf':
        #dist = np.linalg.norm(x-x)
        #tmp = np.exp(-(dist**2)/(2 * sigma))
        tmp = rbf_kernel(x, x, gamma=sigma)
    return tmp


def kernel_smo(name, x, y, degree=3, sigma=0.5):
    """ 
    Various kernels used for SVM.

    Parameters
    ----------
    name : string
        Kernel method used in the algorithm.

    x : array
        Input array used to perform the transformation.

    degree : int, optional (default: 3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    sigma : float, optional (default: 0.5)
        Parameter for RBF kernel.
        Ignored by all other kernels.
    """

    # Linear
    if name == 'linear':
        tmp = np.inner(x, y)

    # Poly
    elif name == 'poly':
        tmp = (1 + np.inner(x, y)) ** degree

    # Radial Basis Function
    elif name == 'rbf':
        dist = np.linalg.norm(x-x)
        tmp = np.exp(-dist/(2 * sigma))

    return tmp


class mySVM:
    """
    Support vector machine (SVM).

    Parameters
    ----------
    kernel : string, optional (default: 'rbf')
        Specifies the kernel type to be used in the algorithm.
        'linear', 'poly', or 'rbf'.

    method: string
        Specifies the method used for solving the constrainted optimization problem for SVMS.
        'CVXOPT', 'SMO'

    C : float, optional (default: 1)
        Penalty parameter C of the error term.
`
    degree : int, optional (default: 3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    sigma : float, optional (default: 1)
        Parameter for RBF kernel

    Attributes
    ----------
    w : array, shape = [n_features]
        Weights assigned to the features.

    b : float
        Intercept in decision function.
    """

    def __init__(self, kernel_name, method, C=1, degree=3, sigma=0.5, max_passes=1):
        self.kernel_name = kernel_name
        self.method = method
        self.C = C
        self.degree = degree
        self.sigma = sigma
        self.max_passes = max_passes

    def fit(self, X, y):
        """ 
        Fit the training data (X,y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The class labels.


        Returns
        -------
        self : returns a trained SVM
        """

        # Check array structure
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        # Number of training samples
        self.m = (self.X.shape[0])

        # CVXOPT
        if self.method == 'CVXOPT':

            # Reshape Matrix (-1, 1)
            y = y.reshape(-1, 1) * 1.

            # H
            H = (np.dot(y, y.T) * kernel_cvxopt(self.kernel_name,
                                                self.X, self.X, self.degree, self.sigma)) * 1.

            # Converting into cvxopt format
            P = matrix(H)
            q = matrix(-np.ones((self.m, 1)))
            G = matrix(-np.eye(self.m))
            h = matrix(np.zeros(self.m))
            A = matrix(y.reshape(1, -1))
            b = matrix(np.zeros(1))

            # Run solver
            sol = solvers.qp(P, q, G, h, A, b)

            # Alphas
            alphas = np.array(sol['x'])

            # Selecting indices which have support vectors.
            S = (alphas > 1e-4).flatten()

            # Weights
            if self.kernel_name == 'linear':
                w = np.dot((alphas * y).T, self.X).reshape(-1, 1)
                b = y[S] - np.dot(X[S], w)

            else:
                w = np.dot((alphas * y).T, self.X).reshape(-1, 1)
                b = y[S] - np.dot(X[S], w)

            self.b = np.mean(b)
            self.w = w
            self.coef_ = self.w
            self.intercept_ = self.b

        # Sequential Minimization Optimization
        elif self.method == 'SMO':

            # Initialize parameters
            alphas = np.zeros(self.m)
            alphas_old = np.zeros(self.m)
            b = 0
            passes = 0
            tol = 10e-5
            random.seed(1)
            while (passes < self.max_passes):

                # Initialize check for while loop
                num_changed_alphas = 0

                # Iterate through each training sample
                for i in range(0, self.m):

                    # Prediction Error_i
                    # Calculate prediction
                    f_xi = alphas[i] * self.y[i] * kernel_smo(
                        self.kernel_name, self.X[i], self.X, self.degree, self.sigma).sum() + b

                    # Error of prediction
                    E_i = f_xi - self.y[i]

                    # Checks KKT constraints
                    if ((self.y[i] * E_i < -tol) & (alphas[i] < self.C)) | ((self.y[i] * E_i > tol) & (alphas[i] > 0)):
                        # Select a random index that is not i
                        all_indices = [j for j in range(self.m)]
                        not_indices = [p for p in all_indices if p not in [i]]
                        j = random.choice(not_indices)

                        # Prediction Error_j

                        # Prediction_j
                        f_xj = alphas[j] * self.y[j] * kernel_smo(
                            self.kernel_name, self.X[j], self.X, self.degree, self.sigma).sum() + b

                        # Error_j
                        E_j = f_xj - self.y[j]

                        # Set old alphas to new alphas
                        alphas_old[i], alphas_old[j] = alphas[i], alphas[j]

                        # Compute L or H bounds
                        if self.y[i] != self.y[j]:
                            L = max(0, alphas[j] - alphas[i])
                            H = min(self.C, self.C + alphas[j] - alphas[i])

                        elif self.y[i] == self.y[j]:
                            L = max(0, alphas[i] + alphas[j] - self.C)
                            H = min(self.C,  alphas[i] + alphas[j])

                        # Condition to continue to next iteration
                        if L == H:
                            continue

                        # Calculate Nu
                        nu = 2 * (kernel_smo(self.kernel_name, self.X[i], self.X[j], self.degree, self.sigma) - kernel_smo(
                            self.kernel_name, self.X[i], self.X[i], self.degree, self.sigma) - kernel_smo(self.kernel_name, self.X[j], self.X[j], self.degree, self.sigma))

                        # Condition to continue to next iteration
                        if (nu >= 0):
                            continue

                        # Compute alpha_j
                        alphas[j] = alphas[j] - ((self.y[j]*(E_i - E_j))/(nu))

                        # Clip alpha_j
                        if (alphas[j] > H):
                            alphas[j] = H

                        elif (L <= alphas[j] <= H):
                            alphas[j] = alphas[j]

                        elif (alphas[j] < L):
                            alphas[j] = L

                        # Condition to continue to next iteration
                        if (np.abs(alphas[j] - alphas_old[j]) < 10e-5):
                            continue

                        # Compute alpha_i
                        alphas[i] = alphas[i] + \
                            (self.y[i] * self.y[j]) * \
                            (alphas_old[j] - alphas[j])

                        # Compute b1 and b2 using (17) and (18) respectively.
                        b1 = (b - E_i - self.y[i]) * (alphas[i] - alphas_old[i]) * kernel_smo(self.kernel_name, self.X[i], self.X[i], self.degree, self.sigma) - (
                            self.y[j] * (alphas[j] - alphas_old[j])) * kernel_smo(self.kernel_name, self.X[i], self.X[j], self.degree, self.sigma)
                        b2 = (b - E_j - self.y[i]) * (alphas[i] - alphas_old[i]) * kernel_smo(self.kernel_name, self.X[i], self.X[j], self.degree, self.sigma) - (
                            self.y[j] * (alphas[j] - alphas_old[j])) * kernel_smo(self.kernel_name, self.X[j], self.X[j], self.degree, self.sigma)

                        # Compute b using b1 and b2
                        if (0 < alphas[i] < self.C):
                            b = b1

                        elif (0 < alphas[j] < self.C):
                            b = b2

                        else:
                            b = (b1 + b2)/2

                        # Add to our while loop condition
                        num_changed_alphas = num_changed_alphas + 1

                # Check our condition to see if it changed
                if (num_changed_alphas == 0):
                    passes = passes + 1

                else:
                    passes = 0

            # Assign our alphas and intercepts
            self.alphas = alphas
            self.b = b
            self.w = ((self.y * self.alphas).T @ self.X).reshape(-1, 1)
            self.coef_ = self.w
            self.intercept_ = self.b

    def predict(self, X_test):
        """ 
        Predict on the testing data (X,y).

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            The input data.


        Returns
        -------
        labels : returns the predicted labels of the testing set.
        """

        # Check array
        self.X_test = np.asarray(X_test)

        # Create empty labels list
        labels = []

        # Convex Optimization
        if self.method == 'CVXOPT':

            # Iterate through each testing sample
            for x in self.X_test:

                # Prediction
                pred = np.sum(np.dot(x, self.w)) + self.b

                # Assigns a label based on the sign of the prediction
                if pred >= 0:
                    labels.append(1)

                else:
                    labels.append(-1)

            # Create an array
            labels = np.array(labels)

        # Sequential Minimization Optimization
        elif self.method == 'SMO':

            # Iterate through each testing sample
            for x in X_test:

                # Prediction
                pred = np.sum(np.dot(x, self.w)) + self.b

                # Assigns a label based on the sign of the prediction
                if pred >= 0:
                    labels.append(1)

                else:
                    labels.append(-1)

            # Create an array
            labels = np.array(labels)

        return labels
