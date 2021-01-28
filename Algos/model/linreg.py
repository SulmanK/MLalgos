#----------------- Packages
import numpy as np
import random

from model.cost import costfunction

#----------------- Function
"""Implementation of the Linear regression regressor either uses the closed-formed solution (Normal)
 or gradient-descent methods (Batch, Stochastic, or Minibatch). """


class myLinearRegression:

    def __init__(self, method, regularization, lambda_, learning_rate, iterations, batch_size):
        self.method = method
        self.regularization = regularization
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        # Fit function
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

        # Concatenate ones to the training set
        x_ones = np.ones((self.X_train.shape[0], 1))
        self.X_train = np.concatenate((x_ones, self.X_train), axis=1)

        # Reshape y_train to m x 1
        self.y_train = self.y_train.reshape((self.y_train.shape[0], 1))

        # Normal Equations
        # Normal equation and no regularization
        if (self.method == 'Normal') & (self.regularization == None):

            # Calculate theta using normal equation
            theta = np.dot(np.linalg.inv(np.dot(np.transpose(self.X_train), self.X_train)),
                           np.dot(np.transpose(self.X_train), self.y_train))

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]

        # Normal equation with L1 regularization
        elif (self.method == 'Normal') & (self.regularization == 'L1 (Lasso)'):
            print("No closed form exists.")

        # Normal equation with L2 regularization
        elif (self.method == 'Normal') & (self.regularization == 'L2 (Ridge)'):

            # Calculate theta using normal equation
            theta = np.dot(np.linalg.inv(np.dot(np.transpose(self.X_train), self.X_train) + self.lambda_ * (np.eye(self.X_train.shape[1]))),
                           np.dot(np.transpose(self.X_train), self.y_train))

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]

        # Batch Gradient Descent

        # BGD and no regularization
        elif (self.method == 'BGD') & (self.regularization == None):

            # Set learning rate
            alpha = self.learning_rate

            # Number of samples
            m = len(self.y_train)

            # Number of iterations for convergence
            iterations_ = self.iterations

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Total Cost
            total_cost = []

            # Iterate through the number of iterations and calculate the cost function for each theta
            for num_ in range(iterations_):

                # Calculate gradient
                hypothesis = np.dot(self.X_train, theta)
                loss = hypothesis - self.y_train
                gradient = np.dot(self.X_train.T, loss)
                theta = theta - (alpha/m)*gradient

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)
               # print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.total_cost = total_cost

        # BGD with L1 regularization
        elif (self.method == 'BGD') & (self.regularization == 'L1 (Lasso)'):
            # Set learning rate
            alpha = self.learning_rate

            # Number of samples
            m = len(self.y_train)

            # Number of iterations for convergence
            iterations_ = self.iterations

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Total Cost
            total_cost = []

            # Iterate through the number of iterations and calculate the cost function for each theta
            for num_ in range(iterations_):

                # Calculate gradient
                hypothesis = np.dot(self.X_train, theta)
                loss = hypothesis - self.y_train
                gradient = np.dot(self.X_train.T, loss)
                reg = (self.lambda_ * np.abs(theta[1::]))
                tmp = np.concatenate(
                    (gradient[0].reshape((1, 1)), gradient[1::]+reg), axis=0)
                theta = theta - (alpha/m)*(tmp)

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)

               # print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.total_cost = total_cost

        # BGD with l2 regularization
        elif (self.method == 'BGD') & (self.regularization == 'L2 (Ridge)'):

            # Set learning rate
            alpha = self.learning_rate

            # Number of samples
            m = len(self.y_train)

            # Number of iterations for convergence
            iterations_ = self.iterations

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Total Cost
            total_cost = []

            # Iterate through the number of iterations and calculate the cost function for each theta
            for num_ in range(iterations_):

                # Calculate gradient
                hypothesis = np.dot(self.X_train, theta)
                loss = hypothesis - self.y_train
                gradient = np.dot(self.X_train.T, loss)
                reg = (self.lambda_ * np.power(theta[1::], 2))
                tmp = np.concatenate(
                    (gradient[0].reshape((1, 1)), gradient[1::]+reg), axis=0)
                theta = theta - (alpha/m)*(tmp)

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)

               # print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.total_cost = total_cost

        # Stochastic Gradient Descent

        # SGD with no regularization
        elif (self.method == 'SGD') & (self.regularization == None):

            # Set learning rate
            alpha = self.learning_rate

            # Number of samples
            m = len(self.y_train)

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Total Cost
            total_cost = []

            # Iterate through the number of training samples and calculate the cost function for each theta
            for num_ in range(m):

                # Select random training samples from index 0 to number of samples - 1
                i = random.randint(0, m-1)
                x = self.X_train[i]
                y = self.y_train[i]
                x = x.reshape((1, x.shape[0]))
                y = y.reshape((y.shape[0], 1))

                # Calculate gradient
                hypothesis = np.dot(x, theta)
                loss = hypothesis - y
                gradient = np.dot(x.T, loss)
                theta = theta - (alpha)*gradient

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)

               # print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.total_cost = total_cost

        # SGD with L1 regularization
        elif (self.method == 'SGD') & (self.regularization == 'L1 (Lasso)'):

            # Set learning rate
            alpha = self.learning_rate

            # Number of samples
            m = len(self.y_train)

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Total Cost
            total_cost = []

            # Iterate through the number of training samples and calculate the cost function for each theta
            for num_ in range(m):

                # Select random training samples from index 0 to number of samples - 1
                i = random.randint(0, m-1)
                x = self.X_train[i]
                y = self.y_train[i]
                x = x.reshape((1, x.shape[0]))
                y = y.reshape((y.shape[0], 1))

                # Calculate gradient
                hypothesis = np.dot(x, theta)
                loss = hypothesis - y
                gradient = np.dot(x.T, loss)
                reg = (self.lambda_ * np.abs(theta[1::]))
                tmp = np.concatenate(
                    (gradient[0].reshape((1, 1)), gradient[1::]+reg), axis=0)
                theta = theta - (alpha)*(tmp)

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)

               # print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.total_cost = total_cost

        # SGD with L2 regularization
        elif (self.method == 'SGD') & (self.regularization == 'L2 (Ridge)'):

            # Set learning rate
            alpha = self.learning_rate

            # Number of samples
            m = len(self.y_train)

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Total Cost
            total_cost = []

            # Iterate through the number of training samples and calculate the cost function for each theta
            for num_ in range(m):

                # Select random training samples from index 0 to number of samples - 1
                i = random.randint(0, m-1)
                x = self.X_train[i]
                y = self.y_train[i]
                x = x.reshape((1, x.shape[0]))
                y = y.reshape((y.shape[0], 1))

                # Calculate gradient
                hypothesis = np.dot(x, theta)
                loss = hypothesis - y
                gradient = np.dot(x.T, loss)
                reg = self.lambda_ * (np.power(theta[1::], 2))
                tmp = np.concatenate(
                    (gradient[0].reshape((1, 1)), gradient[1::]+reg), axis=0)
                theta = theta - (alpha)*(tmp)

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)

              #  print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.cost = total_cost

        # Minibatch Gradient Descent

        # MBG with no regularization
        elif (self.method == 'MBGD') & (self.regularization == None):

            # Learning rate
            alpha = self.learning_rate

            # Mini_batch size
            batch_size_ = self.batch_size

            # Number of iterations
            iterations_ = self.iterations

            # Number of samples
            m = len(self.y_train)

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Cost List
            total_cost = []

            # Iterate through the number of iterations and calculate the cost function for each theta
            for num_ in range(iterations_):

                # Creates new X,y data using the set minibatch size
                indices = np.random.randint(
                    low=0, high=m - 1, size=(batch_size_))
                x = self.X_train[indices]
                y = self.y_train[indices]

                # Calculate gradient
                hypothesis = np.dot(x, theta)
                loss = hypothesis - y
                gradient = np.dot(x.T, loss)
                theta = theta - (alpha/batch_size_)*gradient

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)

               # print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.total_cost = total_cost

        # MBGD with L1 regularization
        elif (self.method == 'MBGD') & (self.regularization == 'L1 (Lasso)'):

            # Learning rate
            alpha = self.learning_rate

            # Mini_batch size
            batch_size_ = self.batch_size

            # Number of iterations
            iterations_ = self.iterations

            # Number of samples
            m = len(self.y_train)

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Total Cost
            total_cost = []

            # Iterate through the number of iterations and calculate the cost function for each theta
            for num_ in range(iterations_):

                # Creates new X,y data using the set minibatch size
                indices = np.random.randint(
                    low=0, high=m - 1, size=(batch_size_))
                x = self.X_train[indices]
                y = self.y_train[indices]

                # Calculate gradient
                hypothesis = np.dot(x, theta)
                loss = hypothesis - y
                gradient = np.dot(x.T, loss)
                reg = (self.lambda_ * np.abs(theta[1::]))
                tmp = np.concatenate(
                    (gradient[0].reshape((1, 1)), gradient[1::]+reg), axis=0)
                theta = theta - (alpha/m)*(tmp)

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)

               # print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.total_cost = total_cost

        # MBGD with l2 regularization
        elif (self.method == 'MBGD') & (self.regularization == 'L2 (Ridge)'):

            # Learning rate
            alpha = self.learning_rate

            # Mini_batch size
            batch_size_ = self.batch_size

            # Number of iterations
            iterations_ = self.iterations

            # Number of samples
            m = len(self.y_train)

            # Reshape theta vector
            theta = np.zeros((self.X_train.shape[1], 1))

            # Total Cost
            total_cost = []

            # Iterate through the number of iterations and calculate the cost function for each theta
            for num_ in range(iterations_):

                # Creates new X,y data using the set minibatch size
                indices = np.random.randint(
                    low=0, high=m - 1, size=(batch_size_))
                x = self.X_train[indices]
                y = self.y_train[indices]

                # Calculate gradient
                hypothesis = np.dot(x, theta)
                loss = hypothesis - y
                gradient = np.dot(x.T, loss)
                reg = (self.lambda_ * np.power(theta[1::], 2))
                tmp = np.concatenate(
                    (gradient[0].reshape((1, 1)), gradient[1::]+reg), axis=0)
                theta = theta - (alpha/m)*(tmp)

                # Use theta to calculate the cost
                cost = costfunction(X=self.X_train, y=self.y_train,
                                    theta=theta, lambda_=self.lambda_,
                                    regularization=self.regularization)

                total_cost.append(cost)

               # print('Cost:', cost)

            # Set the slope as the first theta value and intercept as second.
            self.intercept_ = theta[0]
            self.coef_ = theta[1::]
            self.total_cost = total_cost

        return None

    def predict(self, X_test):
        """Returns the predicted values using testing set"""

        # Return the predicted values
        y_pred_values = []
        self.X_test = np.asarray(X_test)

        # Iterate through each testing sample
        for x in self.X_test:
            # Calculate y_pred = mX + B
            y_pred = np.dot(self.coef_.T, x) + self.intercept_
            y_pred_values.append(y_pred)

        y_pred_values = np.array(y_pred_values)
        return y_pred_values.reshape((y_pred_values.shape[0], 1))

    def mse_score(self, y_true, y_pred):
        """Returns MSE score"""

        # Calculate Mean Squared Error using true and predicted values
        y_true = y_true.reshape((y_true.shape[0]), 1)
        n = len(y_true)
        MSE = (1/n) * (np.sum((y_true - y_pred)**2, axis=0))
        return MSE[0]

    def rsquared_score(self, y_true, y_pred):
        """Returns R-squared"""

        # Calculate Rsquared using true and predicted values
        y_true = y_true.reshape((y_true.shape[0]), 1)
        mean_ytrue = np.mean(y_true)
        ssres = np.sum((y_true - y_pred)**2, axis=0)
        sstot = np.sum((y_true - mean_ytrue)**2, axis=0)
        rsquared = 1 - (ssres/sstot)
        return rsquared[0]
