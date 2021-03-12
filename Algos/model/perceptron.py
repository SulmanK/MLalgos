import numpy as np
import pandas as pd
import random
class myPerceptron:
    """
    Support vector machine (SVM).

    Parameters
    ----------

    method: string
        Specifies the method used for solving update rule for perceptron.
        'SGD', 'SAGD'

    update_rule: string
        Specifies the update rule used.
        'Original', 'Gradient'
`
    learning_rate : float
        Learning rate used for converging to the global minimum.


    Attributes
    ----------
    w : array, shape = [n_features]
        Weights assigned to the features.

    b : float
        Intercept in decision function.
    """

    def __init__(self, method, update_rule, learning_rate=None):
        self.method = method
        self.update = update_rule
        self.lr = learning_rate

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

        # Check X and y datatypes
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        # Assign the number of classes based on unique values of y set
        self.classes = len(np.unique(self.y))

        # Set seed
        random.seed(1)
        # Binary Classification
        if self.classes == 2:

            # Initialize weights by (n + 1 x 1)
            theta = np.zeros((self.X.shape[1] + 1, 1))

            # Number of rows in training set
            m = len(self.y)

            # Fit
            # Variable to collect all weights for SAGD
            total_theta = []

            # Iterate through the number of samples
            for num_ in range(m):

                # Select samples at random
                i = random.randint(0, m-1)
                x = self.X[i]
                y = self.y[i]

                # Add bias to X vector
                x = x.reshape((1, x.shape[0]))
                x_ones = np.ones((x.shape[0], 1))
                x = np.concatenate((x_ones, x), axis=1)

                # Dot product between X @ Theta
                hypothesis = np.dot(x, theta)

                # Positive Label
                if hypothesis >= 0:
                    y_pred = 1

                    # Update on mistake (predicted y is not equal to true y)
                    if y_pred != y:

                        # Original Rosenblatt update rule
                        if self.update == 'Original':
                            x_reshaped = x.reshape((x.shape[1], 1))
                            theta = theta - x_reshaped

                        # Gradient Descent update rule
                        elif self.update == 'Gradient':
                            x_reshaped = x.reshape((x.shape[1], 1))
                            theta = theta + self.lr * (y - y_pred)*x_reshaped

                # Negative Label
                elif hypothesis < 0:
                    y_pred = 0

                    # Update on mistake (predicted y does not equal true y)
                    if y_pred != y:

                        # Original Rosenblatt update rule
                        if self.update == 'Original':
                            x_reshaped = x.reshape((x.shape[1], 1))
                            theta = theta + x_reshaped

                        # Gradient Descent update rule
                        elif self.update == 'Gradient':
                            x_reshaped = x.reshape((x.shape[1], 1))
                            theta = theta + self.lr * (y - y_pred)*x_reshaped

                total_theta.append(theta)

            # Original
            self.theta = theta
            self.coef_ = self.theta[1::]
            self.intercept_ = self.theta[0]

            # Select SGD or SAGD
            if self.method == 'SGD':
                self.theta = theta
                self.coef_ = self.theta[1::]
                self.intercept_ = self.theta[0]

            elif self.method == 'SAGD':
                total_theta = np.array(total_theta)
                total_theta = np.sum(total_theta, axis=2)
                total_theta = np.sum(total_theta, axis=0)
                total_theta = total_theta / m
                self.theta = total_theta
                self.coef_ = self.theta[1::]
                self.intercept_ = self.theta[0]

        # Multiclass Classification
        elif self.classes > 2:

            # Create theta vector with shape of (classes, n + 1)
            theta = np.zeros((self.classes, self.X.shape[1] + 1))

            # Create new labeled data for OVR classification (Assign 0 or 1 for each class)
            tmp_df = pd.DataFrame(data=self.y, columns=['original'])
            for class_ in range(self.classes):
                tmp_df[str(class_)] = tmp_df['original'].apply(
                    lambda x: 1 if x == class_ else 0)
                tmp_df[str(class_)] = pd.to_numeric(tmp_df[str(class_)])

            # Number of samples in training set
            m = len(self.y)

            # Fit
            # Variable to collect all weights for SAGD
            tmp_dict = {k: [] for k in range(self.classes)}

            # Iterate through each sample in the training set
            for num_ in range(m):
                # Select samples at random
                i = random.randint(0, m-1)
                x = self.X[i]
                y = self.y[i]

                # True Label
                label = tmp_df.iloc[i]

                # Add bias vector to X
                x = x.reshape((1, x.shape[0]))
                x_ones = np.ones((x.shape[0], 1))
                x = np.concatenate((x_ones, x), axis=1)

                # Iterate through each class to get coefficients
                for idx, y_label in zip(range(0, self.classes), label.values[1::]):
                    total_theta = []
                    # Assign true label y
                    y = y_label

                    # Dot product X @ Theta
                    hypothesis = np.dot(x, theta[idx, :])

                    # Positive Label
                    if hypothesis >= 0:
                        y_pred = 1

                        # Update on mistake (predicted y does not equal true y)
                        if y_pred != y:

                            # Original Rosenblatt update rule
                            if self.update == 'Original':
                                x_reshaped = x.reshape((x.shape[1]))
                                theta[idx, :] = theta[idx, :] - x_reshaped

                            # Gradient Descent update rule
                            elif self.update == 'Gradient':
                                x_reshaped = x.reshape((x.shape[1]))
                                theta[idx, :] = theta[idx, :] + \
                                    self.lr * (y - y_pred)*x_reshaped

                    # Negative Label
                    elif hypothesis < 0:
                        y_pred = 0

                        # Update on mistake (predicted y does not equal true y)
                        if y_pred != y:

                            # Original Rosenblatt update rule
                            if self.update == 'Original':
                                x_reshaped = x.reshape((x.shape[1]))
                                theta[idx, :] = theta[idx, :] + x_reshaped

                            # Gradient Descent update rule
                            elif self.update == 'Gradient':
                                x_reshaped = x.reshape((x.shape[1]))
                                theta[idx, :] = theta[idx, :] + \
                                    self.lr * (y - y_pred)*x_reshaped

                    # Append values to weight dictionary
                    tmp_dict[idx].append(theta[idx, :])

            # Original
            self.theta = theta
            self.coef_ = self.theta[:, 1::]
            self.intercept_ = self.theta[:, 0]

            # Select SGD or SAGD
            if self.method == 'SGD':
                self.theta = theta
                self.coef_ = self.theta[:, 1::]
                self.intercept_ = self.theta[:, 0]

            elif self.method == 'SAGD':
                total_theta = []

                # Calculate the Average weights of each class
                for k, v in zip(tmp_dict.keys(), tmp_dict.values()):
                    tmp = np.sum(tmp_dict[k], axis=0)/m
                    total_theta.append(tmp)

                total_theta = np.array(total_theta)
                self.theta = total_theta
                self.coef_ = self.theta[:, 1::]
                self.intercept_ = self.theta[:, 0]

    def predict(self, X):
        """ 
        Makes predictions on the testing data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.


        Returns
        -------
        labels: an array of predicted labels
        """

        # Check arrays
        self.X = np.asarray(X)

        # Binary Classification
        if self.classes == 2:

            # Predict List
            y_pred = []

            # Iterate through each sample in testing set
            for x in self.X:

                # Add bias term to X vector
                x = x.reshape((1, x.shape[0]))
                x_ones = np.ones((x.shape[0], 1))
                x = np.concatenate((x_ones, x), axis=1)

                # Dot product X @ Theta
                hypothesis = np.dot(x, self.theta)

                # Positive Labels
                if hypothesis >= 0:
                    y_pred.append(1)

                # Negative Labels
                else:
                    y_pred.append(0)

            # Array of predicted y values
            label = np.array(y_pred)

        # Multiclass Classification
        if self.classes > 2:

            # Predict List
            label = []

            # Iterate through each sample in testing set
            for x in self.X:

                # List to store hypothesis values for each class
                tmp_list = []

                # Add bias term to X vector
                x = x.reshape((1, x.shape[0]))
                x_ones = np.ones((x.shape[0], 1))
                x = np.concatenate((x_ones, x), axis=1)

                # Iterate through each class
                for idx in range(self.classes):
                    # Calculate Hypothesis
                    hypothesis = np.dot(x, self.theta[idx, :])
                    tmp_list.append(hypothesis)

                # Assign the class label with the greatest probability
                max_ = np.argmax(tmp_list)
                label.append(max_)

            # Array of predicted values
            label = np.array(label)

        return label
