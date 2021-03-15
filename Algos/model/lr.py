#----------------- Packages

import numpy as np
import pandas as pd
import random


def sigmoid(z):
    """ 
    Sigmoid function.

    Parameters
    ----------

    z : array
        Input array used to perform the transformation.

    Returns
    ----------
    g : array
        Output array.
    """

    # Sigmoid
    g = 1/(1 + np.exp(-z))
    return g


def softmax(x, y, theta, idx):
    """ 
    Softmax function.

    Parameters
    ----------
    x : array
        Input array used to perform the transformation.


    y : array
        Input array used to perform the transformation.

    theta : array
        Weight matrix used to perform the transformation.

    index : integer
        Index of the weight matrix

    Returns
    ----------
    g : array
        Output array.
    """

    # Number of classes
    classes = len(np.unique(y))
    
    # Shape of array
    shape_ = np.exp(np.dot(x, theta[idx, :])).shape
    
    # Instantiate array of zeros
    sum_ = np.zeros((shape_))

    # Iterate through the number of classes and calculate the softmax function
    for i in range(classes):    
        sum_ += np.exp(np.dot(x, theta[i,: ])) 
    
    # Softmax
    g = np.exp(np.dot(x, theta[idx, :])) / sum_
    
    return g




def log_costfunction(x, y, theta, lambda_, regularization):
    """ 
    Log cost function

    Parameters
    ----------
    x : array
        Input array used to perform the transformation.


    y : array
        Input array used to perform the transformation.

    theta : array
        Weight matrix used to perform the transformation.

    lambda_ : float
        Regularization parameter in shrinking coefficients.
    
    regularization : string
        Regularization method in shrinking coefficients.
    Returns
    ----------
    cost : float
        Output value.
    """

    # Number of samples
    m = y.shape[0]

    # Reshape array
    y = y.reshape((y.shape[0], ))
    

    # No regularization
    if regularization == None:
        tmp = np.dot(-y , np.log(sigmoid(np.dot(x, theta)))) - np.dot((1 - y) , np.log(1 - sigmoid(np.dot(x, theta))))
        cost = np.sum(tmp, axis = 0)/(m)
    
    # L1 (Lasso) Regularization
    elif regularization == 'L1 (Lasso)':
        reg_term = lambda_ * np.abs(theta[1::])
        tmp = np.dot(-y , np.log(sigmoid(np.dot(x, theta)))) - np.dot((1 - y) , np.log(1 - sigmoid(np.dot(x, theta))))
        cost = np.sum(tmp, axis = 0)/m + np.sum(reg_term)/(2*m)
    
    # L2 (Ridge) Regularization
    elif regularization == 'L2 (Ridge)':
        reg_term = lambda_ * np.power(theta[1::], 2)
        tmp = np.dot(-y , np.log(sigmoid(np.dot(x, theta)))) - np.dot((1 - y) , np.log(1 - sigmoid(np.dot(x, theta))))
        cost = np.sum(tmp, axis = 0)/m + np.sum(reg_term)/(2*m)
    
    return cost

#----------------- Function
class myLogisticRegression:
    """
    Logistic Regression

    Parameters
    ----------
    method: string
        Specifies the method used for a logistic regression solver.
        'BGD', 'SGD', 'Newton'
    multi: string
        Specifies the multiclas method for handling more than two classes.
        'OVR', 'Multinomial'

    regularization : string
        Regularization method in shrinking coefficients.

    lambda_ : float, optional (default: 1)
        Penalty parameter C of the error term.
`
    learning_rate : float
        Parameter for the learning rate of gradient descent methods.
    
    iterations : integer
        Number of iterations used for convergence of the gradient descent problem.



    Attributes
    ----------
    theta : array
        Weights assigned to the features.

    fit: object
        Function to fit the training data.

    predict: object
        Function to predict on the test set.
    """


    def __init__(self, method, multi, regularization, lambda_, learning_rate, iterations):
        self.method = method
        self.multi = multi
        self.regularization = regularization
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.iterations = iterations


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

        # Check arrays
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        
        # Concatenate ones to the training set
        x_ones = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((x_ones, self.X), axis=1)

        # Reshape y_train to m x 1
        self.y = self.y.reshape((self.y.shape[0], 1))
        
        # Number of Unique classes
        classes = len(np.unique(self.y))
        self.classes = classes
        
        # Binary Classification
        if self.classes == 2:
            
            # BGD and no regularization
            if (self.method == 'BGD') & (self.regularization == None):

                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros((self.X.shape[1], 1 ))
        
                # Total Cost
                total_cost = []

                # Iterate through the number of iterations and calculate the cost function for each theta
                for num_ in range(iterations_):
                    # Gradient Descent
                    # Hypothesis
                    hypothesis = sigmoid(np.dot(self.X, theta))
                    
                    # Loss
                    loss = (hypothesis - self.y)
                    
                    # Gradient
                    gradient = np.dot(self.X.T, loss)
                    
                    # Update theta
                    theta[0] = theta[0] - (alpha/m) * gradient[0]
                    theta[1::] = theta[1::] - (alpha/m) * gradient[1::]
                    
                    # Use theta to calculate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta, regularization = self.regularization,
                                            lambda_ = self.lambda_)

                    total_cost.append(cost)
                   

            # Set the slope as the first theta value and intercept as second.
                self.intercept_ = theta[0]
                self.coef_ = theta[1::]
                self.total_cost = total_cost
                self.theta = theta
            
            # BGD and L1 (Lasso) regularization
            elif (self.method == 'BGD') & (self.regularization == 'L1 (Lasso)') :
                
                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros((self.X.shape[1], 1 ))
                
                # Total cost
                total_cost = []

                # Iterate through the number of iterations and calculate the cost function for each theta
                for num_ in range(iterations_):
                    # Gradient Descent
                    # Hypothesis
                    hypothesis = sigmoid(np.dot(self.X, theta))
                    
                    # Loss
                    loss = (hypothesis - self.y)
                    
                    # Gradient
                    gradient = np.dot(self.X.T, loss)
                    
                    # Regularization term
                    reg_term = self.lambda_ * (np.abs(theta[1::]))
                    
                    # Update theta
                    theta[0] = theta[0] - (alpha/m)*gradient[0]
                    theta[1::] = theta[1::] - (alpha/m)*(gradient[1::] + reg_term)

                    
                    # Use theta to calculate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta, regularization = self.regularization,
                                            lambda_ = self.lambda_)

                    total_cost.append(cost)

            # Set the slope as the first theta value and intercept as second.
                self.intercept_ = theta[0]
                self.coef_ = theta[1::]
                self.total_cost = total_cost
                self.theta = theta
            
            # BGD and L2 (Ridge) regularization
            elif (self.method == 'BGD') & (self.regularization == 'L2 (Ridge)') :
                  
                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros((self.X.shape[1], 1 ))

                # Total Cost
                total_cost = []

                # Iterate through the number of iterations and calculate the cost function for each theta
                for num_ in range(iterations_):
                    # Gradient Descent
                    # Hypothesis
                    hypothesis = sigmoid(np.dot(self.X, theta))
                    
                    # Loss
                    loss = (hypothesis - self.y)
                    
                    # Gradient
                    gradient = np.dot(self.X.T, loss)
                    
                    # Regularization term
                    reg_term = self.lambda_ * (np.power(theta[1::], 2))
                    
                    # Update theta
                    theta[0] = theta[0] - (alpha/m)*gradient[0]
                    theta[1::] = theta[1::] - (alpha/m)*(gradient[1::] + reg_term)

                    
                    # Use theta to calculate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta, regularization = self.regularization,
                                            lambda_ = self.lambda_)

                    total_cost.append(cost)
                    
            # Set the slope as the first theta value and intercept as second.
                self.intercept_ = theta[0]
                self.coef_ = theta[1::]
                self.total_cost = total_cost
                self.theta = theta
                
            
            # Stochastic Gradient Descent and no regularization
            elif (self.method == 'SGD') & (self.regularization == None):
                
                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros((self.X.shape[1], 1 ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of samples
                for num_ in range(m):
                    # Select random training samples from index 0 to number of samples - 1
                    i = random.randint(0, m-1)
                    x = self.X[i]
                    y = self.y[i]
                    
                    x = x.reshape((1, x.shape[0]))
                    y = y.reshape((y.shape[0], 1))
                    
                    # Gradient Descent
                    # Hypothesis
                    hypothesis = sigmoid(np.dot(x, theta))
                    
                    # Loss
                    loss = (hypothesis - y)
                    
                    # Gradient
                    gradient = np.dot(x.T, loss)
                                        
                    # Update theta
                    theta[0] = theta[0] - (alpha/m)*gradient[0]
                    theta[1::] = theta[1::] - (alpha/m)*(gradient[1::])

                    
                    # Use theta to calculate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta, regularization = self.regularization,
                                            lambda_ = self.lambda_)

                    total_cost.append(cost)
                    
                # Set the slope as the first theta value and intercept as second.
                self.intercept_ = theta[0]
                self.coef_ = theta[1::]
                self.total_cost = total_cost
                self.theta = theta
                    
                    
            # Stochastic Gradient Descent and L1 (Lasso) regularization
            elif (self.method == 'SGD') & (self.regularization == 'L1 (Lasso)'):
                
                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros((self.X.shape[1], 1 ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of samples
                for num_ in range(m):
                    # Select random training samples from index 0 to number of samples - 1
                    i = random.randint(0, m-1)
                    x = self.X[i]
                    y = self.y[i]
                    x = x.reshape((1, x.shape[0]))
                    y = y.reshape((y.shape[0], 1))
                    
                    # Gradient Descent
                    # Hypothesis
                    hypothesis = sigmoid(np.dot(x, theta))
                    
                    # Loss
                    loss = (hypothesis - y)
                    
                    # Gradient
                    gradient = np.dot(x.T, loss)
                    
                    # Regularization term
                    reg_term = self.lambda_ * (np.abs(theta[1::]))
                    
                    # Update theta
                    theta[0] = theta[0] - (alpha/m)*gradient[0]
                    theta[1::] = theta[1::] - (alpha/m)*(gradient[1::] + reg_term)

                    
                    # Use theta to calculate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta, regularization = self.regularization,
                                            lambda_ = self.lambda_)

                    total_cost.append(cost)
                    
                # Set the slope as the first theta value and intercept as second.
                self.intercept_ = theta[0]
                self.coef_ = theta[1::]
                self.total_cost = total_cost
                self.theta = theta
                    

            # Stochastic Gradient Descent and L2 (Ridge) regularization
            elif (self.method == 'SGD') & (self.regularization == 'L2 (Ridge)'):
                
                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros((self.X.shape[1], 1 ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of samples
                for num_ in range(m):
                    # Select random training samples from index 0 to number of samples - 1
                    i = random.randint(0, m-1)
                    x = self.X[i]
                    y = self.y[i]
                    x = x.reshape((1, x.shape[0]))
                    y = y.reshape((y.shape[0], 1))
                    
                    # Gradient Descent
                    # Hypothesis
                    hypothesis = sigmoid(np.dot(x, theta))
                    
                    # Loss
                    loss = (hypothesis - y)
                    
                    # Gradient
                    gradient = np.dot(x.T, loss)
                    
                    # Regularization term
                    reg_term = self.lambda_ * (np.power(theta[1::], 2))
                    
                    # Update theta
                    theta[0] = theta[0] - (alpha/m)*gradient[0]
                    theta[1::] = theta[1::] - (alpha/m)*(gradient[1::] + reg_term)

                    
                    # Use theta to calculate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta, regularization = self.regularization,
                                            lambda_ = self.lambda_)

                    total_cost.append(cost)
                    
                    
                # Set the slope as the first theta value and intercept as second.
                self.intercept_ = theta[0]
                self.coef_ = theta[1::]
                self.total_cost = total_cost
                self.theta = theta
                    
            # Newton-Raphson 
            elif (self.method == 'Newton') & (self.regularization == None):
                # Number of samples
                m = len(self.y)
                
                # Number of iterations for convergence
                iterations_ = self.iterations
                
                # Reshape theta vector
                theta = np.zeros((self.X.shape[1], 1 ))

                # Total Cost
                total_cost = []

                # Iterate through the number of iterations and calculate for each theta
                for num_ in range(iterations_):
                    # Newton's method
                    # Hypothesis
                    hypothesis = sigmoid(np.dot(self.X, theta))
                    
                    # Loss
                    loss = (hypothesis - self.y)
                    
                    # Gradient
                    gradient = np.dot(self.X.T, loss)
                    first_deriv = gradient * 1/m
                    
                    # Hessian                   
                    tmp = hypothesis * (1 - hypothesis)
                    tmp = np.dot(tmp.T, self.X)
                    tmp = np.dot(tmp, self.X.T)
                    tmp = np.sum(tmp)
                    second_deriv = tmp * 1/m
                

                    # Update theta
                    theta[0] = theta[0] - (first_deriv[0])/(second_deriv)
                    theta[1::] = theta[1::] - (first_deriv[1::])/(second_deriv)
                
                
                # Set the slope as the first theta value and intercept as second.
                self.theta = theta
                self.intercept_ = self.theta[0]
                self.coef_ = self.theta[1::]
        
        # Multiclass classification (OVR)
        elif (self.classes > 2) & (self.multi == 'OVR'):
                
            # Create new labeled data
            tmp_df = pd.DataFrame(data = self.y, columns = ['original'])
            for class_ in range(classes):
                tmp_df[str(class_)] = tmp_df['original'].apply(lambda x: 1 if x == class_ else 0)
                tmp_df[str(class_)] = pd.to_numeric(tmp_df[str(class_)])

            # BGD and no regularization
            if (self.method == 'BGD') & (self.regularization == None):

                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []

                # Iterate through the number of iterations and calculate the cost function for each theta
                for num_ in range(iterations_):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]

                        # Gradient Descent
                        # Hypothesis
                        hypothesis = sigmoid(np.dot(self.X,theta[idx, :]))
                        
                        # Loss
                        loss = (hypothesis - y_)
                        
                        # Gradient
                        gradient = np.dot(self.X.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * gradient[0, 1::]

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta

            # BGD and L1 (Lasso) regularization
            elif (self.method == 'BGD') & (self.regularization == 'L1 (Lasso)'):

                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []

                # Iterate through the number of iterations and calculate the cost function for each theta
                for num_ in range(iterations_):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]

                        # Gradient Descent
                        # Hypothesis
                        hypothesis = sigmoid(np.dot(self.X,theta[idx, :]))
                        
                        # Loss
                        loss = (hypothesis - y_)
                        
                        # Gradient
                        gradient = np.dot(self.X.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Regularization
                        reg_term = self.lambda_ * (np.abs(theta[idx, 1::]))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::] + reg_term)

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                
                
            # BGD and L2 (Ridge) regularization
            elif (self.method == 'BGD') & (self.regularization == 'L2 (Ridge)'):

                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []

                # Iterate through the number of iterations and calculate the cost function for each theta
                for num_ in range(iterations_):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]

                        # Gradient Descent
                        # Hypothesis
                        hypothesis = sigmoid(np.dot(self.X,theta[idx, :]))
                        
                        # Loss
                        loss = (hypothesis - y_)
                        
                        # Gradient
                        gradient = np.dot(self.X.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Regularization
                        reg_term = self.lambda_ * (np.power(theta[idx, 1::], 2))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::] + reg_term)

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                
                
            # SGD and no regularization
            elif (self.method == 'SGD') & (self.regularization == None):

                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []

                # Iterate through the number of samples
                for num_ in range(m):
                                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]
                        
                        # Select random training samples from index 0 to number of samples - 1
                        i = random.randint(0, len(y_) - 1)
                        x = self.X[i]
                        y = y_[i]     
                        x = x.reshape((1, x.shape[0]))
                        y = y.reshape((1, 1))
                        
                        # Gradient Descent
                        # Hypothesis
                        hypothesis = sigmoid(np.dot(x, theta[idx, :]))
                        
                        # Loss
                        loss = (hypothesis - y)
                        
                        # Gradient
                        gradient = np.dot(x.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                                                
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::])

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                
                
            # SGD and L1 (Lasso) regularization
            elif (self.method == 'SGD') & (self.regularization == 'L1 (Lasso)'):

                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []

                # Iterate through the number of samples
                for num_ in range(m):
                                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]
                        
                        # Select random training samples from index 0 to number of samples - 1
                        i = random.randint(0, len(y_)-1)
                        x = self.X[i]
                        y = y_[i]
                        x = x.reshape((1, x.shape[0]))
                        y = y.reshape((1, 1))
                        
                        # Gradient Descent
                        # Hypothesis
                        hypothesis = sigmoid(np.dot(x, theta[idx, :]))
                        
                        # Loss
                        loss = (hypothesis - y)
                        
                        # Gradient
                        gradient = np.dot(x.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Regularization
                        reg_term = self.lambda_ * (np.abs(theta[idx, 1::]))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::] + reg_term)

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                
                
            # SGD and L2 (Ridge) regularization
            elif (self.method == 'SGD') & (self.regularization == 'L2 (Ridge)'):

                # Set learning rate
                alpha = self.learning_rate

                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []

                # Iterate through the number of samples
                for num_ in range(m):
                                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]
                        
                        # Select random training samples from index 0 to number of samples - 1
                        i = random.randint(0, len(y_)-1)
                        x = self.X[i]
                        y = y_[i]
                        x = x.reshape((1, x.shape[0]))
                        y = y.reshape((1, 1))
                        
                        # Gradient Descent
                        # Hypothesis
                        hypothesis = sigmoid(np.dot(x, theta[idx, :]))
                        
                        # Loss
                        loss = (hypothesis - y)
                        
                        # Gradient
                        gradient = np.dot(x.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Regularization
                        reg_term = self.lambda_ * (np.power(theta[idx, 1::], 2))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::] + reg_term)

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta

            # Newton method and no regularization
            elif (self.method == 'Newton') & (self.regularization == None):

                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []

                # Iterate through the number of iterations and calculate the cost function for each theta
                for num_ in range(iterations_):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]

                        # Gradient Descent
                        # Hypothesis
                        hypothesis = sigmoid(np.dot(self.X,theta[idx, :]))
                        
                        # Loss
                        loss = (hypothesis - y_)
                        
                        # Gradient
                        gradient = np.dot(self.X.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        first_deriv = gradient * 1/m
                    
                        # Hessian                   
                        tmp = hypothesis * (1 - hypothesis)
                        tmp = np.dot(tmp.T, self.X)
                        tmp = np.dot(tmp, self.X.T)
                        tmp = np.sum(tmp)
                        second_deriv = tmp * 1/m
                                                
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (first_deriv[0, 0])/(second_deriv)
                        theta[idx, 1::] = theta[idx, 1::] - (first_deriv[0, 1::])/(second_deriv)

                        
                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
        
        
        # Multinomial Regression
        elif (self.classes > 2) & (self.multi == 'Multinomial'):
            
            # Create new labeled data
            tmp_df = pd.DataFrame(data = self.y, columns = ['original'])
            for class_ in range(classes):
                tmp_df[str(class_)] = tmp_df['original'].apply(lambda x: 1 if x == class_ else 0)
                tmp_df[str(class_)] = pd.to_numeric(tmp_df[str(class_)])
            
            # Softmax BGD and no regularization
            if (self.method == 'BGD') & (self.regularization == None):
                
                # Set learning rate
                alpha = self.learning_rate
                
                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of iterations
                for num_ in range(iterations_):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]

                        # Gradient Descent
                        # Hypothesis
                        hypothesis = softmax(x = self.X, y = self.y,
                                             theta = theta, idx = idx)
                        
                        # Loss
                        loss = (hypothesis - y_)
                        
                        # Gradient
                        gradient = np.dot(self.X.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                                            
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::])

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                        
            # Softmax BGD and L1 (Lasso) regularization
            elif (self.method == 'BGD') & (self.regularization == 'L1 (Lasso)'):
                
                # Set learning rate
                alpha = self.learning_rate
                
                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of iterations
                for num_ in range(iterations_):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]

                        # Gradient Descent
                        # Hypothesis
                        hypothesis = softmax(x = self.X, y = self.y,
                                             theta = theta, idx = idx)
                        
                        # Loss
                        loss = (hypothesis - y_)
                        
                        # Gradient
                        gradient = np.dot(self.X.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Regularization
                        reg_term = self.lambda_ * (np.abs(theta[idx, 1::]))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::] + reg_term)

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                
            # Softmax BGD and L2 (Ridge) regularization
            elif (self.method == 'BGD') & (self.regularization == 'L2 (Ridge)'):
                
                # Set learning rate
                alpha = self.learning_rate
                
                # Number of samples
                m = len(self.y)

                # Number of iterations for convergence
                iterations_ = self.iterations

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of iterations
                for num_ in range(iterations_):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]

                        # Gradient Descent
                        # Hypothesis
                        hypothesis = softmax(x = self.X, y = self.y,
                                             theta = theta, idx = idx)
                        
                        # Loss
                        loss = (hypothesis - y_)
                        
                        # Gradient
                        gradient = np.dot(self.X.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Regularization
                        reg_term = self.lambda_ * (np.power(theta[idx, 1::], 2))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::] + reg_term)

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = self.X, y = self.y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                
            # Softmax SGD and no regularization
            elif (self.method == 'SGD') & (self.regularization == None):
                
                # Set learning rate
                alpha = self.learning_rate
                
                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of iterations
                for num_ in range(m):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]
                        
                        # Select random training samples from index 0 to number of samples - 1
                        i = random.randint(0, len(y_)-1)
                        x = self.X[i]
                        y = y_[i]
                        x = x.reshape((1, x.shape[0]))
                        y = y.reshape((1, 1))
                        

                        # Gradient Descent
                        # Hypothesis
                        hypothesis = softmax(x = x, y = y,
                                             theta = theta, idx = idx)
                        
                        # Loss
                        loss = (hypothesis - y)
                        
                        # Gradient
                        gradient = np.dot(x.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                                                
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::])

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                
                
                
            # Softmax SGD and L1 (Lasso) regularization
            elif (self.method == 'SGD') & (self.regularization == 'L1 (Lasso)'):
                
                # Set learning rate
                alpha = self.learning_rate
                
                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of iterations
                for num_ in range(m):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]
                        
                        # Select random training samples from index 0 to number of samples - 1
                        i = random.randint(0, len(y_)-1)
                        x = self.X[i]
                        y = y_[i]
                        x = x.reshape((1, x.shape[0]))
                        y = y.reshape((1, 1))
                        
                        # Gradient Descent
                        # Hypothesis
                        hypothesis = softmax(x = x, y = y,
                                             theta = theta, idx = idx)
                        
                        # Loss
                        loss = (hypothesis - y)
                        
                        # Gradient
                        gradient = np.dot(x.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Regularization
                        reg_term = self.lambda_ * (np.abs(theta[idx, 1::]))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::] + reg_term)

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    

                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta

            # Softmax SGD and L2 (Ridge) regularization
            elif (self.method == 'SGD') & (self.regularization == 'L2 (Ridge)'):
                
                # Set learning rate
                alpha = self.learning_rate
                
                # Number of samples
                m = len(self.y)

                # Reshape theta vector
                theta = np.zeros(( classes, self.X.shape[1] ))

                # Total Cost
                total_cost = []
                
                # Iterate through the number of iterations
                for num_ in range(m):
                    
                    # Iterate through each class to get coefficients
                    for idx, y_label in zip(range(0, classes), tmp_df.columns[1::] ):
                        
                        # Class labels 
                        y_ = tmp_df[str(y_label)]
                        
                        # Select random training samples from index 0 to number of samples - 1
                        i = random.randint(0, len(y_)-1)
                        x = self.X[i]
                        y = y_[i]
                        x = x.reshape((1, x.shape[0]))
                        y = y.reshape((1, 1))
                        
                        # Gradient Descent
                        # Hypothesis
                        hypothesis = softmax(x = x, y = y,
                                             theta = theta, idx = idx)
                        
                        # Loss
                        loss = (hypothesis - y)
                        
                        # Gradient
                        gradient = np.dot(x.T, loss)
                        gradient = gradient.reshape((1, theta.shape[1]))
                        
                        # Regularization
                        reg_term = self.lambda_ * (np.power(theta[idx, 1::], 2))
                        
                        # Update Theta
                        theta[idx, 0] = theta[idx, 0] - (alpha/m)*gradient[0, 0]
                        theta[idx, 1::] = theta[idx, 1::] - (alpha/m) * (gradient[0, 1::] + reg_term)

                    # Use theta to calculeate the cost
                    cost = log_costfunction(x = x, y = y,
                                            theta = theta[idx, :], regularization = self.regularization,
                                            lambda_ = self.lambda_)
                    total_cost.append(cost)
                
                    
                # Set the slope as the first theta value and intercept as second.
                self.total_cost = total_cost
                self.theta = theta
                        
                        
    def predict(self, X):
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

        # Check arrays
        self.X = np.asarray(X)
        
        # Concatenate ones to the testing set
        x_ones = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((x_ones, self.X), axis=1)
        
        # Binary Logistic Regression
        if self.classes == 2:
            label = []
            
            # Iterate through each testing sample
            for x in self.X:
                
                # Calculate the probability using the hypothesis function 
                tmp =  sigmoid(np.dot(x, self.theta))
                
                # If the probability is greater than 0.5 threshold, assign it the label of class 1
                if tmp >= 0.50:
                    label.append(1)
                
                # Else assign it the label of class 0
                else:
                    label.append(0)
            
            return np.array(label)
                
        # If the number of classes is greater than 2 and one-versus-rest classification
        elif (self.classes > 2) & (self.multi == 'OVR'):
            label = []
            
            # Iterate through each testing sample
            for x in self.X:
                tmp_list = []
                
                # Iterate through each class
                for i in range(self.classes):
                    # Calculate the probabilities using the hypothesis function
                    tmp = sigmoid(np.dot(x,self.theta[i, :]))
                    tmp_list.append(tmp)
                
                # Assign the class label with the greatest probability
                max_ = np.argmax(tmp_list)
                label.append(max_)
                
            return np.array(label)
        
        # If the number of classes is greater than 2 and multinomial classification
        elif (self.classes > 2) & (self.multi == 'Multinomial'):
            label = []
            
            # Iterate through each testing sample
            for x in self.X:
                tmp_list = []
                
                # Iterate through each class
                for i in range(self.classes):
                    # Calculate the probability using the hypothesis function
                    tmp = softmax(x = x, y = self.y,
                                  theta = self.theta, idx = i)
                    tmp_list.append(tmp)
                
                # Assign the class label with the greatest probability
                max_ = np.argmax(tmp_list)
                label.append(max_)
            
            return np.array(label)
        
        
                
                
        
        
