#----------------- Packages
import numpy as np

#----------------- Function
def costfunction(X, y, theta, lambda_, regularization):
    """Function to calculate the cost function given the training set."""
    # Number of samples
    m = len(y)
    
    if regularization == None:
        # Cost calculation
        hypothesis = np.dot(X, theta)
        loss_sq = (hypothesis - y)**2
        cost = np.sum(loss_sq, axis = 0)/(2*m)
    
    elif regularization == 'L1 (Lasso)':
        # Cost calculation
        hypothesis = np.dot(X, theta)
        reg = lambda_*(np.abs(theta))
        loss_sq = (hypothesis - y)**2
        cost = (np.sum(loss_sq, axis = 0) + reg)/(2*m)
    
    elif regularization == 'L2 (Ridge)':
        # Cost calculation
        hypothesis = np.dot(X, theta)
        reg = lambda_*(np.sum(theta**2))
        loss_sq = (hypothesis - y)**2
        cost = (np.sum(loss_sq, axis = 0) + reg)/(2*m)
        
    return cost