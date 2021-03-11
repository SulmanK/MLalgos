#------------------ Packages
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

#----------------- Function
""""Script to return the training and testing set """

# Create training set
x_neg = np.array([[3,4],[1,4],[2,3]])
y_neg = np.array([-1,-1,-1])
x_pos = np.array([[6,-1],[7,-1],[5,-3]])
y_pos = np.array([1,1,1])

X = np.vstack((x_pos, x_neg))
y = np.concatenate((y_pos,y_neg))

X_train = X
y_train = y