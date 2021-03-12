#------------------ Packages
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

#----------------- Function
def get_data_perceptron(classes):
	""" 
    Loads the data depending on the amount fo classes set in the widget

    Parameters
    ----------
    classes : integer
        The number of classes in the data.


    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,)
        The class labels.
    """

    # Two Classes
	if classes == 2:
		data = load_breast_cancer()
		X = data.data
		y = data.target

	# More than two classes
	else:
		data = load_iris()
		X = data.data
		y = data.target



	# Create a scaling instance
	scaler = StandardScaler()

	# Fit the standardization parameters and scale the data.
	X_scale = scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X_scale, y,
	                                                    test_size=0.2, random_state=42)


	return X_train, X_test, y_train, y_test