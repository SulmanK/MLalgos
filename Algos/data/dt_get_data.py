#------------------ Packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
#----------------- Function

def get_data(method):
	""" 
    Loads the data depending on the method used for the decision tree

    Parameters
    ----------
    method : str
        Method used for solving the decision tree problem
        ['Classification', 'Regression']
	

    Returns
    -------
    X_train : array-like, shape (n_samples, n_features)
        The X training data..
    y_train : array-like, shape (n_samples,)
        The training labels.
    X_test : array-like, shape (n_samples, n_features)
    	The X testing data.
    y_test : array-like, shape (n_samples,)
        The testing labels.

    """


	# Classification
	if method == 'Classification':

		# Column names
		col_names = ['age', 'workclass',
		             'fnlwgt', 'education',
		             'education-num',
		             'marital-status', 'occupation',
		             'relationship', 'race',
		             'sex', 'capital-gain',
		             'capital-loss', 'hours-per-week',
		             'native-country', '>50K, <=50K']

		# Read in dataframe from csv file using col_names list
		df = pd.read_csv('data/adult.data', header=None, names = col_names)

		# Drop education-num column (same as education but encoded)
		df = df.drop(columns = ['education-num'])

		# Select columns
		X = df.iloc[:, 0:12]
		y = df.iloc[:, 13]

		# Subset the data
		X_train, X_test, y_train, y_test = train_test_split(X, y,
		                                          test_size=0.2, random_state=42)

	# Regression
	elif method == 'Regression':

		# Column names
		col_names = ['vendor_name', 'model_name',
		             'MYCT', 'MMIN',
		             'MMAX', 'CACH',
		             'CHMIN', 'CHMAX',
		             'PRP', 'ERP']

		df = pd.read_csv('data/machine.data', header=None, names = col_names)
		
		# Subset the X, y df's
		X = df.iloc[:, 2:7]
		y = df.iloc[:, 8].values

		# Reshape y label (1, -1)
		y = y.reshape(y.shape[0], 1)

		# Create a scaling instance
		scaler = StandardScaler()

		# Fit the standardization parameters and scale the data.
		X_scale = scaler.fit_transform(X)
		y_scale = scaler.fit_transform(y)

		# Dataframe
		X_scale = pd.DataFrame(data = X_scale, columns = col_names[2:7])
		y_scale = pd.DataFrame(data = y_scale)

		X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale,
		                                                    test_size=0.2, random_state=42)

	return X_train, X_test, y_train, y_test