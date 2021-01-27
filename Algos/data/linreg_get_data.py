#------------------ Packages

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#----------------- Function
""""Script to return the training and testing set """

# Column names
col_names = ['vendor_name', 'model_name',
             'MYCT', 'MMIN',
             'MMAX', 'CACH',
             'CHMIN', 'CHMAX',
             'PRP', 'ERP']

# Read in dataframe from csv file using col_names list
#df = pd.read_csv('data/machine.data', header=None)

df = pd.read_csv('machine.data', header=None)
# Subset the X, y df's
X = df.loc[:, 2:7].values

y = df.loc[:, 8].values

# M by 1 shape
y = y.reshape(y.shape[0], 1)

# Create a scaling instance
scaler = StandardScaler()

# Fit the standardization parameters and scale the data.
X_scale = scaler.fit_transform(X)

y_scale = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale,
                                                    test_size=0.2, random_state=42)
