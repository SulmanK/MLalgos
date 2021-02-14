#------------------ Packages
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#----------------- Function
""""Script to return the training and testing set """

# Read in dataframe
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(columns = df.columns[2::])
df = df[['v2', 'v1']]
df.columns = ['Text', 'Output']
     
# Subset our data into a training and testing sets
X = df.iloc[:, 0]
y = df.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)