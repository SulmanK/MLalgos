#------------------ Packages
from sklearn.model_selection import train_test_split
import numpy as np

#----------------- Function
""""Script to return the training and testing set """

# Create X, y data
X = np.linspace(0,1,100)
noise = np.random.normal(loc = 0, scale = .25, size = 100)
y = np.cos(X * 1.5 * np.pi ) 
y_noise = y + noise


# Create training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y_noise,
                                                    test_size=0.2, random_state=42)