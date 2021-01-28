#----------------- Packages
import numpy as np


#----------------- Function
"""Implementation of the K-Nearest-Neighbors classifier using only numpy in Python."""


class KNearestNeighors:

    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        # Fit function
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def get_distance(self, x, distance_metric):
        # Calculating various distance metrics
        # Euclidean distance
        if distance_metric == 'Euclidean':
            distance = np.sqrt(np.sum((self.X - x)**2, axis=1))

        # Manhattan distance
        elif distance_metric == 'Manhattan':
            distance = (np.sum(np.abs(self.X - x), axis=1))

        # if any other distance metric is selected, use euclidean
        else:
            distance = np.sqrt(np.sum((self.X - x)**2, axis=1))

        return distance

    def predict(self, X):
        # Return the predicted labels
        y_pred_labels = []

        for x in X:
            # Calculate the distance between training and test set
            distances = self.get_distance(
                x, distance_metric=self.distance_metric)

            # Sort the distances by the top-kth values
            k_sorted_distance = np.sort(distances)[:self.k]

            # Sort the indices by the top-kth values
            k_sorted_indices = np.argsort(distances)[:self.k]

            # Return labels of those top-kth values
            k_training_labels = self.y[k_sorted_indices]

            # Return the unique and frequency of those labels
            unique, frequency = np.unique(k_training_labels,
                                          return_counts=True)

            # Sort the frequencies
            sorted_frequency = np.argsort(frequency)

            # Return the most frequenct class
            predicted_label = unique[sorted_frequency[-1]]
            y_pred_labels.append(predicted_label)

        return np.array(y_pred_labels)
