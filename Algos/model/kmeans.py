#----------------- Packages
import numpy as np
import random


#----------------- Function


def label_func(R):
    """ 
    Assign labels given the responsibility vector from the K-means algorithm.

    Parameters
    ----------
    R : array-like,
        Responsibility vector containing the labels, whichever index has a value of 1 is the label of that sample.

    Returns
    -------
    self : returns a trained K-means model
    """

    # List
    label_list = []

    # Iterate through each sample in the data
    for r in R:

        # Iterate through each indices of the sample
        for idx, val in enumerate(r):

            # If the sample index is equal to 1
            if val == 1:

                # Assign the label of that index
                label_list.append(idx)

    return label_list


class myKmeans:
    """
    K-means Clustering

    Parameters
    ----------
    method: string
        Specifies the method used for the k-means algorithm.
        'K-means', 'K-means +++'

    k: integer
        Specifies the number of clusters.

    iterations : integer
        Number of iterations used for convergence.



    Attributes
    ----------
    u: array
        Cluster centers of the data.

    fit: object
        Function to fit the training data.

    predict: object
        Function to predict on the test set.
    """

    def __init__(self, method, k, iterations):
        # Method
        self.method = method

        # Number of clusters
        self.k = k

        # Number of iterations
        self.iterations = iterations

    def fit(self, X):
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
        self : returns a trained K-means model
        """

        # Check array
        X = np.asarray(X)

        # Parameters
        # Number of samples
        m = len(X)

        # Reponsibility vector (Which cluster the point belongs to) (m x k)
        r = np.zeros((m, self.k))

        # Mean array (Menas of the clusters) ( k x X)
        u = np.zeros((self.k, X.shape[1]))

        random.seed(1)
        if self.method == 'K-means':

            # K-means algo
            # Randomly assign cluster labels to points
            for i in range(0, m):
                j = random.randint(0, self.k - 1)
                r[i][j] = 1

            # Iterate through the number of iterations
            for iter_ in range(self.iterations):

                # Calculate the mean of the clusters for each cluster
                for num in range(self.k):

                    # Calculate the number of points assigned to a cluster
                    N_k = np.sum(r[:, num])

                    # Calculate the mean of the cluster
                    u[num, :] = (1/N_k) * np.dot(r[:, num], X)

                # Assign the cluster label based on the minimum distance between that point and all the clusters
                for i in range(0, m):

                    # Create list
                    tmp_list = []

                    # Iterate through each cluster
                    for num in range(self.k):

                        # Assign all cluster labels to 0
                        r[i, :] = 0

                        # Calculate the distance between that training point and each cluster mean
                        dist = np.linalg.norm(X[i]-u[num])
                        tmp_list.append(dist)

                    # Find the index with the minimum distance out of all cluster differences
                    tmp_array = tmp_list
                    min_idx = np.where(tmp_array == np.amin(tmp_array))

                    # Assign that cluster index a value of 1
                    r[i][min_idx] = 1

            # Assign training labels
            train_labels = label_func(r)

            # Cluster Centroids
            self.u = u

        elif self.method == 'K-means +++':

            # Distribution of shape (k, m)
            p = np.zeros((self.k, m))

            # Randomly select an index
            n = random.randint(0, m - 1)

            # Assign a random sample as a cluster center
            u[0, :] = X[n]

            # Iterate through the number of clusters from 1 and on
            for num_ in range(1, self.k):
                # Distance matrix (1 x m)
                d = np.zeros((1, m))

                # Create list
                tmp_list = []

                # Iterate through the number of samples
                for i in range(0, m):
                    # Distance metric
                    dist = np.linalg.norm(X[i]-u[num_, :])

                    # Minimum distance
                    d_min = np.min(dist)

                    # Assign distance index with the minimum distance
                    d[0][i] = d_min

                # Assign weights of distributions through the entire number of samples
                for i in range(0, m):
                    p[num_, i] = (d[0][i]**2)/(np.sum(d[0]**2))

                # Select a random index from the distribution
                tmp = np.random.choice(p[num_], 1)
                idx = np.where(p[num_] == tmp)

                # Select the random sample as the cluster center
                u[num_, :] = X[idx]

            # Randomly assign cluster labels to points
            for i in range(0, m):
                j = random.randint(0, self.k - 1)
                r[i][j] = 1

            # Iterate through the number of iterations
            for iter_ in range(self.iterations):

                # Calculate the mean of the clusters for each cluster
                for num in range(self.k):

                    # Calculate the number of points assigned to a cluster
                    N_k = np.sum(r[:, num])

                    # Calculate the mean of the cluster
                    u[num, :] = (1/N_k) * np.dot(r[:, num], X)

                # Assign the cluster label based on the minimum distance between that point and all the clusters
                for i in range(0, m):

                    # Create list
                    tmp_list = []

                    # Iterate through each cluster
                    for num in range(self.k):

                        # Assign all cluster labels to 0
                        r[i, :] = 0

                        # Calculate the distance between that training point and each cluster mean
                        dist = np.linalg.norm(X[i]-u[num])
                        tmp_list.append(dist)

                    # Find the index with the minimum distance out of all cluster differences
                    tmp_array = tmp_list
                    min_idx = np.where(tmp_array == np.amin(tmp_array))

                    # Assign that cluster index a value of 1
                    r[i][min_idx] = 1

            # Assign training labels
            train_labels = label_func(r)

            # Cluster Centroids
            self.u = u

        return train_labels

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

        # Check array
        X = np.asarray(X)

        # Number of samples
        m = len(X)

        # Reponsibility vector (Which cluster the point belongs to) (m x k)
        r = np.zeros((m, self.k))

        # Assign the cluster label based on the minimum distance between that point and all the clusters
        for i in range(0, m):

            # Create list
            tmp_list = []

            # Iterate through each cluster
            for num in range(self.k):

                # Assign all cluster labels to 0
                r[i, :] = 0

                # Calculate the distance between that training point and each cluster mean
                dist = np.linalg.norm(X[i]-self.u[num])
                tmp_list.append(dist)

            # Find the index with the minimum distance out of all cluster differences
            tmp_array = tmp_list
            min_idx = np.where(tmp_array == np.amin(tmp_array))

            # Assign that cluster index a value of 1
            r[i][min_idx] = 1

        # Assign training labels
        pred_labels = label_func(r)

        return pred_labels