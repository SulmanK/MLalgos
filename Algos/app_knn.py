#------------------ Packages

from data.get_data import get_sample_data
from layout.plot_decision_boundary import make_meshgrid, plot_contours, scatter_plot_db
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from model.knn import KNearestNeighors
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import random
import streamlit as st



#----------------- Layout
def app():

	# Title
	st.title('K-Nearest Neighbor Classification')

	# Introduction
	st.markdown('''
		K-nearest neighbor algorithm is widely used for classification. It works in the following way:

	1. Given a new input x.
	2. Find the K closest examples in the training set to x.
	3. Examine their labels and select the majority class of K neighbors.

	In short, we find the k closest such data points across the whole training data and classify based on a majority class of the K nearest training data [2].
	''')

	# Image of KNN
	image = Image.open('KNN_diagram.png')
	st.image(image, caption='How K-NN works',
		 use_column_width=True)

	# Pseudocode / Parameters
	st.markdown('''
	### Psuedocode

	    1. Go through each item in my training set, and calculate the "distance" from that data item to my specific sample.

	    2. Classify the sample as the majority class between K samples in the dataset having minimum distance to the sample.




	### Parameters
	* Size of the neighborhood (K)
	* Distance metric (closeness)

	### Pros and Cons

	''')


	# Split into two columns for PROS and CONS
	col1, col2 = st.beta_columns(2)


	# Pros
	with col1:
	    st.markdown(''' 
		#### Pros

		* K-NN is easy to understand and simple to implement.

		* K-NN is a non-parametric model, where it does make assumptions in what probability distribution the data follows.

		* Works well when similar classes are clustered around certain areas. 

		''')

	# Cons
	with col2:
	    st.markdown(''' 
		#### Cons

		* Computationally intense

		  * D - dimensonality of the data
		  * N - training data size 
		  * K - Number of nearest neighbors
		  * Algorithmic complexity of O(DNK)

		* Suffers from high dimensional data (curse of dimensionality)
		  * The volume of space grows exponentially fast with dimension, so you might have to look quite far away in space to find your nearest neighbor.

		''')

	# Implementation code
	st.markdown('''
			### Implementation

	    class KNearestNeighors:
		# KNN Algorithm implemented using only numpy in python.

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
			if distance_metric == 'euclidean':
			    distance = np.sqrt(np.sum((self.X - x)**2, axis=1))

			# Manhattan distance
			elif distance_metric == 'manhattan':
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


			''')


	# Insert parameters
	st.markdown('''

		#### Comparison Plot between Mine and Sci-kit


		''')

	col1, col2 = st.beta_columns(2)

	# Data parameters
	with col1:
	    st.markdown(''' 
		#### Data parameters

		''')
	    classes = st.radio('Classes', [2, 3, 4])
	    number_of_samples = st.slider(
		'Number of samples', min_value=20, max_value=300)

	# Algorithm parameters
	with col2:
	    st.markdown(''' 
		#### Algorithm parameters
		''')
	    distance_metric = st.radio('Distance metric', ['Euclidean', 'Manhattan'])
	    num_neighbors = st.slider('K', min_value=1, max_value=30)


	# Instantiate of variables

	st.set_option('deprecation.showPyplotGlobalUse', False)

	# Get data
	X_train, X_test, y_train, y_test = get_sample_data(
	    num_classes=classes, num_samples=number_of_samples)

	# Callback function used for populating comparison plots


	@st.cache(allow_output_mutation=True)
	def comparison_plot(X_train, X_test, y_train, y_test):

	    clf = KNearestNeighors(k=num_neighbors, distance_metric=distance_metric)
	    clf.fit(X_train, y_train)
	    y_pred_mine = clf.predict(X_test)

	    neigh = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)
	    neigh.fit(X_train, y_train)
	    y_pred_sklearn = neigh.predict(X_test)

	    accuracy_mine = accuracy_score(y_test, y_pred_mine) * 100
	    accuracy_sklearn = accuracy_score(y_test, y_pred_mine) * 100

	    clf_list = [clf, neigh]

	    fig = scatter_plot_db(X=X_train[:, 0], Y=X_train[:, 1],
				  y_train=y_train, X_test=X_test,
				  num_neighbors=num_neighbors, clf=clf_list,
				  y_pred_mine=y_pred_mine, y_pred_sklearn=y_pred_sklearn,
				  accuracy_mine=accuracy_mine, accuracy_sklearn=accuracy_sklearn
				  )

	    return fig


	# Display the plot
	sl_plot = comparison_plot(X_train=X_train, X_test=X_test,
				  y_train=y_train, y_test=y_test)

	st.pyplot(sl_plot)
