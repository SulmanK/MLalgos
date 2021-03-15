#------------------ Packages
from data.perceptron_get_data import get_data_perceptron
from layout.scatterplot_db_perceptron import ScatterPlotDB_perceptron
from model.perceptron import myPerceptron
from PIL import Image
from sklearn.linear_model import Perceptron

import numpy as np
import pandas as pd
import streamlit as st
#----------------- Layout
def app():
	# Title
	st.title('Perceptron')

	# Supervised Learning
	st.markdown(''' ### Supervised Learning
Firstly, we need to explain supervised learning before we move onto Naive Bayes. 
Supervised learning involves using input features (x) to predict an output (y). 
Given a training set, we employ a learning algorithm to get our hypothesis function. 
Then, we feed our dataset to our hypothesis function to make predictions.  The process described is presented in Figure 1.
	''')

	# Image of Supervised learning
	image = Image.open('SL_diagram.png')
	st.image(image, caption='Figure 1: Supervised learning diagram.',
	         use_column_width=False, width=600)

	# Linear Regression
	st.markdown(r'''
	In supervised learning, when the target variable is continuous, it is called regression, and when it is discrete, it is known as classification. 

	### Perceptron

	Perceptron is a classification algorithm, which uses a hyperplane to separate and classify the data. 

	''')

	image2 = Image.open('hyperplane.png')
	st.image(image2, caption='Figure 2: Hyperplane diagram.',
	         use_column_width=False, width=600)


	st.markdown(r'''


	The loss function of the perceptron algorithm follows the hinge-loss function.

	''')

	image3 = Image.open('hinge_loss.png')
	st.image(image3, caption='Figure 3: Hinge-loss function.',
	         use_column_width=False, width=600)


	st.markdown(r'''

	#### Pseudocode


	    Initialize w = 0
	    While TRUE do
	        m = 0
	        for (x_i, y_i) - do
	            if y_i(w dot x_i) <= 0
	                w = w + yx
	                m = m + 1
	            end if
	        end for
	        if m = 0
	           break
	        end if
	    end while


	#### Update Rules
	(1) Original update
	 
	$
	w = w + x 
	$, if y = 1

	$
	w = w - x 
	$, if y = 0

	(2) Stochastic Gradient descent update 

	$
	w = w - \alpha * (y - y_{pred})*x 
	$







	#### Procedure

	(1) Initialize all weights to zero.

	(2) Iterate through the total number of samples by randomly selecting a training sample in the data set.
	 * If y_predict equals y, continue onto next iteration.
	 * If y_predict does not equal y, update the w using the update rule, continue onto next iteration.








	''')


	# Insert parameters
	st.sidebar.title('Parameters')
	col1 = st.sidebar.beta_columns(1)

	# Widgets
	lr_widget = 0
	method_widget = ''

	classes_widget = st.sidebar.radio(label='Classes', key='Sci-kit_classes',
	                                  options=[2, 3])

	update_widget = st.sidebar.radio(label='Update Rule', key='Sci-kit_update',
	                                 options=['Original', 'Gradient'])

	if update_widget == 'Gradient':

	    col1, col2 = st.sidebar.beta_columns(2)

	    # Data parameters
	    with col1:

	        method_widget = st.sidebar.radio(label='Method', key='Sci-kit_method',
	                                         options=['SGD', 'SAGD'])

	        lr_widget = st.slider('Learning Rate', min_value=0.0,
	                              max_value=1.0, value=0.01,
	                              step=0.01)


	# Split into two columns for Closed-formed solution and Gradient descent.
	col1, col2 = st.beta_columns(2)

	# Pros
	with col1:
	    st.markdown(''' 
	        #### Pros

	        * Works well when there is a clear margin of separation between classes
	        * Ease of impkementation
	        * Complexity time of O(n * d)
	        ''')

	# Cons
	with col2:
	    st.markdown(''' 
	        #### Cons
	        * Cannot classify non-linearly separable data
	        ''')


	# Implementation code
	st.markdown('''
	        ### Implementation
	[Perceptron](https://github.com/SulmanK/MLalgos/blob/main/Algos/model/perceptron.py)

	        ''')


	# Analysis
	st.markdown('''
	        ### Comparison of decision boundaries between my implementation vs sci-kit learn.




	        ''')


	def plots(classes_widget):
	    """Plots for gaussian naive bayes"""

	    X_train, X_test, y_train, y_test = get_data_perceptron(classes_widget)

	    # My SVM
	    mine_clf = myPerceptron(method=method_widget,
	                            update_rule=update_widget, learning_rate=lr_widget)
	    #mine_clf.fit(X = X_train, y = y_train)

	    # SKlearns SVM
	    sk_clf = Perceptron(tol=1e-3, random_state=1)
	    #sk_clf.fit(X = X_train, y = y_train)

	    fig = ScatterPlotDB_perceptron(X_train=X_train, y_train=y_train,
	                                   X_test=X_test, y_test=y_test,
	                                   clf_1=mine_clf, clf_2=sk_clf,
	                                   title_1='Scatterplot with Decision Boundary (Mine)', title_2='Scatterplot with Decision Boundary (Sklearn)')

	    return fig


	# Streamlit plot figures
	sp_perceptron_plot = plots(classes_widget=classes_widget)
	st.pyplot(sp_perceptron_plot)
