#------------------ Packages
from data.lr_get_data import get_data_lr
from layout.scatterplot_db_lr import ScatterPlotDB_lr
from layout.plot_cost_lr import plot_cost_gd_lr
from model.lr import myLogisticRegression
from PIL import Image

from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import streamlit as st
#----------------- Layout
def app():

  
  # Title
  st.title('Logistic Regression')

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

  # Logistic Regression
  st.markdown(r'''
  In supervised learning, when the target variable is continuous, it is called regression, and when it is discrete, it is known as classification. 

  ### Logistic Regression

  Logistic Regression is a classification algorithm, which uses a hyperplane to separate and classify the data. 

  ''')

  image2 = Image.open('hyperplane.png')
  st.image(image2, caption='Figure 2: Hyperplane diagram.',
           use_column_width=False, width=600)


  st.markdown(r'''


  The loss function of the perceptron algorithm follows the sigmoid function.

  ''')


  image3 = Image.open('sigmoid_loss.png')
  st.image(image3, caption='Figure 3: Sigmoid plot.',
           use_column_width=False, width=600)


  st.markdown(r'''
      We need to represent the hypothesis function.
      $$ 
      h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2
      $$ 


      $$
      h(x) = \frac{1}{1 + e^{-\sum_{i = 0}^{d} \theta_i x_i}} = \frac{1}{1 + e^{-\theta^Tx}}
      $$

      $
      \theta_i
      $
       's are the parameters and 
      $
      x_i
      $
      's are input features

      Next, we'll need a method to measure our predictions, let's make h(x) close to y. Our loss function is
      $$
      Loss = h_\theta(x) -y
      $$

      We'll use the binary cross-entropy model to get our cost function which is a measure of our loss function over our entire training set.

      $$
      J(\theta) = \sum_{i = 1}^{m} y^{(i)} log (h(x^{(i)})) + (1 - y^{(i)}) log( 1 - h(x^{(i)}))
      $$

      Our goal is to minimize our cost function (the sum of our loss function across the training set is minimized).
      We'll be examining two iterative methods to achieve this, newton's method and various gradient descent methods.

      ### Newton-Raphson
      Minimizing J by taking its derivatives with respect to 
      $
      \theta_j
      $
      and setting them to zero.

      This can be achieved by using newton's method in finding the roots.

      $$ 
      \theta := \theta - \frac{J^{`}(\theta)}{J^{``}(\theta)}
      $$


      ### Gradient Descent 
      Gradient Descent is a search algorithm that starts with an initial guess for
      $
      \theta
      $
      , it changes 
      $
      \theta
      $
      to make 
      $
      J(\theta)
      $ 
      smaller until it converges to a value of 
      $
      \theta
      $
      that minimizes
      $
      J(\theta)
      $
      . An example of the trajectory of a gradient descent algorithm is presented in Figure 4.




    ''')
  # Image of GD
  image = Image.open('GD_diagram.png')
  st.image(image, caption='Figure 4: Gradient descent diagram.',
           use_column_width=False, width=600)


  st.markdown(r'''

  $$
  \theta_j := \theta_j - \alpha \frac{d}{d\theta_j}J(\theta)
  $$ 

  $$
  \theta_j := \theta_j - \alpha (y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}
  $$ 

  The 
  $
  \alpha
  $ 
  parameter is our learning rate, the rate at which our theta updates for convergence.
  We do not want to set it too high or it will never converge to the optimum point and drift towards infinity.

  In our case, we'll use two different gradient descent methods: Batch gradient descent and Stochastic gradient descent.

  #### Batch Gradient Descent

  Repeats the update rule above on all training examples on each iteration for a specified number of iterations.

  #### Stochastic Gradient Descent

  Repeats the update rule on a single training example selected at random for the total number of samples in the training set.


  ### Multiclass Classification

  There are two distinct methods for classifying multi-labeled datasets.

  * One-Versus-Rest 

  * Multinomial (Softmax Regression)


  ### One-Versus-Rest

  Procedure:

  (1) Perform binary classification for each class.

  (2) Compare the predicted probabilities of each class for each sample in the testing set.

  (3) Select the class with the maximum probability and assign that label.


  ### Softmax Regression
  Softmax regression is the multiclass version of logistic regression.

  For multiclass classification, the cost function is modified to support additional classes.
  $$ 
  J(\theta) = - \left[ \sum_{i = 1}^m \sum_{k = 0}^1 \{ y^{(i)} = k, log P(y^{(i)} = k | x^{(i)} ; \theta)   \} \right]
  $$

  The probabilties are normalized across each class.


  $$
  P(y^{(i)} = k | x^{(i)} ; \theta) = \frac{exp(\theta^{(k)T} x^{(i)})}{\sum_{j = 1}^Kexp(\theta^{(k)T} x^{(i)})}
  $$

  ''')




  # Widgets
  lr_widget = None
  regularization_widget = None
  iterations_widget = None
  method_widget = None
  multi_widget = None
  lambda_widget = None

  # Insert parameters
  st.sidebar.title('Parameters')
  col1, col2 = st.sidebar.beta_columns(2)

  # Data parameters
  with col1:

      classes_widget = st.radio(label='Classes', key='Sci-kit_classes',
                                options=[2, 3])

  if classes_widget == 3:

      with col2:

          multi_widget = st.radio(label='Multi Method', key='Sci-kit_multi',
                                  options=['OVR', 'Multinomial'])


  # Algorithmic Parameters

  # Two Classes
  if classes_widget == 2:

      method_widget = st.sidebar.radio(label='Method', key='Method1',
                                       options=['Newton', 'BGD', 'SGD'])
      # Regularization if not newton
      if method_widget != 'Newton':

          regularization_widget = st.sidebar.radio(label='Regularization', key='Regularization',
                                                   options=[None, 'L1 (Lasso)', 'L2 (Ridge)'])
  # Three classes
  elif classes_widget == 3:

      # One-versus-Rest Classification
      if multi_widget == 'OVR':

          method_widget = st.sidebar.radio(label='Method', key='Method2',
                                           options=['Newton', 'BGD', 'SGD'])

      # Multinomial
      elif multi_widget == 'Multinomial':

          method_widget = st.sidebar.radio(label='Method', key='Method2',
                                           options=['BGD', 'SGD'])

      # Regularization if not newton
      if method_widget != 'Newton':
          regularization_widget = st.sidebar.radio(label='Regularization', key='Regularization',
                                                   options=[None, 'L1 (Lasso)', 'L2 (Ridge)'])


  # Regularization
  if regularization_widget != None:
      lambda_widget = st.sidebar.slider('Penalty coefficient', min_value=0.0,
                                        max_value=10.0, value=0.0,
                                        step=0.01)
  if method_widget != 'SGD':
      iterations_widget = st.sidebar.slider('Iterations', min_value=1,
                                            max_value=1500, value=100,
                                            step=1)

  lr_widget = st.sidebar.slider('Learning rate', min_value=0.0,
                                max_value=1.0, value=0.01,
                                step=0.01)


  # Split into two columns for Closed-formed solution and Gradient descent.
  col1, col2 = st.beta_columns(2)

  # Pros
  with col1:
      st.markdown(''' 
          #### Pros

          * Performs well when the dataset is linearly separable.
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
          [Logistic Regression](https://github.com/SulmanK/MLalgos/blob/main/Algos/model/lr.py)

          ''')


  # Analysis
  st.markdown('''
          ### Decision boundary of my logistic regression implementation.




          ''')


  def plots(classes_widget):
      """Plots for gaussian naive bayes"""

      X_train, X_test, y_train, y_test = get_data_lr(classes_widget)

      # My SVM
      mine_clf = myLogisticRegression(method=method_widget, multi=multi_widget,
                                      regularization=regularization_widget, lambda_=lambda_widget,
                                      learning_rate=lr_widget, iterations=iterations_widget)

      fig_db = ScatterPlotDB_lr(X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test,
                                clf_1=mine_clf,
                                title_1='Scatterplot with Decision Boundary (Mine)')

      # Cost Function,
      if method_widget != 'Newton':

          # BGD
          if method_widget == 'BGD':
              fig_cost = plot_cost_gd_lr(X_train=X_train, y_train=y_train,
                                         clf=mine_clf, method=method_widget,
                                         iterations=iterations_widget)
          # SGD
          elif method_widget == 'SGD':
              fig_cost = plot_cost_gd_lr(X_train=X_train, y_train=y_train,
                                         clf=mine_clf, method=method_widget,
                                         iterations=len(X_train))

      else:
          fig_cost = None
      return fig_db, fig_cost


  # Streamlit plot figures
  sp_lr_plot, cost_lr_plot = plots(classes_widget=classes_widget)
  st.pyplot(sp_lr_plot)

  if (method_widget != 'Newton') & (cost_lr_plot != None):
      st.markdown('''
       ### Convergence of gradient descent methods 

      ''')

      st.pyplot(cost_lr_plot)
