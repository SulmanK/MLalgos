#------------------ Packages
from data.lwr_get_data import X_train, y_train, X_test, y_test
from layout.scatterplot_lwr import ScatterPlot_LWR
from model.lwr import myLocallyWeightedRegression
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
#----------------- Layout
def app():
    # Title
    st.title('Locally Weighted Regression')

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

    # Locally Weighted Regression
    st.markdown(r'''
    In supervised learning, when the target variable is continuous, it is called regression, and when it is discrete, it is known as classification. 

    ### Locally Weighted Regression

    Locally Weighted Regression is a non-parametric method that modifies least squares regression with an additional weight parameter that measures distances between subsets of the data, similar to a k-nearest-neighbors model.

    ''')

    image2 = Image.open('lwr_diagram.png')
    st.image(image2, caption='Figure 2: Locally weighted regression diagram.',
             use_column_width=False, width=600)


    st.markdown(r'''
    For least squares regression, the cost function was the following.

    $$ 
    J = \sum_{i = 1}^{m} (y^{(i)} - \theta^T x^{(i)})^2
    $$

    For locally weighted regression, we modify the above cost function with the weights term.

    $$ 
    J = \sum_{i = 1}^{m} w^{(i)}(y^{(i)} - \theta^T x^{(i)})^2
    $$

    There are a variety of options you can choose for weights, we'll examine the gaussian weight function.

    $$
    w^{(i)} = exp\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)
    $$

    The main parameter of LWR is 
    $
    \tau
    $
    . It's the bandwidth parameter, which controls the contribution of weights depending on the distance between the points.

    Similarly to linear regression, there are multiple solutions to minimizing the cost function. We'll be using closed form solution.

    $$
    \theta = (X^T W X)^{-1} (X^T W Y)
    $$
    ''')


    # Insert parameters
    st.sidebar.title('Parameters')
    col1 = st.sidebar.beta_columns(1)


    col1, col2 = st.sidebar.beta_columns(2)


    # Algorithm parameters
    with col1:
        tau_widget = st.slider(
            'Bandwidth (Tau)', min_value=0.0, max_value=1.0, value=0.90)


    # Algorithm parameters
    with col2:
        degree_widget = st.slider('Degree of polynomial',
                                  min_value=1, max_value=10, value=3)


    # Split into two columns for Closed-formed solution and Gradient descent.
    col1, col2 = st.beta_columns(2)

    # Pros
    with col1:
        st.markdown(''' 
            #### Pros

            * Allows us to put less care into selecting the features in order to avoid overfitting 
            * Very interpretable
            * Complexity time of O(N^2 * d)
            ''')

    # Cons
    with col2:
        st.markdown(''' 
            #### Cons
            * Requires to keep the entire training set in order to make future predictions
            * The number of parameters grows linearly with the size of the training se
            * Computationally expensive
            ''')


    # Implementation code
    st.markdown('''
            ### Implementation
            [Locally Weighted Regression](https://github.com/SulmanK/MLalgos/blob/main/Algos/model/lwr.py)

            ''')


    # Analysis
    st.markdown('''
            ### Analysis




            ''')


    def plots(X_train, y_train, X_test, y_test, degree, tau):
        """Plots for gaussian naive bayes"""

        reg = myLocallyWeightedRegression(tau=tau, degree=degree)

        fig = ScatterPlot_LWR(X_train=X_train, y_train=y_train,
                              X_test=X_test, y_test=y_test,
                              reg=reg, tau=tau, degree=degree)

        return fig


    # Streamlit plot figures
    sp_lwr_plot = plots(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        degree=degree_widget, tau=tau_widget)
    st.pyplot(sp_lwr_plot)
