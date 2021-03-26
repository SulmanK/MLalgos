#------------------ Packages
from data.dt_get_data import get_data
from layout.scatterplot_dt import ScatterPlot_DT
from layout.scatterplot_db_dt import ScatterPlotDB_DT
from model.dt import myDecisionTree
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
#----------------- Layout
def app():
    # Title
    st.title('Decision Trees')

    # Supervised Learning
    st.markdown(''' ### Supervised Learning
    Firstly, we need to explain supervised learning before we move onto Decision Trees. 
    Supervised learning involves using input features (x) to predict an output (y). 
    Given a training set, we employ a learning algorithm to get our hypothesis function. 
    Then, we feed our dataset to our hypothesis function to make predictions.  The process described is presented in Figure 1.
    ''')

    # Image of Supervised learning
    image = Image.open('assets/SL_diagram.png')
    st.image(image, caption='Figure 1: Supervised learning diagram.',
             use_column_width=False, width=600)

    # Decision Tree
    st.markdown(r'''
    In supervised learning, when the target variable is continuous, it is called regression, and when it is discrete, it is known as classification. 

    ### Decision Trees

    Decision Tree is a non-parametric method that uses a tree-like structure to make 'decisions' by splitting nodes into sub-nodes.
    The splitting is performed indefinitely until there are no features left to continue splitting on.
    There are various methods to conduct the splitting in which differ for classification and regression trees.
    In general, we want to minimize whichever splitting method used. 

    ''')


    image2 = Image.open('assets/tree_layout.jpeg')
    st.image(image2, caption='Figure 2: Decision tree diagram.',
             use_column_width=False, width=600)

    st.markdown(r'''
    #### Procedure
    (1) Iterate through each feature 

    - Calculate the weighted average of the splits
    - Calculate the optimal and suboptimal splits

    (2) Select the feature which gives the least weighted average.

    (3) Continue steps (1,2) until there are no features left or data labels to continue splitting on. 

    #### Classification

    ##### Entropy

    Entropy is the measure of disorder in the dataset.
    In other words, how the data is distributed into each respective class.

    $$
    p =  \frac{Number\_of\_Samples_{i^{th} class}}{Total\_Samples}
    $$

    $$ 
    E = -p * log(p)
    $$

    ##### Gini
    Gini is the probability of correctly labeling a randomly chosen element if it was randomly labeled according to the distribution of labels in the node.

    $$
    Gini = \sum_{i = 1}^{n} p_i^2
    $$

    In our case, we take the gini impurity, which measures the homogenity of the node.

    $$ 
    Gini\_impurity = 1 - Gini
    $$



    #### Regression
    ##### Mean Squared Error
    As noted by its name, MSE measures the quality of an estimator. 

    $$ 
    MSE = \frac{1}{n}\sum_{i = 1}^n(Y_i - \hat{Y_i})^2
    $$
    , where n is the number of samples, $Y_i$ is the prediction, and $\hat{Y_i}$ is the average of predictions.

    ''')




    #Widgets
    method_widget = None
    splitting_method_widget = None


    # Insert parameters
    st.sidebar.title('Parameters')
    col1, col2 = st.sidebar.beta_columns(2)

    # Algorithmic
    with col1:

        method_widget = st.radio(label='Method', key='Methods',
                                  options=['Classification', 'Regression'])

    if method_widget == 'Classification':

        X_train, X_test, y_train, y_test = get_data(method_widget)

        with col2:

            splitting_method_widget = st.radio(label='Splitting Method', key='Splitting Method',
                                    options=['Entropy', 'Gini'])

    elif method_widget == 'Regression':

        X_train, X_test, y_train, y_test = get_data(method_widget)

        with col2:

            splitting_method_widget = st.radio(label='Splitting Method', key='Splitting Method',
                                    options=['MSE'])



    # Split into two columns for Closed-formed solution and Gradient descent.
    col1, col2 = st.beta_columns(2)

    # Pros
    with col1:
        st.markdown(''' 
            #### Pros

            * Easy to understand and intepret
            * Non-parametric model: no assumptions about the data
            * Complexity time of O(d * nlog(n))
            ''')

    # Cons
    with col2:
        st.markdown(''' 
            #### Cons
            * Overfitting
                * Mitigate this by limiting tree depth or pruning

            ''')

    # Implementation code
    st.markdown('''
            ### Implementation


            ''')


    # Analysis
    st.markdown('''
            ### Analysis




            ''')


    def plots(method, splitting_method, X_train, y_train, X_test, y_test):
        """Plots for gaussian naive bayes"""


        if method == 'Classification':
            clf = myDecisionTree(method = method , splitting_method = splitting_method)

            fig = ScatterPlotDB_DT(X_train = X_train, y_train = y_train,
                                   X_test = X_test, y_test = y_test,
                                   clf_1 = clf, title_1 = 'Scatterplot with Labels (Mine')


        elif method == 'Regression':

            reg = myDecisionTree(method = method , splitting_method = splitting_method)

            fig = ScatterPlot_DT(X_train=X_train, y_train=y_train,
                                  X_test=X_test, y_test=y_test,
                                  reg = reg
                                  )

        return fig


    # Streamlit plot figures
    if method_widget == 'Classification':
        st.markdown('''
            #### Scatterplot with Labels




            ''')


        dt_plot = plots(method = method_widget, splitting_method = splitting_method_widget,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                            )
        st.pyplot(dt_plot)

    elif method_widget == 'Regression':
        st.markdown(r'''
            #### Scatterplot

            The coefficient of determination (
            $
            r^2 
            $
            ) does not seem to represent the data well.
            This can be attributed to how my decision tree implementation handles integer features, as it's only doing a midpoint split instead of comparing various percentiles of the feature.



            ''')


        dt_plot = plots(method = method_widget, splitting_method = splitting_method_widget,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                            )
        st.pyplot(dt_plot)