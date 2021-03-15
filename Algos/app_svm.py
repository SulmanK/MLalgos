#------------------ Packages
from data.svm_get_data import X_train, y_train
from layout.scatterplot_db_svm import ScatterPlotDB_svm
from model.svm import mySVM
from PIL import Image
from sklearn.svm import SVC

import numpy as np
import pandas as pd
import streamlit as st
#----------------- Layout
def app():
    # Title
    st.title('Support Vector Machine')

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

    ### Support Vector Machine

    Support Vector Machine is a classifier, which uses a hyperplane to separate the data.
    A linear subspace to split by the number of classes across all the dimensions.
    In finding the optimal hyperplane, we need to define an important parameter of a hyperplane, known as the geometric margin (w).

    ''')

    image2 = Image.open('SVM_diagram.png')
    st.image(image2, caption='Figure 2: Support Vector Machine diagram.',
             use_column_width=False, width=600)


    st.markdown(r'''
    To find the most optimal hyperplane, we will have to use constraint-based optimization techniques to solve for it (convex optimization).


    $$ 
    min_{w,b} = ||w||^2  
    $$

    $$
    subject \quad to \: y_i (w \bullet x + b) - 1 \geq 0, i = 1, 2, ...m 
    $$

    When dealing with constraint-based optimizations, we must adhere to the KKT conditions.

    (1) 
    $$
    \frac{d}{d\hat{w}}L(\hat{w}, b, \lambda) = \hat{w} - \sum_{i} \lambda_i y_i \bar{x_i} = 0 
    $$

    (2)

    $$
    \frac{d}{d\hat{w}}L(\hat{w}, b, \lambda) = \hat{w} - \sum_{i} \lambda_i y_i \bar{x_i} = 0 
    $$

    (3) 
    $$ 
    y_i [(\bar{w}, \bar{x}), + b ] - 1 \geq 0 
    $$

    (4)
    $$
    \lambda_i \geq 0 
    $$

    (5)
    $$
    \lambda_i(y_i [(\bar{w}, \bar{x}), + b ] - 1) \geq 0
    $$

    There are two distinct ways to solve this problem.

    (1) Primal 

    (2) Wolfe Dual


    #### Primal

    Model the problem as a lagrangian multiplier problem:

    $$
    \nabla f(x) - \alpha \nabla g(x) = 0 
    $$

    where 
    $
    \alpha
    $
    is called the Lagrange Multiplier

    From equation (1), the SVM problem is formulated as:

    $$
    L(w, b, \alpha) =  \frac{1}{2} ||w||^2 - \sum_{i = 1}^m a_i[y_i (w \bullet x + b) - 1]
    $$

    $$
    min_{w,b} \: max \: L(w,b,a)
    $$

    $$
    subject \: to \: \alpha_i  \geq 0, i = 1, 2, ...m 
    $$

    #### Dual

    We combine KKT (1) and (2):

    $$
    L(\alpha, b) = \sum_{i = 1}^m \alpha_i - \frac{1}{2}\sum_{i = 1}^m \sum_{j = 1}^m \alpha_i \alpha_j y_i y_j (x_i \bullet x_j) 
    $$

    $$
    max_\alpha \: \sum_{i = 1}^m \alpha_i - \frac{1}{2}\sum_{i = 1}^m \sum_{j = 1}^m \alpha_i \alpha_j y_i y_j (x_i \bullet x_j) 
    $$

    $$
    subject \: to \: \alpha_i  \geq 0, i = 1, 2, ...m, \sum_{i = 1}^m \alpha_i y_i = 0
    $$

    The main advantage of the Dual method is the problem can be easily solved with the kernel trick - the idea that we can classify non-linearly separable data by mapping them into kernels. 

    There are two ways we can solve this method using two specific approaches:

    (1) CVXOPT (Convex optimization programming solver)

    (2) SMO (Sequential minimization optimization) algorithm
    ''')



    # Insert parameters
    st.sidebar.title('Parameters')
    col1 = st.sidebar.beta_columns(1)

    # Widgets
    degree_widget = 0
    sigma_widget = 0.5

    method_widget = st.sidebar.radio(label='Method', key='Sci-kit_regularization',
                              options=['CVXOPT', 'SMO'])

    if method_widget == 'CVXOPT':


        col1, col2 = st.sidebar.beta_columns(2)

            # Data parameters
        with col1:
            kernel_widget = st.radio('Kernel', ['linear'])

            if kernel_widget == 'poly':
                degree_widget = st.slider('Degree of polynomial', min_value=1, max_value=10, value=3)

            elif kernel_widget == 'rbf':
                sigma_widget = st.slider('Sigma', min_value=0.0, max_value=25.0, value=0.50)


        # Algorithm parameters
        with col2:
            regularization_widget = st.slider('Regularization', min_value=0.0, max_value=10.0, value=1.00)

    elif method_widget == 'SMO':


        col1, col2 = st.sidebar.beta_columns(2)

            # Data parameters
        with col1:
            kernel_widget = st.radio('Kernel', ['linear'])

            if kernel_widget == 'poly':
                degree_widget = st.slider('Degree of polynomial', min_value=1, max_value=10, value=3)

            elif kernel_widget == 'rbf':
                sigma_widget = st.slider('Sigma', min_value=0.0, max_value=25.0, value=0.50)


        # Algorithm parameters
        with col2:
            regularization_widget = st.slider('Regularization', min_value=0.0, max_value=10.0, value=1.00)



    ### CVXOPT Markdown
    if method_widget == 'CVXOPT':

        st.markdown(r'''
            ##### CVOPT

        CVOPT requires the problem to reframed in the following way.

        $$
        minimize \; \: \frac{1}{2}x^T Px + q^Tx 
        $$

        $$
        subject \; to  \; \; \; \; \; \; \; \; \; \; \; Gx \leq h 
        $$

        $$
        \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \;
        Ax = b
        $$


        $
        P_{ij} = y_i y_j ( x_i \bullet x_j )
        $

        $
        q^T \lambda = \sum_{i}^m q_i \lambda_i  = - \sum_i^m \lambda_i
        $

        $
        G = \begin{bmatrix}
        -1 & 0 & 0 \\
        0 & -1 & 0 \\
        0 & 0 & -1
        \end{bmatrix}
        $
        , Identity matrix of size (m)

        $
        h = \begin{bmatrix}
        0 \\
        0  \\
        0
        \end{bmatrix}
        $
        , matrix of zeros with size m

        $
        A = \begin{bmatrix}
        y_1 y_2 y_3
        \end{bmatrix}
        $
        , y_labels with size m

        $
        b = \begin{bmatrix}
        0
        \end{bmatrix}
        $

        Then, we input our parameters into the solver to obtain the weights and intercept terms from fitting.


        ''')

    # SMO Markdown
    elif method_widget == 'SMO':

            st.markdown(r'''

    ##### SMO
    Presented below is the pseudocode described for the simplified SMO algorithm described in this paper. [Simplified SMO](cs229.stanford.edu/materials/smo.pdf)

    ###### Algorithm
        
        Input:
            C: regularization parameter
            tol: numerical tolerance
            max_passes: max # of times to iterate over alpha's without changing


        Output:
            alphas: Lagrange multipliers for solution
            b: threshold for solution

        * Initialize alphas = 0, b = 0
        * Initialize passes = 0
        * while (passes < max_passes)
            * num_changed_alphas = 0
            * for i = 1,...m,
                * Calculate Prediction Error i
                * if ((y[i]*E[i] < -tol & alpha[i] < C) || (y[i]*E_i > tol & alpha[i] >0)
                    * Select j != i randomly
                    * Calculate Prediction Error j
                    * Store old alphas 
                    * Compute L and H (boundary conditions)
                    * if (L == H)
                        continue to next i.
                    * Compute eta
                    * If (eta >= 0)
                        continue to next i.
                    * Compute and clip new values for alpha[j]
                    * if (|alpha[j] - alpha_old[j]| < 1e-5)
                        continue to next i
                    * Determine value for alpha[i]
                    * Compute b1 and b2
                    * Compute b
                    * num_changed_alphas := num_changed_alphas + 1
                * end if
            * end for
            * if (num_changed_alphas == 0)
                passes := passes + 1
            * else
                passes := 0
        * end while

    ''')

    # Split into two columns for Closed-formed solution and Gradient descent.
    col1, col2 = st.beta_columns(2)

    # Pros
    with col1:
        st.markdown(''' 
            #### Pros

            * Works well when there is a clear margin of separation between classes
            * Highly effective in high dimensional spaces
            * Complexity time of O(max(n,d) min (n,d)^2)
            ''')

    # Cons
    with col2:
        st.markdown(''' 
            #### Cons
            * Not suitable for large datasets
            * No probabilistic explanation for the classification
            ''')


    # Implementation code
    st.markdown('''
            ### Implementation
            [Support Vector Machine](https://github.com/SulmanK/MLalgos/blob/main/Algos/model/svm.py)

            ''')


    # Analysis
    st.markdown('''
            ### Comparison of decision boundaries between my implementation vs sci-kit learn.




            ''')

    def plots(X_train, y_train):
        """Plots for gaussian naive bayes"""
        

        # My SVM
        mine_clf = mySVM(kernel_name = kernel_widget, method = method_widget,
                         C = regularization_widget, degree = degree_widget,
                         sigma = sigma_widget)

        mine_clf.fit(X = X_train, y = y_train)


        # SKlearns SVM
        sk_clf = SVC(kernel = kernel_widget, 
                         C = regularization_widget, degree = degree_widget,
                         gamma = sigma_widget)

        sk_clf.fit(X = X_train, y = y_train)



        print(sk_clf.class_weight_)

        fig = ScatterPlotDB_svm(X_train = X_train, y_train = y_train,
                                clf_1 = mine_clf, clf_2 = sk_clf,
                                title_1 = 'Scatterplot with Decision Boundary (Mine)', title_2 = 'Scatterplot with Decision Boundary (Sklearn)')
        
        return fig

    # Streamlit plot figures
    sp_svm_plot = plots(X_train = X_train, y_train = y_train)
    st.pyplot(sp_svm_plot) 
