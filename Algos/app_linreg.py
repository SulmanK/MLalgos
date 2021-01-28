# ------------------ Packages
from data.linreg_get_data import X_train, X_test, y_train, y_test
from layout.plot_cost import plot_cost_gd
from layout.plot_predicted_dist import plot_pred_histogram
from model.linreg import myLinearRegression
from PIL import Image
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error


import numpy as np
import streamlit as st


# ----------------- Layout
def app():

    # Title
    st.title('Linear Regression')

    # Supervised Learning
    st.markdown(''' 
        ### Supervised Learning
        Firstly, we need to explain supervised learning before we move onto linear regression.
        Supervised learning involves using input features (x) to predict an output (y).
        Given a training set, we employ a learning algorithm to get our hypothesis function.
        Then, we feed our dataset to our hypothesis function to make predictions.  The process described is presented in Figure 1.
         ''')

    # Image of Supervised learning
    image = Image.open('SL_diagram.png')
    st.image(image, caption='Figure 1: Supervised learning diagram.',
             use_column_width=False, width=600)

    # Linear Regression
    st.markdown('''
         In supervised learning, when the target variable is continuous, it is called regression, and when it is discrete, it is known as classification.

         # Linear Regression
         We need to represent the hypothesis function.
         '''
                )
    st.markdown(r'''
             $$
             h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2
             $$


             $$
             h(x) = \sum_{i = 0}^{d} \theta_i x_i = \theta^Tx
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

             We'll use the ordinary least squares regression model to get our cost function which is a measure of our loss function over our entire training set.

             $$
             J(\theta) =  \frac{1}{2} \sum_{i = 1}^{m} (h_\theta(x^{(i)})- y^{(i)})^2
             $$

             Our goal is to minimize our cost function (the sum of our loss function across the training set is minimized).
             We'll be examining two solutions to achieve this, a closed-form solution and various gradient descent methods.

             # Closed-form Solution (Normal)
             Closed-form solution minimizes J by taking its derivatives with respect to
             $
             \theta_j
             $
             's and setting them to zero.

             $$
             \theta = (X^TX)^{-1}X^Ty
             $$

             # Gradient Descent
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
             . An example of the trajectory of a gradient descent algorithm is presented in Figure 2.

             ''')

    # Image of GD
    image = Image.open('GD_diagram.png')
    st.image(image, caption='Figure 2: Gradient descent diagram.',
             use_column_width=False, width=600)

    # Linear Regression Cost Function / Gradient Descent
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

         In our case, we'll use three different gradient descent methods: Batch gradient descent, Stochastic gradient descent and Minibatch gradient descent.

         # Batch Gradient Descent

         Repeats the update rule above on all training examples on each iteration for a specified number of iterations.

         # Stochastic Gradient Descent

         Repeats the update rule on a single training example selected at random for the total number of samples in the training set.

         # Minibatch Gradient Descent

         Compromise between minibatch and stochastic gradient descent.
         Repeats the update rule above on a randomly selected subset of the training samples (minibatch) for a specified number of iterations.

         ''')

    # Metrics
    st.markdown(r'''

         # Metrics

         We'll be observing two metrics - Mean Squared Error and Coefficient of Determination
         $
         R^2
         $


         # Mean Squared Error

         Calculates the mean squared errors of the predictions to true values.

         $$
         MSE = \frac{1}{n} \sum_{i = 1}^{n} (Y_{true(i)} - Y_{pred(i)})^2
         $$

         # Coefficient of Determination

         Describes the variance with respect to the mean of the true values and variance in respect to our predictions.

         $$
         R^2 = 1 - \frac{SS_res}{SS_tot} = 1 - \frac{\sum_{i = 1}^{m} (y_{true(i)} - y_{pred (i)} )^2 }{\sum_{i = 1}^{m} (y_{true(i)} - y_{mean} )^2}
         $$


             ''')

    # Regularization
    st.markdown(r'''
         # Regularization

         Let's say our model is giving us fantastic results but we are afraid of overfitting.
         We can penalize our loss function by adding a
         $
         L_1
         $
         (Lasso) or
         $
         L_1
         $
         (Ridge) parameters. We multiply these parameters by
         $
         \lambda
         $
         a scalar value, the optimal value is found by cross-validation methods.

         # Lasso

         $$
         \theta_j := \theta_j - \alpha (\frac{d}{d\theta_j}J(\theta) + \lambda |\theta_j|   )
         $$

          It minimizes the usual sum of squared errors, with a bound on the sum of the absolute values of the coefficients.
          Lasso shrinks coefficients to zero as presented in Figure 3. There is no closed-form solution with the Lasso penalty term.

         ''')

    # Image of Lasso
    image = Image.open('Lasso_diagram.png')
    st.image(image, caption='Figure 3: Lasso diagram.',
             use_column_width=False, width=600)

    # Ridge
    st.markdown(r'''

         # Ridge

         $$
         \theta = (X^TX + \lambda * I_{matrix}  )^{-1}X^Ty
         $$

         $$
         \theta_j := \theta_j - \alpha (\frac{d}{d\theta_j}J(\theta) + \lambda \theta_j^2   )
         $$

         It minimizes the usual sum of squared errors, with a bound on the sum of the squared values of the coefficients.
         Ridge shrinks coefficients to low non-zero values as presented in Figure 4.



             ''')

    # Image of Ridge
    image = Image.open('Ridge_diagram.png')
    st.image(image, caption='Figure 4: Ridge diagram.',
             use_column_width=False, width=600)

    # Split into two columns for Closed-formed solution and Gradient descent.
    col1, col2 = st.beta_columns(2)

    # Pros
    with col1:
        st.markdown('''
                 # Closed-formed Solution

                 * Ease of implementation.
                 * Sufficient for smaller datasets.

                 ''')

    # Cons
    with col2:
        st.markdown('''
                 # Gradient Descent

                 * Computational complexity - faster to find solutions in some cases.

                     * Large datasets - it is more efficient to use a form of gradient descent such as SGD.

                     * Sparse data.
                 ''')

    # Implementation code
    st.markdown('''
                 # Implementation

         [Linear Regression](https://github.com/SulmanK/MLalgos/blob/main/Algos/model/linreg.py)


                 ''')

    # Insert parameters
    st.markdown('''

             # Comparing the distributions of the testing set to the predicted values.

             Use the sidebar widgets to adjust various parameters of the linear regression methods discussed.
             We'll be comparing sci-kit learns implementation to my own.

             # Histograms of various linear regression methods

             ''')

    st.sidebar.title('Parameters')
    col1, col2 = st.sidebar.beta_columns(2)

    with col1:
        st.markdown('''
                # Sci-kit Learn

                ''')

        sk_regularization = st.radio(label='Regularization', key='Sci-kit_regularization',
                                     options=[None, 'L1 (Lasso)', 'L2 (Ridge)'])

        if sk_regularization != None:

            sk_iterations = st.slider(
                'Iterations', key='sk_iterations',
                min_value=0, max_value=3000,
                value=1500, step=1)

            sk_lambda_ = st.slider(
                'Penalty coefficient', key='Sci-kit_lambda',
                min_value=0.0, max_value=10.0,
                value=0.0, step=0.01)

        else:
            sk_lambda_ = 0
            sk_iterations = 0

        st.markdown('''
            # Batch Gradient Descent (BGD)

            ''')

        bgd_learning_rate = st.slider(
            'Learning rate', key='bgd_LR',
            min_value=0.0, max_value=1.0,
            value=0.01, step=0.01)

        bgd_iterations = st.slider(
            'Iterations', key='bgd_iterations',
            min_value=0, max_value=3000,
            value=1500, step=1)

        bgd_regularization = st.radio(label='Regularization', key='BGD_regularization',
                                      options=[None, 'L1 (Lasso)', 'L2 (Ridge)'])

        if bgd_regularization != None:
            bgd_lambda_ = st.slider(
                'Penalty coefficient', key='BGD_lambda',
                min_value=0.0, max_value=10.0,
                value=0.0, step=0.01)

        elif bgd_regularization == None:
            bgd_lambda_ = 0

        st.markdown('''
            # Minibatch Gradient Descent (MBGD)

            ''')

        mbgd_learning_rate = st.slider(
            'Learning rate', key='mbgd_LR',
            min_value=0.0, max_value=1.0,
            value=0.01, step=0.01)

        mbgd_iterations = st.slider(
            'Iterations', key='mbgd_iterations',
            min_value=0, max_value=3000,
            value=1500, step=1)

        mbgd_batch_size = st.slider(
            'Batch size', key='mbgd_batch_size',
            min_value=0, max_value=len(X_train),
            value=50, step=1)

        mbgd_regularization = st.radio(label='Regularization', key='MBGD_regularization',
                                       options=[None, 'L1 (Lasso)', 'L2 (Ridge)'])

        if mbgd_regularization != None:
            mbgd_lambda_ = st.slider(
                'Penalty coefficient', key='MBGD_lambda',
                min_value=0.0, max_value=10.0,
                value=0.0, step=0.01)

        else:
            mbgd_lambda_ = 0

    # Seond column
    with col2:
        st.markdown('''
                # Normal (Closed-form solution)

                ''')

        normal_regularization = st.radio(label='Regularization', key='normal_regularization',
                                         options=[None,  'L2 (Ridge)'])

        if normal_regularization != None:
            normal_lambda_ = st.slider(
                'Penalty coefficient', key='Normal_lambda',
                min_value=0.0, max_value=10.0,
                value=0.0, step=0.01)

        else:
            normal_lambda_ = 0

        st.markdown('''
                # Stochastic Gradient Descent (SGD)

                ''')

        sgd_learning_rate = st.slider(
            'Learning rate', key='sgd_LR',
            min_value=0.0, max_value=1.0,
            value=0.01, step=0.01)

        sgd_regularization = st.radio(label='Regularization', key='SGD_reg',
                                      options=[None, 'L1 (Lasso)', 'L2 (Ridge)'])

        if sgd_regularization != None:
            sgd_lambda_ = st.slider(
                'Penalty coefficient', key='sgd_lambda_',
                min_value=0.0, max_value=10.0,
                value=0.0, step=0.01)
        else:
            sgd_lambda_ = 0

    # Instantiate of variables
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Callback function used for populating comparison plots

    @st.cache(allow_output_mutation=True)
    def comparison_plot_dist(X_train, X_test, y_train, y_test):

        # Instantiate models

        # Sci-kit learn models
        if sk_regularization == None:
            reg_sk = LinearRegression().fit(X_train, y_train)
            y_pred_sk = reg_sk.predict(X_test)
            mse_sk = mean_squared_error(y_test, y_pred_sk)
            r2_sk = r2_score(y_test, y_pred_sk)

        elif sk_regularization == 'L1 (Lasso)':
            reg_sk = Lasso(alpha=sk_lambda_, max_iter=sk_iterations).fit(
                X_train, y_train)
            y_pred_sk = reg_sk.predict(X_test)
            mse_sk = mean_squared_error(y_test, y_pred_sk)
            r2_sk = r2_score(y_test, y_pred_sk)

        elif sk_regularization == 'L2 (Ridge)':
            reg_sk = Ridge(alpha=sk_lambda_, max_iter=sk_iterations).fit(
                X_train, y_train)
            y_pred_sk = reg_sk.predict(X_test)
            mse_sk = mean_squared_error(y_test, y_pred_sk)
            r2_sk = r2_score(y_test, y_pred_sk)

        # My models
        # Normal
        reg_normal = myLinearRegression(method='Normal', regularization=normal_regularization,
                                        lambda_=normal_lambda_, learning_rate=None,
                                        iterations=None, batch_size=None)

        reg_normal.fit(X_train, y_train)
        y_pred_normal = reg_normal.predict(X_test)
        mse_normal = reg_normal.mse_score(y_test, y_pred_normal)
        r2_normal = reg_normal.rsquared_score(y_test, y_pred_normal)

        # BGD
        reg_bgd = myLinearRegression(method='BGD', regularization=bgd_regularization,
                                     lambda_=bgd_lambda_, learning_rate=bgd_learning_rate,
                                     iterations=bgd_iterations, batch_size=None)

        reg_bgd.fit(X_train, y_train)
        cost_bgd = reg_bgd.total_cost
        y_pred_bgd = reg_bgd.predict(X_test)
        mse_bgd = reg_bgd.mse_score(y_test, y_pred_bgd)
        r2_bgd = reg_bgd.rsquared_score(y_test, y_pred_bgd)

        # SGD
        reg_sgd = myLinearRegression(method='SGD', regularization=sgd_regularization,
                                     lambda_=sgd_lambda_, learning_rate=sgd_learning_rate,
                                     iterations=None, batch_size=None)

        reg_sgd.fit(X_train, y_train)
        cost_sgd = reg_sgd.total_cost
        y_pred_sgd = reg_sgd.predict(X_test)
        mse_sgd = reg_sgd.mse_score(y_test, y_pred_sgd)
        r2_sgd = reg_sgd.rsquared_score(y_test, y_pred_sgd)

        # MBGD
        reg_mbgd = myLinearRegression(method='MBGD', regularization=mbgd_regularization,
                                      lambda_=mbgd_lambda_, learning_rate=mbgd_learning_rate,
                                      iterations=mbgd_iterations, batch_size=mbgd_batch_size)

        reg_mbgd.fit(X_train, y_train)
        cost_mbgd = reg_mbgd.total_cost
        y_pred_mbgd = reg_mbgd.predict(X_test)
        mse_mbgd = reg_mbgd.mse_score(y_test, y_pred_mbgd)
        r2_mbgd = reg_mbgd.rsquared_score(y_test, y_pred_mbgd)

        # Create figure
        fig1 = plot_pred_histogram(true_value=y_test,
                                  plot_sk=y_pred_sk, plot_normal=y_pred_normal, plot_bgd=y_pred_bgd, plot_sgd=y_pred_sgd, plot_mbgd=y_pred_mbgd,
                                  mse_sk=mse_sk, mse_normal=mse_normal, mse_bgd=mse_bgd, mse_sgd=mse_sgd, mse_mbgd=mse_mbgd,
                                  rsq_sk=r2_sk, rsq_normal=r2_normal, rsq_bgd=r2_bgd, rsq_sgd=r2_sgd, rsq_mbgd=r2_mbgd)


        fig2 = plot_cost_gd(cost_bgd = cost_bgd, cost_sgd = cost_sgd, cost_mbgd = cost_mbgd,
              bgd_iterations = bgd_iterations, sgd_iterations = len(X_train), mbgd_iterations = mbgd_iterations)


        return fig1, fig2

    # Display the plot
    sl1_plot, sl2_plot = comparison_plot_dist(X_train=X_train, X_test=X_test,
                                   y_train=y_train, y_test=y_test)

    st.pyplot(sl1_plot)


    st.markdown('''

             # Convergence of gradient descent methods

             ''')

    st.pyplot(sl2_plot)
