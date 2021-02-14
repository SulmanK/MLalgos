#------------------ Packages
from layout.confusion_mat_plot import confusion_matrix_dict, confusion_matrix_plot
from layout.metrics import log_loss_mine
from layout.roc_curves import ROC_ratios, roc_curves_side_by_side
from layout.scatterplot_mat_db import scatterplot_matrix_db
from model.naivebayes import myNaiveBayes
from PIL import Image
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np
import pandas as pd
import streamlit as st

#----------------- Layout
def app():
    # Title
    st.title('Naive Bayes Classifier')

    # Supervised Learning
    st.markdown(''' ### Supervised Learning
    Firstly, we need to explain supervised learning before we move onto Naive Bayes. 
    Supervised learning involves using input features (x) to predict an output (y). 
    Given a training set, we employ a learning algorithm to get our hypothesis function. 
    Then, we feed our dataset to our hypothesis function to make predictions.  The process described is presented in Figure 1.
    ''')

    # Image of Supervised learning
    image = Image.open('assets/SL_diagram.png')
    st.image(image, caption='Figure 1: Supervised learning diagram.',
             use_column_width=False, width=600)

    # Linear Regression
    st.markdown(r'''
    In supervised learning, when the target variable is continuous, it is called regression, and when it is discrete, it is known as classification. 

    ### Naive Bayes Classifier
    The Naive Bayes classifier takes on a probabilistic approach for classification. The Bayes component arises from Bayes Theorem as such:

    $$
    P(A | B) = \frac{P(A,B)}{P(B)} = \frac{P(B|A) * P(A)}{P(B)}
    $$

    Generative Classifiers learn a model of the joint probability 
    $
    p(x,y)
    $
    , of the inputs 
    $
    x
    $
    and the output
    $
    y
    $
    , and make their predictions by using Bayes rule to calculate the 
    $
    p(y|x)
    $
    and then picking the most likely 
    $
    y
    $
    . Discriminative classifiers model the posterior 
    $
    p(y|x)
    $
    directly or learn a direct map from inputs 
    $
    x
    $
    to the class labels. It is generally more advisable to use discriminative models because you would rather solve for 
    $
    p(x|y)
    $
    instead of calculating an intermediate step such as 
    $
    p(y|x)
    $
    .

    The Naive portion assumes independence between every pair of feauture in the data.


    Now, let's derive the formula for the Naive Bayes classifier.

    Let 
    $
    (x_1, x_2, ..., x_p)
    $
    be a feature vector and
    $
    y
    $
    be the class label corresponding to this feature vector. Applying Bayes' theorem,

    $$
    P(y|X) = \frac{P(X,y)}{P(X)} = \frac{P(X|y) * P(y)}{P(X)}
    $$

    where 
    $
    X
    $
    is given as 
    $
    X = (x_1, x_2, ..., x_p)
    $
    . By substituting for 
    $
    X
    $
    and expanding using the chain rule we get, 

    $$
    P(y|x_1, x_2, ..., x_p) = \frac{P(x_1, x_2, ..., x_p,y)}{P(X)} = \frac{P(x_1, x_2, ..., x_p|y) * P(y)}{P(x_1, x_2, ..., x_p)}
    $$

    Since, 
    $
    (x_1, x_2, ..., x_p)
    $
    are independent of each other,

    $$
    P(y|x_1, x_2, ..., x_p) = \frac{P(y) * \prod^{p}_{i = 1} P(x_i | y)}{\prod^{p}_{i = 1}P(x_i)}
    $$

    Then, we utilize the MAP (Naximum A Posteriori) decision rule, which assigns the label to the class with the highest posterior.

    $$
    \hat{y} = p(X,p) = p(y, x_1, x_2, ..., x_p) = argmax_{k = [1,2,...K]}P(y) * \prod^{p}_{i = 1}P(x_i|y)
    $$
    We calculate probability for all K classes using the above function and take the maximum value to classify a new point belonging to that class.


    #### Types of Naive Bayes Classifiers
    **Gaussian** : It assumes the features take up a continous value and are sampled from a gaussian distribution. The parameters of the Gaussian are the mean and standard deviation of each feature.
    The conditional probability of each feature is given by
    $$
    P(x_i|y)= \frac{1}{\sqrt{2\pi\sigma2_y}}exp(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2})
    $$

    **Multinomial**: It assumes the features take up discrete counts, and the features are the frequency of the words present in the dataset. 
    $$
    \hat{\theta_{pi}} = \frac{N_{yi} + \alpha}{N_{y} + \alpha * n}
    $$

    $
    N_{yi} = \sum_{x_{train}}x_i
    $
    is the number of times feature 
    $
    i
    $
    appears in a sample of class 
    $
    y
    $
    in the training set. 

    $
    N_y = \sum_{i = 1}^n N_{yi}
    $
    is the total count of all features for class 
    $
    y
    $

    $
    \alpha
    $
    is the smoothing factor, which accounts for all features not present in the training set and prevents zero probabilities.  

    #### Procedure 
    1. Calculate the prior probability of each class (the fraction of each class in the training set) 
    2. For each class: Calculate the conditional probability of each feature in the training set. 
    3. For each sample in the testing set: Take the product of feature's conditional probability calculated in Step 2 and multiply with the prior probability in Step 1 for each class.
    4. Select the maximum probability from Step 3 and assign the label accordingly.
    '''
                )


    # Metrics
    st.markdown(r'''

    ### Metrics

    We'll be observing two metrics - Log-Loss and Reciever Operating Characteristics


    #### Log-Loss

    Log-Loss quantifies the accuracy of a classifier by penalising false classifications.
    It measures the uncertainty of the probabilities calculated by the model to the true labels.

    For Binary Classification, the equation is as follows


    $$
    L_{log}(y,p) = -(ylog(p) + (1 -y)log(1-p))
    $$

    #### Reciever Operating Characteristics

    The ROC curve plots the false positive rate against the true positive rate at varied thresholds set by the classifier.
    In order to calculate the false positive and true positive rates, we need to calculate the number of False Positive, False Negatives, True Positive, and True Negatives.

    FP = True Label (Class 0) | Predicted Label (Class 1)

    FN = True Label (Class 1) | Predicted Label (Class 0)

    TP = True Label (Class 1) | Predicted Label (Class 1)

    FN = True Label (Class 0) | Predicted Label (Class 0)

    $$
    FPR = \frac{FP}{TN + FP}
    $$

    $$
    TPR = \frac{TP}{FN + TP}
    $$



    The area under the curve (AUC) provides an aggregate measure of performance across all possible classification thresholds.


        ''')


    # Split into two columns for Closed-formed solution and Gradient descent.
    col1, col2 = st.beta_columns(2)

    # Pros
    with col1:
        st.markdown(''' 
            #### Pros

            * Ease of implementation.
            * Computational complexity O(C x D)
                * C = # of classes
                * D = # of dimensions (features)

            ''')

    # Cons
    with col2:
        st.markdown(''' 
            #### Cons

            * Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent. 
            * Performance is sensitive to skewed data — that is, when the training data is not representative of the class distributions in the overall population. In this case, the prior estimates will be incorrect.
            ''')

    # Implementation code
    st.markdown('''
            ### Implementation




            ''')


    # Insert parameters


    st.sidebar.title('Parameters')
    col1 = st.sidebar.beta_columns(1)

    # Widgets


    method = st.sidebar.radio(label='Method', key='Sci-kit_regularization',
                              options=['Gaussian', 'Multinomial'])


    if method == 'Multinomial':

        multi_alpha = st.sidebar.slider(
            'Smoothing factor', key='alpha',
            min_value=0.0, max_value=1.0,
            value=1.0, step=0.01)


    # Analysis
    if method == 'Gaussian':

        st.markdown('''

            ### Comparison of decision boundaries between my implementation vs sci-kit learn. 

            ''')


    if method == 'Multinomial':

        st.markdown('''

            ### Comparison of Confusion Matrices between my implementation and sci-kit learn.

            ''')


    # Instantiate of variables

    st.set_option('deprecation.showPyplotGlobalUse', False)


    # Callback function used for populating comparison plots


    # @st.cache(allow_output_mutation=True)
    if method == 'Gaussian':

        from data.nb_gauss_get_data import X_train, X_test, y_train, y_test

        def gaussian_plots(X_train, X_test, y_train, y_test, clf):
            """Plots for gaussian naive bayes"""
            # Sklearns Gaussian NB classifier

            if clf == 'Mine':
                mine_gaussianNB = myNaiveBayes(method='Gaussian', alpha=None)
                mine_gaussianNB.fit(X=X_train, y=y_train)
                y_pred = mine_gaussianNB.predict(X=X_test)
                predict_prob = mine_gaussianNB.predict_proba
                fig = scatterplot_matrix_db(X_train=X_train, y_train=y_train,
                                            clf=mine_gaussianNB, title='Scatterplot Matrix with Decision Boundaries (Mine)')

            elif clf == 'Sklearn':
                sk_gaussianNB = GaussianNB()
                sk_gaussianNB.fit(X=X_train, y=y_train)
                y_pred = sk_gaussianNB.predict(X=X_test)
                predict_prob = sk_gaussianNB.predict_proba(X=X_test)

                fig = scatterplot_matrix_db(X_train=X_train, y_train=y_train,
                                            clf=sk_gaussianNB, title='Scatterplot Matrix with Decision Boundaries (Sklearn)')

            return fig, predict_prob

        # Instantiate plot functions
        mine_spm_plot, mine_predict_prob = gaussian_plots(X_train=X_train, X_test=X_test,
                                                          y_train=y_train, y_test=y_test,
                                                          clf='Mine')

        sk_spm_plot, sk_predict_prob = gaussian_plots(X_train=X_train, X_test=X_test,
                                                      y_train=y_train, y_test=y_test,
                                                      clf='Sklearn')

        # Streamlit app
        st.pyplot(mine_spm_plot)
        st.pyplot(sk_spm_plot)

        # Log Loss Calculation
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(y_train))
        y_test_discrete = le.transform(y_test)
        y_test_discrete = y_test_discrete.reshape((y_test_discrete.shape[0], 1))

        mine_log_loss = log_loss(y_test_discrete, mine_predict_prob)
        sk_log_loss = log_loss(y_test_discrete, sk_predict_prob)

        results = f"The log loss of my implementation is **{mine_log_loss:.3g}** and sci-kit learns is  **{sk_log_loss:.3g}**."

        st.markdown(results)


    elif method == 'Multinomial':

        from data.nb_multi_get_data import X_train, X_test, y_train, y_test

        def multinomial_plots(X_train, X_test, y_train, y_test, alpha):
            """ Create plots for multinomial naive bayes classifier"""

            # Instantiate my classifier
            mine_multinomialNB = myNaiveBayes(method='Multinomial', alpha=alpha)
            mine_multinomialNB.fit(X=X_train, y=y_train)
            mine_y_pred = mine_multinomialNB.predict(X=X_test)
            mine_predict_prob = mine_multinomialNB.predict_proba

           # Scikit Learn Vectorization
            cv = CountVectorizer(encoding='latin-1',
                                 strip_accents='ascii', analyzer='word')

            # Create counts vector for training set
            counts = cv.fit_transform(X_train)

            # Create dataframe of counts vector
            df_counts = pd.DataFrame(
                counts.A, columns=cv.get_feature_names(), index=X_train.index)
            df_counts['Output'] = y_train

            # Prediction
            pred_counts = cv.transform(X_test)

            # DF Counts
            df_pred_counts = pd.DataFrame(
                pred_counts.A, columns=cv.get_feature_names(), index=X_test.index)

            class_prior_ham = len(
                df_counts[df_counts['Output'] == 'ham'])/len(df_counts)
            class_prior_spam = len(
                df_counts[df_counts['Output'] == 'spam'])/len(df_counts)

            # Sk learn
            sk_multinomialNB = MultinomialNB(alpha=alpha)
            sk_multinomialNB.fit(X=df_counts.iloc[:, 0:7649], y=y_train)
            sk_y_pred = sk_multinomialNB.predict(df_pred_counts)
            sk_predict_prob = sk_multinomialNB.predict_proba(X=df_pred_counts)

            # Figure objects
            fig_cm = confusion_matrix_plot(y_test=y_test, y_pred_1=mine_y_pred,
                                           y_pred_2=sk_y_pred)

            fig_roc = roc_curves_side_by_side(
                y_test=y_test, pred_prob_1=mine_predict_prob, pred_prob_2=sk_predict_prob)

            return fig_cm, fig_roc, mine_predict_prob, sk_predict_prob

        # Instantiate plot functions
        cm_plot, roc_plot, mine_predict_prob, sk_predict_prob = multinomial_plots(X_train=X_train, X_test=X_test,
                                                                                  y_train=y_train, y_test=y_test,
                                                                                  alpha=multi_alpha)

        # Streamlit app
        st.pyplot(cm_plot)

        st.markdown('''

            ### Comparison of Reciever Operating Characteristic Curves between my implementation and sci-kit learn.

            ''')
        st.pyplot(roc_plot)

        # Log Loss Calculation
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(y_train))
        y_test_discrete = le.transform(y_test)
        y_test_discrete = y_test_discrete.reshape((y_test_discrete.shape[0], 1))
        mine_log_loss = log_loss_mine(
            prediction_prob=mine_predict_prob, true_value=y_test_discrete)
        sk_log_loss = log_loss_mine(prediction_prob=sk_predict_prob,
                                    true_value=y_test_discrete)
        results = f"The log loss of my implementation is **{mine_log_loss:.3g}** and sci-kit learns is  **{sk_log_loss:.3g}**."

        st.markdown(results)
