#------------------ Packages
from data.kmeans_get_data import get_sample_data
from layout.scatterplot_kmeans import ScatterPlot_Kmeans
from model.kmeans import myKmeans
from PIL import Image
import streamlit as st


#----------------- Layout
def app():

    # Title
    st.title('K-means Clustering')

    # Introduction
    st.markdown(r'''
    Firstly, we need to explain unsupervised learning before we move onto K-means clustering. Unsupervised learning involves using input features (x) to predict an output (y). We try to find an underlying pattern or structure given unlabeled data.

    K-means Clustering is an iterative algorithm that loops until it converges to a solution. There are two methods in applying K-means clustering:

    (1) Standard

    (2) K-means +++

    #### Standard

    ##### Pseudocode

        Input: X array, K (number of clusters) integer, u = 0 (cluster center)
        for n ~ 1 to N:
            r_n = [0, 0, ..., 0]
            k = RandomInteger (1, K) 
            r_nk = 1 
        end for
        repeat
            for k ~ 1 to K:
                N_k = \sum_{n = 1}^N r_{nk} 
                 \mu_k = \frac{1}{N_k} \sum_{n = 1}^{N} r_{nk} x_n
            end for
            for n ~ 1 to N:
                r_n = [0, 0, ..., 0]       
                k =  arg min_k||x_n - \mu_k||^2 
                r_{nk} = 1 
            end for
        Return labels from r (responsibility vector) and u cluster means

    #### K-means +++
    K-means +++ is very similar to the standard algorithm.
    However, there is one modification in how the cluster centers are initialized.
    We set the first cluster centers as a random sample from the data. Then, we loop over the rest of the cluster centers.

    ##### Pseudocode
        Input: X array, K (number of clusters) integer, u = 0
        n = RandomInteger(1, N) 
        u_1 = x[n]
        for k ~ 2 to K:
            for n ~ 1 to N:
                d_n = min ||x_n - \mu_k ||^2 
            end for
            for n ~ 1 to N:
                 p_n = \frac{d_n^2}{\sum_n * d_n^2} 
            end for
         n = Discrete(p_1, p2, ..., p_N) 
         u_k = x_n 
        end for 
        Return cluster means

        Begin the rest of the K-means Algorithm

    ''')

    # Split into two columns for PROS and CONS
    col1, col2 = st.beta_columns(2)

    # Pros
    with col1:
        st.markdown(''' 
            #### Pros

            * Relatively simple to implement.

            * Scales to large data sets.

            * Guarantees convergence.

            ''')

    # Cons
    with col2:
        st.markdown(''' 
            #### Cons

        * Computationally intense
            * t - number of iterations

            * k - Number of cluster centers
            
            * n - number of points
            
            * d - number of dimensions
            
            * Algorithmic complexity of O(t  k  n  d)

        * Choosing k manually.

        * Clustering outliers.

        * Scaling with number of dimensions.
            ''')

    # Implementation code
    st.markdown('''
            ### Implementation
            [K-means Clustering](https://github.com/SulmanK/MLalgos/blob/main/Algos/model/kmeans.py)



            ''')

    # Insert parameters
    st.markdown('''

        #### Scatterplot of K-means Clustering


        ''')

    col1, col2 = st.sidebar.beta_columns(2)

    # Data parameters
    with col1:
        st.markdown(''' 
            #### Data parameters

            ''')
        classes_widget = st.radio('Classes', [2, 3, 4])
        number_of_samples_widget = st.slider(
            'Number of samples', min_value=20, max_value=300, value=50)

    # Algorithm parameters
    with col2:
        st.markdown(''' 
            #### Algorithm parameters
            ''')

        method_widget = st.radio('Method', ['K-means', 'K-means +++'])
        num_cluster_widget = st.slider('K', min_value=1, value=2, max_value=4)
        iterations_widget = st.slider(
            'Iterations', min_value=3, value=100, max_value=1000)

    # Get data
    X_train, X_test, y_train, y_test = get_sample_data(
        num_classes=classes_widget, num_samples=number_of_samples_widget)

    # Callback function used for populating comparison plots

    @st.cache(allow_output_mutation=True)
    def plots(X_train, X_test, y_train, y_test):

        fig = ScatterPlot_Kmeans(X_train=X_train, X_test=X_test,
                                 method=method_widget, k=num_cluster_widget,
                                 iterations=iterations_widget,
                                 )

        return fig

    # Display the plot
    sl_plot = plots(X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)

    st.pyplot(sl_plot)
