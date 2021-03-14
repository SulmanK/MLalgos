import app_knn
import app_linreg
import app_lwr
import app_nb
import app_svm
import app_perceptron
import streamlit as st
PAGES = {
    "Linear Regression": app_linreg,
    'Locally Weighted Regression': app_lwr,
    "K-Nearest Neighbors": app_knn,
    'Naive Bayes Classifier': app_nb,
    'Perceptron': app_perceptron,
    'Support Vector Machines': app_svm,
    
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

