import app_knn
import app_linreg
import app_nb
import app_svm
import streamlit as st
PAGES = {
    "K-Nearest Neighbors": app_knn,
    "Linear Regression": app_linreg,
    'Naive Bayes Classifier': app_nb,
    'Support Vector Machines': app_svm
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

