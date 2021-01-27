import app_knn
import app_linreg
import streamlit as st
PAGES = {
    "K-Nearest Neighbors": app_knn,
    "Linear Regression": app_linreg
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
