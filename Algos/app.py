import app_knn
import app_linreg
import streamlit as st
PAGES = {
    "App1": app1,
    "App2": app2
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()