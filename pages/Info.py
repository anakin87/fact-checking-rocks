import streamlit as st

from app_utils.frontend_utils import build_sidebar

build_sidebar()

with open("README.md", "r") as fin:
    readme = fin.read().rpartition("---")[-1]

st.markdown(readme, unsafe_allow_html=True)
