import streamlit as st


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# Small callback to reset the interface in case the text of the question changes
def reset_results(*args):
    st.session_state.answer = None
    st.session_state.results = None
    st.session_state.raw_json = None


              