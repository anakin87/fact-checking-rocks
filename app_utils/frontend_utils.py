import streamlit as st


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# Small callback to reset the interface in case the text of the question changes
def reset_results(*args):
    st.session_state.answer = None
    st.session_state.results = None
    st.session_state.raw_json = None

entailment_html_messages = {'entailment': 'The knowledge base seems to <span style="color:green">confirm</span> your statement',
                            'contradiction': 'The knowledge base seems to <span style="color:red">contradict</span> your statement',
                            'neutral': 'The knowledge base is <span style="color:darkgray">neutral</span> about your statement'}
