import streamlit as st
import pandas as pd

entailment_html_messages = {
    "entailment": 'The knowledge base seems to <span style="color:green">confirm</span> your statement',
    "contradiction": 'The knowledge base seems to <span style="color:red">contradict</span> your statement',
    "neutral": 'The knowledge base is <span style="color:darkgray">neutral</span> about your statement',
}


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


# Small callback to reset the interface in case the text of the question changes
def reset_results(*args):
    st.session_state.answer = None
    st.session_state.results = None
    st.session_state.raw_json = None


def highlight_cols(s):
    coldict = {"con": "#FFA07A", "neu": "#E5E4E2", "ent": "#a9d39e"}
    if s.name in coldict.keys():
        return ["background-color: {}".format(coldict[s.name])] * len(s)
    return [""] * len(s)


def create_df_for_relevant_snippets(docs):
    rows = []
    urls = {}
    for doc in docs:
        row = {
            "Title": doc.meta["name"],
            "Relevance": f"{doc.score:.3f}",
            "con": f"{doc.meta['entailment_info']['contradiction']:.2f}",
            "neu": f"{doc.meta['entailment_info']['neutral']:.2f}",
            "ent": f"{doc.meta['entailment_info']['entailment']:.2f}",
            "Content": doc.content,
        }
        urls[doc.meta["name"]] = doc.meta["url"]
        rows.append(row)
        df = pd.DataFrame(rows).style.apply(highlight_cols)
    return df, urls
