import streamlit as st
import pandas as pd
import plotly.graph_objects as go


entailment_html_messages = {
    "entailment": 'The knowledge base seems to <span style="color:green">confirm</span> your statement',
    "contradiction": 'The knowledge base seems to <span style="color:red">contradict</span> your statement',
    "neutral": 'The knowledge base is <span style="color:darkgray">neutral</span> about your statement',
}


def build_sidebar():
    sidebar = """
    <h1 style='text-align: center'>Fact Checking 🎸 Rocks!</h1>
    <div style='text-align: center'>
    <i>Fact checking baseline combining dense retrieval and textual entailment</i>
    <p><br/><a href='https://github.com/anakin87/fact-checking-rocks'>Github project</a> - Based on <a href='https://github.com/deepset-ai/haystack'>Haystack</a></p>
    <p><small><a href='https://en.wikipedia.org/wiki/List_of_mainstream_rock_performers'>Data crawled from Wikipedia</a></small></p>
    </div>
    """
    st.sidebar.markdown(sidebar, unsafe_allow_html=True)


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


# Small callback to reset the interface in case the text of the question changes
def reset_results(*args):
    st.session_state.answer = None
    st.session_state.results = None
    st.session_state.raw_json = None


def create_ternary_plot(entailment_data):
    """
    Create a Plotly ternary plot for the given entailment dict.
    """
    hover_text = ""
    for label, value in entailment_data.items():
        hover_text += f"{label}: {value}<br>"

    fig = go.Figure(
        go.Scatterternary(
            {
                "cliponaxis": False,
                "mode": "markers",
                "a": [i for i in map(lambda x: x["entailment"], [entailment_data])],
                "b": [i for i in map(lambda x: x["contradiction"], [entailment_data])],
                "c": [i for i in map(lambda x: x["neutral"], [entailment_data])],
                "hoverinfo": "text",
                "text": hover_text,
                "marker": {
                    "color": "#636efa",
                    "size": [0.01],
                    "sizemode": "area",
                    "sizeref": 2.5e-05,
                    "symbol": "circle",
                },
            }
        )
    )

    fig.update_layout(
        {
            "ternary": {
                "sum": 1,
                "aaxis": makeAxis("Entailment", 0),
                "baxis": makeAxis("<br>Contradiction", 45),
                "caxis": makeAxis("<br>Neutral", -45),
            }
        }
    )
    return fig


def makeAxis(title, tickangle):
    return {
        "title": title,
        "titlefont": {"size": 20},
        "tickangle": tickangle,
        "tickcolor": "rgba(0,0,0,0)",
        "ticklen": 5,
        "showline": False,
        "showgrid": True,
    }


def create_df_for_relevant_snippets(docs):
    """
    Create a dataframe that contains all relevant snippets.
    Also returns the URLs
    """
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


def highlight_cols(s):
    coldict = {"con": "#FFA07A", "neu": "#E5E4E2", "ent": "#a9d39e"}
    if s.name in coldict.keys():
        return ["background-color: {}".format(coldict[s.name])] * len(s)
    return [""] * len(s)
