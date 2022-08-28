import random
import time
import logging
from json import JSONDecodeError

import streamlit as st

from app_utils.backend_utils import load_statements, query
from app_utils.frontend_utils import (
    set_state_if_absent,
    reset_results,
    entailment_html_messages,
    create_df_for_relevant_snippets,
    create_ternary_plot,
    build_sidebar,
)
from app_utils.config import RETRIEVER_TOP_K


def main():
    statements = load_statements()
    build_sidebar()

    # Persistent state
    set_state_if_absent("statement", "Elvis Presley is alive")
    set_state_if_absent("answer", "")
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_statement_requested", False)

    st.write("# Fact Checking 🎸 Rocks!")
    st.write()
    st.markdown(
        """
    ##### Enter a factual statement about [Rock music](https://en.wikipedia.org/wiki/List_of_mainstream_rock_performers) and let the AI check it out for you...
    """
    )
    # Search bar
    statement = st.text_input(
        "", value=st.session_state.statement, max_chars=100, on_change=reset_results
    )
    col1, col2 = st.columns(2)
    col1.markdown(
        "<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True
    )
    col2.markdown(
        "<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True
    )
    # Run button
    run_pressed = col1.button("Run")
    # Random statement button
    if col2.button("Random statement"):
        reset_results()
        statement = random.choice(statements)
        # Avoid picking the same statement twice (the change is not visible on the UI)
        while statement == st.session_state.statement:
            statement = random.choice(statements)
        st.session_state.statement = statement
        st.session_state.random_statement_requested = True
        # Re-runs the script setting the random statement as the textbox value
        # Unfortunately necessary as the Random statement button is _below_ the textbox
        # Adapted for Streamlit>=1.12
        if hasattr(st, "scriptrunner"):
            raise st.scriptrunner.script_runner.RerunException(
                st.scriptrunner.script_requests.RerunData("")
            )
        else:
            raise st.runtime.scriptrunner.script_runner.RerunException(
                st.runtime.scriptrunner.script_requests.RerunData("")
            )
    else:
        st.session_state.random_statement_requested = False
    run_query = (
        run_pressed or statement != st.session_state.statement
    ) and not st.session_state.random_statement_requested

    # Get results for query
    if run_query and statement:
        time_start = time.time()
        reset_results()
        st.session_state.statement = statement
        with st.spinner("🧠 &nbsp;&nbsp; Performing neural search on documents..."):
            try:
                st.session_state.results = query(statement, RETRIEVER_TOP_K)
                print(statement)
                time_end = time.time()
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                print(f"elapsed time: {time_end - time_start}")
            except JSONDecodeError as je:
                st.error(
                    "👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?"
                )
                return
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    # Display results
    if st.session_state.results:
        results = st.session_state.results
        docs, agg_entailment_info = results["documents"], results["agg_entailment_info"]

        # show different messages depending on entailment results
        max_key = max(agg_entailment_info, key=agg_entailment_info.get)
        message = entailment_html_messages[max_key]
        st.markdown(f"<br/><h4>{message}</h4>", unsafe_allow_html=True)

        st.markdown(f"###### Aggregate entailment information:")
        col1, col2 = st.columns([2, 1])
        agg_entailment_info = results["agg_entailment_info"]
        fig = create_ternary_plot(agg_entailment_info)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write(results["agg_entailment_info"])

        st.markdown(f"###### Most Relevant snippets:")
        df, urls = create_df_for_relevant_snippets(docs)
        st.dataframe(df)
        str_wiki_pages = "Wikipedia source pages: "
        for doc, url in urls.items():
            str_wiki_pages += f"[{doc}]({url}) "
        st.markdown(str_wiki_pages)


main()
