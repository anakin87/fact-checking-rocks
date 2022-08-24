import streamlit as st

import time
import logging
from json import JSONDecodeError
# from markdown import markdown
# from annotated_text import annotation
# from urllib.parse import unquote
import random
import pandas as pd

from app_utils.backend_utils import load_statements, query
from app_utils.frontend_utils import set_state_if_absent, reset_results, entailment_html_messages 
from app_utils.config import RETRIEVER_TOP_K


def main():


    statements = load_statements()

    # Persistent state
    set_state_if_absent('statement', "Elvis Presley is alive")
    set_state_if_absent('answer', '')
    set_state_if_absent('results', None)
    set_state_if_absent('raw_json', None)
    set_state_if_absent('random_statement_requested', False)


    ## MAIN CONTAINER
    st.write("# Fact checking 🎸 Rocks!")
    st.write()
    st.markdown("""
    ##### Enter a factual statement about [Rock music](https://en.wikipedia.org/wiki/List_of_mainstream_rock_performers) and let the AI check it out for you...
    """)
    # Search bar
    statement = st.text_input("", value=st.session_state.statement,
                             max_chars=100, on_change=reset_results)
    col1, col2 = st.columns(2)
    col1.markdown(
        "<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown(
        "<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
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
        # raise st.script_runner.RerunException(
        #     st.script_request_queue.RerunData(None))
    else:
        st.session_state.random_statement_requested = False
    run_query = (run_pressed or statement != st.session_state.statement) \
        and not st.session_state.random_statement_requested

    # Get results for query
    if run_query and statement:
        time_start = time.time()
        reset_results()
        st.session_state.statement = statement
        with st.spinner("🧠 &nbsp;&nbsp; Performing neural search on documents..."):
            try:
                st.session_state.results = query(
                    statement, RETRIEVER_TOP_K)
                time_end = time.time()
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                print(f'elapsed time: {time_end - time_start}')
            except JSONDecodeError as je:
                st.error(
                    "👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    # Display results
    if st.session_state.results:
        results = st.session_state.results
        docs, agg_entailment_info = results['documents'], results['agg_entailment_info']
        print(results)
        
        max_key = max(agg_entailment_info, key=agg_entailment_info.get)
        message = entailment_html_messages[max_key]
        st.markdown(f'<h4>{message}</h4>', unsafe_allow_html=True)
        st.markdown(f'###### Aggregate entailment information:')
        st.write(results['agg_entailment_info'])
        st.markdown(f'###### Relevant snippets:')
        
        # colms = st.columns((2, 5, 1, 1, 1, 1))
        # fields = ["Page title",'Content', 'Relevance', 'contradiction', 'neutral', 'entailment']
        # for col, field_name in zip(colms, fields):
        #     # header
        #     col.write(field_name)
        df = []
        for doc in docs:
        #     col1, col2, col3, col4, col5, col6 = st.columns((2, 5, 1, 1, 1, 1))
        #     col1.write(f"[{doc.meta['name']}]({doc.meta['url']})")
        #     col2.write(f"{doc.content}")
        #     col3.write(f"{doc.score:.3f}")
        #     col4.write(f"{doc.meta['entailment_info']['contradiction']:.2f}")
        #     col5.write(f"{doc.meta['entailment_info']['neutral']:.2f}")
        #     col6.write(f"{doc.meta['entailment_info']['entailment']:.2f}")
            
            #         'con': f"{doc.meta['entailment_info']['contradiction']:.2f}",
            #         'neu': f"{doc.meta['entailment_info']['neutral']:.2f}",
            #         'ent': f"{doc.meta['entailment_info']['entailment']:.2f}",
            #         # 'url': doc.meta['url'], 
            #         'Content': doc.content} 
            # 
            # 
            # 
            row = {'Title': doc.meta['name'],
                    'Relevance': f"{doc.score:.3f}",
                    'con': f"{doc.meta['entailment_info']['contradiction']:.2f}",
                    'neu': f"{doc.meta['entailment_info']['neutral']:.2f}",
                    'ent': f"{doc.meta['entailment_info']['entailment']:.2f}",
                    # 'url': doc.meta['url'], 
                    'Content': doc.content}
            df.append(row)
        st.dataframe(pd.DataFrame(df))#.style.apply(highlight))


    #     if len(st.session_state.results['answers']) == 0:
    #         st.info("""🤔 &nbsp;&nbsp; Haystack is unsure whether any of 
    # the documents contain an answer to your question. Try to reformulate it!""")

    #     for result in st.session_state.results['answers']:
    #         result = result.to_dict()
    #         if result["answer"]:
    #             if alert_irrelevance and result['score'] < LOW_RELEVANCE_THRESHOLD:
    #                 alert_irrelevance = False
    #                 st.write("""
    #                 <h4 style='color: darkred'>Attention, the 
    #                 following answers have low relevance:</h4>""",
    #                          unsafe_allow_html=True)

    #         answer, context = result["answer"], result["context"]
    #         start_idx = context.find(answer)
    #         end_idx = start_idx + len(answer)
    #         # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
    #         st.write(markdown("- ..."+context[:start_idx] +
    #                 str(annotation(answer, "ANSWER", "#3e1c21", "white")) + 
    #                 context[end_idx:]+"..."), unsafe_allow_html=True)
    #         source = ""
    #         name = unquote(result['meta']['name']).replace('_', ' ')
    #         url = result['meta']['url']
    #         source = f"[{name}]({url})"
    #         st.markdown(
    #             f"**Score:** {result['score']:.2f} -  **Source:** {source}")

# def make_pretty(styler):
#     styler.set_caption("Weather Conditions")
#     # styler.format(rain_condition)
#     styler.format_con(lambda v: v.float(v))
#     styler.background_gradient(axis=None, vmin=0, vmax=1, cmap="YlGnBu")
#     return styler

def highlight(s):
    return ['background-color: red']*5
main()