from curses.ascii import NL
from logging import exception
import streamlit as st

INDEX_DIR = "data/index"
STATEMENTS_PATH = "data/statements.txt"

RETRIEVER_MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"
RETRIEVER_MODEL_FORMAT = "sentence_transformers"
RETRIEVER_TOP_K = 5

try:
    NLI_MODEL = st.secrets['NLI_MODEL']
except:
    NLI_MODEL = "valhalla/distilbart-mnli-12-1"
print(f'Used NLI model: {NLI_MODEL}')
