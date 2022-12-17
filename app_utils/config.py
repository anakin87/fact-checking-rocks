import streamlit as st

INDEX_DIR = "data/index"
STATEMENTS_PATH = "data/statements.txt"

RETRIEVER_MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"
RETRIEVER_MODEL_FORMAT = "sentence_transformers"
RETRIEVER_TOP_K = 5

# In HF Space, we use microsoft/deberta-v2-xlarge-mnli
# for local testing, a smaller model is better
try:
    NLI_MODEL = st.secrets["NLI_MODEL"]
except:
    NLI_MODEL = "valhalla/distilbart-mnli-12-1"
print(f"Used NLI model: {NLI_MODEL}")
