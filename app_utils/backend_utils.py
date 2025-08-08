import shutil
from typing import List

from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PromptNode
from haystack.pipelines import Pipeline
import streamlit as st

from haystack_entailment_checker import EntailmentChecker
from app_utils.config import (
    STATEMENTS_PATH,
    INDEX_DIR,
    RETRIEVER_MODEL,
    RETRIEVER_MODEL_FORMAT,
    NLI_MODEL,
    PROMPT_MODEL,
)


@st.cache_data
def load_statements():
    """Load statements from file"""
    with open(STATEMENTS_PATH) as fin:
        statements = [
            line.strip() for line in fin.readlines() if not line.startswith("#")
        ]
    return statements


# cached to make index and models load only at start
@st.cache_resource
def start_haystack():
    """
    load document store, retriever, entailment checker and create pipeline
    """
    shutil.copy(f"{INDEX_DIR}/faiss_document_store.db", ".")
    document_store = FAISSDocumentStore(
        faiss_index_path=f"{INDEX_DIR}/my_faiss_index.faiss",
        faiss_config_path=f"{INDEX_DIR}/my_faiss_index.json",
    )
    print(f"Index size: {document_store.get_document_count()}")
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=RETRIEVER_MODEL,
        model_format=RETRIEVER_MODEL_FORMAT,
    )
    entailment_checker = EntailmentChecker(
        model_name_or_path=NLI_MODEL,
        use_gpu=False,
        entailment_contradiction_threshold=0.5,
    )

    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=entailment_checker, name="ec", inputs=["retriever"])

    prompt_node = PromptNode(model_name_or_path=PROMPT_MODEL, max_length=150, task_name="text2text-generation")

    return pipe, prompt_node


pipe, prompt_node = start_haystack()

# the pipeline is not included as parameter of the following function,
# because it is difficult to cache
@st.cache_resource
def check_statement(statement: str, retriever_top_k: int = 5):
    """Run query and verify statement"""
    params = {"retriever": {"top_k": retriever_top_k}}
    return pipe.run(statement, params=params)


@st.cache_resource
def explain_using_llm(
    statement: str, documents: List[Document], entailment_or_contradiction: str
) -> str:
    """Explain entailment/contradiction, by prompting a LLM"""
    premise = " \n".join([doc.content.replace("\n", ". ") for doc in documents])
    if entailment_or_contradiction == "entailment":
        verb = "entails"
    elif entailment_or_contradiction == "contradiction":
        verb = "contradicts"

    prompt = f"Premise: {premise}; Hypothesis: {statement}; Please explain in detail why the Premise {verb} the Hypothesis. Step by step Explanation:"

    print(prompt)
    return prompt_node(prompt)[0]
