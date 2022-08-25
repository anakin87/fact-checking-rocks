import shutil

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import Pipeline
import streamlit as st

from app_utils.entailment_checker import EntailmentChecker
from app_utils.config import (
    STATEMENTS_PATH,
    INDEX_DIR,
    RETRIEVER_MODEL,
    RETRIEVER_MODEL_FORMAT,
    NLI_MODEL,
)


@st.cache()
def load_statements():
    """Load statements from file"""
    with open(STATEMENTS_PATH) as fin:
        statements = [
            line.strip() for line in fin.readlines() if not line.startswith("#")
        ]
    return statements


# cached to make index and models load only at start
@st.cache(
    hash_funcs={"builtins.SwigPyObject": lambda _: None}, allow_output_mutation=True
)
def start_haystack():
    """
    load document store, retriever, reader and create pipeline
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

    entailment_checker = EntailmentChecker(model_name_or_path=NLI_MODEL, use_gpu=False)

    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=entailment_checker, name="ec", inputs=["retriever"])
    return pipe


pipe = start_haystack()

# the pipeline is not included as parameter of the following function,
# because it is difficult to cache
@st.cache(persist=True, allow_output_mutation=True)
def query(statement: str, retriever_top_k: int = 5):
    """Run query and verify statement"""
    params = {"retriever": {"top_k": retriever_top_k}}
    results = pipe.run(statement, params=params)

    scores, agg_con, agg_neu, agg_ent = 0, 0, 0, 0
    for i, doc in enumerate(results["documents"]):
        scores += doc.score
        ent_info = doc.meta["entailment_info"]
        con, neu, ent = (
            ent_info["contradiction"],
            ent_info["neutral"],
            ent_info["entailment"],
        )
        agg_con += con * doc.score
        agg_neu += neu * doc.score
        agg_ent += ent * doc.score

        # if in the first documents there is a strong evidence of entailment/contradiction,
        # there is no need to consider less relevant documents
        if max(agg_con, agg_ent) / scores > 0.5:
            results["documents"] = results["documents"][: i + 1]
            break

    results["agg_entailment_info"] = {
        "contradiction": round(agg_con / scores, 2),
        "neutral": round(agg_neu / scores, 2),
        "entailment": round(agg_ent / scores, 2),
    }
    return results
