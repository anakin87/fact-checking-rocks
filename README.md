---
title: Fact Checking rocks!
emoji: 🎸
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.10.0
app_file: Rock_fact_checker.py
pinned: false
license: apache-2.0
---

# Fact Checking rocks! &nbsp; [![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/anakin87/fact-checking-rocks) [![Generic badge](https://img.shields.io/github/stars/anakin87/fact-checking-rocks?label=Github&style=social)](https://github.com/anakin87/fact-checking-rocks)

## *Fact checking baseline combining dense retrieval and textual entailment*

### Idea 💡
This project aims to show that a naive and simple baseline for fact checking can be built by combining dense retrieval and a textual entailment task (based on Natural Language Inference models).

### System description
This project is strongly based on [Haystack](https://github.com/deepset-ai/haystack), an open source NLP framework to realize search system. The main components of our system are an indexing pipeline and a search pipeline.

#### Indexing pipeline
* [Crawling](https://github.com/anakin87/fact-checking-rocks/blob/321ba7893bbe79582f8c052493acfda497c5b785/notebooks/get_wikipedia_data.ipynb): Crawl data from Wikipedia, starting from the page [List of mainstream rock performers](https://en.wikipedia.org/wiki/List_of_mainstream_rock_performers) and using the [python wrapper](https://github.com/goldsmith/Wikipedia)
* [Indexing through Haystack](https://github.com/anakin87/fact-checking-rocks/blob/321ba7893bbe79582f8c052493acfda497c5b785/notebooks/indexing.ipynb)
  * Preprocess the downloaded documents into chunks consisting of 2 sentences
  * Chunks with less than 10 words are discarded, because not very informative
  * Instantiate a [FAISS](https://github.com/facebookresearch/faiss) Document store and store the passages on it
  * Create embeddings for the passages, using a Sentence Transformer model and save them in FAISS. It seems that the retrieval task will involve [*asymmetric semantic search*](https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search) (statements to be verified are usually shorter than inherent passages), therefore I choose the model `msmarco-distilbert-base-tas-b`.
  * Save FAISS index

#### Search pipeline
