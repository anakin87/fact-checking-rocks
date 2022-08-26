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

# Fact Checking 🎸 Rocks! &nbsp; [![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/anakin87/fact-checking-rocks) [![Generic badge](https://img.shields.io/github/stars/anakin87/fact-checking-rocks?label=Github&style=social)](https://github.com/anakin87/fact-checking-rocks)

## *Fact checking baseline combining dense retrieval and textual entailment*

### Idea 💡
This project aims to show that a *naive and simple baseline* for fact checking can be built by combining dense retrieval and a textual entailment task (based on Natural Language Inference models).
In a nutshell, the flow is as follows:
* the users enters a factual statement
* the relevant passages are retrieved from the knowledge base using dense retrieval
* the system computes the text entailment between each relevant passage and the statement, using a Natural Language Inference model
* the entailment scores are aggregated to produce a summary score.

### System description 🪄
This project is strongly based on [🔎 Haystack](https://github.com/deepset-ai/haystack), an open source NLP framework to realize search system. The main components of our system are an indexing pipeline and a search pipeline.


#### Indexing pipeline
* [Crawling](https://github.com/anakin87/fact-checking-rocks/blob/321ba7893bbe79582f8c052493acfda497c5b785/notebooks/get_wikipedia_data.ipynb): Crawl data from Wikipedia, starting from the page [List of mainstream rock performers](https://en.wikipedia.org/wiki/List_of_mainstream_rock_performers) and using the [python wrapper](https://github.com/goldsmith/Wikipedia)
* [Indexing](https://github.com/anakin87/fact-checking-rocks/blob/321ba7893bbe79582f8c052493acfda497c5b785/notebooks/indexing.ipynb)
  * preprocess the downloaded documents into chunks consisting of 2 sentences
  * chunks with less than 10 words are discarded, because not very informative
  * instantiate a [FAISS](https://github.com/facebookresearch/faiss) Document store and store the passages on it
  * create embeddings for the passages, using a Sentence Transformer model and save them in FAISS. The retrieval task will involve [*asymmetric semantic search*](https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search) (statements to be verified are usually shorter than inherent passages), therefore I choose the model `msmarco-distilbert-base-tas-b`.
  * save FAISS index

#### Search pipeline

* the user enters a factual statement
* compute the embedding of the user statement using the same Sentence Transformer (`msmarco-distilbert-base-tas-b`)
* retrieve the K most relevant text passages stored in FAISS (along with their relevance scores)
* **text entailment task**: compute the text entailment between each text passage (premise) and the user statement (hypotesis), using a Natural Language Inference model (`microsoft/deberta-v2-xlarge-mnli`). For every text passage, we have 3 scores (summing to 1): entailment, contradiction, neutral. *(For this task, I developed a custom Haystack node: `EntailmentChecker`)*
* aggregate the text entailment scores: compute the weighted average of them, where the weight is the relevance score. **Now it is possible to tell if the knowledge base confirms, is neutral or disproves the user statement.**
* *empirical consideration: if in the first N documents (N<K),  there is a strong evidence of entailment/contradiction (partial aggregate scores > 0.5), it is better not to consider less relevant documents*

### Limits and possible improvements ✨

