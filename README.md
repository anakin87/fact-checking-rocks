---
title: Fact Checking rocks!
emoji: 🎸
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.10.0
app_file: Rock_fact_checker.py
pinned: true
models: [sentence-transformers/msmarco-distilbert-base-tas-b, microsoft/deberta-v2-xlarge-mnli]
license: apache-2.0
---

# Fact Checking 🎸 Rocks! &nbsp; [![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/anakin87/fact-checking-rocks) [![Generic badge](https://img.shields.io/github/stars/anakin87/fact-checking-rocks?label=Github&style=social)](https://github.com/anakin87/fact-checking-rocks)

## *Fact checking baseline combining dense retrieval and textual entailment*

  - [Idea 💡](#idea)
  - [System description 🪄](#system-description)
    - [Indexing pipeline](#indexing-pipeline)
    - [Search pipeline](#search-pipeline)
  - [Limits and possible improvements ✨](#limits-and-possible-improvements)
  - [Repository structure 📁](#repository-structure)
  - [Installation 💻](#installation)

### Idea
💡 This project aims to show that a *naive and simple baseline* for fact checking can be built by combining dense retrieval and a textual entailment task.
In a nutshell, the flow is as follows:
* the user enters a factual statement
* the relevant passages are retrieved from the knowledge base using dense retrieval
* the system computes the text entailment between each relevant passage and the statement, using a Natural Language Inference model
* the entailment scores are aggregated to produce a summary score.

### System description
🪄 This project is strongly based on [🔎 Haystack](https://github.com/deepset-ai/haystack), an open source NLP framework to realize search system. The main components of our system are an indexing pipeline and a search pipeline.


#### Indexing pipeline
* [Crawling](https://github.com/anakin87/fact-checking-rocks/blob/321ba7893bbe79582f8c052493acfda497c5b785/notebooks/get_wikipedia_data.ipynb): Crawl data from Wikipedia, starting from the page [List of mainstream rock performers](https://en.wikipedia.org/wiki/List_of_mainstream_rock_performers) and using the [python wrapper](https://github.com/goldsmith/Wikipedia)
* [Indexing](https://github.com/anakin87/fact-checking-rocks/blob/321ba7893bbe79582f8c052493acfda497c5b785/notebooks/indexing.ipynb)
  * preprocess the downloaded documents into chunks consisting of 2 sentences
  * chunks with less than 10 words are discarded, because not very informative
  * instantiate a [FAISS](https://github.com/facebookresearch/faiss) Document store and store the passages on it
  * create embeddings for the passages, using a Sentence Transformer model and save them in FAISS. The retrieval task will involve [*asymmetric semantic search*](https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search) (statements to be verified are usually shorter than inherent passages), therefore I choose the model `msmarco-distilbert-base-tas-b`
  * save FAISS index.

#### Search pipeline

* the user enters a factual statement
* compute the embedding of the user statement using the same Sentence Transformer used for indexing (`msmarco-distilbert-base-tas-b`)
* retrieve the K most relevant text passages stored in FAISS (along with their relevance scores)
* **text entailment task**: compute the text entailment between each text passage (premise) and the user statement (hypotesis), using a Natural Language Inference model (`microsoft/deberta-v2-xlarge-mnli`). For every text passage, we have 3 scores (summing to 1): entailment, contradiction and neutral. *(For this task, I developed a custom Haystack node: `EntailmentChecker`)*
* aggregate the text entailment scores: compute the weighted average of them, where the weight is the relevance score. **Now it is possible to tell if the knowledge base confirms, is neutral or disproves the user statement.**
* *empirical consideration: if in the first N passages (N<K),  there is strong evidence of entailment/contradiction (partial aggregate scores > 0.5), it is better not to consider (K-N) less relevant documents.*

### Limits and possible improvements
 ✨ As mentioned, the current approach to fact checking is simple and naive. Some **structural limits of this approach**:
  * there is **no statement detection**. In fact, the statement to be verified is chosen by the user. In real-world applications, this step is often necessary.
  * **Wikipedia is taken as a source of truth**. Unfortunately, Wikipedia does not contain universal knowledge and there is no real guarantee that it is a source of truth. There are certainly very interesting approaches that view a snapshot of the entire web as an uncurated source of knowledge (see [Facebook Research SPHERE](https://arxiv.org/abs/2112.09924)).
  * Several papers and even our experiments show a general effectiveness of **dense retrieval** in retrieving textual passages for evaluating the user statement. However, there may be cases in which the most useful textual passages for fact checking do not emerge from the simple semantic similarity with the statement to be verified.
  * **no organic evaluation** was performed, but only manual experiments.

While keeping this simple approach, some **improvements** could be made:
* For reasons of simplicity and infrastructural limitations, the retrieval uses only a very small portion of the Wikipedia data (artists pags from the [List of mainstream rock performers](https://en.wikipedia.org/wiki/List_of_mainstream_rock_performers)). With these few data available, in many cases the knowledge base remains neutral even with respect to statements about rock albums/songs. Certainly, fact checking **quality could improve by expanding the knowledge base** and possibly extending it to the entire Wikipedia.
* Both the retriever model and the Natural Language Inference model are general purpose models and have not been fine-tuned for our domain. Undoubtedly they can **show better performance if fine-tuned in the rock music domain**. Particularly, the retriever model might be adapted with low effort, using [Generative Pseudo Labelling](https://haystack.deepset.ai/guides/gpl).

### Repository structure
* [Rock_fact_checker.py](Rock_fact_checker.py) and [pages folder](./pages/): multi-page Streamlit web app
* [app_utils folder](./app_utils/): python modules used in the web app
* [notebooks folder](./notebooks/): Jupyter/Colab notebooks to get Wikipedia data and index the text passages (using Haystack)
* [data folder](./data/): all necessary data, including original Wikipedia data, FAISS Index and prepared random statements

### Installation
💻 To install this project locally, follow these steps:
* `git clone https://github.com/anakin87/fact-checking-rocks`
* `cd fact-checking-rocks`
* `pip install -r requirements.txt`

To run the web app, simply type: `streamlit run Rock_fact_checker.py`