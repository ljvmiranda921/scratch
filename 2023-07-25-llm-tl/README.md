<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Benchmarking Tagalog datasets on LLMs

Accompanying spaCy project for my blog post, [*Do large language models work on Tagalog?*](https://ljvmiranda921.github.io/notebook/2023/10/18/llm-tagalog/).
Here, I used [spacy-llm](https://github.com/explosion/spacy-llm) to access different LLMs.
I highly-recommend checking the [documentation](https://spacy.io/api/large-language-models) on how to use the framework.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `process-datasets` | Process the datasets and convert them into spaCy format |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/treebank/UD_Tagalog-Ugnayan/` | Git | Treebank data for UD_Tagalog-Ugnayan. Originally sourced from *Parsing in the absence of related languages: Evaluating low-resource dependency parsers in Tagalog* by Aquino and de Leon (2020). |
| `assets/treebank/UD_Tagalog-TRG/` | Git | Treebank data for UD_Tagalog-TRG. Originally sourced from the thesis, *A treebank prototype for Tagalog*, at the University of Tübingen by Samson (2018). |
| `assets/hatespeech.zip` | URL | Contains 10k tweets with 4.2k testing and validation data labeled as hate speech or non-hate speech (text categorization). Based on *Hate speech in Philippine election-related tweets: Automatic detection and classification using natural language processing* by Cabasag et al. (2019) |
| `assets/dengue.zip` | URL | Contains tweets on dengue labeled with five different categories. Tweets can be categorized to multiple categories at the same time (multilabel text categorization). Based on *Monitoring dengue using Twitter and deep learning techniques* by Livelo and Cheng (2018). |
| `assets/calamancy_gold.tar.gz` | URL | Contains the annotated TLUnified corpora in spaCy format with PER, ORG, LOC as entity labels (named entity recognition). Annotated by three annotators with IAA (Cohen's Kappa) of 0.78. Corpora was based from *Improving Large-scale Language Models and Resources for Filipino* by Cruz and Cheng (2021). |
| `scripts/process_dengue.py` | URL | Processing script for the Dengue dataset |
| `scripts/process_hatespeech.py` | URL | Processing script for the Hatespeech dataset |
| `scripts/split_treebank.py` | URL | Processing script for the UD treebanks |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->