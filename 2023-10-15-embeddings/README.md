<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Visualizing Tagalog NER embeddings

This project aims to visualize Tagalog NER embeddings.
I ran RoBERTa Tagalog on the entities for the whole dataset, then clustered them using t-SNE, and visualized the results.
You can find the output in my [blog post](https://ljvmiranda921.github.io/notebook/2023/11/20/tagalog-ner-embeddings/).


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `setup` | Setup corpora and model |
| `embed` | Embed the documents and get the vectors for each named entity |
| `plot` | Plot the chart (highly-customized for blog post) |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `setup` &rarr; `embed` &rarr; `plot` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/calamancy_gold.tar.gz` | URL | Contains the annotated TLUnified corpora in spaCy format with PER, ORG, LOC as entity labels (named entity recognition). Annotated by three annotators with IAA (Cohen's Kappa) of 0.78. Corpora was based from *Improving Large-scale Language Models and Resources for Filipino* by Cruz and Cheng (2021). |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->