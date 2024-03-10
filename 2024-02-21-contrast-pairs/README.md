<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Examining contrast pairs in datasets

I'm curious if lexical-based distances (e.g., get the word embeddings and then
cosine distance) correlate with quality-based distances (e.g., rank distance)
in preference data. My hypothesis is that they are not correlated in some
domains like OpenQA, but they *can* be correlated in some like summarization
or coding.


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
| `get-dist-histogram` | Visualize cosine distances between preference pairs from an embedding model. |
| `get-dist-ranking` | Run experiment to get cosine distances for each rank |
| `get-pearson-correlation` | Run experiment to get pearson correlation between distances |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->