<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Examining contrast pairs in datasets

I'm curious if lexical-based distances (e.g., get the word embeddings and then
cosine distance) correlate with quality-based distances (e.g., rank distance)
in preference data. My hypothesis is that they are not correlated in some
domains like OpenQA, but they *can* be correlated in some like summarization
or coding.

| Domain        | Source                                                                                                                                               |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Summarization | [Open AI's Summarize from Human Feedback dataset](https://huggingface.co/datasets/openai/summarize_from_feedback)                                    |
| OpenQA        | [Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)                                                          |
| ClosedQA      | [Argilla's Cleaned Ultrafeedback, Flan-v2 subset](https://huggingface.co/datasets/argilla/ultrafeedback-multi-binarized-quality-preferences-cleaned) |
| Mixed         | [AlpacaFarm Human Preferences](https://huggingface.co/datasets/tatsu-lab/alpaca_farm/viewer/alpaca_human_preference)                                 |

Then, for each prompt-preference pair (x, y_w, y_l), I computed two distance metrics:
* The **lexical distance**, which is the cosine distance of the BERT
  embeddings of the chosen and rejected response, including the prompt (x +
  y_w, x + y_l).  
* The **quality distance**, which differs based on the dataset. For datasets with
  ordinal ranking, I computed for the linear difference between ranks. 

Finally, I measured the Pearson correlation between the two distance measures.
Below are my findings:


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
| `embed-dataset` | Embed dataset using an embedding model. |
| `get-correlation` | Compute Pearson correlation for each task. |
| `visualize-embeddings` | Visualize the chosen and rejected pairs for each task. |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->