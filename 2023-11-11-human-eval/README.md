<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Am I smarter than a text generator?

Revisiting human evaluation benchmarks in language models.


This step requires a [Prodigy](https://prodigy.ai) installation.
First, you need to download the dataset and specify the annotation interface (choice/textbox) it will be converted into.

```sh
weasel run convert-hf-to-prodigy . --vars.dataset piqa --vars.interface choice
```

Then, you can run [Prodigy](https://prodigy.ai) to start annotating (by default, it opens in https://localhost:8080):

```sh
weasel run annotate-lm-task . --vars.dataset piqa --vars.interface choice
```

This saves all your annotations to the `humaneval_{dataset}_{interface}`
Prodigy dataset, which you can then export using [`prodigy db-out`](https://prodi.gy/docs/recipes#db-out) command.
Alternatively, you can also run the `export` command:

```sh
weasel export-annotations . --vars.dataset piqa
```

This will generate a file: `annotations/piqa/humaneval_piqa_choice.jsonl` that contains the human annotations that you can use for gold evaluation.
This JSONL file will also be used later on for LM evaluation. In the custom task, the nested meta.doc features will be used.


We want to compare how well we annotated based on the gold-standard data.
To do this, you can run the `evaluate-gold` command and supplying the dataset we want:

```sh
weasel run evaluate-gold . --vars.dataset piqa
```


LM evaluation is done via Eleuther AI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository (`big-refactor` branch).
You need to install its dependencies via:

```sh
weasel run setup-eleuther-harness
```

It requires a YAML file (and an optional `utils.py`) to define a task.
Since we're only concerned with a particular subset (e.g., only the annotated instances of PIQA), we define our own in the `custom_harness_tasks` directory.
We can then run an evaluation by running the `evaluate-lm` command:

```sh
weasel run evaluate-lm . --vars.dataset piqa
```

Under the hood, what it does is that it runs the harness repo's `lm_eval` script, and passes the task supplied in `custom_harness_tasks`.
Note that each custom implementation of a harness task compares the HF dataset from the annotated one,
so it is **important** that you already have exported the annotations from Prodigy in the `annotations/piqa/*.jsonl` directory.

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
| `setup-eleuther-harness` | Install necessary dependencies to run EleutherAI harness |
| `convert-hf-to-prodigy` | Download dataset from HuggingFace and convert them into Prodigy format |
| `annotate-lm-task` | Annotate a dataset using Prodigy. Requires a Prodigy installation |
| `export-annotations` | Export your annotations into JSONL format for later evaluation |
| `evaluate-gold` | Evaluate against gold dataset |
| `evaluate-lm` | Evaluate an LM on a given task |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `lm-evaluation-harness` | Git | The framework to use for evaluation. Custom tasks are defined in `custom_harness_tasks`. |
| [`annotations/piqa/humaneval_piqa_choice.jsonl`](annotations/piqa/humaneval_piqa_choice.jsonl) | Local | PIQA Human annotations by Lj |
| [`annotations/hellaswag/humaneval_hellaswag_choice.jsonl`](annotations/hellaswag/humaneval_hellaswag_choice.jsonl) | Local | HellaSwag Human annotations by Lj |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->