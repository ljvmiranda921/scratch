<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Am I smarter than a text generator?

Some Prodigy annotation tasks to try out some common LLM benchmarks
First, you need to download the dataset and specify the annotation interface (choice/textbox) it will be converted into.

```sh
weasel run setup . --vars.dataset piqa --vars.interface choice
```

Then, you can run [Prodigy](https://prodigy.ai) to start annotating:

```sh
weasel run annotate . --vars.dataset piqa --vars.interface choice
```

This saves all your annotations to the `humaneval_{dataset}_{interface}`
Prodigy dataset, which you can then export using [`prodigy db-out`](https://prodi.gy/docs/recipes#db-out) command.


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
| `download` | Download dataset from HuggingFace and convert them into Prodigy format |
| `annotate` | Annotate a dataset using Prodigy. |
| `export` | Export into JSONL format |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`annotations/piqa/humaneval_piqa_choice.jsonl`](annotations/piqa/humaneval_piqa_choice.jsonl) | Local | PIQA Human annotations by Lj |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->