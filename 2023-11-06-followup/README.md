<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Are you sure? Asking LLMs follow-up questions

This project aims to evaluate how LLMs will change their response in
knowledge-based tasks when presented with challenging, "are you sure?"
questions.


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
| `process-datasets` | Convert all datasets into a single format. |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/boolq/train.jsonl` | URL | Training dataset for BoolQ (Clark et al., 2019). |
| `assets/boolq/dev.jsonl` | URL | Validation dataset for BoolQ (Clark et al., 2019). |
| `assets/strategyqa/train.json` | URL | Training dataset for StrategyQA (Geva et al., 2021). |
| `assets/strategyqa/dev.json` | URL | Validation dataset for StrategyQA (Geva et al., 2021). |
| `assets/boolqcs/dev.json` | URL | Validation dataset for the BoolQ constrast sets (Gardner et al., 2020). |
| `assets/boolqnp/train.jsonl` | URL | Natural perturbations for the BoolQ training dataset (Khashabi et al., 2020). |
| `assets/boolqnp/dev.jsonl` | URL | Natural perturbations for the BoolQ validation dataset (Khashabi et al., 2020). |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->