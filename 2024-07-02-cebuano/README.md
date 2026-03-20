<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: IFT exploration

The goal here is to try and practice finetuning an existing base model.
I'll use Aya-23 8B and a Cebuano dataset (from the Aya collection) as an example.
Another thing I can do is to create a custom interface using Prodigy for preference annotations.


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
| `classify-aya` | Classify bot-like vs natural texts using the SEACrowd model. |
| `finetune-t4` | Finetune Aya-23 8B on a T4 instance |
| `finetune-a100` | Finetune Aya-23 8B on an A100 instance |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->