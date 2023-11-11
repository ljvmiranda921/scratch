from enum import Enum
from pathlib import Path

import srsly
import typer
from datasets import load_dataset
from wasabi import msg


class Interface(str, Enum):
    choice = "choice"  # https://prodi.gy/docs/api-interfaces#choice
    textbox = "textbox"  # https://prodi.gy/docs/api-interfaces#text


class Split(str, Enum):
    train = "train"
    test = "test"
    validation = "validation"


def download_piqa(
    # fmt: off
    output_path: Path = typer.Argument(..., help="Path to save the JSONL file."),
    split: Split = typer.Option(Split.validation, help="Dataset split to convert."),
    interface: Interface = typer.Option(Interface.choice, help="Prodigy interface to use.")
    # fmt: on
):
    """Download PIQA dataset from HuggingFace and convert it into Prodigy format."""
    examples = load_dataset("piqa", split=split.value)

    annotation_tasks = []
    for eg in examples:
        if interface == Interface.choice.value:
            annotation_tasks.append(
                {
                    "text": eg.get("goal"),
                    "options": [
                        {"id": "sol1", "text": eg.get("sol1")},
                        {"id": "sol2", "text": eg.get("sol2")},
                    ],
                }
            )
        elif interface == Interface.textbox.value:
            annotation_tasks.append(
                {
                    "text": eg.get("goal"),
                    "field_id": "user_input",
                    "field_label": "",
                    "field_rows": 5,
                    "field_placeholder": "Type here...",
                    "field_autofocus": False,
                }
            )
        else:
            msg.fail("Unknown annotation interface.", exits=True)

    srsly.write_jsonl(output_path, annotation_tasks)
    msg.good(f"Saved {len(annotation_tasks)} annotation tasks to {output_path}")


if __name__ == "__main__":
    typer.run(download_piqa)
