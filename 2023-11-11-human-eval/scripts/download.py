from enum import Enum
from pathlib import Path

import srsly
import typer
from datasets import load_dataset
from wasabi import msg

from .hellaswag import HellaSwagDataset
from .piqa import PIQADataset
from .utils import Interface, Split


class Dataset(str, Enum):
    piqa = "piqa"
    hellaswag = "hellaswag"


DATASETS = {
    Dataset.piqa.value: PIQADataset,
    Dataset.hellaswag.value: HellaSwagDataset,
}


def download(
    # fmt: off
    output_path: Path = typer.Argument(..., help="Path to save the JSONL file."),
    dataset: Dataset = typer.Option(Dataset.piqa, help="Dataset to download."),
    split: Split = typer.Option(Split.validation, help="Dataset split to convert."),
    interface: Interface = typer.Option(Interface.choice, help="Prodigy interface to use."),
    # fmt: on
):
    """Download datasets from HuggingFace and convert them into Prodigy format"""
    examples = load_dataset(dataset.value, split=split.value)

    # Get converter
    converter = DATASETS[dataset.value].convert_to_prodigy
    annotation_tasks = converter(examples, interface.value)
    srsly.write_jsonl(output_path, annotation_tasks)
    msg.good(f"Saved {len(annotation_tasks)} annotation tasks to {output_path}")


if __name__ == "__main__":
    typer.run(download)
