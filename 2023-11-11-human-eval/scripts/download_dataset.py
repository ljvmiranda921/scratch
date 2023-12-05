from pathlib import Path
from typing import Callable

import srsly
import typer
from datasets import load_dataset
from wasabi import msg

from .readers import Dataset, get_dataset_reader
from .utils import Interface, Split


def download_dataset(
    # fmt: off
    output_path: Path = typer.Argument(..., help="Path to save the JSONL file."),
    dataset: Dataset = typer.Option(Dataset.piqa, help="Dataset to download."),
    split: Split = typer.Option(Split.validation, help="Dataset split to convert."),
    interface: Interface = typer.Option(Interface.choice, help="Prodigy interface to use."),
    # fmt: on
):
    """Download datasets from HuggingFace and convert them into Prodigy format"""
    dataset_reader = get_dataset_reader(dataset)
    config = dataset_reader.hf_config
    examples = load_dataset(dataset.value, config, split=split.value)

    # Get converter
    converter: Callable = dataset_reader.convert_to_prodigy
    annotation_tasks = converter(examples, interface.value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(output_path, annotation_tasks)
    msg.good(f"Saved {len(annotation_tasks)} annotation tasks to {output_path}")


if __name__ == "__main__":
    typer.run(download_dataset)
