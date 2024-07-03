from typing import Optional

import typer
from transformers import pipeline
from datasets import load_dataset
from wasabi import msg

id2label = {
    "LABEL_0": "Human-translated",
    "LABEL_1": "Machine-translated",
    "LABEL_2": "Natural",
}


def main(
    # fmt: off
    dataset_name: str = typer.Argument(..., help="Dataset to test on."),
    split: str = typer.Option("train", help="Split to get the instances from."),
    subset: Optional[str] = typer.Option(None, help="Subset to get the data."),
    column_name: str = typer.Option("text", help="Column name to source the texts."),
    model: str = typer.Option("SEACrowd/mdeberta-v3_sea_translationese", help="Classifier to use."),
    # fmt: on
):
    """Get distribution of bot-like texts in a dataset"""
    msg.info("Loading the dataset")
    dataset = load_dataset(dataset_name, name=subset, split=split)
    breakpoint()

    msg.info("Loading classification model")
    pipe = pipeline(
        "text-classification",
        model=model,
        device="cuda:0",
    )
    breakpoint()


if __name__ == "__main__":
    typer.run(main)
