from typing import Optional

import typer
import torch
from transformers import pipeline
from wasabi import msg

id2label = {
    "LABEL_0": "Human-translated",
    "LABEL_1": "Machine-translated",
    "LABEL_2": "Natural",
}


def main(
    # fmt: off
    dataset: str = typer.Argument(..., help="Dataset to test on."),
    split: Optional[str] = typer.Option("train", help="Split to get the instances from."),
    column_name: Optional[str] = typer.Option("text", help="Column name to source the texts."),
    # fmt: on
):
    """Get distribution of bot-like texts in a dataset"""
    msg.info("Loading classification model")
    pipe = pipeline(
        "text-classification",
        model="SEACrowd/mdeberta-v3_sea_translationese",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    breakpoint()


if __name__ == "__main__":
    typer.run(main)
