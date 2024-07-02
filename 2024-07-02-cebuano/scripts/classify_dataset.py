from typing import Optional

import typer
import torch
from transformers import pipeline
from wasabi import msg


def main(
    # fmt: off
    dataset: str = typer.Argument(..., help="Dataset to test on."),
    split: Optional[str] = typer.Option(None, help="Split to get the instances from."),
    column_name: Optional[str] = typer.Option("text", help="Column name to source the texts."),
    # fmt: on
):
    """Get distribution of bot-like texts in a dataset"""
    pipe = pipeline(
        "text-classification",
        model="SEACrowd/mdeberta-v3_sea_translationese",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    typer.run(main)
