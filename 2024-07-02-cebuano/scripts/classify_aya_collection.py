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
    split: str = typer.Option("train", help="Split to get the instances from."),
    sample: Optional[int] = typer.Option(None, help="If set, will select random instances."),
    seed: int = typer.Option(42, help="Random seed for shuffling."),
    # fmt: on
):
    """Get distribution of bot-like texts in the Aya Collection dataset"""
    msg.info("Loading the dataset")
    dataset_name = "CohereForAI/aya_collection_language_split"
    dataset = load_dataset(dataset_name, name="cebuano", split=split)
    if sample:
        dataset = dataset.shuffle(seed=seed).select(range(sample))
    breakpoint()

    msg.info("Loading classification model")
    model_name = "SEACrowd/mdeberta-v3_sea_translationese"
    pipe = pipeline(
        "text-classification",
        model=model_name,
        device="cuda:0",
    )
    breakpoint()


if __name__ == "__main__":
    typer.run(main)
