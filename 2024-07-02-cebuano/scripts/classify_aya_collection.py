from collections import Counter
from pathlib import Path
from typing import Optional

import typer
from datasets import Dataset, load_dataset
from transformers import pipeline
from wasabi import msg

id2label = {
    "LABEL_0": "Human-translated",
    "LABEL_1": "Machine-translated",
    "LABEL_2": "Natural",
}


def main(
    # fmt: off
    output_path: Path = typer.Argument(..., help="Path to save the statistics as JSON file."),
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
        msg.text(f"Getting {sample} samples (seed={seed})")
        dataset = dataset.shuffle(seed=seed).select(range(sample))
    df = dataset.to_pandas()
    print(df["dataset_name"].value_counts().to_markdown(tablefmt="github"))

    msg.info("Loading classification model")
    model_name = "SEACrowd/mdeberta-v3_sea_translationese"
    pipe = pipeline(
        "text-classification",
        model=model_name,
        device="cuda:0",
        batch_size=8,
    )

    preds_dataset = {}
    for name, group_df in df.groupby("dataset_name"):
        texts = group_df["inputs"].to_list()
        msg.text(f"Classifying {name} dataset (len={len(texts)})")
        preds = [id2label[pred.get("label")] for pred in pipe(texts)]
        preds_dataset[name] = Counter(preds)
    print(preds_dataset)


if __name__ == "__main__":
    typer.run(main)
