from pathlib import Path
from typing import Dict, List, Tuple, Literal
from operator import attrgetter

import torch
import typer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from wasabi import msg

from scripts.preprocessors import compute_elo_rankings


def main():
    aspects = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    for aspect in aspects:
        chosen_texts, sorted_rejected = _preprocess_helpsteer(aspect)


def _preprocess_helpsteer(aspect: str) -> Tuple[List[Tuple], List[List[Tuple]]]:
    dataset = load_dataset("nvidia/HelpSteer", split="train")
    df = dataset.to_pandas()

    chosen_texts = []
    rejected_texts = []
    for _, instance in df.groupby("prompt"):
        instance = instance[["response", aspect]]
        responses = sorted(
            list(instance.itertuples(index=False)),
            key=attrgetter(aspect),
            reverse=True,
        )

        chosen_texts.append(responses[0])
        rejected_texts.append(responses[1:])

    return chosen_texts, rejected_texts


if __name__ == "__main__":
    typer.run(main)
