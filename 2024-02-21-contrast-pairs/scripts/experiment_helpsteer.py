from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import typer
from datasets import load_dataset
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from wasabi import msg

from scripts.preprocessors import compute_elo_rankings


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    output_dir = Path("embeddings/get-help-steer")

    aspects = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    for aspect in aspects:
        output_dir = output_dir / aspect
        output_dir.mkdir(parents=True, exist_ok=True)
        msg.text(f"Processing for '{aspect}' aspect")
        chosen_texts, sorted_rejected = _preprocess_helpsteer(aspect)
        chosen_embs = model.encode([text.response for text in chosen_texts])
        np.save(output_dir / f"chosen_{aspect}.npy", chosen_embs)
        for rejection_type in ("next", "mid", "last"):
            if rejection_type == "next":
                rejected_texts = [rej[0] for rej in sorted_rejected]
            if rejection_type == "last":
                rejected_texts = [rej[-1] for rej in sorted_rejected]
            if rejection_type == "mid":
                mid_idx = round(np.mean(np.arange(len(sorted_rejected))))
                rejected_texts = [rej[mid_idx] for rej in sorted_rejected]
            rejected_embs = model.encode([text.response for text in rejected_texts])
            np.save(output_dir / f"chosen_{aspect}_{rejection_type}.npy", rejected_embs)


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
