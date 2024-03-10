from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from wasabi import msg

from scripts.preprocessors import DATASET_PREPROCESSORS


class Rank(str, Enum):
    next = "next"
    last = "last"
    mid = "mid"


def main(
    # fmt: off
    dataset_name: str = typer.Argument(..., help="HuggingFace dataset name."),
    output_dir: Path = typer.Argument(..., help="Directory to save the embeddings."),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace namespace for the embedding model."),
    reduce_dims: bool = typer.Option(False, help="Reduce dimensions using t-SNE."),
    bottom_idx: Optional[int] = typer.Option(None, help="Bottom index ranking to use for rejected responses. Prioritized over --bottom-rank."),
    bottom_rank: Optional[Rank] = typer.Option(None, help="Bottom rank for rejected responses. The --bottom-idx is prioritized over this."),
    # fmt: on
):

    if dataset_name not in DATASET_PREPROCESSORS:
        msg.fail(
            f"No preprocessor found for {dataset_name}. Available: {', '.join(DATASET_PREPROCESSORS.keys())}",
            exits=1,
        )

    rejected_idx = None
    if bottom_rank:
        rejected_idx = bottom_rank
    if bottom_idx:
        rejected_idx = bottom_idx
    options = {"rejected_idx": rejected_idx} if rejected_idx else {}

    chosen, rejected = DATASET_PREPROCESSORS.get(dataset_name)(**options)

    # Get the embeddings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(embedding_model, device=device)
    msg.info(f"Embedding the sentences using {embedding_model}")
    embeddings = {
        "chosen": model.encode(
            chosen,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        "rejected": model.encode(
            rejected,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
    }

    # Perform dimensionality reduction
    if reduce_dims:
        msg.info("Applying dimensionality reduction (n_components=2)")
        model = TSNE(n_components=2, verbose=3, metric="cosine")
        combined = np.vstack((embeddings.get("chosen"), embeddings.get("rejected")))
        reduced_dims = model.fit_transform(combined)
        num_chosen = len(embeddings["chosen"])
        embeddings["chosen"] = reduced_dims[:num_chosen]
        embeddings["rejected"] = reduced_dims[num_chosen:]

    # Save the embeddings
    subdir = output_dir / dataset_name.replace("/", "___")
    subdir.mkdir(exist_ok=True, parents=True)
    for k, v in embeddings.items():
        output_file = subdir / f"{k}.npy"
        np.save(output_file, v)
        msg.good(f"Saved embeddings to {output_file}")


if __name__ == "__main__":
    typer.run(main)
