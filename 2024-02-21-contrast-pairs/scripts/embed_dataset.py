from pathlib import Path

import numpy as np
import typer
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from wasabi import msg

from scripts.preprocessors import DATASET_PREPROCESSORS


def main(
    # fmt: off
    dataset_name: str = typer.Argument(..., help="HuggingFace dataset name."),
    output_dir: Path = typer.Argument(..., help="Directory to save the embeddings."),
    embedding_model: str = typer.Option("sentence-transformers/all-mpnet-base-v2", help="HuggingFace namespace for the embedding model."),
    include_prompt: bool = typer.Option(False, help="Include prompt in the chosen and rejected texts."),
    # fmt: on
):

    if dataset_name not in DATASET_PREPROCESSORS:
        msg.fail(
            f"No preprocessor found for {dataset_name}. Available: {', '.join(DATASET_PREPROCESSORS.keys())}",
            exits=1,
        )
    chosen, rejected = DATASET_PREPROCESSORS.get(dataset_name)(include_prompt)

    # Get the embeddings
    model = SentenceTransformer(embedding_model)
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
    msg.info("Applying dimensionality reduction (n_components=2)")
    model = TSNE(n_components=2, verbose=3)
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
