from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from wasabi import msg

from scripts.preprocessors import DATASET_PREPROCESSORS

ranked_datasets = [
    "openai/summarize_from_feedback",
    "stanford/SHP",
    "berkeley-nest/Nectar",
]


def main():
    output_dir = Path("embeddings/get-dist-ranking")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    for dataset_name in ranked_datasets:
        msg.divider(f"Embedding dataset: {dataset_name}")
        chosen, _ = DATASET_PREPROCESSORS.get(dataset_name)()
        ranks = ["next", "mid", "last"]

        if dataset_name == "berkeley-nest/Nectar":
            ranks += list(range(1, 7 + 1))

        rejected_texts = {
            rank: DATASET_PREPROCESSORS.get(dataset_name)(rejected_idx=rank)[1]
            for rank in ranks
        }

        chosen_encodings = model.encode(
            chosen,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        subdir = output_dir / dataset_name.replace("/", "___")
        subdir.mkdir(exist_ok=True, parents=True)
        output_file = subdir / "chosen.npy"
        np.save(output_file, chosen_encodings)
        msg.good(f"Saved chosen embeddings to {output_file}.")

        for rank, rejected in rejected_texts.items():
            msg.info(f"Embedding rank: {rank}")
            rejected_encodings = model.encode(
                rejected,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            output_file = subdir / f"rejected_{rank}.npy"
            np.save(output_file, rejected_encodings)
            msg.good(f"Saved rejected ({rank}) embeddings to {output_file}.")


if __name__ == "__main__":
    main()
