import torch
from sentence_transformers import SentenceTransformer
from scripts.preprocessors import compute_elo_rankings
from datasets import load_dataset
import pandas as pd
from typing import List
from scipy.spatial.distance import cosine


def get_text_ratings(dataset_name: str, model: SentenceTransformer):
    if dataset_name == "openai/summarize_from_feedback":
        dataset = load_dataset(
            "openai/summarize_from_feedback",
            name="comparisons",
            split="train",
        )
        dataset_df = dataset.to_pandas()
        dataset_df["id"] = dataset_df["info"].apply(lambda x: x.get("id"))

        rows: List[str, float, float] = []
        for _, instances in dataset_df.groupby("id"):
            matchups = [
                (
                    instance["summaries"][0].get("text"),
                    instance["summaries"][1].get("text"),
                    instance["choice"],
                )
                for _, instance in instances.iterrows()
            ]
            ranked = compute_elo_rankings(matchups)
            chosen_text, _ = ranked[0]
            chosen_emb = model.encode(chosen_text)
            for text, elo_ratings in ranked:
                text_emb = model.encode(text, normalize_embeddings=True)
                dist_from_chosen = cosine(chosen_emb, text_emb)
                rows.append([text, elo_ratings, dist_from_chosen])

        return pd.DataFrame(rows, columns=["text", "elo_ratings", "dist_from_chosen"])


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    df = get_text_ratings("openai/summarize_from_feedback", model=model)
    breakpoint()


if __name__ == "__main__":
    main()
