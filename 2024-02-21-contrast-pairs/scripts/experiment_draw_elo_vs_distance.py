from typing import List

import pandas as pd
import torch
import typer
from datasets import load_dataset
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from scripts.preprocessors import compute_elo_rankings

app = typer.Typer()


def _get_text_ratings(dataset_name: str, model: SentenceTransformer):
    if dataset_name == "openai/summarize_from_feedback":
        dataset = load_dataset(
            "openai/summarize_from_feedback",
            name="comparisons",
            split="train",
        )
        dataset_df = dataset.to_pandas()
        dataset_df["id"] = dataset_df["info"].apply(lambda x: x.get("id"))

        rows: List[str, float, float] = []
        for _, instances in tqdm(dataset_df.groupby("id")):
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

    if dataset_name == "stanfordnlp/SHP":
        dataset = load_dataset("stanfordnlp/SHP", split="train").filter(
            lambda x: x["domain"] == "explainlikeimfive_train"
        )

        dataset_df = dataset.to_pandas()
        rows: List[str, float, float] = []
        for _, instances in tqdm(dataset.groupby("post_id")):
            matchups = [
                (instance["human_ref_A"], instance["human_ref_B"], instance["labels"])
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


@app.command("embed")
def embed():
    """Embed datasets, compute distance, and elo ranking"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    datasets = [
        # "openai/summarize_from_feedback",
        "stanfordnlp/SHP",
        # "berkeley-nest/Nectar",
    ]
    for dataset in datasets:
        df = _get_text_ratings(dataset, model=model)
        file_name = dataset.replace("/", "___")
        df.to_csv(f"outputs/{file_name}.csv", index=False)


if __name__ == "__main__":
    app()
