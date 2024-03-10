"""Script for running experiments on drawing Elo rating and distance"""

from typing import List

import pandas as pd
import plotly.express as px
import torch
import typer
from datasets import load_dataset
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from wasabi import msg

from scripts.preprocessors import compute_elo_rankings

app = typer.Typer()


DATASETS = [
    "openai/summarize_from_feedback",
    "stanfordnlp/SHP",
    # "berkeley-nest/Nectar",
]


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
        for _, instances in tqdm(dataset_df.groupby("post_id")):
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
    for dataset in DATASETS:
        df = _get_text_ratings(dataset, model=model)
        file_name = dataset.replace("/", "___")
        df.to_csv(f"outputs/{file_name}.csv", index=False)


@app.command("visualize")
def visualize():
    """Visualize results, compute for pearson correlation"""

    layout_properties = {
        "autosize": False,
        "width": 500,
        "height": 500,
        "font_family": "CMU Sans Serif",
        "title_font_family": "CMU Sans Serif",
        "title_font_size": 24,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis_title": "Elo rating",
        "yaxis_title": "Cosine distance",
    }

    for dataset in DATASETS:
        file_name = dataset.replace("/", "___")
        df = pd.read_csv(f"outputs/{file_name}.csv").dropna()
        elo_ratings = df["elo_ratings"].tolist()
        cosine_distance = df["dist_from_chosen"].tolist()
        msg.text(pearsonr(elo_ratings, cosine_distance))

        layout_properties["title_text"] = file_name

        fig = px.scatter(
            df,
            x="elo_ratings",
            y="dist_from_chosen",
            trendline="ols",
            color_discrete_sequence=["#a00000"],
            opacity=0.50,
        )
        fig.update_layout(**layout_properties)
        fig.update_yaxes(range=[0, 2])
        fig.show()


if __name__ == "__main__":
    app()
