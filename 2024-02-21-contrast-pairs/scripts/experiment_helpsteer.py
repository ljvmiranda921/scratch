from operator import attrgetter
from pathlib import Path
from typing import List, Tuple

import math
import numpy as np
import torch
import typer
from datasets import load_dataset
from plotly.figure_factory import create_distplot
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from wasabi import msg

app = typer.Typer()


@app.command("embed")
def embed():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    output_dir = Path("embeddings/get-help-steer")

    aspects = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    for aspect in aspects:
        aspect_dir = output_dir / aspect
        aspect_dir.mkdir(parents=True, exist_ok=True)
        msg.text(f"Processing for '{aspect}' aspect")
        chosen_texts, sorted_rejected = _preprocess_helpsteer(aspect)

        chosen_embs = model.encode([text.response for text in chosen_texts])
        np.save(aspect_dir / f"chosen_{aspect}.npy", chosen_embs)
        for rejection_type in ("next", "mid", "last"):
            if rejection_type == "next":
                rejected_texts = [rej[0] for rej in sorted_rejected]
            if rejection_type == "last":
                rejected_texts = [rej[-1] for rej in sorted_rejected]
            if rejection_type == "mid":
                mid_idx = lambda x: math.ceil(np.mean(np.arange(len(x))))  # noqa
                rejected_texts = [rej[mid_idx(rej)] for rej in sorted_rejected]
            rejected_embs = model.encode([text.response for text in rejected_texts])
            np.save(
                aspect_dir / f"rejected_{aspect}_{rejection_type}.npy", rejected_embs
            )


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

        if len(responses[0]) > 1 and len(responses[1:]) >= 1:
            chosen_texts.append(responses[0])
            rejected_texts.append(responses[1:])

    return chosen_texts, rejected_texts


@app.command("visualize")
def visualize():
    output_dir = Path("embeddings/get-help-steer")
    aspects = reversed(
        ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    )

    hist_data = []
    group_labels = []
    for aspect in aspects:
        aspect_dir = output_dir / aspect
        chosen_embeddings = np.load(aspect_dir / f"chosen_{aspect}.npy")
        rejected_embeddings = np.load(aspect_dir / f"rejected_{aspect}_last.npy")
        distances = [
            cosine(chosen, rejected)
            for chosen, rejected in zip(chosen_embeddings, rejected_embeddings)
        ]
        hist_data.append(distances)
        group_labels.append(aspect.title())

    layout_properties = {
        "autosize": False,
        "width": 720,
        "height": 540,
        "font_family": "CMU Sans Serif",
        "title_font_family": "CMU Sans Serif",
        "title_font_size": 24,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis_title": "Cosine distance",
        "yaxis_title": "Probability density",
    }

    colors = {
        "silver": "#adadc9",
        "pewter": "#696880",
        "stone_gray": "#928E85",
        "slate_gray": "#708090",
        "crimson": "#a00000",
        # "medium_gray": "#bebebe",
    }

    fig = create_distplot(
        hist_data,
        group_labels,
        bin_size=0.1,
        show_rug=False,
        show_hist=False,
        colors=list(list(colors.values())),
    )
    fig.update_layout(
        legend={
            "title": "Preference aspects",
            "orientation": "h",
            "yref": "paper",
            "y": -0.3,
        },
        **layout_properties,
    )

    outfile = Path("outputs") / "distance_helpsteer_plot.html"
    fig.write_html(outfile, include_plotlyjs="cdn")
    fig.show()


if __name__ == "__main__":
    app()
