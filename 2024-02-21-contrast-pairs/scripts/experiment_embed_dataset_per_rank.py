from pathlib import Path

import numpy as np
import torch
import typer
from plotly.figure_factory import create_distplot
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from wasabi import msg

from scripts.preprocessors import DATASET_PREPROCESSORS

app = typer.Typer()

ranked_datasets = [
    # "openai/summarize_from_feedback",
    "stanford/SHP",
    "berkeley-nest/Nectar",
]


@app.command("embed")
def embed():
    """Embed chosen and preference pairs into embeddings per rank."""
    output_dir = Path("embeddings/get-dist-ranking")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    for dataset_name in ranked_datasets:
        msg.divider(f"Embedding dataset: {dataset_name}")
        chosen, _ = DATASET_PREPROCESSORS.get(dataset_name)()
        ranks = ["last", "mid", "next"]

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


@app.command("visualize")
def visualize():
    """Visualize cosine distance distribution per rank."""
    data_dir = Path("embeddings/get-dist-ranking")
    output_dir = Path("outputs")

    for dataset_name in ranked_datasets:
        msg.text(f"Visualizing {dataset_name}...")
        dataset_dir = data_dir / dataset_name.replace("/", "___")

        embeddings = {}
        ranks = ["last", "mid", "next"]

        embeddings["chosen"] = np.load(list(dataset_dir.glob("chosen.npy"))[0])
        for rank in ranks:
            embeddings[f"rejected_{rank}"] = np.load(
                list(dataset_dir.glob(f"rejected_{rank}.npy"))[0]
            )

        hist_data = []
        group_labels = []
        for rank in ranks:
            group_labels.append(f"chosen and rejected ({rank})")
            hist_data.append(
                [
                    cosine(chosen, rejected)
                    for chosen, rejected in zip(
                        embeddings["chosen"], embeddings[f"rejected_{rank}"]
                    )
                ]
            )
        breakpoint()

        layout_properties = {
            "autosize": False,
            "width": 720,
            "height": 540,
            "font_family": "CMU Sans Serif",
            "title_font_family": "CMU Sans Serif",
            "title_font_size": 24,
            "title_text": dataset_name,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis_title": "Cosine distance",
            "yaxis_title": "Probability density",
            "legend_traceorder": "reversed",
        }

        fig = create_distplot(
            hist_data,
            group_labels,
            bin_size=0.1,
            show_rug=False,
        )
        fig.update_layout(
            legend={
                "title": "Pairs",
                "orientation": "h",
                "yref": "paper",
                "y": -0.3,
            },
            **layout_properties,
        )

        outfile = "distance_rank_plot_" + dataset_name.replace("/", "___") + ".html"
        fig.write_html(output_dir / outfile, include_plotlyjs="cdn")


if __name__ == "__main__":
    app()
