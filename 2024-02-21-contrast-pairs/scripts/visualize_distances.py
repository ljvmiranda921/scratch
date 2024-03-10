from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import typer
from plotly.figure_factory import create_distplot
from scipy.spatial.distance import cosine
from wasabi import msg


def visualize_distances(
    # fmt: off
    embeddings_dir: Path = typer.Argument(..., help="Directory to the embeddings dictionary."),
    outdir: Path = typer.Argument(..., help="Directory to save the plots.", dir_okay=True),
    show: bool = typer.Option(False, help="If set, then show the plot in the browser."),
    # fmt: on
):
    datasets: Dict[str, pd.DataFrame] = {}
    dataset_dirs = list(embeddings_dir.iterdir())

    msg.info(f"Reading UMAP embeddings from {embeddings_dir}")
    for dataset_dir in dataset_dirs:
        msg.text(f"Reading {dataset_dir}...")
        files = list(dataset_dir.glob("*.npy"))
        embeddings = {}
        for file in files:
            if file.name == "chosen.npy":
                embeddings["chosen"] = np.load(file)
            if file.name == "rejected.npy":
                embeddings["rejected"] = np.load(file)

        # Get cosine distance
        dists = [
            cosine(chosen, rejected)
            for chosen, rejected in zip(embeddings["chosen"], embeddings["rejected"])
        ]

        # Save values
        dataset_name = dataset_dir.name.replace("___", "/")
        datasets[dataset_name] = {
            "embeddings": embeddings,
            "distances": np.array(dists),
        }

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

    msg.info("Plotting the distances")

    hist_data = []
    group_labels = []
    for dataset, results in datasets.items():
        hist_data.append(results.get("distances"))
        group_labels.append(dataset)

    fig = create_distplot(hist_data, group_labels, bin_size=0.1, show_rug=False)
    fig.update_layout(
        legend={
            "title": "Datasets",
            "orientation": "h",
            "yref": "paper",
            "y": -0.3,
        },
        **layout_properties,
    )
    outfile = outdir / "distance_hist_plot.html"
    fig.write_html(outfile, include_plotlyjs="cdn")
    if show:
        fig.show()


if __name__ == "__main__":
    typer.run(visualize_distances)
