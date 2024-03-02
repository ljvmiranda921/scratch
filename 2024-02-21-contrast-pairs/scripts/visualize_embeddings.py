from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import typer
from wasabi import msg

from scripts.preprocessors import DATASET_PREPROCESSORS


def visualize_embeddings(
    # fmt: off
    embeddings_dir: Path = typer.Argument(..., help="Directory to the embeddings dictionary."),
    outdir: Path = typer.Argument(..., help="Directory to save the plots.", dir_okay=True),
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

        dataset_name = dataset_dir.name.replace("___", "/")
        preprocessor = DATASET_PREPROCESSORS.get(dataset_name)
        chosen, rejected = preprocessor()
        df = pd.DataFrame(
            {
                # fmt: off
                "x": np.hstack((embeddings["chosen"][:, 0], embeddings["rejected"][:, 0])),
                "y": np.hstack((embeddings["chosen"][:, 1], embeddings["rejected"][:, 1])),
                "text": chosen + rejected,
                "type": ["chosen"] * len(chosen) + ["rejected"] * len(rejected),
                # fmt: on
            }
        )
        datasets[dataset_name] = df

    layout_properties = {
        "autosize": False,
        "width": 720,
        "height": 540,
        "font_family": "CMU Sans Serif",
        "title_font_family": "CMU Sans Serif",
        "title_font_size": 24,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis_title": None,
        "yaxis_title": None,
    }

    msg.info("Plotting the embeddings")
    for dataset, df in datasets.items():
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="type",
            template="simple_white",
            hover_name="text",
            title=dataset,
            color_discrete_map={"chosen": "#a00000", "rejected": "#adadc9"},
        )
        fig.update_layout(legend_title="Preference", **layout_properties)
        outfile = outdir / f"{dataset.replace('/', '___')}.html"
        fig.write_html(outfile, include_plotlyjs="cdn")
        msg.good(f"Saved HTML plot to {outfile}!")


if __name__ == "__main__":
    typer.run(visualize_embeddings)
