import itertools
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import srsly
import typer
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option

BRAND_COLOR = "#fffff8"  # dirty-white for blog
# Set matplotlb parameters
plt.rcParams.update(
    {
        "text.usetex": True,
        "axes.facecolor": BRAND_COLOR,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.pad": 8,
        "ytick.major.pad": 8,
    }
)


def plot_heatmap(
    # fmt: off
    input_dir: Path = typer.Argument(..., dir_okay=True, help="Path to the metrics/cross directory."),
    output_path: Optional[Path] = typer.Option(None, "--output-path", "--output", "-o", help="Path to save the output plot."),
    metric: str = typer.Option("cats_macro_auc", "--metric", "-m", help="Metric to plot a heatmap for."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Set verbosity."),
    # fmt: on
):
    """Plot a heatmap for the blog post"""
    guidelines = sorted([d.stem for d in input_dir.iterdir()])

    # Parse all the JSONL files from our metrics/cross directory
    msg.text(f"Found {len(guidelines)} guidelines: {guidelines}", show=verbose)
    results: Dict[str, Dict[str, Any]] = {
        d.stem: fetch_results(d) for d in input_dir.iterdir()
    }

    # Prepare data for plotting
    msg.text(f"Using metric: {metric}", show=verbose)
    scores = [
        results.get(g1).get(g2).get(metric)
        for g1, g2 in itertools.product(guidelines, guidelines)
    ]
    n_guides = len(guidelines)
    scores = np.array(scores).reshape(n_guides, n_guides)

    # Plotting proper
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(BRAND_COLOR)
    im = ax.imshow(scores, cmap="Greys")

    ax.set_xticks(np.arange(n_guides), labels=[g.title() for g in guidelines])
    ax.set_yticks(np.arange(n_guides), labels=[g.title() for g in guidelines])
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    ax.set_ylabel("Guideline used as reference", labelpad=20)
    ax.set_xlabel("Guideline used for prediction", labelpad=20)
    ax.set_title(f"Cross-guideline evaluation ({metric})", pad=20)

    for i in range(n_guides):
        for j in range(n_guides):
            color = "white" if scores[i, j] > 0.5 else "k"
            text = ax.text(
                j,
                i,
                f"{scores[i, j] * 100 :.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=15,
            )
    fig.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, format="png")
        msg.good(f"Saved to {output_path}")


def fetch_results(directory: Path) -> Dict[str, Any]:
    parse = lambda x: x.stem.split("-")[2]
    results = {parse(fp): srsly.read_json(fp) for fp in directory.glob("**/*.jsonl")}
    return results


if __name__ == "__main__":
    typer.run(plot_heatmap)
