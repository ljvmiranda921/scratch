from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import typer
import numpy as np

pylab.rcParams.update(
    {
        "legend.fontsize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "larger",
        "ytick.labelsize": "larger",
    }
)
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif"})


RESULTS = {
    "Dengue": {
        "GPT-4": (62.04, 0.20),
        "GPT-3.5": (51.21, 0.38),
        "Claude-1": (35.85, 0.02),
        "Command": (39.27, 0.64),
        "Dolly-v2 7b": (27.26, 0.40),
        "Falcon 7b": (14.77, 0.35),
        "StableLM 7b": (15.56, 0.08),
        "OpenLLaMa 7b": (15.24, 0.43),
        "calamanCy (large)": (68.42, 0.01),
        "calamanCy (transformer)": (72.45, 0.02),
        "XLM-RoBERTa": (67.20, 0.01),
        "mBERT": (71.07, 0.04),
    },
    "Hatespeech": {
        "GPT-4": (45.74, 1.16),
        "GPT-3.5": (73.90, 0.27),
        "Claude 1": (58.70, 0.03),
        "Command": (16.38, 0.88),
        "Dolly v2 7b": (32.30, 0.18),
        "Falcon 7b": (33.00, 0.11),
        "StableLM 7b": (32.17, 0.24),
        "OpenLLaMa 7b": (32.18, 0.73),
        "calamanCy (large)": (75.62, 0.02),
        "calamanCy (transformer)": (78.25, 0.06),
        "XLM-RoBERTa": (77.57, 0.01),
        "mBERT": (76.40, 0.02),
    },
    "TLUnified-NER": {
        "GPT-4": (65.89, 0.44),
        "GPT-3.5": (53.05, 0.42),
        "Claude-1": (58.88, 0.03),
        "Command": (25.48, 0.11),
        "Dolly-v2 7b": (13.07, 0.14),
        "Falcon 7b": (8.65, 0.04),
        "StableLM 7b": (0.25, 0.03),
        "OpenLLaMa 7b": (15.09, 0.48),
        "calamanCy (large)": (88.90, 0.01),
        "calamanCy (transformer)": (90.34, 0.02),
        "XLM-RoBERTa": (88.03, 0.03),
        "mBERT": (87.40, 0.02),
    },
}


def plot_results(outfile: Path = typer.Argument(..., help="Path to save the output.")):
    """Plot the results and save it to disk."""
    models = list(RESULTS.get("Dengue").keys())

    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    for ax, (dataset, scores) in zip(axs, RESULTS.items()):
        models = list(scores.keys())
        avgs = [avg for avg, _ in scores.values()]
        stds = [std for _, std in scores.values()]
        colors = ["#828282"] * 8 + ["#a00000"] * 4
        ax.bar(np.arange(len(models)), avgs, yerr=stds, color=colors)

        # Format the graph
        ax.set_title(dataset)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=90)
        ax.set_ylim(top=100)

        # Hide the right and top splines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

    fig.text(
        0.01,
        0.60,
        "F1-score",
        ha="center",
        rotation="vertical",
        **{"fontsize": "x-large"}
    )
    fig.tight_layout()
    plt.savefig(outfile, transparent=True)


if __name__ == "__main__":
    typer.run(plot_results)
