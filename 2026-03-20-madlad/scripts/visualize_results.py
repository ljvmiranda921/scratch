"""Visualize classification results as heatmaps (one per category group)."""

import glob
import os
import re
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prompts import FORMAT_LABELS, SIB200_LABELS, TOPIC_LABELS

# Crest color scheme
CREST_COLORS = ["#FFE2C8", "#FFC392", "#FD8153", "#DD3025"]
CREST_CMAP = mcolors.LinearSegmentedColormap.from_list("crest", CREST_COLORS)

# fmt: off
LANG_NAMES = {
    "lo": "Lao", "fo": "Faroese", "ba": "Bashkir", "tk": "Turkmen",
    "sn": "Shona", "su": "Sundanese", "pap": "Papiamento", "ig": "Igbo",
    "zu": "Zulu", "xh": "Xhosa", "ny": "Nyanja", "yo": "Yoruba",
    "st": "Southern Sotho", "lus": "Mizo", "oc": "Occitan", "as": "Assamese",
    "tl": "Tagalog", "ceb": "Cebuano",
}
# fmt: on


def load_all_classified(
    classified_dir: str = "data/classified",
) -> dict[str, pd.DataFrame]:
    """Load classified CSVs, keeping the largest file per language."""
    files = sorted(glob.glob(f"{classified_dir}/*.csv"))
    lang_files: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for f in files:
        match = re.match(r"(.+?)_\d{8}_\d{6}_classified\.csv", f.split("/")[-1])
        if match:
            lang = match.group(1)
            df = pd.read_csv(f)
            lang_files[lang].append((len(df), f))

    result = {}
    for lang, entries in sorted(lang_files.items()):
        _, best_file = max(entries, key=lambda x: x[0])
        result[lang] = pd.read_csv(best_file)
    return result


def build_heatmap_data(
    lang_dfs: dict[str, pd.DataFrame],
    categories: list[str],
    column: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build heatmap matrix for a single category group.

    Returns (matrix, row_labels, col_labels).
    """
    lang_codes = sorted(lang_dfs.keys())
    languages = [LANG_NAMES.get(c, c) for c in lang_codes]
    matrix = np.zeros((len(lang_codes), len(categories)))

    for i, lang in enumerate(lang_codes):
        df = lang_dfs[lang]
        n = len(df)
        if n == 0:
            continue
        for j, cat in enumerate(categories):
            matrix[i, j] = (df[column] == cat).sum() / n

    return matrix, languages, categories


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
):
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.5), max(4, n_rows * 1.0)))

    im = ax.imshow(matrix, cmap=CREST_CMAP, aspect="auto", vmin=0, vmax=1.0)

    # X-axis on top, slanted
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="left", fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Y-axis
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate cells with percentage
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            val = 100 * val
            if val > 0:
                ax.text(
                    j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7, color="k"
                )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% of documents", fontsize=9)

    ax.set_title(title, pad=40, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, ax


def main():
    lang_dfs = load_all_classified()
    if not lang_dfs:
        print("No classified data found in data/classified/")
        return

    groups = [
        ("topic", list(TOPIC_LABELS.keys()), "Topic Classification"),
        ("format", list(FORMAT_LABELS.keys()), "Format Classification"),
        ("sib200", list(SIB200_LABELS.keys()), "SIB-200 Classification"),
    ]

    for column, categories, title in groups:
        matrix, row_labels, col_labels = build_heatmap_data(
            lang_dfs, categories, column
        )
        fig, ax = plot_heatmap(matrix, row_labels, col_labels, title)
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/heatmap_{column}.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Saved {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
