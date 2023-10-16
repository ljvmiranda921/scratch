from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional

import spacy
import srsly
import typer
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option

ENTITY_TYPES = ["PER", "ORG", "LOC"]
SPAN_PROPERTIES = ["paren", "all_caps", "initial", "plain"]
DEFAULT_TSNE_COORDS_PATH = "tsne.coords"


@dataclass
class Example:
    """Class for keeping track of an example"""

    text: str
    span_text: str
    label: str
    start_char: int
    end_char: int
    ctx_vector: np.ndarray

    # Other properties, assign the first applicable category
    paren: bool  # if preceded by a left parenthesis
    all_caps: bool  # if it consists of all capitals
    initial: bool  # if it is the first token in a sentence
    plain: bool  # catch-all category

    # t-SNE coordinates for plotting
    tsne_x: float = 0.0
    tsne_y: float = 0.0


def plot(
    # fmt: off
    embeddings: Path = Arg(..., help="Path to a spaCy file containing span embeddings."),
    outdir: Path = Arg(..., help="Directory to save the plots.", dir_okay=True),
    coords_path: Optional[Path] = Opt(None, help="If provided, will use coords from an external file."),
    # fmt: on
):
    """Plot entity embeddings."""
    if not coords_path:
        # Read file
        nlp = spacy.blank("xx")
        doc_bin = DocBin().from_disk(embeddings)
        docs = doc_bin.get_docs(nlp.vocab)

        # Compute the t-SNE coordinates
        _examples = _compute_properties(docs)
        msg.text("Obtaining t-SNE plot")
        X = np.vstack([eg.ctx_vector for eg in _examples])
        model = TSNE(n_components=2, random_state=0)
        fit_X = model.fit_transform(X)
        examples = []
        for eg, coord in zip(_examples, fit_X):
            eg = asdict(replace(eg, tsne_x=coord[0], tsne_y=coord[1]))
            eg.pop("ctx_vector")  # to save space
            examples.append(eg, None)

        msg.info(f"Processed {len(examples)} entities from {embeddings}")
        srsly.write_msgpack(DEFAULT_TSNE_COORDS_PATH, examples)
        msg.good(f"Saved coordinates into a binary file: {DEFAULT_TSNE_COORDS_PATH}")
    else:
        msg.info(f"Using coordinates path from '{coords_path}'")
        examples = srsly.read_msgpack(coords_path)

    # Plot based on (1) entity type or (2) span properties
    df = pd.DataFrame(examples)
    _plot_all(df, outdir)
    # _plot_by_ent(df, outdir, label="PER")
    # _plot_by_ent(df, outdir, label="ORG")
    # _plot_by_ent(df, outdir, label="LOC")


def _compute_properties(docs: Iterable[Doc]) -> Iterable[Example]:
    examples = []
    for doc in tqdm(docs):
        for ent in doc.ents:
            # We only want the first applicable category for
            # per-label plots. This looks a bit ugly.
            if ent.start != 0 and doc[ent.start - 1].text == "(":
                props = (1, 0, 0, 0)
            elif all([token.is_upper for token in ent]):
                props = (0, 1, 0, 0)
            elif ent.start == 0:
                props = (0, 0, 1, 0)
            else:
                props = (0, 0, 0, 1)

            # Create display text for later visualization
            window = 15
            start = max(0, ent.start_char - window)
            end = min(len(doc.text), ent.end_char + window)
            prefix = doc.text[start : ent.start_char]
            suffix = doc.text[ent.end_char : end]
            if start > 0:
                prefix = "..." + prefix
            if end < len(doc.text) - 1:
                suffix = suffix + "..."
            display_text = f"{prefix}<b>{ent.text}</b>{suffix}"

            # Create Example class that contains all computed props
            eg = Example(
                text=doc.text,
                display_text=display_text,
                span_text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                ctx_vector=doc.user_data[
                    ("._.", "ctx_vector", ent.start_char, ent.end_char)
                ],
                paren=bool(props[0]),
                all_caps=bool(props[1]),
                initial=bool(props[2]),
                plain=bool(props[3]),
            )
            examples.append(eg)
    return examples


def _plot_all(df: pd.DataFrame, outdir: Path):
    """Plot all points and color code them based on entity type."""
    fig = px.scatter(
        df,
        x="tsne_x",
        y="tsne_y",
        color="label",
        template="simple_white",
        hover_name="span_text",
        hover_data=["display_text", "span_text", "label"],
    )
    fig.show()


# def _plot_by_ent(df: pd.DataFrame, outdir: Path, label: str):
#     """Plot points per entity type that corresponds to a span property."""
#     fig, ax = plt.subplots(1, 1)
#     filtered_examples = [eg for eg in examples if eg["label"] == label]

#     for prop, color in zip(SPAN_PROPERTIES, ("red", "blue", "green", "black")):
#         x = [eg["tsne_x"] for eg in filtered_examples if eg[prop]]
#         y = [eg["tsne_y"] for eg in filtered_examples if eg[prop]]
#         ax.plot(x, y, marker="o", linestyle="", color=color, alpha=0.4, label=prop)
#     ax.legend()
#     fig.tight_layout()
#     outfile = outdir / f"per_label_{label}.png"
#     plt.savefig(outfile, transparent=True)
#     msg.good(f"Saving plot for label '{label}' in {outfile}")


if __name__ == "__main__":
    typer.run(plot)
