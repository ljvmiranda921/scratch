from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional

import spacy
import srsly
import typer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
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
        _examples = _get_properties(docs)
        msg.text("Obtaining t-SNE plot")
        X = np.vstack([eg.ctx_vector for eg in _examples])
        model = TSNE(n_components=2, random_state=0)
        fit_X = model.fit_transform(X)
        examples = []
        for eg, coord in zip(_examples, fit_X):
            eg = asdict(replace(eg, tsne_x=coord[0], tsne_y=coord[1]))
            eg.pop("ctx_vector")  # to save space
            examples.append(eg)

        msg.info(f"Processed {len(examples)} entities from {embeddings}")
        srsly.write_msgpack(DEFAULT_TSNE_COORDS_PATH, examples)
        msg.good(f"Saved coordinates into a binary file: {DEFAULT_TSNE_COORDS_PATH}")
    else:
        msg.info(f"Using coordinates path from {coords_path}")
        examples = srsly.read_msgpack(coords_path)

    # Plot based on (1) entity type or (2) span properties
    _plot_all(examples, outdir)
    _plot_by_ent(examples, outdir, label="PER")
    _plot_by_ent(examples, outdir, label="ORG")
    _plot_by_ent(examples, outdir, label="LOC")


def _get_properties(docs: Iterable[Doc]) -> Iterable[Example]:
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

            eg = Example(
                text=doc.text,
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


def _plot_all(examples: Iterable[Dict], outdir: Path):
    """Plot all points and color code them based on entity type."""
    fig, ax = plt.subplots(1, 1)

    for entity_type, color in zip(ENTITY_TYPES, ["red", "green", "blue"]):
        x = [eg["tsne_x"] for eg in examples if eg["label"] == entity_type]
        y = [eg["tsne_y"] for eg in examples if eg["label"] == entity_type]
        ax.plot(
            x,
            y,
            marker="o",
            linestyle="",
            color=color,
            alpha=0.4,
            label=entity_type,
        )
    ax.legend()
    fig.tight_layout()
    plt.savefig(outdir / "test.png", transparent=True)


def _plot_by_ent(examples: Iterable[Example], outdir: Path, label: str):
    """Plot points per entity type that corresponds to a span property."""
    fig, ax = plt.subplots(1, 1)
    filtered_examples = [eg for eg in examples if eg["level"] == label]

    for prop, color in zip(SPAN_PROPERTIES, ("red", "blue", "green", "black")):
        x = [eg["tsne_x"] for eg in filtered_examples if eg[prop]]
        y = [eg["tsne_y"] for eg in filtered_examples if eg[prop]]
        ax.plot(x, y, marker="o", linestyle="", color=color, alpha=0.4, label=prop)
    ax.legend()
    fig.tight_layout()
    plt.savefig(outdir / "test.png", transparent=True)


if __name__ == "__main__":
    typer.run(plot)
