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
    display_text: str
    span_text: str
    label: str
    start_char: int
    end_char: int
    ctx_vector: np.ndarray
    props: str

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
            eg.pop("ctx_vector", None)  # to save space
            examples.append(eg)

        msg.info(f"Processed {len(examples)} entities from {embeddings}")
        srsly.write_msgpack(DEFAULT_TSNE_COORDS_PATH, examples)
        msg.good(f"Saved coordinates into a binary file: {DEFAULT_TSNE_COORDS_PATH}")
    else:
        msg.info(f"Using coordinates path from '{coords_path}'")
        examples = srsly.read_msgpack(coords_path)

    # Prepare plotting variables
    df = pd.DataFrame(examples)
    colors = {
        "crimson": "#a00000",
        "silver": "#adadc9",
        "pewter": "#696880",
        "stone_gray": "#928E85",
        "slate_gray": "#708090",
        "medium_gray": "#bebebe",
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
        "xaxis_title": None,
        "yaxis_title": None,
    }

    # Plot all points and label them by entity type
    fig_all_points = px.scatter(
        df,
        x="tsne_x",
        y="tsne_y",
        color="label",
        template="simple_white",
        hover_name="span_text",
        hover_data=["display_text", "span_text", "label"],
        title="All labels",
        color_discrete_map={
            "PER": colors.get("crimson"),
            "ORG": colors.get("silver"),
            "LOC": colors.get("pewter"),
        },
    )
    fig_all_points.update_layout(legend_title="Entity label", **layout_properties)
    fig_all_points.write_html(outdir / "fig_all_points.html", include_plotlyjs="cdn")

    # Plot points per entity type that corresponds to a span property
    for entity_label in ENTITY_TYPES:
        df_per_label = df.query(f"label == '{entity_label}'")
        fig_per_label = px.scatter(
            df_per_label,
            x="tsne_x",
            y="tsne_y",
            color="props",
            template="simple_white",
            hover_name="span_text",
            hover_data=["display_text", "span_text", "label"],
            title=f"{entity_label} properties",
            color_discrete_map={
                "paren": colors.get("slate_gray"),
                "all_caps": colors.get("silver"),
                "initial": colors.get("pewter"),
                "plain": colors.get("crimson"),
            },
        )
        fig_per_label.update_layout(legend_title="Properties", **layout_properties)
        fig_per_label.write_html(
            outdir / f"fig_per_label_{entity_label}.html", include_plotlyjs="cdn"
        )
    msg.good(
        f"Successfully drew the charts. Please check the output in the '{outdir}' directory."
    )


def _compute_properties(docs: Iterable[Doc]) -> Iterable[Example]:
    examples = []
    for doc in tqdm(docs):
        for ent in doc.ents:
            # We only want the first applicable category for
            # per-label plots. This looks a bit ugly.
            if ent.start != 0 and doc[ent.start - 1].text == "(":
                props = "paren"  # if preceded by a left parenthesis
            elif all([token.is_upper for token in ent]):
                props = "all_caps"  # if entity consists of all capital letters
            elif ent.start == 0:
                props = "initial"  # if it is the first token in a sentence
            else:
                props = "plain"  # catch-all category

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
                props=props,
            )
            examples.append(eg)
    return examples


if __name__ == "__main__":
    typer.run(plot)
