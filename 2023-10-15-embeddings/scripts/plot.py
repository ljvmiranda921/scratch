from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Tuple

import spacy
import typer
import pandas as pd
import numpy as np
from spacy.tokens import Doc, DocBin
from sklearn.manifold import TSNE
from tqdm import tqdm
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


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
    # fmt: on
):
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
    examples = [
        replace(eg, tsne_x=coord[0], tsne_y=coord[1])
        for eg, coord in zip(_examples, fit_X)
    ]
    msg.info(f"Processed {len(examples)} entities from {embeddings}")

    # Plot
    df = pd.DataFrame([asdict(eg) for eg in examples]).drop(columns=["ctx_vector"])
    breakpoint()

    ## Plot everything

    ## Plot PER entities

    ## Plot ORG entities

    ## Plot LOC entities


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
                props = (0, 0, 0, 0)

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


if __name__ == "__main__":
    typer.run(plot)
