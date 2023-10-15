from dataclasses import dataclass
from pathlib import Path
from typing import Optional


import spacy
import typer
import numpy as np
from spacy.tokens import Doc, DocBin
from srsly import write_msgpack
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


@dataclass
class Example:
    """Class for keeping track of an example"""

    text: str
    label: str
    char_start: int
    char_end: int
    ctx_vector: np.ndarray
    tsne_coord: np.ndarray

    # Other properties, assign the first applicable category
    paren: bool  # if preceded by a left parenthesis
    all_caps: bool  # if it consists of all capitals
    initial: bool  # if it is the first token in a sentence
    plain: bool  # catch-all category


def plot(
    # fmt: off
    embeddings: Path = Arg(..., help="Path to a spaCy file containing span embeddings."),
    outdir: Path = Arg(..., help="Directory to save the plots.", dir_okay=True),
    save_tsne_coords: Optional[Path] = Opt(None, help="If provided, will save t-SNE coordinates in a serializable msgpack format."),
    # fmt: on
):
    # Read file
    nlp = spacy.blank("xx")
    doc_bin = DocBin().from_disk(embeddings)
    docs = list(doc_bin.get_docs(nlp.vocab))
    msg.good(f"Found {len(docs)} from {embeddings}")

    # Format documents into a human-readable table
    for doc in docs:
        for ent in doc.ents
