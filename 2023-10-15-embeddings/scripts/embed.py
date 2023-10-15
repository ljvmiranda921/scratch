from pathlib import Path
from typing import Iterable

import spacy
import typer
from spacy.tokens import Doc, DocBin
from srsly import write_msgpack
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def embed(
    # fmt: off
    corpus: Path = Arg(..., help="Path to the corpus directory containing spaCy files."),
    outfile: Path = Arg(..., help="Path to save the embeddings."),
    model: str = Opt("tl_calamancy_trf", "-m", "--model", help="Model to use for embedding."),
    lang: str = Opt("tl", "-l", "--lang", help="Language code."),
    # fmt: on
):
    """Get embeddings for each span label in each document"""
    nlp = spacy.load(model)

    # Combine corpus
    files = corpus.glob("*.spacy")
    docs = []
    for file in files:
        doc_bin = DocBin().from_disk(file)
        docs.extend(list(doc_bin.get_docs(nlp.vocab)))
    msg.text(f"Found {len(docs)} documents in '{corpus}'")


if __name__ == "__main__":
    typer.run(embed)
