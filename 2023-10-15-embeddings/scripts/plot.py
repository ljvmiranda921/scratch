from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


import spacy
import typer
import numpy as np
from spacy.tokens import Doc, DocBin
from srsly import write_msgpack
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

    tsne_coord: Tuple[float, float] = (0, 0)


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
    docs = doc_bin.get_docs(nlp.vocab)

    # Format documents into a human-readable table
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
                paren=props[0],
                all_caps=props[1],
                initial=props[2],
                plain=props[3],
            )
            examples.append(eg)

    msg.info(f"Processed {len(examples)} entities from {embeddings}")


if __name__ == "__main__":
    typer.run(plot)
