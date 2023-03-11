from pathlib import Path
from typing import Dict, Optional

import spacy
import srsly
import typer
from spacy.tokens import Doc, DocBin
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def evaluate(
    # fmt: off
    reference: Path = Arg(..., help="Path to the JSONL/spaCy reference file."),
    predicted: Path = Arg(..., help="Path to the JSONL predictions."),
    output_path: Optional[Path] = Opt(None, "--output-path", "-o", help="Path to save the output metrics."),
    lang: str = Opt("en", "--lang", "-l", help="Language code."),
    # fmt: on
):
    nlp = spacy.blank(lang)

    if reference.suffix == ".jsonl":
        msg.text("Found JSONL files, converting records to spaCy Doc objects")
        records = srsly.read_jsonl(reference)
        ref_docs = [to_spacy_doc(record) for record in records]
    else:
        ref_doc_bin = DocBin().from_disk(reference)
        ref_docs = ref_doc_bin.get_docs(nlp.vocab)


def to_spacy_doc(record: Dict) -> Doc:
    pass


if __name__ == "__main__":
    typer.run(evaluate)
