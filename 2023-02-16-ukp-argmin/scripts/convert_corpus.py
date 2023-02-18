from pathlib import Path
from typing import Dict, List

import spacy
import typer
from spacy import Language
from spacy.tokens import Doc, DocBin
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def read_tsv(tsv_file) -> List[Dict[str, str]]:
    """Read the TSV file and yield an iterable of records"""
    records = []
    next(tsv_file)  # skip first line (headers)
    for line in tsv_file:
        text, label, dataset = line.strip().split("\t")[4:]  # skip a few columns
        record = {"text": text, "label": label, "dataset": dataset}
        records.append(record)
    return records


def convert_record(nlp: Language, record: Dict[str, str], categories: List[str]) -> Doc:
    """Convert a record from the TSV into a spaCy Doc object"""
    doc = nlp.make_doc(record.get("text"))
    # All categories under than the true value gets 0
    doc.cats = {category: 0 for category in categories}
    doc.cats[record.get("label")] = 1
    return doc


def convert_corpus(
    # fmt: off
    input_path: Path = Arg(..., help="Path to the TSV file of UKP annotations."),
    output_dir: Path = Opt(CORPUS_DIR, dir_okay=True, help="Path to the output directory to store the spaCy files."),
    # fmt: on
):
    """Convert the raw annotations into the spaCy format"""
    records = read_tsv(input_path.open("r"))
    msg.info(f"Found {len(records)} records in {input_path}")

    datasets = set([record.get("dataset") for record in records])
    categories = set([record.get("label") for record in records])

    nlp = spacy.blank("en")
    corpus: Dict[str, List[Doc]] = {
        dataset: [
            convert_record(nlp, record, categories)
            for record in records
            if record.get("dataset") == dataset
        ]
        for dataset in datasets
    }

    for dataset, docs in corpus.items():
        doc_bin = DocBin(docs=docs)
        output_path = output_dir / f"{dataset}.spacy"
        doc_bin.to_disk(output_dir / f"{dataset}.spacy")
        msg.good(f"Saved {len(docs)} docs to {output_path}")


if __name__ == "__main__":
    typer.run(convert_corpus)
