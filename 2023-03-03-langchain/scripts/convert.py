from pathlib import Path
from typing import Dict, List

import spacy
import srsly
import typer
from spacy.tokens import Doc, DocBin
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def convert_corpus(
    # fmt: off
    input_path: Path = Arg(..., help="Path to the TSV file of UKP annotations."),
    output_dir: Path = Arg(CORPUS_DIR, dir_okay=True, help="Path to the output directory to store the spaCy files."),
    jsonl: bool = Opt(False, "--jsonl", help="If set to true, records are converted to JSONL files"),
    # fmt: on
):
    """Convert the raw annotations into the spaCy format"""
    records = read_tsv(input_path.open("r", encoding="utf-8"))
    msg.info(f"Found {len(records)} records in {input_path}")
    corpora = records_to_spacy(records) if not jsonl else records_to_jsonl(records)

    # Save to disk
    for dataset, corpus in corpora.items():
        if not jsonl:
            output_path = output_dir / f"{dataset}.spacy"
            corpus.to_disk(output_path)
        else:
            output_path = output_dir / f"{dataset}.jsonl"
            srsly.write_jsonl(output_path, corpus)

        msg.good(f"Saved {len(corpus)} records to {output_path}")


def read_tsv(tsv_file) -> List[Dict[str, str]]:
    """Read the TSV file and yield an iterable of records"""
    records = []
    next(tsv_file)  # skip first line (headers)
    for line in tsv_file:
        text, label, dataset = line.strip().split("\t")[4:]  # skip a few columns
        record = {"text": text, "label": label, "dataset": dataset}
        records.append(record)
    return records


def records_to_spacy(
    records: List[Dict[str, str]], lang: str = "en"
) -> Dict[str, DocBin]:
    datasets: List[str] = set([record.get("dataset") for record in records])
    categories: List[str] = set([record.get("label") for record in records])
    nlp = spacy.blank(lang)

    def to_spacy(record) -> Doc:
        doc = nlp.make_doc(record.get("text"))
        # All cats other than the true value gets 0
        doc.cats = {category: 0 for category in categories}
        doc.cats[record.get("label")] = 1
        return doc

    corpora = {
        dataset: DocBin(
            docs=[
                to_spacy(record)
                for record in records
                if record.get("dataset") == dataset
            ]
        )
        for dataset in datasets
    }
    return corpora


def records_to_jsonl(records: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    datasets: List[str] = set([record.get("dataset") for record in records])
    corpora = {
        dataset: [record for record in records if record.get("dataset") == dataset]
        for dataset in datasets
    }
    return corpora


if __name__ == "__main__":
    typer.run(convert_corpus)
