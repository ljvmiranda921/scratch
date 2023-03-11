from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
import srsly
import typer
from spacy import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from wasabi import msg

from .parsers import LABELS

Arg = typer.Argument
Opt = typer.Option


def evaluate(
    # fmt: off
    reference: Path = Arg(..., help="Path to the JSONL reference file."),
    predicted: Path = Arg(..., help="Path to the JSONL predictions file."),
    output_path: Optional[Path] = Opt(None, "--output-path", "-o", help="Path to save the output metrics."),
    normalize_labels: bool = Opt(False, "--normalize-labels", "--normalize", "-nl", help="Normalize the labels into binary (Argument / NoArgument)."),
    lang: str = Opt("en", "--lang", "-l", help="Language code."),
    # fmt: on
):
    """Evaluate annotation guidelines with respect to gold-standard data or against each other"""

    # Perform conversion to spaCy doc objects. The reference
    # annotations can be in spaCy or JSONL format. The predictions
    # are always JSONL because that's what we got from Prodigy.
    nlp = spacy.blank(lang)
    ref_records = srsly.read_jsonl(reference)
    ref_docs = [
        to_spacy_doc(
            nlp,
            record,
            get_labels(reference),
            normalize_labels=normalize_labels,
        )
        for record in ref_records
    ]

    pred_records = srsly.read_jsonl(predicted)
    pred_docs = [
        to_spacy_doc(
            nlp,
            record,
            get_labels(predicted),
            normalize_labels=normalize_labels,
        )
        for record in pred_records
    ]

    # Create Example objects for evaluation
    examples = [Example(pred, ref) for pred, ref in zip(pred_docs, ref_docs)]
    msg.text(f"Found {len(examples)} examples")
    labels = get_labels(predicted)
    if normalize_labels:
        labels = [normalize(label) for label in labels]
    scores = Scorer.score_cats(examples, attr="cats", labels=labels, multi_label=False)
    msg.text(title="Scores", text=scores)
    if output_path:
        srsly.write_json(output_path, scores)
        msg.good(f"Saved metrics to {output_path}")


def get_labels(path: Path) -> List[str]:
    if path.name == "test.jsonl":  # check if gold-standard data
        annotation_guideline = "stab2018"
    else:
        _, _, annotation_guideline = path.stem.split("-")
    return LABELS.get(annotation_guideline).get("labels")


def to_spacy_doc(
    nlp: Language,
    record: Dict[str, Any],
    labels: List[str],
    *,
    normalize_labels: bool = False,
) -> Doc:
    """Convert a JSONL record into a spaCy Doc object"""
    doc = nlp.make_doc(record.get("text"))
    label = record.get("accept")[0]

    if normalize_labels:
        labels = [normalize(label) for label in labels]
        label = normalize(label)

    doc.cats = {label: 0 for label in set(labels)}
    doc.cats[label] = 1
    return doc


def normalize(label: str) -> str:
    # Normalize different labels into these two
    NORMALIZE_MAP = {
        "Argument": ["Accept", "Claim", "Argument_for", "Argument_against"],
        "NoArgument": ["Reject", "No claim", "NoArgument"],
    }

    for normalized_label in NORMALIZE_MAP:
        if label in NORMALIZE_MAP[normalized_label]:
            return normalized_label


if __name__ == "__main__":
    typer.run(evaluate)
