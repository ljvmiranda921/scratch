from enum import Enum
from pathlib import Path
from typing import Dict, List, Any

import srsly
import spacy
import typer
from spacy.tokens import Doc
from spacy.scorer import Scorer
from spacy.training import Example
from datasets import load_dataset
from wasabi import msg

app = typer.Typer()

CLASS_LABELS = ["sol1", "sol2"]


class Interface(str, Enum):
    choice = "choice"  # https://prodi.gy/docs/api-interfaces#choice
    textbox = "textbox"  # https://prodi.gy/docs/api-interfaces#text


class Split(str, Enum):
    train = "train"
    test = "test"
    validation = "validation"


@app.command()
def download(
    # fmt: off
    output_path: Path = typer.Argument(..., help="Path to save the JSONL file."),
    split: Split = typer.Option(Split.validation, help="Dataset split to convert."),
    interface: Interface = typer.Option(Interface.choice, help="Prodigy interface to use."),
    # fmt: on
):
    """Download PIQA dataset from HuggingFace and convert it into Prodigy format."""
    examples = load_dataset("piqa", split=split.value)

    annotation_tasks = []
    for eg in examples:
        if interface == Interface.choice.value:
            annotation_tasks.append(
                {
                    "text": eg.get("goal"),
                    "options": [
                        {"id": "sol1", "text": eg.get("sol1")},
                        {"id": "sol2", "text": eg.get("sol2")},
                    ],
                    "meta": {"label": CLASS_LABELS[eg.get("label")]},
                }
            )
        elif interface == Interface.textbox.value:
            annotation_tasks.append(
                {
                    "text": eg.get("goal"),
                    "field_id": "user_input",
                    "field_label": "",
                    "field_rows": 5,
                    "field_placeholder": "Type here...",
                    "field_autofocus": False,
                    "meta": {"label": CLASS_LABELS[eg.get("label")]},
                }
            )
        else:
            msg.fail("Unknown annotation interface.", exits=True)

    srsly.write_jsonl(output_path, annotation_tasks)
    msg.good(f"Saved {len(annotation_tasks)} annotation tasks to {output_path}")


@app.command()
def evaluate(
    # fmt: off
    references: Path = typer.Argument(..., help="Path to the gold-standard data."),
    predictions: Path = typer.Argument(..., help="Path to the human-annotated results."),
    # fmt: on
):
    """Compare results on gold-standard data."""
    nlp = spacy.blank("en")

    # Get reference documents
    ref_records = list(srsly.read_jsonl(references))
    ref_labels = [rec.get("meta").get("label") for rec in ref_records]
    ref_docs = [
        to_spacy_doc(nlp, rec, label, CLASS_LABELS)
        for rec, label in zip(ref_records, ref_labels)
    ]

    # Get predicted documents
    pred_records = list(srsly.read_jsonl(predictions))
    pred_labels = list([rec.get("accept")[0] for rec in pred_records])
    pred_docs = [
        to_spacy_doc(nlp, rec, label, CLASS_LABELS)
        for rec, label in zip(pred_records, pred_labels)
    ]

    # Create spacy Examples
    examples = [Example(pred, ref) for pred, ref in zip(pred_docs, ref_docs)]
    msg.text(f"Found {len(examples)} examples")
    scores = Scorer.score_cats(
        examples, attr="cats", labels=CLASS_LABELS, multi_label=False
    )
    msg.text(title="Scores", text=scores)


def to_spacy_doc(
    nlp: "spacy.language.Language",
    record: Dict[str, Any],
    label: str,
    class_labels: List[str],
) -> Doc:
    doc = nlp.make_doc(record.get("text"))
    doc.cats = {class_label: 0 for class_label in set(class_labels)}
    doc.cats[label] = 1
    return doc


if __name__ == "__main__":
    app()
