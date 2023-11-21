from enum import Enum
from pathlib import Path

import spacy
import typer
from spacy.scorer import Scorer
from spacy.training import Example
from wasabi import msg

from .hellaswag import HellaSwagDataset
from .piqa import PIQADataset


class Dataset(str, Enum):
    piqa = "piqa"
    hellaswag = "hellaswag"


DATASETS = {
    Dataset.piqa.value: PIQADataset,
    Dataset.hellaswag.value: HellaSwagDataset,
}


def evaluate_gold(
    # fmt: off
    dataset: Dataset = typer.Argument(Dataset.piqa, help="Dataset to evaluate."),
    references: Path = typer.Argument(..., help="Path to the gold-standard data."),
    predictions: Path = typer.Argument(..., help="Path to the human-annotated results."),
    multi_label: bool = typer.Option(False, help="Set true if task is multilabel."),
    # fmt: on
):
    """Compare results on gold-standard data."""
    nlp = spacy.blank("en")

    ref_docs = DATASETS[dataset.value].get_reference_docs(nlp, references)
    pred_docs = DATASETS[dataset.value].get_predicted_docs(nlp, predictions)

    # Create spacy Examples
    examples = [Example(pred, ref) for pred, ref in zip(pred_docs, ref_docs)]
    msg.text(f"Found {len(examples)} examples")
    scores = Scorer.score_cats(
        examples,
        attr="cats",
        labels=DATASETS[dataset.value].CLASS_LABELS,
        multi_label=multi_label,
    )
    msg.text(title="Scores", text=scores)


if __name__ == "__main__":
    typer.run(evaluate_gold)
