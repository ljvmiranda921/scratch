from pathlib import Path

import spacy
import typer
from spacy.scorer import Scorer
from spacy.training import Example
from wasabi import msg

from .datasets import Dataset, get_dataset_reader


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
    dataset_reader = get_dataset_reader(dataset)

    ref_docs = dataset_reader.get_reference_docs(nlp, references)
    pred_docs = dataset_reader.get_predicted_docs(nlp, predictions)

    if dataset_reader.task_type == "multi_choice":
        # Create spacy Examples
        examples = [Example(pred, ref) for pred, ref in zip(pred_docs, ref_docs)]
        msg.text(f"Found {len(examples)} examples")
        scores = Scorer.score_cats(
            examples,
            attr="cats",
            labels=dataset_reader.class_labels,
            multi_label=multi_label,
        )
        msg.text(title="Scores", text=scores)
    elif dataset_reader.task_type == "sentence_completion":
        pass
    else:
        msg.fail("Unknown task type.")


if __name__ == "__main__":
    typer.run(evaluate_gold)
