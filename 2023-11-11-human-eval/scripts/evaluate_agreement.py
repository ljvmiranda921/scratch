from pathlib import Path
from typing import Any, Dict, List, Optional

import srsly
import typer
from wasabi import msg

from .readers import get_dataset_reader, Dataset


def evaluate_agreement(
    # fmt: off
    lm_outputs: Path = typer.Argument(..., help="Path to the LM output from EleutherAI when you pass --output_path and --log_samples."),
    human_outputs: Path = typer.Argument(..., help="Path to human annotations via Prodigy."),
    dataset: Dataset = typer.Option(..., help="Dataset name for parsing."),
    output_path: Optional[Path] = typer.Option(None, help="If set, save the agreement values in a JSONL file."),
    compute_metrics: bool = typer.Option(False, help="If set, will compute human-agreement metrics and display them as the output."),
    # fmt: on
):
    """Compare LM evaluations to human annotations."""
    # It's weird because the output is a JSONL file but the format is JSON.
    lm_outputs = list(srsly.read_json(lm_outputs))
    human_outputs = list(srsly.read_jsonl(human_outputs))

    reader = get_dataset_reader(dataset)

    if len(lm_outputs) != len(human_outputs):
        msg.warn(
            "Number of documents in two files not the same! "
            f"{len(lm_outputs)} != {len(human_outputs)}"
        )

    docs: List[Dict[str, Any]] = []
    for lm, human in zip(lm_outputs, human_outputs):
        # They're just the same to be honest, but we just wanna make sure
        gold_lm = lm.get("target")
        gold_human = human.get("meta").get("doc").get("label")
        if gold_human != gold_lm:
            msg.fail(
                f"Gold labels do not match for doc_id '{lm.get('doc_id')}'!",
                exits=1,
            )
        gold = reader.class_labels[gold_human]  # just use gold_human

        if reader.task_type == "multi_choice":
            logits = [probs for probs, _ in lm.get("filtered_resps")]
            lm_ans = reader.class_labels[logits.index(max(logits))]
            human_ans = human.get("accept")[0]
            docs.append(
                {
                    "human": human_ans,
                    "lm": lm_ans,
                    "gold": gold,
                    "text": human.get("text"),
                    "options": human.get("options"),
                    "lm_doc_id": lm.get("doc_id"),
                }
            )

    if compute_metrics:
        n_counts = len(human_outputs)
        metrics = {
            # fmt: off
            "human_gold_ref": sum([doc.get("human") == doc.get("gold") for doc in docs]) / n_counts,
            "lm_gold_ref": sum([doc.get("lm") == doc.get("gold") for doc in docs]) / n_counts,
            "lm_human_ref": sum([doc.get("lm") == doc.get("human") for doc in docs]) / n_counts,
            # fmt: on
        }
        msg.table(metrics, divider=True, header=["Metric", "Score"])

    if output_path:
        pass


if __name__ == "__main__":
    typer.run(evaluate_agreement)
