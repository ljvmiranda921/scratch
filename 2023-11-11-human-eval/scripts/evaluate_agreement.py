from pathlib import Path
from typing import Any, Dict, List, Optional

import srsly
import typer
import pandas as pd
from wasabi import msg

from .readers import Dataset, get_dataset_reader


def evaluate_agreement(
    # fmt: off
    lm_outputs: Path = typer.Argument(..., help="Path to the LM output from EleutherAI when you pass --output_path and --log_samples."),
    human_outputs: Path = typer.Argument(..., help="Path to human annotations via Prodigy."),
    dataset: Dataset = typer.Option(..., help="Dataset name for parsing."),
    output_path: Optional[Path] = typer.Option(None, help="If set, save the agreement values in a CSV file."),
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
        if reader.task_type == "multi_choice":
            # A hacky patch for Winogrande
            if dataset == Dataset.winogrande:
                answer_to_num = {"1": 0, "2": 1}
                gold_human = answer_to_num.get(
                    human.get("meta").get("doc").get("answer")
                )
                gold_lm = gold_human
            elif dataset == Dataset.logiqa:
                gold_lm = int(lm.get("target"))
                gold_human = int(human.get("meta").get("doc").get("correct_option"))
            elif dataset == Dataset.truthfulqa:
                gold_lm = int(lm.get("target"))
                gold_human = int(
                    human.get("meta")
                    .get("doc")
                    .get("mc1_targets")
                    .get("labels")
                    .index(1)
                )
            else:
                # They're just the same to be honest, but we just wanna make sure
                gold_lm = int(lm.get("target"))
                gold_human = int(human.get("meta").get("doc").get("label"))

            if gold_human != gold_lm:
                msg.fail(
                    f"Gold labels do not match for doc_id '{lm.get('doc_id')}'!"
                    f" {gold_human} ({type(gold_human)}) != {gold_lm} ({type(gold_lm)})",
                    exits=1,
                )
            gold = reader.class_labels[gold_human]  # just use gold_human

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
        # Additional information
        export(docs, output_path)


def export(docs: List[Dict[str, Any]], output_path: Path):
    def _get_text_from_id(id_value: str, options: List[Dict[str, Any]]) -> str:
        for option in options:
            if option.get("id") == id_value:
                return option.get("text")

    updated_docs = []
    for doc in docs:
        options = doc.get("options")
        human = doc.get("human")
        lm = doc.get("lm")
        gold = doc.get("gold")
        doc["both_correct"] = (human == gold) & (lm == gold)
        doc["both_wrong"] = (human != gold) & (lm != gold)
        doc["human_correct_lm_wrong"] = (human == gold) & (lm != gold)
        doc["human_wrong_lm_correct"] = (human != gold) & (lm == gold)
        doc["human_answer"] = _get_text_from_id(human, options)
        doc["lm_answer"] = _get_text_from_id(lm, options)
        doc["gold_answer"] = _get_text_from_id(gold, options)
        updated_docs.append(doc)

    df = pd.DataFrame(docs)
    # fmt: off
    df = df[["both_correct", "both_wrong", "human_correct_lm_wrong", "human_wrong_lm_correct", "human_answer", "lm_answer", "gold_answer", "text", "options"]]
    # fmt: on
    df.to_csv(output_path, index=False)
    msg.good(f"File saved to {output_path}")


if __name__ == "__main__":
    typer.run(evaluate_agreement)
