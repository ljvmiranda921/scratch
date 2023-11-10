"""Contains dataset converters and a utility CLI"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import srsly

QAPairs = Iterable[Dict[str, Any]]


def _process_boolq(input_dir: Path) -> QAPairs:
    examples = srsly.read_jsonl(input_dir / "dev.jsonl")
    return [
        {
            "question": eg.get("question"),
            "answer": bool(eg.get("answer")),
            "passage": eg.get("passage"),
        }
        for eg in examples
    ]


def _process_boolqcs(input_dir: Path) -> QAPairs:
    examples = srsly.read_json(input_dir / "dev.json")["data"][1:]
    pairs = []
    for eg in examples:
        passage = eg.get("paragraph")
        for pt in eg.get("perturbed_questions"):
            pairs.append(
                {
                    "question": pt.get("perturbed_q"),
                    "answer": True if pt.get("answer").lower() == "true" else False,
                    "passage": passage,
                }
            )
    return pairs


def _process_boolqnp(input_dir: Path) -> QAPairs:
    examples = srsly.read_jsonl(input_dir / "dev.jsonl")
    pairs = []
    for eg in examples:
        if not eg.get("is_seed_question"):
            pairs.append(
                {
                    "question": eg.get("question"),
                    "answer": True if eg.get("hard_label").lower() == "true" else False,
                    "passage": eg.get("passage"),
                }
            )
    return pairs


def _process_strategyqa(input_dir: Path) -> QAPairs:
    examples = srsly.read_json(input_dir / "dev.json")
    pairs = []
    for eg in examples:
        pairs.append(
            {
                "question": eg.get("question"),
                "answer": eg.get("answer"),
                "passage": "\n".join(eg.get("facts")),
            }
        )
    return pairs


DATASETS: Dict[str, Callable[[Path], Dict[str, QAPairs]]] = {
    "boolq": _process_boolq,
    "boolqcs": _process_boolqcs,
    "boolqnp": _process_boolqnp,
    "strategyqa": _process_strategyqa,
}
