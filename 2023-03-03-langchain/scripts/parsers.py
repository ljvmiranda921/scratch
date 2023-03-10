"""Parsers for different annotation guidelines

Each annotation guideline has their own set of labels. We want to parse them
properly. GPT-3 is still an LLM, so it can return any arbitrary string of text.
I tried to be robust, but for some cases I just decided that a naive parser
will work.

Although some tasks are binary classification problems. I opted to parse them as
an exclusive multilabel classification problem.
"""

from typing import Any, Callable, Dict, List

from pathlib import Path
from prodigy.util import msg

LABELS = {
    # fmt: off
    "levy2018": {"labels": ["Accept", "Reject"], "default": "Reject"},
    "morante2020": {"labels": ["Claim", "No claim"], "default": "No claim"},
    "shnarch2018": {"labels": ["Accept", "Reject"], "default": "Reject"},
    "stab2018": {"labels": ["Non-argument", "Supporting argument", "Opposing argument"], "default": "Non-argument"},
    # fmt: on
}


def get_parser(file: Path) -> Callable:
    if file.stem not in LABELS.keys():
        msg.fail(f"Cannot find parser for {file}. Available: {LABELS.keys()}", exit=1)

    labels = LABELS.get(file.stem).get("labels")
    default = LABELS.get(file.stem).get("default")
    return make_naive_parser(labels=labels, default=default)


def make_naive_parser(
    labels: List[str],
    default: str,
) -> Callable:
    def _naive_parser(response: str) -> Dict[str, Any]:
        for label in labels:
            if label.lower() in response.strip().lower():
                accept = label
                break
            else:
                msg.warn(f"Cannot parse: '{response}'. Will set to '{default}'")
                accept = default

        return {
            "options": [{"id": label, "text": label} for label in labels],
            "answer": "accept",
            "accept": [accept],
        }

    return _naive_parser
