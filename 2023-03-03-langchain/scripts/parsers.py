"""Parsers for different annotation guidelines

Each annotation guideline has their own set of labels. We want to parse them
properly. GPT-3 is still an LLM, so it can return any arbitrary string of text.
I tried to be robust, but for some cases I just decided that a naive parser
will work.

Although some tasks are binary classification problems. I opted to parse them as
an exclusive multilabel classification problem.
"""

from typing import Any, Callable, Dict, List, Optional

from pathlib import Path
from prodigy.util import msg
from prodigy.types import TaskType


def get_parser(file: Path) -> Callable:
    if file.stem not in PARSERS.keys():
        msg.fail(f"Cannot find parser for {file}. Available: {PARSERS.keys()}", exit=1)
    parser = PARSERS.get(file.stem)
    return parser


def _parse_levy2018():
    pass


def _parse_morante2020():
    pass


def _parse_shnarch2018():
    pass


def _parse_stab2018(
    response: str,
    example: Optional[TaskType] = None,
    labels: List[str] = ["Non-argument", "Supporting argument", "Opposing argument"],
) -> Dict[str, Any]:
    for label in labels:
        if label.lower() in response.strip().lower():
            accept = label
            break
        else:
            msg.warn(f"Cannot parse: '{response}'. Will set to 'Non-argument'")
            accept = "Non-argument"
    return {
        "options": [{"id": label, "text": label} for label in labels],
        "answer": "accept",
        "accept": [accept],
    }


PARSERS: Dict[str, Callable] = {
    "levy2018": _parse_levy2018,
    "morante2020": _parse_morante2020,
    "shnarch2018": _parse_shnarch2018,
    "stab2018": _parse_stab2018,
}
