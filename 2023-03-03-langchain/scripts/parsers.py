"""Parsers for different annotation guidelines

Each annotation guideline has their own set of labels. We want to parse them
properly. GPT-3 is still an LLM, so it can return any arbitrary string of text.
"""

from typing import Callable, Dict

from pathlib import Path
from prodigy.util import msg


def get_parser(file: Path) -> Callable:
    if file.stem not in PARSERS.keys():
        msg.fail(f"Cannot find parser for {file}. Available: {PARSERS.keys()}", exit=1)

    parser = PARSERS.get(file.name)
    return parser


def _parse_levy2018():
    pass


def _parse_morante2020():
    pass


def _parse_shnarch2018():
    pass


def _parse_stab2018():
    pass


PARSERS: Dict[str, Callable] = {
    "levy2018": _parse_levy2018,
    "morante2020": _parse_morante2020,
    "shnarch2018": _parse_shnarch2018,
    "stab2018": _parse_stab2018,
}
