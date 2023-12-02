from enum import Enum
from typing import Any, Dict, List

import spacy
from spacy.tokens import Doc


class Interface(str, Enum):
    """Prodigy interface for the annotator"""

    choice = "choice"  # https://prodi.gy/docs/api-interfaces#choice
    textbox = "textbox"  # https://prodi.gy/docs/api-interfaces#text


class Split(str, Enum):
    """Dataset split"""

    train = "train"
    test = "test"
    validation = "validation"


def make_textcat_doc(
    nlp: "spacy.language.Language",
    record: Dict[str, Any],
    label: str,
    class_labels: List[str],
) -> Doc:
    """Make spaCy Doc given a multi-choice task"""
    doc = nlp.make_doc(record.get("text"))
    doc.cats = {class_label: 0 for class_label in set(class_labels)}
    doc.cats[label] = 1
    return doc


def make_sentence_completion_doc(
    nlp: "spacy.language.Language",
    record: Dict[str, Any],
    label: str,
) -> Doc:
    """Make spaCy Doc for a sentence completion task"""
    doc = nlp.make_doc(record.get("text"))
    doc._.target = label
    return doc
