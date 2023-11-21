from typing import Any, Dict, Iterable, List

import srsly
from datasets import Dataset
from spacy.tokens import Doc
from wasabi import msg

from ..utils import Interface, make_doc


class LAMBADADataset:
    TASK_TYPE = "sentence_completion"

    @classmethod
    def convert_to_prodigy(
        cls, examples: "Dataset", interface: str
    ) -> List[Dict[str, Any]]:
        """Convert each HuggingFace example into Prodigy instances"""
        if interface == Interface.choice.value:
            msg.fail(
                "Annotation interface 'choice' unavailable for this dataset.", exits=1
            )
        annotation_tasks = []
        for eg in examples:
            text = eg.get("text").rsplit(" ", 1)[0] + " ____. ->"
            label = eg.get("text").rsplit(" ", 1)[1]
            annotation_tasks.append({"text": text, "meta": {"label": label}})
        return annotation_tasks

    @classmethod
    def get_reference_docs(
        cls, nlp, references: Iterable["srsly.util.JSONOutput"]
    ) -> List[Doc]:
        pass

    @classmethod
    def get_predicted_docs(
        cls, nlp, predictions: Iterable["srsly.util.JSONOutput"]
    ) -> List[Doc]:
        pass
