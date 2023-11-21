from typing import Any, Dict, Iterable, List

import srsly
from datasets import Dataset
from spacy.tokens import Doc
from wasabi import msg

from ..utils import Interface, make_doc


class PIQADataset:
    CLASS_LABELS = ["sol1", "sol2"]

    @classmethod
    def convert_to_prodigy(
        cls, examples: "Dataset", interface: str
    ) -> List[Dict[str, Any]]:
        """Convert each HuggingFace example into Prodigy instances"""
        annotation_tasks = []
        for eg in examples:
            if interface == Interface.choice.value:
                annotation_tasks.append(
                    {
                        "text": eg.get("goal"),
                        "options": [
                            {"id": "sol1", "text": eg.get("sol1")},
                            {"id": "sol2", "text": eg.get("sol2")},
                        ],
                        "meta": {"label": cls.CLASS_LABELS[eg.get("label")]},
                    }
                )
            elif interface == Interface.textbox.value:
                annotation_tasks.append(
                    {
                        "text": eg.get("goal"),
                        "field_id": "user_input",
                        "field_label": "",
                        "field_rows": 5,
                        "field_placeholder": "Type here...",
                        "field_autofocus": False,
                        "meta": {"label": cls.CLASS_LABELS[eg.get("label")]},
                    }
                )
            else:
                msg.fail("Unknown annotation interface.", exits=True)
        return annotation_tasks

    @classmethod
    def get_reference_docs(
        cls, nlp, references: Iterable["srsly.util.JSONOutput"]
    ) -> List[Doc]:
        ref_records = list(srsly.read_jsonl(references))
        ref_labels = [rec.get("meta").get("label") for rec in ref_records]
        return [
            make_doc(nlp, rec, label, cls.CLASS_LABELS)
            for rec, label in zip(ref_records, ref_labels)
        ]

    @classmethod
    def get_predicted_docs(
        cls, nlp, predictions: Iterable["srsly.util.JSONOutput"]
    ) -> List[Doc]:
        pred_records = list(srsly.read_jsonl(predictions))
        pred_labels = list([rec.get("accept")[0] for rec in pred_records])
        return [
            make_doc(nlp, rec, label, cls.CLASS_LABELS)
            for rec, label in zip(pred_records, pred_labels)
        ]
