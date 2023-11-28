from typing import Any, Dict, Iterable, List, Optional

import srsly
from datasets import Dataset
from spacy.language import Language
from spacy.tokens import Doc
from wasabi import msg

from ..utils import Interface, make_doc
from .base import DatasetReader


class PIQA(DatasetReader):
    @property
    def class_labels(self) -> Optional[List[str]]:
        return ["sol1", "sol2"]

    @property
    def task_type(self) -> str:
        return "multi_choice"

    @property
    def hf_config(self) -> str:
        return "plain_text"

    def get_prompt(self, eg: Dict[str, Any]) -> str:
        """Construct the prompt

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (str): the prompt for that given example.
        """
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/piqa.py#L60
        prompt = "Question: " + eg.get("goal") + "\nAnswer:"
        return prompt

    def get_targets(self, eg: Dict[str, Any]) -> List[str]:
        """Get the targets for evaluating LM performance

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (List[str]): a list of targets to compute evals.
        """
        return [eg.get(label) for label in self.class_labels]

    def convert_to_prodigy(
        self, examples: "Dataset", interface: Interface
    ) -> List[Dict[str, Any]]:
        """Convert each HuggingFace example into Prodigy instances

        examples (datasets.Dataset): a particular split from a Huggingface dataset
        interface (Interface): the Prodigy annotation interface to build task examples upon.
        RETURNS (List[Dict[str, Any]]): an iterable containing all annotation tasks formatted for Prodigy.
        """
        annotation_tasks = []
        for eg in examples:
            if interface == Interface.choice:
                annotation_tasks.append(
                    {
                        "text": self.get_prompt(eg),
                        "options": [
                            {"id": "sol1", "text": eg.get("sol1")},
                            {"id": "sol2", "text": eg.get("sol2")},
                        ],
                        "meta": {
                            "label": self.class_labels[eg.get("label")],
                            "doc": eg,
                        },
                    }
                )
            elif interface == Interface.textbox:
                annotation_tasks.append(
                    {
                        "text": self.get_prompt(eg),
                        "meta": {
                            "label": self.class_labels[eg.get("label")],
                            "doc": eg,
                        },
                    }
                )
            else:
                msg.fail("Unknown annotation interface.", exits=True)
        return annotation_tasks

    def get_reference_docs(
        self, nlp: Language, references: Iterable["srsly.util.JSONOutput"]
    ) -> List[Doc]:
        """Get reference documents to compare human annotations against

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        references (Iterable[srsly.util.JSONOutput]): dictionary-like containing relevant information for evals.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        ref_records = list(srsly.read_jsonl(references))
        ref_labels = [rec.get("meta").get("label") for rec in ref_records]
        return [
            make_doc(nlp, rec, label, self.class_labels)
            for rec, label in zip(ref_records, ref_labels)
        ]

    def get_predicted_docs(
        self, nlp, predictions: Iterable["srsly.util.JSONOutput"]
    ) -> List[Doc]:
        """Get predicted documents to compare on the gold-reference data.

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        predictions (Iterable[srsly.util.JSONOutput]): dictionary-like containing relevant information for evals.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        pred_records = list(srsly.read_jsonl(predictions))
        pred_labels = list([rec.get("accept")[0] for rec in pred_records])
        return [
            make_doc(nlp, rec, label, self.class_labels)
            for rec, label in zip(pred_records, pred_labels)
        ]
