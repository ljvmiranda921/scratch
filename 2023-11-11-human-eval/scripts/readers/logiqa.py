from pathlib import Path
from typing import Any, Dict, List, Optional

import srsly
from datasets import Dataset
from spacy.language import Language
from spacy.tokens import Doc
from wasabi import msg

from ..utils import Interface, make_textcat_doc
from .base import DatasetReader


class LogiQA(DatasetReader):
    @property
    def class_labels(self) -> Optional[List[str]]:
        return ["0", "1", "2", "3"]

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
        choices = ["a", "b", "c", "d"]
        prompt = "Passage: " + eg.get("context") + "\n"
        prompt += "Question: " + eg.get("query") + "\nChoices:\n"
        for choice, option in zip(choices, eg.get("options")):
            prompt += f"{choice.upper()}. {option}\n"
        prompt += "Answer:"
        return prompt

    def get_targets(self, eg: Dict[str, Any]) -> List[str]:
        """Get the targets for sentence completion

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (List[str]): a list of targets to compute evals.
        """
        pass

    def convert_to_prodigy(
        self, examples: "Dataset", interface: Interface
    ) -> List[Dict[str, Any]]:
        """Convert each HuggingFace example into Prodigy instances

        examples (datasets.Dataset): a particular split from a Huggingface dataset
        interface (Interface): the Prodigy annotation interface to build task examples upon.

        RETURNS (List[Dict[str, Any]]): an iterable containing all annotation tasks formatted for Prodigy.
        """
        annotation_tasks = []
        if interface == Interface.textbox:
            msg.fail("Interface 'textbox' is not applicable for this dataset.", exits=1)

        for eg in examples:
            annotation_tasks.append(
                {
                    "text": self.get_prompt(eg),
                    "options": [
                        {"id": id, "text": choice}
                        for id, choice in zip(self.class_labels, eg.get("options"))
                    ],
                    "meta": {"label": str(eg.get("correct_option")), "doc": eg},
                }
            )

        return annotation_tasks

    def get_reference_docs(self, nlp: Language, references: Path) -> List[Doc]:
        """Get reference documents to compare human annotations against

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        references (Path): Path to the examples file.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        ref_records = list(srsly.read_jsonl(references))
        ref_labels = [
            str(rec.get("meta").get("doc").get("correct_option")) for rec in ref_records
        ]
        return [
            make_textcat_doc(nlp, rec, label, self.class_labels)
            for rec, label in zip(ref_records, ref_labels)
        ]

    def get_predicted_docs(self, nlp: Language, predictions: Path) -> List[Doc]:
        pred_records = list(srsly.read_jsonl(predictions))
        pred_labels = list([rec.get("accept")[0] for rec in pred_records])
        return [
            make_textcat_doc(nlp, rec, label, self.class_labels)
            for rec, label in zip(pred_records, pred_labels)
        ]
