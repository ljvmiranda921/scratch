from typing import Any, Dict, Iterable, List

import srsly
from datasets import Dataset
from spacy.tokens import Doc
from wasabi import msg

from ..utils import Interface, make_doc
from .base import DatasetReader


class Lambada(DatasetReader):
    @property
    def task_type(self) -> str:
        return "sentence_completion"

    @property
    def hf_config(self) -> str:
        return "plain_text"

    def get_prompt(self, eg: Dict[str, Any]) -> str:
        """Construct the prompt

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (str): the prompt for that given example.
        """
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/lambada_cloze.py#L36-L37
        return eg.get("text").rsplit(" ", 1)[0] + " ___. ->"

    def get_targets(self, eg: Dict[str, Any]) -> List[str]:
        """Get the target for sentence completion

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (List[str]): a list of targets to compute evals.
        """
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/lambada_cloze.py#L45-L46
        return [eg.get("text").rsplit(" ", 1)[1]]

    def convert_to_prodigy(
        self, examples: "Dataset", interface: Interface
    ) -> List[Dict[str, Any]]:
        """Convert each HuggingFace example into Prodigy instances

        examples (datasets.Dataset): a particular split from a Huggingface dataset
        interface (Interface): the Prodigy annotation interface to build task examples upon.
        RETURNS (List[Dict[str, Any]]): an iterable containing all annotation tasks formatted for Prodigy.
        """
        if interface == Interface.choice.value:
            msg.fail(
                "Annotation interface 'choice' unavailable for this dataset.", exits=1
            )
        annotation_tasks = []
        for eg in examples:
            annotation_tasks.append(
                {
                    "text": self.get_prompt(eg),
                    "meta": {"label": [self.get_targets(eg)], "doc": eg},
                }
            )
        return annotation_tasks

    def get_reference_docs(
        self, nlp, references: Iterable["srsly.util.JSONOutput"]
    ) -> List[Doc]:
        """Get reference documents to compare human annotations against

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        references (Iterable[srsly.util.JSONOutput]): dictionary-like containing relevant information for evals.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        ...

    def get_predicted_docs(
        self, nlp, predictions: Iterable["srsly.util.JSONOutput"]
    ) -> List[Doc]:
        """Get predicted documents to compare on the gold-reference data.

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        predictions (Iterable[srsly.util.JSONOutput]): dictionary-like containing relevant information for evals.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        ...
