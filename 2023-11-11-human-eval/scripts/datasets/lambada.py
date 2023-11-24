from typing import Any, Dict, Iterable, List

import srsly
from datasets import Dataset
from spacy.tokens import Doc
from wasabi import msg

from ..utils import Interface, make_doc


class LAMBADADataset:
    TASK_TYPE = "sentence_completion"
    HF_CONFIG = "plain_text"

    @classmethod
    def get_prompt(cls, eg: Dict[str, Any]) -> str:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/lambada_cloze.py#L36-L37
        return eg.get("text").rsplit(" ", 1)[0] + " ___. ->"

    @classmethod
    def get_target(cls, eg: Dict[str, Any]) -> str:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/lambada_cloze.py#L45-L46
        return eg.get("text").rsplit(" ", 1)[1]

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
            annotation_tasks.append(
                {"text": cls.get_prompt(eg), "meta": {"label": cls.get_target(eg)}}
            )
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
