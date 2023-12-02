from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import srsly
from datasets import Dataset
from spacy.tokens import Doc
from wasabi import msg

from ..utils import Interface, make_textcat_doc
from .base import DatasetReader


class Winogrande(DatasetReader):
    @property
    def class_labels(self) -> Optional[List[str]]:
        return ["option1", "option2"]

    @property
    def task_type(self) -> str:
        return "multi_choice"

    @property
    def hf_config(self) -> str:
        return "winogrande_xl"

    def get_prompt(self, eg: Dict[str, Any]) -> str:
        """Construct the prompt

        Use for sentence completion task (textbox)

        This evaluation of Winogrande uses partial evaluation as described by
        Trinh & Le in Simple Method for Commonsense Reasoning (2018).
        See: https://arxiv.org/abs/1806.02847

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (str): the prompt for that given example.
        """
        breakpoint()

        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/winogrande.py#L66-L70
        def _partial_ctx(eg: Dict[str, Any], option: str):
            pronoun_loc = eg.get("sentence").index("_")
            return eg.get("sentence")[:pronoun_loc] + option

        return _partial_ctx(eg, eg.get(f"option{eg.get('answer')}"))

    def get_targets(self, eg: Dict[str, Any]) -> List[str]:
        """Get the targets for sentence completion

        This evaluation of Winogrande uses partial evaluation as described by
        Trinh & Le in Simple Method for Commonsense Reasoning (2018).
        See: https://arxiv.org/abs/1806.02847

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (List[str]): a list of targets to compute evals.
        """

        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/winogrande.py#L76-L79
        def _partial_target(eg: Dict[str, Any]) -> str:
            pronoun_loc = eg.get("sentence").index("_") + 1
            return " " + eg.get("sentence")[pronoun_loc:].strip()

        return [_partial_target(eg)]

    def convert_to_prodigy(
        self, examples: "Dataset", interface: Interface
    ) -> List[Dict[str, Any]]:
        """Convert each HuggingFace example into Prodigy instances

        examples (datasets.Dataset): a particular split from a Huggingface dataset
        interface (Interface): the Prodigy annotation interface to build task examples upon.

        RETURNS (List[Dict[str, Any]]): an iterable containing all annotation tasks formatted for Prodigy.
        """
        breakpoint()
        annotation_tasks = []
        for eg in examples:
            if interface == Interface.choice:
                annotation_tasks.append(
                    {
                        "text": eg.get("sentence"),
                        "options": [
                            {"id": "option1", "text": eg.get("option1")},
                            {"id": "option2", "text": eg.get("option2")},
                        ],
                        "meta": {
                            "label": self.class_labels[int(eg.get("answer")) - 1],
                            "doc": eg,
                        },
                    }
                )
            elif interface == Interface.textbox:
                annotation_tasks.append(
                    {
                        "text": self.get_prompt(eg),
                        "meta": {"label": [self.get_targets(eg)], "doc": eg},
                    }
                )
            else:
                msg.fail("Unknown annotation interface.", exits=True)
        return annotation_tasks

    def get_reference_docs(self, nlp, references: Path) -> List[Doc]:
        """Get reference documents to compare human annotations against

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        references (Path): Path to the examples file.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        ...

    def get_predicted_docs(self, nlp, predictions: Path) -> List[Doc]:
        """Get predicted documents to compare on the gold-reference data.

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        predictions (Path): Path to the examples file.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        ...
