from typing import Any, Dict, Iterable, List

import srsly
from datasets import Dataset
from spacy.tokens import Doc
from wasabi import msg

from ..utils import Interface, make_doc


class WinograndeDataset:
    CLASS_LABELS = ["option1", "option2"]
    TASK_TYPE = "multi_choice"
    HF_CONFIG = "winogrande_debiased"

    @classmethod
    def get_prompt(cls, eg: Dict[str, Any]) -> str:
        """Use for sentence completion task (textbox)

        This evaluation of Winogrande uses partial evaluation as described by
        Trinh & Le in Simple Method for Commonsense Reasoning (2018).
        See: https://arxiv.org/abs/1806.02847
        """

        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/winogrande.py#L66-L70
        def _partial_ctx(eg: Dict[str, Any], option: str):
            pronoun_loc = eg.get("sentence").index("_")
            return eg.get("sentence")[:pronoun_loc] + option

        return _partial_ctx(eg, eg.get(f"option{eg.get('answer')}"))

    @classmethod
    def get_target(cls, eg: Dict[str, Any]) -> str:
        """Use for sentence completion task (textbox)

        This evaluation of Winogrande uses partial evaluation as described by
        Trinh & Le in Simple Method for Commonsense Reasoning (2018).
        See: https://arxiv.org/abs/1806.02847
        """

        # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/winogrande.py#L76-L79
        def _partial_target(eg: Dict[str, Any]) -> str:
            pronoun_loc = eg.get("sentence").index("_") + 1
            return " " + eg.get("sentence")[pronoun_loc:].strip()

        return _partial_target(eg)

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
                        "text": eg.get("sentence"),
                        "options": [
                            {"id": "option1", "text": eg.get("option1")},
                            {"id": "option2", "text": eg.get("option2")},
                        ],
                        "meta": {"label": cls.CLASS_LABELS[int(eg.get("answer")) - 1]},
                    }
                )
            elif interface == Interface.textbox.value:
                annotation_tasks.append(
                    {"text": cls.get_prompt(eg), "meta": {"label": cls.get_target(eg)}}
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
