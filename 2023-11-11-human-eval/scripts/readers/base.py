import abc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset
from spacy.language import Language
from spacy.tokens import Doc

from ..utils import Interface


class DatasetReader(abc.ABC):
    """Dataset reader implementation"""

    @abc.abstractproperty
    def class_labels(self) -> Optional[List[str]]:
        """Class labels to get the options from"""
        return None

    @abc.abstractproperty
    def task_type(self) -> str:
        """Task type to implement"""
        ...

    @abc.abstractproperty
    def hf_config(self) -> str:
        """HuggingFace configuration to use"""
        ...

    @abc.abstractmethod
    def get_prompt(self, eg: Dict[str, Any]) -> str:
        """Construct the prompt

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (str): the prompt for that given example.
        """
        ...

    @abc.abstractmethod
    def get_targets(self, eg: Dict[str, Any]) -> List[str]:
        """Get the targets for evaluating LM performance

        eg (Dict[str, Any]): an example from the dataset.
        RETURNS (List[str]): a list of targets to compute evals.
        """
        ...

    @abc.abstractmethod
    def convert_to_prodigy(
        self, examples: "Dataset", interface: Interface
    ) -> List[Dict[str, Any]]:
        """Convert each HuggingFace example into Prodigy instances

        examples (datasets.Dataset): a particular split from a Huggingface dataset
        interface (Interface): the Prodigy annotation interface to build task examples upon.
        RETURNS (List[Dict[str, Any]]): an iterable containing all annotation tasks formatted for Prodigy.
        """
        ...

    @abc.abstractmethod
    def get_reference_docs(self, nlp: Language, references: Path) -> List[Doc]:
        """Get reference documents to compare human annotations against

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        references (Path): Path to the examples file.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        ...

    @abc.abstractmethod
    def get_predicted_docs(self, nlp, predictions: Path) -> List[Doc]:
        """Get predicted documents to compare on the gold-reference data.

        nlp (Language): a spaCy language pipeline to obtain the vocabulary.
        predictions (Path): Path to the examples file.
        RETURNS (List[Doc]): list of spaCy Doc objects for later evaluation.
        """
        ...
