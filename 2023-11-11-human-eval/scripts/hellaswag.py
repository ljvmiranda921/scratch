from pathlib import Path
from typing import Any, Dict, List

import spacy
import srsly
import typer
from datasets import load_dataset
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example
from wasabi import msg

from .utils import Interface, Split

app = typer.Typer()


class HellaSwagDataset:
    pass
