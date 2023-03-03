import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Tuple

from dotenv import load_dotenv
from langchain.document_loaders import PagedPDFSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from prodigy.core import recipe
from prodigy.util import log, msg


@recipe(
    # fmt: off
    "langchain.textcat",
    dataset=("Dataset to save answers to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    labels=("Labels (comma delimited)", "option", "L", lambda s: s.split(",")),
    guideline=("Path to the PDF annotation guideline", "option", "G", Path),
    model=("GPT-3 model to use for completion", "option", "m", str),
    temperature=("Temperature parameter to control LLM generation", "option", "t", float),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    # fmt: on
)
def langchain_ner(
    dataset: str,
    source: Union[str, Iterable[Dict]],
    labels: List[str],
    guideline: Path,
    model: str = "text-davinci-003",
    temperature: float = 0.7,
    batch_size: int = 10,
    loader: Optional[str] = None,
):
    """Perform zero-shot annotation using GPT-3 with the aid of an annotation guideline."""

    api_key, _ = get_api_credentials()

    if not labels:
        msg.fail("No --labels argument set", exits=0)
    if not guideline.exists():
        msg.fail(f"Cannot find path to the annotation guideline ({guideline})", exits=0)

    loader = PagedPDFSplitter(guideline)
    llm = OpenAI(openai_api_key=api_key, model_name=model, temperature=temperature)


def get_api_credentials() -> Tuple[str, str]:
    """Obtain OpenAI API credentials from a .env file"""
    load_dotenv()
    api_key = os.getenv("PRODIGY_OPENAI_KEY")
    api_org = os.getenv("PRODIGY_OPENAI_ORG")
    return api_key, api_org
