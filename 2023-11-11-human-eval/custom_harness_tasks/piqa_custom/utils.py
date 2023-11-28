from pathlib import Path

import srsly
from datasets import Dataset


def process_docs(dataset: "Dataset", data_files: Path):
    annotated_texts = [eg.get("text") for eg in srsly.read_jsonl(data_files)]

    def _helper(doc):
        if doc in annotated_texts:
            return doc

    return dataset.map(_helper)
