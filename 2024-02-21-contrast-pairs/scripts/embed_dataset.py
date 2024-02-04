from pathlib import Path
from typing import List, Tuple

import numpy as np
import typer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from wasabi import msg


def main(
    # fmt: off
    dataset_name: str = typer.Argument(..., help="HuggingFace dataset name."),
    output_dir: Path = typer.Argument(..., help="Directory to save the embeddings."),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace namespace for the embedding model."),
    include_prompt: bool = typer.Option(False, help="Include prompt in the chosen and rejected texts."),
    # fmt: on
):

    dataset_preprocessors = {
        "openai/summarize_from_feedback": _preprocess_openai_summarize,
        "stanford/SHP": _preprocess_stanford_shp,
        "argilla/ultrafeedback-multi-binarized-quality-preferences-cleaned": _preprocess_argilla_ultrafeedback,
        "tatsu-lab/alpaca_farm": _preprocess_tatsulab_alpacafarm,
    }

    if dataset_name not in dataset_preprocessors:
        msg.fail(
            f"No preprocessor found for {dataset_name}. Available: {', '.join(dataset_preprocessors.keys())}",
            exits=1,
        )
    chosen, rejected = dataset_preprocessors.get(dataset_name)(include_prompt)

    # Get the embeddings
    model = SentenceTransformer(embedding_model)
    msg.info(f"Embedding the sentences using {embedding_model}")
    embeddings = {
        "chosen": model.encode(chosen, show_progress_bar=True),
        "rejected": model.encode(rejected, show_progress_bar=True),
    }
    # Save the embeddings
    subdir = output_dir / dataset_name.replace("/", "___")
    subdir.mkdir(exist_ok=True, parents=True)
    for k, v in embeddings.items():
        output_file = subdir / f"{k}.npy"
        np.save(output_file, v)
        msg.good(f"Saved embeddings to {output_file}")


def _preprocess_openai_summarize(include_prompt: bool) -> Tuple[List[str], List[str]]:
    """Preprocess OpenAI's Summarize from Human Feedback dataset"""
    dataset = load_dataset(
        "openai/summarize_from_feedback", name="comparisons", split="train"
    )

    chosen_texts = []
    rejected_texts = []
    for example in dataset:
        prompt = example["info"].get("post")
        choice = example["choice"]

        chosen = example["summaries"][choice].get("text")
        rejected = example["summaries"][1 - choice].get("text")

        chosen_texts.append(prompt + " " + chosen if include_prompt else chosen)
        rejected_texts.append(prompt + " " + rejected if include_prompt else rejected)

    return chosen_texts, rejected_texts


def _preprocess_stanford_shp(include_prompt: bool) -> Tuple[List[str], List[str]]:
    """Preprocess the explaimlikeimfive_train subset from Stanford SHP"""
    dataset = load_dataset("stanfordnlp/SHP", split="train").filter(
        lambda x: x["domain"] == "explainlikeimfive_train"
    )

    chosen_texts = []
    rejected_texts = []
    for example in dataset:
        prompt = example["history"]
        ref_chosen, ref_rejected = ("A", "B") if example["labels"] == 0 else ("B", "A")
        chosen_texts.append(
            prompt + " " + example[f"human_ref_{ref_chosen}"]
            if include_prompt
            else example[f"human_ref_{ref_chosen}"]
        )
        rejected_texts.append(
            prompt + " " + example[f"human_ref_{ref_rejected}"]
            if include_prompt
            else example[f"human_ref_{ref_rejected}"]
        )

    return chosen_texts, rejected_texts


def _preprocess_argilla_ultrafeedback(
    include_prompt: bool,
) -> Tuple[List[str], List[str]]:
    """Preprocess the Flan-v2 subset of Argilla's cleaned Ultrafeedback dataset"""
    dataset = load_dataset(
        "argilla/ultrafeedback-multi-binarized-quality-preferences-cleaned",
        split="train",
    ).filter(lambda x: x["source"] == "flan_v2_niv2")

    chosen_texts = []
    rejected_texts = []
    for example in dataset:
        prompt = example.get("prompt")
        chosen = example.get("chosen")[0].get("content")
        rejected = example.get("rejected")[0].get("content")

        chosen_texts.append(prompt + " " + chosen if include_prompt else chosen)
        rejected_texts.append(prompt + " " + rejected if include_prompt else rejected)

    return chosen_texts, rejected_texts


def _preprocess_tatsulab_alpacafarm(include_prompt: bool):
    """Preprocess Tatsu Lab's AlpacaFarm dataset"""
    dataset = load_dataset(
        "tatsu-lab/alpaca_farm",
        "alpaca_human_preference",
        split="preference",
    )

    chosen_texts = []
    rejected_texts = []
    for example in dataset:
        prompt = (
            example.get("instruction") + " " + example.get("input")
            if example.get("input")
            else example.get("instruction")
        )

        preference = example.get("preference")
        chosen = example.get(f"output_{preference}")
        rejected = example.get(f"output_{2 - preference + 1}")

        chosen_texts.append(prompt + " " + chosen if include_prompt else chosen)
        rejected_texts.append(prompt + " " + rejected if include_prompt else rejected)

    return chosen_texts, rejected_texts


if __name__ == "__main__":
    typer.run(main)
