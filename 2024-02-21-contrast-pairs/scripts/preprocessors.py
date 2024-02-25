from typing import List, Tuple

from datasets import load_dataset


def preprocess_openai_summarize(include_prompt: bool) -> Tuple[List[str], List[str]]:
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


def preprocess_stanford_shp(include_prompt: bool) -> Tuple[List[str], List[str]]:
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


def preprocess_argilla_ultrafeedback(
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
        chosen = example.get("chosen")[1].get("content")
        rejected = example.get("rejected")[1].get("content")

        chosen_texts.append(prompt + " " + chosen if include_prompt else chosen)
        rejected_texts.append(prompt + " " + rejected if include_prompt else rejected)

    return chosen_texts, rejected_texts


def preprocess_tatsulab_alpacafarm(include_prompt: bool):
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


DATASET_PREPROCESSORS = {
    "openai/summarize_from_feedback": preprocess_openai_summarize,
    "stanford/SHP": preprocess_stanford_shp,
    "argilla/ultrafeedback-multi-binarized-quality-preferences-cleaned": preprocess_argilla_ultrafeedback,
    "tatsu-lab/alpaca_farm": preprocess_tatsulab_alpacafarm,
}
