from typing import List, Tuple

from datasets import load_dataset
from tqdm import tqdm


def compute_elo_rankings(
    matchups: List[Tuple[str, str, int]], initial_elo_rating: int = 1000, k: int = 32
) -> List[str]:
    """Compute ELO rankings given a list of matchups"""
    elo_ratings = {}

    for p1, p2, result in matchups:
        if p1 not in elo_ratings:
            elo_ratings[p1] = initial_elo_rating
        if p2 not in elo_ratings:
            elo_ratings[p2] = initial_elo_rating

        # Compute expected scores
        p1_expected = 1 / (1 + 10 ** ((elo_ratings[p2] - elo_ratings[p1]) / 400))
        p2_expected = 1 - p1_expected

        # Update Elo ratings based on result
        if result == 0:  # Player 1 wins
            elo_ratings[p1] += k * (1 - p1_expected)
            elo_ratings[p2] += k * (0 - p2_expected)
        else:
            elo_ratings[p1] += k * (0 - p1_expected)
            elo_ratings[p2] += k * (1 - p2_expected)

    # Rank players on descending order
    ranked_players = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    return ranked_players


def preprocess_openai_summarize() -> Tuple[List[str], List[str]]:
    """Preprocess OpenAI's Summarize from Human Feedback dataset"""
    dataset = load_dataset(
        "openai/summarize_from_feedback", name="comparisons", split="train"
    )

    df = dataset.to_pandas()
    df["id"] = df["info"].apply(lambda x: x.get("id"))

    chosen_texts = []
    rejected_texts = []
    for _, instances in tqdm(df.groupby("id")):
        matchups = [
            (
                instance["summaries"][0].get("text"),
                instance["summaries"][1].get("text"),
                instance["choice"],
            )
            for _, instance in instances.iterrows()
        ]
        ranked = compute_elo_rankings(matchups)

        # Get best and worst
        chosen_texts.append(ranked[0][0])
        rejected_texts.append(ranked[-1][0])

    return chosen_texts, rejected_texts


def preprocess_stanford_shp() -> Tuple[List[str], List[str]]:
    """Preprocess the explaimlikeimfive_train subset from Stanford SHP"""
    dataset = load_dataset("stanfordnlp/SHP", split="train").filter(
        lambda x: x["domain"] == "explainlikeimfive_train"
    )

    df = dataset.to_pandas()
    chosen_texts = []
    rejected_texts = []
    for _, instances in tqdm(df.groupby("post_id")):
        matchups = [
            (instance["human_ref_A"], instance["human_ref_B"], instance["labels"])
            for _, instance in instances.iterrows()
        ]
        ranked = compute_elo_rankings(matchups)
        # Get best and worst
        chosen_texts.append(ranked[0][0])
        rejected_texts.append(ranked[-1][0])

    return chosen_texts, rejected_texts


def preprocess_argilla_ultrafeedback() -> Tuple[List[str], List[str]]:
    """Preprocess the Flan-v2 subset of Argilla's cleaned Ultrafeedback dataset"""
    dataset = load_dataset(
        "argilla/ultrafeedback-multi-binarized-quality-preferences-cleaned",
        split="train",
    ).filter(lambda x: x["source"] == "flan_v2_niv2")
    chosen_texts = []
    rejected_texts = []
    for example in dataset:
        chosen_texts.append(example.get("chosen")[1].get("content"))
        rejected_texts.append(example.get("rejected")[1].get("content"))

    return chosen_texts, rejected_texts


def preprocess_tatsulab_alpacafarm():
    """Preprocess Tatsu Lab's AlpacaFarm dataset"""
    dataset = load_dataset(
        "tatsu-lab/alpaca_farm",
        "alpaca_human_preference",
        split="preference",
    )

    chosen_texts = []
    rejected_texts = []
    for example in dataset:
        preference = example.get("preference")
        chosen_texts.append(example.get(f"output_{preference}"))
        rejected_texts.append(example.get(f"output_{2 - preference + 1}"))
    return chosen_texts, rejected_texts


def preprocess_berkeley_nest_nectar():
    """Preprocess Berkeley NEST's Nectar dataset"""
    dataset = load_dataset("berkeley-nest/Nectar", split="train")
    dataset = dataset.filter(lambda eg: eg["turns"] == 1)

    chosen_texts = []
    rejected_texts = []
    for example in dataset:
        answers = example["answers"]
        for answer in answers:
            if answer.get("rank") == 1:
                chosen_texts.append(answer.get("answer"))
            if answer.get("rank") == len(answers):
                rejected_texts.append(answer.get("answer"))

    return chosen_texts, rejected_texts


DATASET_PREPROCESSORS = {
    "openai/summarize_from_feedback": preprocess_openai_summarize,
    "stanford/SHP": preprocess_stanford_shp,
    "argilla/ultrafeedback-multi-binarized-quality-preferences-cleaned": preprocess_argilla_ultrafeedback,
    "tatsu-lab/alpaca_farm": preprocess_tatsulab_alpacafarm,
    "berkeley-nest/Nectar": preprocess_berkeley_nest_nectar,
}
