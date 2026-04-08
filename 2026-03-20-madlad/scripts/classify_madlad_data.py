"""Download MADLAD dataset and classify it using a model."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRANSLATE_TOKENIZER = AutoProcessor.from_pretrained("google/translategemma-4b-it")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Classify MADLAD Data")
    parser.add_argument("-l", "--language", type=str, help="Language code for the specific MADLAD subsplit.")
    parser.add_argument("-T", "--translategemma_lang_code", type=str, default=None, help="TranslateGemma language code (defaults to --language).")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = load_madlad(args.language, split="clean_docs")
    src_lang = args.translategemma_lang_code or args.language
    df["translation"] = df["text"].apply(lambda text: translate(text, src_lang=src_lang))  # fmt: skip
    breakpoint()


def translate(
    text: str,
    src_lang: str,
    tgt_lang: str = "en",
    base_url: str = "http://localhost:8080",
) -> str:
    """Translate text using translategemma via llama-server."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": src_lang,
                    "target_lang_code": tgt_lang,
                    "text": text,
                }
            ],
        }
    ]
    prompt = TRANSLATE_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = requests.post(
        f"{base_url}/completion",
        json={"prompt": prompt, "n_predict": 512, "temperature": 0.0},
    )
    response.raise_for_status()
    return response.json()["content"].strip()


def load_madlad(lang: str, split: str = "clean_docs") -> pd.DataFrame:
    """Load MADLAD-400 data for a given language code."""
    api = HfApi()
    files = api.list_repo_tree(
        "allenai/MADLAD-400",
        path_in_repo=f"data-v1p5/{lang}",
        repo_type="dataset",
    )

    local_dir = Path("data") / lang
    local_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for f in files:
        if not f.rfilename.endswith(".jsonl.gz"):
            continue
        if split != "all" and split not in f.rfilename:
            continue
        path = hf_hub_download(
            "allenai/MADLAD-400",
            filename=f.rfilename,
            repo_type="dataset",
            local_dir=local_dir,
        )
        paths.append(path)

    df = pd.concat([pd.read_json(p, lines=True) for p in paths], ignore_index=True)
    return df


if __name__ == "__main__":
    main()
