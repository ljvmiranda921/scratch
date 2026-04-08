"""Download MADLAD dataset and classify it using a model."""

import argparse
import asyncio
import logging
from pathlib import Path

import aiohttp
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

TRANSLATE_TOKENIZER = AutoProcessor.from_pretrained("google/translategemma-4b-it")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Classify MADLAD Data")
    parser.add_argument("-l", "--language", type=str, help="Language code for the specific MADLAD subsplit.")
    parser.add_argument("-T", "--translategemma_lang_code", type=str, default=None, help="TranslateGemma language code (defaults to --language).")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size for long-running tasks like translation and classification.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = load_madlad(args.language, split="clean_docs")
    src_lang = args.translategemma_lang_code or args.language
    df["translation"] = asyncio.run(
        translate_all(
            df["text"].tolist(),
            src_lang=src_lang,
            batch_size=args.batch_size,
        )
    )


async def translate_all(
    texts: list[str],
    src_lang: str,
    tgt_lang: str = "en",
    batch_size: int = 8,
    base_url: str = "http://localhost:8080",
) -> list[str]:
    """Translate texts in async batches using translategemma via llama-server."""
    translations = []
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i : i + batch_size]
            tasks = [
                translate(session, text, src_lang, tgt_lang, base_url) for text in batch
            ]
            results = await asyncio.gather(*tasks)
            translations.extend(results)
    return translations


async def translate(
    session: aiohttp.ClientSession,
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
    async with session.post(
        f"{base_url}/completion",
        json={"prompt": prompt, "n_predict": 512, "temperature": 0.0},
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data["content"].strip()


def load_madlad(lang: str, split: str = "clean_docs") -> pd.DataFrame:
    """Load MADLAD-400 data for a given language code."""
    api = HfApi()
    files = list(
        api.list_repo_tree(
            "allenai/MADLAD-400",
            path_in_repo=f"data-v1p5/{lang}",
            repo_type="dataset",
        )
    )
    files = [
        f
        for f in files
        if f.rfilename.endswith(".jsonl.gz")
        and (split == "all" or split in f.rfilename)
    ]

    local_dir = Path("data") / lang
    local_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for f in tqdm(files, desc=f"Downloading {lang}"):
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
