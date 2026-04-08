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
    parser.add_argument("-n", "--limit", type=int, default=None, help="Limit number of instances to process.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = load_madlad(args.language, split="clean_docs")
    if args.limit:
        df = df.head(args.limit)
    logging.info(f"Number of documents: {len(df)}")
    src_lang = args.translategemma_lang_code or args.language
    df["translation"] = asyncio.run(
        batch_translate(
            df["text"].tolist(),
            src_lang=src_lang,
            batch_size=args.batch_size,
        )
    )


async def batch_translate(
    texts: list[str],
    src_lang: str,
    tgt_lang: str = "en",
    batch_size: int = 8,
    base_url: str = "http://localhost:8080",
) -> list[str]:
    """Translate texts in async batches using translategemma via llama-server."""
    semaphore = asyncio.Semaphore(batch_size)
    pbar = tqdm(total=len(texts), desc="Translating")

    async def _translate_with_limit(idx, text):
        async with semaphore:
            result = await translate(session, text, src_lang, tgt_lang, base_url)
            pbar.update(1)
            return idx, result

    async with aiohttp.ClientSession() as session:
        tasks = [_translate_with_limit(i, text) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks)

    pbar.close()
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


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
    local_dir = Path("data") / lang
    existing = sorted(local_dir.glob("**/*.jsonl.gz")) if local_dir.exists() else []
    if split != "all":
        existing = [p for p in existing if split in p.name]

    if existing:
        log.info(
            f"Found {len(existing)} cached files in {local_dir}, skipping download"
        )
        paths = existing
    else:
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
