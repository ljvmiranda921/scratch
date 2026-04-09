"""Download MADLAD dataset, translate, and classify using Azure OpenAI."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import aiohttp
import pandas as pd
from openai import AsyncAzureOpenAI
from tqdm import tqdm
from transformers import AutoProcessor

from prompts import CLASSIFY_SYSTEM_PROMPT, build_classify_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for lib in ("huggingface_hub", "transformers", "httpx", "openai"):
    logging.getLogger(lib).setLevel(logging.WARNING)
log = logging.getLogger(__name__)

TRANSLATE_TOKENIZER = AutoProcessor.from_pretrained("google/translategemma-4b-it")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Classify MADLAD Data (LM-based)")
    parser.add_argument("-l", "--language", type=str, help="Language code for the specific MADLAD subsplit.")
    parser.add_argument("-T", "--translategemma_lang_code", type=str, default=None, help="TranslateGemma language code (defaults to --language).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for long-running tasks like translation and classification.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances to process.")
    parser.add_argument("--shuffle", action="store_true", default=False, help="If set, will shuffle the instances before running the classification pipeline.")
    parser.add_argument("--truncate", type=int, default=None, help="Truncate input text to this many characters before translation.")
    parser.add_argument("--azure_endpoint", type=str, default=None, help="Azure OpenAI endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT env var.")
    parser.add_argument("--azure_deployment", type=str, default=None, help="Azure OpenAI deployment name. Falls back to AZURE_OPENAI_DEPLOYMENT env var.")
    parser.add_argument("--api_version", type=str, default="2024-12-01-preview", help="Azure OpenAI API version.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = load_madlad(args.language, split="clean_docs")
    if args.limit:
        df = df.head(args.limit)
    if args.shuffle:
        logging.info("Shuffling the dataset")
        df = df.sample(frac=1).reset_index(drop=True)
    logging.info(f"Number of documents: {len(df)}")

    # Translation first
    src_lang = args.translategemma_lang_code or args.language
    texts = df["text"].tolist()
    if args.truncate:
        logging.info(f"Truncating texts to {args.truncate} characters")
        texts = [t[: args.truncate] for t in texts]
    df["translation"] = asyncio.run(
        batch_translate(
            texts,
            src_lang=src_lang,
            batch_size=args.batch_size,
        )
    )

    # Classify translated text using Azure OpenAI
    results = asyncio.run(
        batch_classify(
            df["translation"].tolist(),
            azure_endpoint=args.azure_endpoint,
            azure_deployment=args.azure_deployment,
            api_version=args.api_version,
            batch_size=args.batch_size,
        )
    )
    df["topic"] = [r.get("topic", "") for r in results]
    df["format"] = [r.get("format", "") for r in results]
    df["sib200"] = [r.get("sib200", "") for r in results]

    breakpoint()


async def batch_classify(
    texts: list[str],
    azure_endpoint: str | None = None,
    azure_deployment: str | None = None,
    api_version: str = "2024-12-01-preview",
    batch_size: int = 8,
) -> list[dict]:
    """Classify texts using Azure OpenAI in async batches."""
    client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_version=api_version,
    )
    semaphore = asyncio.Semaphore(batch_size)
    pbar = tqdm(total=len(texts), desc="Classifying")

    async def _classify_with_limit(idx: int, text: str):
        async with semaphore:
            result = await classify(client, text, azure_deployment)
            pbar.update(1)
            return idx, result

    tasks = [_classify_with_limit(i, text) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)

    pbar.close()
    await client.close()
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


async def classify(
    client: AsyncAzureOpenAI,
    text: str,
    deployment: str | None = None,
) -> dict:
    """Classify a single document into topic, format, and SIB-200 category."""
    user_prompt = build_classify_prompt(text)
    response = await client.chat.completions.create(
        model=deployment or "gpt-4o",
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        log.warning(f"Failed to parse LM response: {content}")
        return {"topic": "", "format": "", "sib200": ""}


# ---------------------------------------------------------------------------
# Translation (reused from classify_madlad_data.py)
# ---------------------------------------------------------------------------


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
    # Strip leading BOS token to avoid double-BOS (llama-server adds its own)
    bos = TRANSLATE_TOKENIZER.tokenizer.bos_token
    if bos and prompt.startswith(bos):
        prompt = prompt[len(bos):]
    async with session.post(
        f"{base_url}/completion",
        json={"prompt": prompt, "n_predict": 512, "temperature": 0.0},
    ) as response:
        if not response.ok:
            body = await response.text()
            raise RuntimeError(
                f"llama-server returned {response.status} for text ({len(text)} chars): {body}"
            )
        data = await response.json()
        return data["content"].strip()


# ---------------------------------------------------------------------------
# Data loading (reused from classify_madlad_data.py)
# ---------------------------------------------------------------------------


def load_madlad(lang: str, split: str = "clean_docs") -> pd.DataFrame:
    """Load MADLAD-400 data for a given language code."""
    from huggingface_hub import HfApi, hf_hub_download

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
