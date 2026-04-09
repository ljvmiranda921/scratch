"""Download MADLAD dataset, translate, and classify using Azure OpenAI."""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from openai import APIError, AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoProcessor

from prompts import (
    FORMAT_SYSTEM_PROMPT,
    SIB200_SYSTEM_PROMPT,
    TOPIC_SYSTEM_PROMPT,
    FormatAnnotation,
    SIB200Annotation,
    TopicAnnotation,
    build_format_prompt,
    build_sib200_prompt,
    build_topic_prompt,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
for lib in ("huggingface_hub", "transformers", "httpx", "openai"):
    logging.getLogger(lib).setLevel(logging.WARNING)
log = logging.getLogger(__name__)

load_dotenv()

TRANSLATE_TOKENIZER = AutoProcessor.from_pretrained("google/translategemma-4b-it")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Classify MADLAD Data (LM-based)")
    parser.add_argument("-l", "--language", type=str, help="Language code for the specific MADLAD subsplit.")
    parser.add_argument("-T", "--translategemma_lang_code", type=str, default=None, help="TranslateGemma language code (defaults to --language).")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Language model to use for classification.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for long-running tasks like translation and classification.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances to process.")
    parser.add_argument("--shuffle", action="store_true", default=False, help="If set, will shuffle the instances before running the classification pipeline.")
    parser.add_argument("--truncate", type=int, default=None, help="Truncate input text to this many characters before translation.")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure OpenAI instead of OpenAI. Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env.")
    parser.add_argument("--azure_api_version", type=str, default="2024-12-01-preview", help="Azure OpenAI API version.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between batches (for rate limiting).")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries per request on rate limit errors.")
    parser.add_argument("--resume", type=str, default=None, help="Path to an existing output CSV to resume from (skips already-annotated rows).")
    parser.add_argument("--output_dir", type=str, default="data/classified", help="Directory to save classified CSV.")
    # fmt: on
    return parser.parse_args()


def _build_client(args) -> AsyncOpenAI:
    if args.use_azure:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_endpoint or not azure_key:
            log.error("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")
            sys.exit(1)
        return AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=args.azure_api_version,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY is not set!")
            sys.exit(1)
        return AsyncOpenAI(api_key=api_key)


def main():
    args = get_args()

    df = load_madlad(args.language, split="clean_docs")
    if args.limit:
        df = df.head(args.limit)
    if args.shuffle:
        log.info("Shuffling the dataset")
        df = df.sample(frac=1).reset_index(drop=True)
    log.info(f"Number of documents: {len(df)}")

    # Translation first
    src_lang = args.translategemma_lang_code or args.language
    texts = df["text"].tolist()
    if args.truncate:
        log.info(f"Truncating texts to {args.truncate} characters")
        texts = [t[: args.truncate] for t in texts]
    df["translation"] = asyncio.run(
        batch_translate(
            texts,
            src_lang=src_lang,
            batch_size=args.batch_size,
        )
    )

    # Set up output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        output_path = Path(args.resume)
        if not output_path.exists():
            log.error(f"Resume file not found: {output_path}")
            sys.exit(1)
        existing_df = pd.read_csv(output_path)
        done_indices = set(existing_df.index.tolist())
        log.info(f"Resuming from {output_path} ({len(done_indices)} rows already done)")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{args.language}_{timestamp}_classified.csv"
        done_indices = set()

    client = _build_client(args)

    # Run three classification passes
    for task_name, system_prompt, build_fn, response_model, label_col, reasoning_col in [
        ("topic", TOPIC_SYSTEM_PROMPT, build_topic_prompt, TopicAnnotation, "topic", "topic_reasoning"),
        ("format", FORMAT_SYSTEM_PROMPT, build_format_prompt, FormatAnnotation, "format", "format_reasoning"),
        ("sib200", SIB200_SYSTEM_PROMPT, build_sib200_prompt, SIB200Annotation, "sib200", "sib200_reasoning"),
    ]:
        log.info(f"Classifying: {task_name}")
        requests = []
        for idx, row in df.iterrows():
            if idx in done_indices:
                continue
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_fn(row["translation"])},
            ]
            requests.append((idx, messages))

        if not requests:
            log.info(f"  {task_name}: nothing to do, all rows already annotated")
            continue

        results = asyncio.run(
            submit_async_requests(
                requests,
                client=client,
                model=args.model,
                response_model=response_model,
                batch_size=args.batch_size,
                delay_between_batches=args.delay,
                max_retries=args.max_retries,
                desc=f"Classifying {task_name}",
            )
        )
        results_map = {r["_idx"]: r for r in results}
        df[label_col] = df.index.map(lambda i: results_map.get(i, {}).get("label", ""))
        df[reasoning_col] = df.index.map(lambda i: results_map.get(i, {}).get("reasoning", ""))

    # Save final output
    df.to_csv(output_path, index=False)
    log.info(f"Done! Results saved to {output_path}")


# ---------------------------------------------------------------------------
# LM classification
# ---------------------------------------------------------------------------


async def submit_async_requests(
    messages: list[tuple[int, list[dict]]],
    *,
    client: AsyncOpenAI,
    model: str,
    response_model: type[BaseModel],
    batch_size: int = 8,
    delay_between_batches: float = 1.0,
    max_retries: int = 5,
    desc: str = "Processing",
) -> list[dict[str, Any]]:
    """Submit async requests in batches, with delay between batches."""
    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]
    all_results = []

    for batch_num, batch in enumerate(tqdm_asyncio(batches, desc=desc, unit="batch"), start=1):
        tasks = [
            _process_single(client, idx, msgs, model=model, response_model=response_model, max_retries=max_retries)
            for idx, msgs in batch
        ]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)

        if batch_num < len(batches):
            await asyncio.sleep(delay_between_batches)

    return all_results


async def _process_single(
    client: AsyncOpenAI,
    idx: int,
    messages: list[dict],
    *,
    model: str,
    response_model: type[BaseModel],
    max_retries: int = 5,
) -> dict[str, Any]:
    """Process a single request with structured output and retry on rate limits."""
    for attempt in range(max_retries):
        try:
            response = await client.beta.chat.completions.parse(
                messages=messages,
                model=model,
                response_format=response_model,
            )
            parsed = response.choices[0].message.parsed.model_dump()
            return {"_idx": idx, **parsed}
        except RateLimitError as e:
            wait = 2**attempt
            log.warning(f"Rate limited on idx={idx}, retrying in {wait}s (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(wait)
        except APIError as e:
            wait = 2**attempt
            log.warning(f"API error on idx={idx}, retrying in {wait}s (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(wait)
    log.error(f"Failed after {max_retries} retries for idx={idx}")
    return {"_idx": idx, "reasoning": "", "label": ""}


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
