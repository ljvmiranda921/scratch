"""Download MADLAD dataset and classify it using a model."""

import argparse
import asyncio
import logging
from pathlib import Path

import aiohttp
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for lib in ("huggingface_hub", "transformers", "httpx"):
    logging.getLogger(lib).setLevel(logging.WARNING)
log = logging.getLogger(__name__)

TRANSLATE_TOKENIZER = AutoProcessor.from_pretrained("google/translategemma-4b-it")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Classify MADLAD Data")
    parser.add_argument("-l", "--language", type=str, help="Language code for the specific MADLAD subsplit.")
    parser.add_argument("-T", "--translategemma_lang_code", type=str, default=None, help="TranslateGemma language code (defaults to --language).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for long-running tasks like translation and classification.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances to process.")
    parser.add_argument("--shuffle", action="store_true", default=False, help="If set, will shuffle the instances before running the classification pipeline.")
    parser.add_argument("--include_url", action="store_true", default=False, help="If set, prepend URL to text for classification. If not set, use the NoURL model variants.")
    parser.add_argument("--truncate", type=int, default=None, help="Truncate input text to this many characters before translation.")
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

    # Classify on translated text
    urls = df["url"].tolist() if args.include_url else None

    topic_tokenizer, topic_model, topic_id2label, device = _load_classifier(
        "TopicClassifier", args.include_url
    )
    df["topic"] = classify_topic(
        df["translation"].tolist(),
        topic_tokenizer,
        topic_model,
        topic_id2label,
        device,
        urls=urls,
    )

    format_tokenizer, format_model, format_id2label, device = _load_classifier(
        "FormatClassifier", args.include_url
    )
    df["format"] = classify_format(
        df["translation"].tolist(),
        format_tokenizer,
        format_model,
        format_id2label,
        device,
        urls=urls,
    )

    breakpoint()


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
        prompt = prompt[len(bos) :]
    async with session.post(
        f"{base_url}/completion",
        json={"prompt": prompt, "n_predict": 512, "temperature": 0.0},
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data["content"].strip()


def _load_classifier(name: str, include_url: bool) -> tuple:
    """Load a WebOrganizer classifier and its tokenizer.

    Returns (tokenizer, model, id2label, device).
    """
    model_name = f"WebOrganizer/{name}" if include_url else f"WebOrganizer/{name}-NoURL"
    log.info(f"Loading classifier: {model_name}")
    # MPS has issues with these custom models, so only use CUDA or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_memory_efficient_attention=False,
            unpad_inputs=False,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    id2label = model.config.id2label
    return tokenizer, model, id2label, device


@torch.no_grad()
def classify_topic(
    texts: list[str],
    tokenizer,
    model,
    id2label: dict,
    device: torch.device,
    urls: list[str] | None = None,
) -> list[str]:
    """Classify documents into topic categories."""
    results = []
    for i, text in enumerate(tqdm(texts, desc="Classifying topics")):
        input_text = f"{urls[i]}\n\n{text}" if urls else text
        inputs = tokenizer([input_text], return_tensors="pt", truncation=True).to(device)
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model(**inputs)
        pred = outputs.logits.softmax(dim=-1).argmax(dim=-1).item()
        results.append(id2label[pred])
    return results


@torch.no_grad()
def classify_format(
    texts: list[str],
    tokenizer,
    model,
    id2label: dict,
    device: torch.device,
    urls: list[str] | None = None,
) -> list[str]:
    """Classify documents into format categories."""
    results = []
    for i, text in enumerate(tqdm(texts, desc="Classifying formats")):
        input_text = f"{urls[i]}\n\n{text}" if urls else text
        inputs = tokenizer([input_text], return_tensors="pt", truncation=True).to(device)
        seq_len = inputs["input_ids"].shape[1]
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model(**inputs)
        pred = outputs.logits.softmax(dim=-1).argmax(dim=-1).item()
        results.append(id2label[pred])
    return results


if __name__ == "__main__":
    main()
