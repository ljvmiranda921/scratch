from typing import Any, List, Tuple

import torch
from transformers import AutoTokenizer, Pipeline, pipeline
from wasabi import msg

TASK = "text-generation"
# Use Instruct models
MODELS = {
    "falcon": {
        "name": "tiiuae/falcon-7b-instruct",
        "url": "https://huggingface.co/tiiuae/falcon-7b-instruct",
    },
    "mpt": {
        "name": "mosaicml/mpt-7b-instruct",
        "url": "https://huggingface.co/mosaicml/mpt-7b-instruct",
    },
    "llama2-together": {
        "name": "togethercomputer/Llama-2-7b-32K-Instruct",
        "url": "https://huggingface.co/togethercomputer/Llama-2-7B-32K-Instruct",
    },
    "llama2-meta": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "url": "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf",
    },
    "mistral": {
        "name": "mistralai/Mistral-7b-Instruct-v0.1",
        "url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1",
    },
}


def init_pipeline(model: str) -> "Pipeline":
    """Initialize a model pipeline

    model (str): a model's shorthand name.

    RETURNS (Pipeline): a transformer pipeline used for `prompt_model`
    """
    # https://huggingface.co/transformers/v4.10.1/main_classes/pipelines.html#transformers.TextGenerationPipeline
    tokenizer = AutoTokenizer.from_pretrained(model)
    init_params = {
        "task": TASK,
        "model": model.get("name"),
        "tokenizer": tokenizer,
        "device_map": "auto",
    }
    if model == "falcon":
        init_params["torch_dtype"] = torch.bfloat16
    pipe = pipeline(**init_params)
    msg.good(f"Initializing pipeline with the config: {init_params}")
    return pipe


def prompt_model(
    prompt: str,
    pipeline: "Pipeline",
    top_k: int = 1,
    max_new_tokens: int = 3,
    verbose: bool = False,
) -> Tuple[List[bool], List[Any]]:
    """Prompt a model.

    prompt (str): the prompt to pass to the model.
    pipeline (transformers.Pipeline): a transformer pipeline created from `init_pipeline`.
    top_k (int): return k generations, default is 1.
    max_new_tokens (int): number of tokens to generate, default is 3.
    verbose (bool): show additional information.

    RETURNS (Tuple[List, List]): the parsed output and the raw generated text.
    """

    # Create the pipeline and run the prompt
    sequences: List[Any] = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        do_sample=True,
        return_full_text=False,
    )

    answers = [parse_generated_text(seq) for seq in sequences]
    if verbose:
        msg.divider(show=verbose)
        msg.info(f"PROMPT: {prompt}", show=verbose)
        for ans, seq in zip(answers, sequences):
            msg.text(ans, seq, show=verbose)

    return answers, sequences


def parse_generated_text(text: str, default: bool = False) -> bool:
    """Perform best-effort parsing on the text to return a single y/n answer"""
    if text.lower() == "yes" or "yes" in text.lower():
        return True
    elif text.lower() == "no" or "no" in text.lower():
        return False
    else:
        return default
