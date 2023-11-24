import itertools
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import typer
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from wasabi import msg

from .datasets import Dataset, get_dataset_reader
from .utils import Split


class LanguageModel(str, Enum):
    bert_uncased = "bert-base-uncased"  # for local testing only
    llama2_together = "togethercomputer/Llama-2-7B-32K"
    llama2_meta = "meta-llama/Llama-2-7b-hf"
    vicuna_thebloke = "TheBloke/vicuna-13b-v1.3.0-GPTQ"
    falcon = "tiiuae/falcon-7b"
    mpt = "mosaicml/mpt-7b"
    mistral = "mistralai/Mistral-7B-v0.1"


def evaluate_llm(
    # fmt: off
    dataset_name: Dataset = typer.Argument(Dataset.piqa, help="Dataset to evaluate."), 
    model: LanguageModel = typer.Argument(LanguageModel.bert_uncased, help="Language model to use for evaluation."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to save computed metrics (metrics.json) and raw outputs (outputs.jsonl)."),
    batch_size: int = typer.Option(16, help="Batch size during inference. Higher values may speed things up, but might require more compute."),
    split: Split = typer.Option(Split.validation, help="Dataset split to convert."),
    seed: int = typer.Option(42, help="Set the random seed."),
    verbose: bool = typer.Option(False, help="Show additional information."),
    # fmt: on
):
    """Evaluate a language model on a given dataset"""
    torch.manual_seed(seed)

    # Load the dataset
    msg.info(f"Downloading dataset '{dataset_name.value}' ({split.value})")
    dataset = get_dataset_reader(dataset_name)
    examples = load_dataset(dataset_name.value, dataset.hf_config, split=split.value)

    # Initialize the model and tokenizer
    msg.info(f"Initializing the model and tokenizer using '{model.value}'")
    tokenizer = AutoTokenizer.from_pretrained(model.value)
    model = AutoModelForCausalLM.from_pretrained(model.value)

    # Perform inference for each batch
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    batch_accuracy = []
    for batch in tqdm(minibatch(examples, size=batch_size)):
        for eg in batch:
            # Create a dictionary that will keep all the fields we need.
            # No need to be fancy. Copy allenai/data-efficient-finetuning
            # implementation and variable names for easy comparison.
            ranked_clf_instance: Dict[str, Any] = {
                # fmt: off
                # The formatted prompt
                # (e.g., "Question: May went to the store. She went to where?\n Answer: )"
                "prompt_and_input_pretokenized": dataset.get_prompt(eg),
                "prompt_and_input": tokenizer(dataset.get_prompt(eg), return_tensors="pt"),
                # The options for multi-choice and the target for sentence-completion
                # (e.g., ["store", "school", "sea"])
                "answer_options_pretokenized": dataset.get_targets(eg),
                "answer_options": tokenizer(dataset.get_targets(eg), return_tensors="pt"),
                # The label / correct answer
                "correct_answer_index_value": eg.get("label"),
                "correct_answer_index": torch.LongTensor([eg.get("label")])
                # fmt: on
            }

            answer_option_ids = ranked_clf_instance["answer_options"].input_ids
            answer_option_ids[answer_option_ids == 0] = -100
            correct_answer_ids = answer_option_ids[
                torch.arange(answer_option_ids.shape[0]),
                ranked_clf_instance["correct_answer_index"].squeeze(),
            ]
            output = model.forward(
                input_ids=ranked_clf_instance["prompt_and_input"].input_ids,
                labels=correct_answer_ids,
                return_dict=True,
            )

            breakpoint()

            # We will run the model for all options.
            # Instead of fwd-passing them one by one, we will pass them as a batch
            prompts: List[str] = dataset.get_targets(eg)
            inputs = tokenizer(prompts, return_tensors="pt")
            outputs = model.forward(
                inputs.input_ids,
                labels=option_ids,
                return_dict=True,
            )

            # Compute the loss for both options using cross-entropy
            logits = outputs.logits.detach()
            losses = loss_fn(logits.permute([0, 2, 1]), option_ids)
            losses = losses.sum(dim=-1)

            # Compute the accuracy based on the loss
            min_loss = None
            best_option_id = 0
            for j, option_loss in enumerate(losses):
                if min_loss is None or min_loss > option_loss:
                    min_loss = option_loss
                    best_option_id = j
            compute_accuracy(correct_option_id == best_option_id)


def minibatch(items: Iterable[Any], size: int) -> Iterable[Any]:
    """Iterate over batches of items

    size (int): the size of the batch
    RETURNS (Iterable[Any]): iterable containing items on a given batch.
    """
    size = itertools.repeat(size)
    items = iter(items)
    while True:
        batch_size = next(size)
        batch = list(itertools.islice(items, int(batch_size)))
        if len(batch) == 0:
            break
        yield list(batch)


if __name__ == "__main__":
    typer.run(evaluate_llm)
