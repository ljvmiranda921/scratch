import os
import re
from pathlib import Path

import bitsandbytes as bnb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
import typer
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TrainingArguments, logging
from trl import SFTTrainer


def main(
    # fmt: off
    output_dir: Path = typer.Argument(..., help="Path to save the finetuned model."),
    dataset_name: str = typer.Option("CohereForAI/aya_collection_language_split", help="Source dataset to use for finetuning."),
    model_name: str = typer.Option("CohereForAI/aya-23-8b", help="Model name to finetune."),
    quantize_4bit: bool = typer.Option(False, "--quantize-4bit", "-q", help="Quantize model to 4-bit when training."),
    use_grad_checkpointing: bool = typer.Option(False, "--use-grad-checkpointing", "-g", help="Use gradient checkpointing during training."),
    train_batch_size: int = typer.Option(2, "--train-batch-size", "-b", help="Set batch size during training"),
    train_max_seq_len: int = typer.Option(512, "--max-seq-len", "-l", help="Set max sequence length when training."),
    use_flash_attn: bool = typer.Option(False, "--use-flash-attn", "-f", help="Use flash attention. Useful for training on a larger machine."),
    grad_acc_steps: int = typer.Option(16, "--grad-acc-steps", "-g", help="Set gradient accumulation steps."),
    # fmt: on
):
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if quantize_4bit
        else None
    )

    attn_implementation = "flash_attention_2" if use_flash_attn else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


if __name__ == "__main__":
    typer.run(main)
