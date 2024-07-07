from pathlib import Path

import torch
import typer
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from wasabi import msg


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

    msg.info(f"Loading model configuration for {model_name}...")
    attn_implementation = "flash_attention_2" if use_flash_attn else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    msg.info(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, "cebuano", split="train")

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation=grad_acc_steps,
        gradient_checkpointing=use_grad_checkpointing,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=1e-3,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
    )

    peft_config = LoraConfig(
        lora_alpha=32,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=train_max_seq_len,
        tokenizer=tokenizer,
        args=training_arguments,
        formatting_func=formatting_prompts_func,
    )

    trainer.train()

    msg.info("Saving adapter model to disk...")
    trainer.model.save_pretrained(save_directory="aya-qlora-ceb")
    model.config.use_cache = True
    model.eval()


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["inputs"])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{example['inputs'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['targets'][i]}"
        output_texts.append(text)
    return output_texts


if __name__ == "__main__":
    typer.run(main)
