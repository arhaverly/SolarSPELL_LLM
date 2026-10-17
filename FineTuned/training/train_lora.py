"""Step 3: LoRA fine-tune Llama 3.2 1B Instruct on the Alpaca Q&A pairs.

Ported from nx-sol-ai "Model Training File.ipynb", with the same
hyperparameters (r=8, alpha=32, q/v projections, lr 2e-4, 5 epochs, 512
tokens, 8-bit base). Fixes relative to the notebook:
  * the HF token comes from the HF_TOKEN environment variable, not the source
  * padding is masked out of the loss and an explicit EOS is appended, so the
    model learns to stop instead of learning to predict padding
  * uses the Instruct tokenizer that matches the Instruct base model

Needs a CUDA GPU. Llama 3.2 is gated: accept the license on Hugging Face first.

Usage: HF_TOKEN=hf_... python train_lora.py qa_alpaca.json --output lora_llama
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LENGTH = 512


def alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Prompt format the model is trained on; app/llm.py must match it."""
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="Alpaca-format JSON from generate_qa.py")
    parser.add_argument("--output", default="lora_llama")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        token=token,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    def tokenize(example):
        text = alpaca_prompt(example["instruction"], example["input"]) + example["output"] + tokenizer.eos_token
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, add_special_tokens=True)
        tokens["labels"] = [
            token_id if mask else -100
            for token_id, mask in zip(tokens["input_ids"], tokens["attention_mask"])
        ]
        return tokens

    dataset = load_dataset("json", data_files=args.dataset)["train"]
    split = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    columns = dataset.column_names
    train = split["train"].map(tokenize, remove_columns=columns)
    val = split["test"].map(tokenize, remove_columns=columns)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="steps",
            eval_steps=250,
            logging_steps=100,
            save_steps=500,
            save_total_limit=3,
            num_train_epochs=args.epochs,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            label_names=["labels"],
        ),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print(f"Saved adapter to {args.output}/final")


if __name__ == "__main__":
    main()
