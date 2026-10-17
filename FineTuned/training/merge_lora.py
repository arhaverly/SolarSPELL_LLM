"""Merge a LoRA adapter into Llama 3.2 1B Instruct and save a full HF model.

Works with adapters from train_lora.py and with the original nx-sol-ai
checkpoint (e.g. lora_llama/checkpoint-4000). The base is loaded in fp16
rather than 8-bit, since quantized weights can't be merged into.

Usage: HF_TOKEN=hf_... python merge_lora.py lora_llama/final merged/
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter")
    parser.add_argument("output")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, token=token)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL, token=token).save_pretrained(args.output)
    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
