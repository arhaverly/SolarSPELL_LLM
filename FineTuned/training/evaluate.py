"""Step 5 (optional): score a fine-tuned model against held-out Q&A pairs.

Ported from the evaluation cells of "Model Training File.ipynb" (F1, exact
match, BLEU, ROUGE-L, BERTScore). Fix relative to the notebook: only the newly
generated tokens are scored; the notebook decoded the prompt too, so the
question text inflated every metric.

Usage:
  HF_TOKEN=hf_... python evaluate.py qa_eval.json --adapter lora_llama/final
  python evaluate.py qa_eval.json --model merged/            # merged model
"""

import argparse
import json
import os

import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import BASE_MODEL, alpaca_prompt


def token_f1(pred: str, truth: str) -> float:
    pred_tokens, truth_tokens = pred.lower().split(), truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision, recall = len(common) / len(pred_tokens), len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def load(args):
    token = os.environ.get("HF_TOKEN")
    path = args.model or BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(path, token=token)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto", token=token)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    return tokenizer, model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--adapter", help="LoRA adapter applied to the base model")
    parser.add_argument("--model", help="merged model directory (instead of base + adapter)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    tokenizer, model = load(args)
    examples = json.load(open(args.dataset, encoding="utf-8"))
    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rougeL"])

    f1s, exact, bleus, rouge_ls, predictions, truths = [], [], [], [], [], []
    for example in examples:
        inputs = tokenizer(alpaca_prompt(example["instruction"], example.get("input", "")), return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        truth = example["output"]

        f1s.append(token_f1(prediction, truth))
        exact.append(int(prediction.lower() == truth.strip().lower()))
        bleus.append(sentence_bleu([truth.lower().split()], prediction.lower().split(), smoothing_function=smoothie))
        rouge_ls.append(rouge.score(truth, prediction)["rougeL"].fmeasure)
        predictions.append(prediction)
        truths.append(truth)

    _, _, bert_f1 = bert_score(predictions, truths, lang="en")
    mean = lambda values: sum(values) / len(values)
    print(f"Examples:      {len(examples)}")
    print(f"Token F1:      {mean(f1s):.4f}")
    print(f"Exact match:   {mean(exact):.4f}")
    print(f"BLEU:          {mean(bleus):.4f}")
    print(f"ROUGE-L:       {mean(rouge_ls):.4f}")
    print(f"BERTScore F1:  {bert_f1.mean().item():.4f}")


if __name__ == "__main__":
    main()
