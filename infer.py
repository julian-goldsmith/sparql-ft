#!/usr/bin/env python3
"""Single inference against a logged MLflow model.

Usage (from sparql-finetune/):
    python infer.py "List all subclasses of foaf:Person"       # newest run
    python infer.py --uri runs:/<run_id>/model "question"      # specific run

Requires the run to have a "model" artifact (the mlflow.transformers.log_model
call in train_sparql.py). Loads merged weights — no PEFT assembly needed.
"""
import argparse

import mlflow
import torch

from yago_prompts import SYSTEM_PROMPT
from validate import strip_reasoning

# Quick transformers test cell
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 8192  # SPARQL gen default

BASE_MODEL_NAME = "Qwen/Qwen3.5-4B"        # or Qwen/Qwen3-1.7B for max speed
MODEL = "./output/qwen3.5-4b-sparql/checkpoint-292/"
#MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LORA_R = 32                          # rank; 16-64. 16 is a fine start on 4B
LORA_ALPHA = 64                     # 2x rank

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"Using model {args.model} on device {device}")

    #tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    #model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map=device)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16
    ).to("xpu")
    model = PeftModel.from_pretrained(model, args.model)

    # If Qwen3's thinking mode leaks in, add:
    #   chat_template_kwargs={"enable_thinking": False}
    # strip_reasoning below catches stray <think> blocks anyway.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy: deterministic, best for parse checks
        )
    q, r = strip_reasoning(tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    print('Reasoning')
    print('====================================')
    print(r)
    print()
    print()
    print('Query')
    print('====================================')
    print(q)
    print()
    print()


if __name__ == "__main__":
    main()
