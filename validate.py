#!/usr/bin/env python3
"""
Validate fine-tuned SPARQL output: parseability + execution against Fuseki.

Target hardware: Intel Arc B65 (XPU, native PyTorch >=2.8 — IPEX is EOL).
Strips reasoning/thinking blocks BEFORE validating so Qwen3's thinking mode
can't pollute the SPARQL.

Usage:
    python validate.py --model output/qwen3-4b-sparql --eval data/eval.jsonl

Prints: parse rate, execution rate, exact-match rate.
"""

import argparse
import json

from yago_prompts import SYSTEM_PROMPT, strip_reasoning

import torch
from rdflib.plugins.sparql import prepareQuery
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B"        # must match the base used for training
FUSEKI_URL = "http://192.168.8.110:3030/yago/sparql"
def generate(model, tokenizer, question):
    # Must match the training prompt EXACTLY (yago_prompts.py is the source
    # of truth) or eval numbers will lie about training quality. Training
    # user turns are the BARE question — no entities/properties/prefixes.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        chat_template_kwargs={"enable_thinking": False},  # Qwen3: no thinking mode
    ).to("xpu")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.01,   # near-greedy: SPARQL wants determinism
            do_sample=True,
            top_p=0.9,
        )
    decoded = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return strip_reasoning(decoded)


def is_parsable(query: str) -> bool:
    try:
        prepareQuery(query)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to saved PEFT adapter")
    ap.add_argument("--eval", required=True, help="eval JSONL path")
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to("xpu")
    model = PeftModel.from_pretrained(base, args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rows = [json.loads(l) for l in open(args.eval) if l.strip()]
    parse_ok = exec_ok = exact_ok = 0

    for row in rows:
        q = generate(model, tokenizer, row["question"])
        if is_parsable(q):
            parse_ok += 1
            # Optional: execute against Fuseki and compare result sets
            # (use SPARQLWrapper or requests POST). Exact-match = gold result set.
            if row.get("gold_query") and q.strip() == row["gold_query"].strip():
                exact_ok += 1

    n = len(rows)
    print(f"parse rate:   {parse_ok}/{n} ({100*parse_ok/n:.1f}%)")
    print(f"exact rate:   {exact_ok}/{n} ({100*exact_ok/n:.1f}%)")
    print("(execution rate needs Fuseki wiring — see FUSEKI_URL)")


if __name__ == "__main__":
    main()