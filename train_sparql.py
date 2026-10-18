#!/usr/bin/env python3
"""
Train a small Qwen3 model to generate SPARQL queries (Text2SPARQL).

Target hardware: Intel Arc B65 (XPU) — bf16 supported in hardware.
Stack: PEFT + TRL on native PyTorch XPU (>=2.8). MLflow for experiment tracking.

IMPORTANT: Intel Extension for PyTorch (IPEX) is EOL (archived Mar 2026).
XPU support is now upstream in PyTorch itself — no IPEX needed.

NOT usable on Intel Arc (all CUDA-only):
- Unsloth (Triton/CUDA kernels)
- bitsandbytes 4-bit QLoRA
- paged_adamw_8bit optimizer
- IPEX (discontinued — use native torch.xpu instead)

Install:
    pip install torch --index-url https://download.pytorch.org/whl/xpu
    pip install transformers datasets peft trl accelerate mlflow

Usage:
    python train_sparql.py
    mlflow ui          # then open http://localhost:5000 to view runs
"""

import torch, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import mlflow

from yago_prompts import (YAGO_NS, PREFIXES, KNOWN_NS, SYSTEM_PROMPT,
                          EXTRACT_SYSTEM, QUESTION_GEN_SYSTEM, ANSWER_SYSTEM,
                          PROPERTY_HINTS, strip_reasoning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
#MODEL_NAME = "Qwen/Qwen3.5-4B-Base"        # or Qwen/Qwen3-1.7B for max speed
MODEL_NAME = "Qwen/Qwen3.5-0.8B"        # or Qwen/Qwen3-1.7B for max speed
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "output/qwen3.5-0.8b-sparql"
EVAL_PATH = "data/eval.jsonl"       # optional, used for post-train parse check

LORA_R = 64                          # rank; 16-64. 16 is a fine start on 4B
LORA_ALPHA = 128                     # 2x rank
MAX_SEQ_LEN = 2048                   # bump to 4096 if you have the headroom
BATCH_SIZE = 16                      # safe on 32GB; try 4-8
GRAD_ACCUM = 2                       # effective batch = BATCH_SIZE * GRAD_ACCUM
EPOCHS = 3
LR = 3e-4                           # Would be 1e-5ish for full SFT, but LoRA likes larger LRs.

# MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"   # local, no server needed
MLFLOW_EXPERIMENT = "qwen3.5-0.8b-sparql"

# ---------------------------------------------------------------------------
# Load model — bf16 on XPU (Arc has native bf16; no quantization needed)
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to("xpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer_no_pad = AutoTokenizer.from_pretrained(MODEL_NAME, add_bos_token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
if False:
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()   # expect ~0.5-2% trainable

# Note: no ipex.optimize() — IPEX is EOL. Native torch.xpu handles bf16.
# If you want graph-level speedups, try torch.compile(model, backend="xpu").

# ---------------------------------------------------------------------------
# Data: JSONL, one {"messages": [...]} per line (see data_format.md)
# ---------------------------------------------------------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Only compute loss on assistant turns (don't waste capacity predicting the prompt)
response_template = "<|im_start|>assistant"
#collator = DataCollatorForCompletionOnlyLM(
    #response_template=response_template, tokenizer=tokenizer
#)

# ---------------------------------------------------------------------------
# Trainer — bf16=True (Arc native), plain adamw (bitsandbytes is CUDA-only)
# report_to="mlflow" makes HF auto-log train loss every logging step.
# ---------------------------------------------------------------------------
#FIXME training_args = TrainingArguments(
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=1,
    logging_first_step=True,
    save_strategy="epoch",
    bf16=True,                          # Intel Arc native
    fp16=False,
    gradient_checkpointing=True,        # True to save VRAM from activations.
    packing=True,
    packing_strategy="wrapped",
    #optim="adamw_torch",               # NOT paged_adamw_8bit (CUDA-only)
    optim="adamw_torch_fused",          # NOT paged_adamw_8bit (CUDA-only)
    max_grad_norm=1.0,
    report_to="mlflow",
    seed=42,
    # Doesn't work quite right: torch_compile=True,

    completion_only_loss=True,
    max_length=MAX_SEQ_LEN,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # FIXME data_collator=collator,
    # Qwen3 has a "thinking" mode; force it OFF so the model outputs
    # plain SPARQL, not wrapped in <|thinking_start|> blocks.
    #chat_template_kwargs={"enable_thinking": False},
)

# ---------------------------------------------------------------------------
# MLflow run: log config, train, log artifacts + a quick parse-rate check
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name=f"{MODEL_NAME}-r{LORA_R}") as run:
    # Log hyperparameters (params are immutable once logged)
    mlflow.log_params({
        "base_model": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "epochs": EPOCHS,
        "lr": LR,
        "max_len": MAX_SEQ_LEN,
        "device": "xpu",
    })

    # Train — HF auto-logs train_loss via report_to="mlflow"
    trainer.train()

    # Save adapter, then log it as an artifact
    trainer.save_model(OUTPUT_DIR)
    mlflow.transformers.log_model(transformers_model={"model": trainer.model, "tokenizer": tokenizer_no_pad}, name="model", save_pretrained=False)
    mlflow.log_artifact(__file__, artifact_path="code")

    # Optional: quick post-train parse-rate check on a small eval set
    if EVAL_PATH and os.path.exists(EVAL_PATH):
        import json
        from validate import strip_reasoning, is_parsable

        rows = [json.loads(l) for l in open(EVAL_PATH) if l.strip()][:50]
        ok = 0
        for row in rows:
            print(row['question'])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row['question']},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
                #chat_template_kwargs={"enable_thinking": False}
            ).to("xpu")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_SEQ_LEN,
                    temperature=0.01, do_sample=True, top_p=0.9,
                )
            q = strip_reasoning(
                tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            )
            if is_parsable(q):
                ok += 1
        parse_rate = ok / len(rows)
        mlflow.log_metric("parse_rate", parse_rate)
        print(f"Post-train parse rate on {len(rows)} eval rows: {parse_rate:.1%}")

print(f"Done. Run logged to MLflow ({MLFLOW_TRACKING_URI}). Adapter saved to {OUTPUT_DIR}")
print(f"View with: mlflow ui")
