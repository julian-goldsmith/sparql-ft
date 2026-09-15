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

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import mlflow

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.5-4B-Base"        # or Qwen/Qwen3-1.7B for max speed
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "output/qwen3-4b-sparql"
EVAL_PATH = "data/eval.jsonl"       # optional, used for post-train parse check

LORA_R = 16                          # rank; 16-64. 16 is a fine start on 4B
LORA_ALPHA = 32                      # 2x rank
MAX_SEQ_LEN = 2048                   # bump to 4096 if you have the headroom
BATCH_SIZE = 2                       # safe on 32GB; try 4-8
GRAD_ACCUM = 8                       # effective batch = BATCH_SIZE * GRAD_ACCUM
EPOCHS = 3
LR = 2e-4

# MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"   # local, no server needed
MLFLOW_EXPERIMENT = "qwen3.5-4b-sparql"

# ---------------------------------------------------------------------------
# Load model — bf16 on XPU (Arc has native bf16; no quantization needed)
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to("xpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
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
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, tokenizer=tokenizer
)

# ---------------------------------------------------------------------------
# Trainer — bf16=True (Arc native), plain adamw (bitsandbytes is CUDA-only)
# report_to="mlflow" makes HF auto-log train loss every logging step.
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                          # Intel Arc native
    fp16=False,
    gradient_checkpointing=True,
    optim="adamw_torch",                # NOT paged_adamw_8bit (CUDA-only)
    max_grad_norm=1.0,
    report_to="mlflow",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    max_seq_length=MAX_SEQ_LEN,
    data_collator=collator,
    packing=False,
    # Qwen3 has a "thinking" mode; force it OFF so the model outputs
    # plain SPARQL, not wrapped in <|thinking_start|> blocks.
    chat_template_kwargs={"enable_thinking": False},
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
        "max_seq_len": MAX_SEQ_LEN,
        "device": "xpu",
    })

    # Train — HF auto-logs train_loss via report_to="mlflow"
    trainer.train()

    # Save adapter, then log it as an artifact
    trainer.save_model(OUTPUT_DIR)
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="adapter")

    # Optional: quick post-train parse-rate check on a small eval set
    if EVAL_PATH and os.path.exists(EVAL_PATH):
        import json
        from validate import strip_reasoning, is_parsable

        rows = [json.loads(l) for l in open(EVAL_PATH) if l.strip()][:50]
        ok = 0
        for row in rows:
            messages = [
                {"role": "system", "content": (
                    "You are an expert SPARQL generator for the Yago knowledge graph. "
                    "Output ONLY the SPARQL query."
                )},
                {"role": "user", "content": (
                    f"Question: {row['question']}\n\nEntities:\n{row['entities']}"
                    f"\n\nPrefixes:\n{row['prefixes']}"
                )},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", chat_template_kwargs={"enable_thinking": False}
            ).to("xpu")
            with torch.no_grad():
                out = model.generate(
                    inputs, max_new_tokens=512,
                    temperature=0.01, do_sample=True, top_p=0.9,
                )
            q = strip_reasoning(
                tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
            )
            if is_parsable(q):
                ok += 1
        parse_rate = ok / len(rows)
        mlflow.log_metric("parse_rate", parse_rate)
        print(f"Post-train parse rate on {len(rows)} eval rows: {parse_rate:.1%}")

    mlflow.log_artifact(__file__, artifact_path="code")

print(f"Done. Run logged to MLflow ({MLFLOW_TRACKING_URI}). Adapter saved to {OUTPUT_DIR}")
print(f"View with: mlflow ui")
